# %% [markdown] {"id":"1043dad9","_kg_hide-output":true}
# ##### Copyright 2025 Google LLC.

# %% [code]
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% [markdown] {"id":"de5da83d"}
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://ai.google.dev/gemma/docs/functiongemma/finetuning-with-functiongemma"><img src="https://ai.google.dev/static/site-assets/images/docs/notebook-site-button.png" height="32" width="32" />View on ai.google.dev</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemma/docs/functiongemma/finetuning-with-functiongemma.ipynb""><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/google/generative-ai-docs/blob/main/site/en/gemma/docs/functiongemma/finetuning-with-functiongemma.ipynb"><img src="https://www.kaggle.com/static/images/logos/kaggle-logo-transparent-300.png" height="32" width="70"/>Run in Kaggle</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://console.cloud.google.com/vertex-ai/colab/import/https%3A%2F%2Fraw.githubusercontent.com%2Fgoogle%2Fgenerative-ai-docs%2Fmain%2Fsite%2Fen%2Fgemma%2Fdocs%2Ffunctiongemma%2Ffinetuning-with-functiongemma.ipynb"><img src="https://ai.google.dev/images/cloud-icon.svg" width="40" />Open in Vertex AI</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/google/generative-ai-docs/blob/main/site/en/gemma/docs/functiongemma/finetuning-with-functiongemma.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>

# %% [markdown] {"id":"64ce3b40"}
# # Fine-tuning with FunctionGemma

# %% [markdown] {"id":"297c692d"}
# This guide demonstrates how to fine-tune FunctionGemma for tool calling.
# 
# While FunctionGemma is natively capable of calling tools. But true capability comes from two distinct skills: the mechanical knowledge of how to use a tool (syntax) and the cognitive ability to interpret *why* and *when* to use it (intent).
# 
# Models, especially smaller ones, have fewer parameters available to retain complex intent understanding. This is why we need to fine-tune them
# 
# Common use cases for fine-tuning tool calling include:
# 
# - **Model Distillation**: Generating synthetic training data with a larger model and fine-tuning a smaller model to replicate the specific workflow efficiently.
# - **Handling Non-Standard Schemas**: Overcoming base model struggles with legacy, highly complex data structures or proprietary format not found in public data, such as handling [domain-specific mobile actions](https://ai.google.dev/gemma/docs/mobile-actions).
# - **Optimizing Context Usage**: "Baking" tool definitions into the model's weights. This allows you to use shorthand descriptions in your prompts, freeing up the context window for the actual conversation.
# - **Resolving Selection Ambiguity**: Biasing the model toward specific enterprise policies, such as prioritizing an internal knowledge base over an external search engine.
# 
# In this example, we will focus specifically on managing tool selection ambiguity.

# %% [markdown] {"id":"b65e9a4e"}
# ## Setup development environment
# 
# The first step is to install Hugging Face Libraries, including TRL, and datasets to fine-tune open model, including different RLHF and alignment techniques.

# %% [code] {"_kg_hide-output":true}
# Install Pytorch & other libraries
%pip install torch tensorboard

# Install Hugging Face libraries
%pip install transformers datasets accelerate evaluate trl protobuf sentencepiece

# COMMENT IN: if you are running on a GPU that supports BF16 data type and flash attn, such as NVIDIA L4 or NVIDIA A100
#% pip install flash-attn

# %% [code] {"id":"589cca76","outputId":"83908341-eb33-46ae-dd91-82baaa8a4dd2","execution":{"iopub.status.busy":"2025-12-18T13:40:10.141629Z","iopub.execute_input":"2025-12-18T13:40:10.14189Z","iopub.status.idle":"2025-12-18T13:40:10.605895Z","shell.execute_reply.started":"2025-12-18T13:40:10.141862Z","shell.execute_reply":"2025-12-18T13:40:10.605245Z"},"_kg_hide-output":true}
import os
# Prevent VRAM fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import kagglehub
checkpoint_dir = "/kaggle/working/"
base_model = kagglehub.model_download("google/functiongemma/transformers/functiongemma-270m-it")
learning_rate = 2e-4 #@param {type:"number"}

# %% [markdown] {"id":"ec79a57b"}
# ## Prepare the fine-tuning dataset
# 
# You will use the following example dataset, which contains sample conversations requiring a choice between two tools: `search_knowledge_base` and `search_google`.
# 

# %% [code] {"id":"d77b7883","execution":{"iopub.status.busy":"2025-12-18T13:40:10.606825Z","iopub.execute_input":"2025-12-18T13:40:10.607115Z","iopub.status.idle":"2025-12-18T13:40:10.618303Z","shell.execute_reply.started":"2025-12-18T13:40:10.607078Z","shell.execute_reply":"2025-12-18T13:40:10.617325Z"}}
import json
from datasets import Dataset

# Load the custom Meeseeks ChatML dataset
dataset_path = "/kaggle/input/datasets/voidedset/meeseeks/meeseeks_yaml_dataset_chatml.jsonl"
print(f"[*] Loading dataset from: {dataset_path}")

records = []
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
print(f"[+] Loaded {len(records)} training records.")

dataset = Dataset.from_list(records)

# Split dataset into 90% training samples and 10% test samples
dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
print(f"Train split size: {len(dataset['train'])}")
print(f"Val split size: {len(dataset['test'])}")

# %% [markdown] {"id":"a35e812d"}
# **Important Note on Dataset Distribution**
# 
# When using `shuffle=False` to your own custom datasets, ensure your source data is pre-mixed. If the distribution is unknown or sorted, you should use `shuffle=True` to ensure the model learns a balanced representation of all tools during training.
# 

# %% [markdown] {"id":"0d01561b"}
# ## Fine-tune FunctionGemma using TRL and the SFTTrainer
# 
# You are now ready to fine-tune your model. Hugging Face TRL [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) makes it straightforward to supervise fine-tune open LLMs. The `SFTTrainer` is a subclass of the `Trainer` from the `transformers` library and supports all the same features,
# 
# The following code loads the FunctionGemma model and tokenizer from Hugging Face.

# %% [code] {"id":"aae37bdc","outputId":"d9ea1ee5-b861-405f-a261-061cb2e34c0a","execution":{"iopub.status.busy":"2025-12-18T13:40:18.031148Z","iopub.execute_input":"2025-12-18T13:40:18.031383Z","iopub.status.idle":"2025-12-18T13:40:47.387602Z","shell.execute_reply.started":"2025-12-18T13:40:18.031361Z","shell.execute_reply":"2025-12-18T13:40:47.386898Z"},"_kg_hide-output":true}
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Configure PEFT LoRA Adapter
from peft import LoraConfig, get_peft_model, TaskType
print("[*] Configuring PEFT LoRA adapter...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print(f"Device: {model.device}")
print(f"DType: {model.dtype}")

# Print formatted user prompt
print("--- dataset input ---")
print(json.dumps(dataset["train"][0], indent=2))
debug_msg = tokenizer.apply_chat_template(dataset["train"][0]["messages"], tools=dataset["train"][0].get("tools"), add_generation_prompt=False, tokenize=False)
print("--- Formatted prompt ---")
print(debug_msg)

# %% [markdown] {"id":"3f5da901"}
# ## Before fine-tune
# 
# The output below shows that the out-of-the-box capabilities may not be good enough for this use case.

# %% [code] {"id":"1aea9ba6","outputId":"ab7f75d0-0073-43a6-925e-e816dc5f03d2","execution":{"iopub.status.busy":"2025-12-18T13:40:47.388538Z","iopub.execute_input":"2025-12-18T13:40:47.389148Z","iopub.status.idle":"2025-12-18T13:41:20.022704Z","shell.execute_reply.started":"2025-12-18T13:40:47.389116Z","shell.execute_reply":"2025-12-18T13:41:20.021566Z"}}
def check_success_rate():
  success_count = 0
  for idx, item in enumerate(dataset['test']):
    messages = [
        item["messages"][0],
        item["messages"][1],
    ]

    inputs = tokenizer.apply_chat_template(
        messages, 
        tools=item.get("tools"), 
        add_generation_prompt=True, 
        return_dict=True, 
        return_tensors="pt"
    )

    out = model.generate(
        **inputs.to(model.device), 
        pad_token_id=tokenizer.eos_token_id, 
        max_new_tokens=128,
        do_sample=False  # Greedy decoding: deterministic and avoids NaN sampling crashes
    )
    output = tokenizer.decode(out[0][len(inputs["input_ids"][0]) :], skip_special_tokens=False)

    print(f"{idx+1} Prompt: {item['messages'][1]['content']}")
    print(f"  Output: {output}")

    # Find expected tool call name(s)
    expected_tools = []
    for msg in item['messages']:
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            for tc in msg['tool_calls']:
                expected_tools.append(tc['function']['name'])

    correct = False
    if expected_tools:
        # Check if the output contains the expected tool name(s)
        for tname in expected_tools:
            if f"call:{tname}" in output:
                correct = True
    else:
        # If no tool calls were expected, the assistant output should be plain text
        if "call:" not in output:
            correct = True

    if correct:
      print("  `-> ✅ correct!")
      success_count += 1
    else:
      print(f"  -> ❌ wrong (expected: {expected_tools})")

  print(f"Success : {success_count} / {len(dataset['test'])}")

# check_success_rate()  # Commented out to skip check before fine-tuning

# %% [markdown] {"id":"72fea798"}
# ## Training
# 
# Before you can start your training, you need to define the hyperparameters you want to use in a `SFTConfig` instance.

# %% [code] {"id":"893edb1b","execution":{"iopub.status.busy":"2025-12-18T13:41:20.023908Z","iopub.execute_input":"2025-12-18T13:41:20.024222Z","iopub.status.idle":"2025-12-18T13:41:20.115668Z","shell.execute_reply.started":"2025-12-18T13:41:20.024196Z","shell.execute_reply":"2025-12-18T13:41:20.115018Z"},"_kg_hide-output":true}
from trl import SFTConfig

torch_dtype = model.dtype

args = SFTConfig(
    output_dir=checkpoint_dir,              # directory to save and repository id
    max_length=1350,                        # max sequence length for model and packing of the dataset
    packing=False,                          # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=5,                     # number of training epochs
    per_device_train_batch_size=2,          # batch size per device during training
    gradient_accumulation_steps=2,          # Keep effective batch size at 4
    gradient_checkpointing=True,            # Safe from VRAM OOM
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=5,                        # log every 5 steps
    eval_strategy="epoch",                  # evaluate checkpoint every epoch
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=learning_rate,            # learning rate
    fp16=True,                              # use float16 precision
    bf16=False,
    lr_scheduler_type="cosine",             # cosine lr scheduler is best for LoRA
    push_to_hub=False,                      # push model to hub
    report_to="none",                       # no reporting overhead
)

# %% [markdown] {"id":"de13f89f"}
# You now have every building block you need to create your `SFTTrainer` to start the training of your model.

# %% [code] {"id":"38a75701","outputId":"344db792-78b3-4e24-8e36-64bfc2ab8147","execution":{"iopub.status.busy":"2025-12-18T13:41:20.116562Z","iopub.execute_input":"2025-12-18T13:41:20.116841Z","iopub.status.idle":"2025-12-18T13:41:25.803231Z","shell.execute_reply.started":"2025-12-18T13:41:20.116803Z","shell.execute_reply":"2025-12-18T13:41:25.802597Z"}}
from trl import SFTTrainer

# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,
)

# %% [markdown] {"id":"a3f888d6"}
# Start training by calling the `train()` method.

# %% [code] {"id":"4257a651","outputId":"c97197e1-20f3-4f2c-8650-9e8ba0b63038","execution":{"iopub.status.busy":"2025-12-18T13:41:25.804353Z","iopub.execute_input":"2025-12-18T13:41:25.80472Z","iopub.status.idle":"2025-12-18T13:41:47.134046Z","shell.execute_reply.started":"2025-12-18T13:41:25.804678Z","shell.execute_reply":"2025-12-18T13:41:47.133376Z"}}
# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# %% [markdown] {"id":"d2a14291"}
# To plot the training and validation losses, you would typically extract these values from the `TrainerState` object or the logs generated during training.
# 
# Libraries like Matplotlib can then be used to visualize these values over training steps or epochs. The x-asis would represent the training steps or epochs, and the y-axis would represent the corresponding loss values.

# %% [code] {"id":"1b163975","outputId":"74835231-7040-4f1f-def5-0b4922ac85c6","execution":{"iopub.status.busy":"2025-12-18T13:41:47.135699Z","iopub.execute_input":"2025-12-18T13:41:47.136139Z","iopub.status.idle":"2025-12-18T13:41:47.352235Z","shell.execute_reply.started":"2025-12-18T13:41:47.136114Z","shell.execute_reply":"2025-12-18T13:41:47.351599Z"}}
import matplotlib.pyplot as plt

# Access the log history
log_history = trainer.state.log_history

# Extract training / validation loss
train_losses = [log["loss"] for log in log_history if "loss" in log]
epoch_train = [log["epoch"] for log in log_history if "loss" in log]
eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]
epoch_eval = [log["epoch"] for log in log_history if "eval_loss" in log]

# Plot the training loss
plt.plot(epoch_train, train_losses, label="Training Loss")
plt.plot(epoch_eval, eval_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown] {"id":"911e7961"}
# ## Test Model Inference
# 
# After the training is done, you'll want to evaluate and test your model. You can load different samples from the test dataset and evaluate the model on those samples.
# 

# %% [code] {"id":"d688ee11","outputId":"48d09971-4912-473d-e421-0da96b22d731","execution":{"iopub.status.busy":"2025-12-18T13:41:47.353234Z","iopub.execute_input":"2025-12-18T13:41:47.353594Z","iopub.status.idle":"2025-12-18T13:42:07.123157Z","shell.execute_reply.started":"2025-12-18T13:41:47.353554Z","shell.execute_reply":"2025-12-18T13:42:07.122431Z"}}
check_success_rate()

# %% [markdown] {"id":"72c7a051"}
# ## Summary and next steps
# 
# You learned how to fine-tune FunctionGemma to resolve **tool selection ambiguity**, a scenario where a model must choose between overlapping tools (e.g., internal vs. external search) based on specific enterprise policies. By utilizing the **Hugging Face TRL library** and `SFTTrainer`, the tutorial walked through the process of preparing a dataset, configuring hyperparameters, and executing a supervised fine-tuning loop.
# 
# The results illustrate the critical difference between a "capable" base model and a "production-ready" fine-tuned model:
# 
# - **Before Fine-tuning**: The base model struggled to adhere to the specific policy, often failing to call tools or choosing the wrong one, resulting in a low success rate (e.g., 2/20).
# - **After Fine-tuning**: After training for 8 epochs, the model learned to correctly distinguish between queries requiring search_knowledge_base versus search_google, improving the success rate (e.g., 16/20).
# 
# Now that you have a fine-tuned model, consider the following steps to move toward production:
# 
# - **Expand the Dataset**: The current dataset was a small, synthetic split (50/50) used for demonstration. For a robust enterprise application, curate a larger, more diverse dataset that covers edge cases and rare policy exceptions.
# - **Evaluation with RAG**: Integrate the fine-tuned model into a Retrieval Augmented Generation (RAG) pipeline to verify that the `search_knowledge_base` tool calls actually retrieve relevant documents and result in accurate final answers.
# 
# Check out the following docs next:
# 
# - [Full function calling sequence with FunctionGemma](https://ai.google.dev/gemma/docs/functiongemma/full-function-calling-sequence-with-functiongemma)
# - [Finetune FunctionGemma for Mobile Actions](https://github.com/google-gemini/gemma-cookbook/blob/main/FunctionGemma/%5BFunctionGemma%5DFinetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb) in the Gemma Cookbook
# 
