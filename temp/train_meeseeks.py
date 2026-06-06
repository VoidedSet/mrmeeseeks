# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers",
#     "peft",
#     "trl",
#     "datasets",
#     "accelerate",
# ]
# ///
"""
Fine-tuning script for training Mr Meeseeks (FunctionGemma-270m-it) on custom dataset.
Supports lightweight LoRA/PEFT training which runs comfortably in under 2GB VRAM on GTX 1650.

Usage:
  python temp/train_meeseeks.py --epochs 5 --batch-size 4
"""

import os
import sys
import json
import argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# Try to import SFTConfig from TRL (newer versions), fallback to standard SFTTrainer arguments
try:
    from trl import SFTTrainer, SFTConfig
    HAS_SFT_CONFIG = True
except ImportError:
    from trl import SFTTrainer
    HAS_SFT_CONFIG = False

def load_and_merge_datasets(files: list[str]) -> Dataset:
    records = []
    for fpath in files:
        if not os.path.exists(fpath):
            print(f"Dataset file not found: {fpath}, skipping.")
            continue
        print(f"Loading dataset: {fpath}")
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                    
    if not records:
        raise ValueError("No records loaded. Make sure the dataset files exist and are not empty.")
    
    print(f"Loaded a total of {len(records)} training records.")
    return Dataset.from_list(records)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune FunctionGemma for Mr Meeseeks.")
    parser.add_argument("--model-id", default="google/functiongemma-270m-it", help="Base model ID on Hugging Face")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (higher for LoRA)")
    parser.add_argument("--output-dir", default="temp/meeseeks-functiongemma-ft", help="Checkpoint output directory")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA and perform full fine-tuning (not recommended for 4GB VRAM)")
    args = parser.parse_args()

    # Define dataset files to merge
    dataset_files = [
        "temp/meeseeks-finetune-dataset.jsonl",
        "temp/meeseeks-finetune-dataset-extended.jsonl"
    ]
    
    # 1. Load and prepare dataset
    raw_dataset = load_and_merge_datasets(dataset_files)
    
    # 2. Load tokenizer and model
    print(f"Loading tokenizer and model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # GTX 1650 doesn't support bfloat16 natively (runs on slow emulation), so we use float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="eager"
    )
    
    # 3. Format dataset using FunctionGemma's chat template
    print("Formatting dataset with chat templates...")
    def format_chat_template(sample):
        # Apply the chat template containing developer, user, assistant, and tool declarations
        text = tokenizer.apply_chat_template(
            sample["messages"],
            tools=sample.get("tools"),
            add_generation_prompt=False,
            tokenize=False
        )
        return {"text": text}
    
    formatted_dataset = raw_dataset.map(format_chat_template)
    
    # Train/Test Split
    split_dataset = formatted_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    print(f"Train split size: {len(split_dataset['train'])}")
    print(f"Val split size: {len(split_dataset['test'])}")
    
    # 4. Configure PEFT / LoRA (Required for GTX 1650 4GB VRAM)
    if not args.no_lora:
        print("Configuring PEFT (LoRA)...")
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
    else:
        print("WARNING: Performing full fine-tuning. This might exceed 4GB VRAM.")

    # 5. Define Training Arguments
    print("Setting up trainer...")
    training_kwargs = {
        "output_dir": args.output_dir,
        "max_seq_length": 512,  # Limit seq length to save VRAM
        "packing": False,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_checkpointing": True,  # Critical to save activation VRAM
        "optim": "adamw_torch_fused" if device == "cuda" else "adamw_torch",
        "logging_steps": 5,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": args.lr,
        "fp16": True if dtype == torch.float16 else False,
        "bf16": False,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "report_to": "none"
    }

    if HAS_SFT_CONFIG:
        trainer_args = SFTConfig(**training_kwargs)
    else:
        trainer_args = TrainingArguments(**training_kwargs)

    # 6. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        dataset_text_field="text",
        processing_class=tokenizer,
    )
    
    # 7. Start Training
    print("Starting training...")
    trainer.train()
    
    # 8. Save the model
    print(f"Saving fine-tuned model weights to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
