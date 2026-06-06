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

def main():
    parser = argparse.ArgumentParser(description="Fine-tune FunctionGemma for Mr Meeseeks.")
    parser.add_argument("--model-id", default="google/functiongemma-270m-it", help="Base model ID on Hugging Face")
    parser.add_argument("--dataset", default="temp/meeseeks_yaml_dataset_chatml.jsonl", help="Dataset file path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (higher for LoRA)")
    parser.add_argument("--output-dir", default="temp/meeseeks-functiongemma-ft", help="Checkpoint output directory")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA and perform full fine-tuning (not recommended for 4GB VRAM)")
    args = parser.parse_args()

    # Load tokenizer first to format records
    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Load and merge dataset files
    dataset_files = [args.dataset]
    
    records = []
    for fpath in dataset_files:
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

    # Pre-format datasets using chat templates in python to avoid PyArrow nested schemas bugs
    print("Formatting dataset with chat templates...")
    formatted_records = []
    for r in records:
        try:
            text = tokenizer.apply_chat_template(
                r["messages"],
                tools=r.get("tools"),
                add_generation_prompt=False,
                tokenize=False
            )
            formatted_records.append({"text": text})
        except Exception as e:
            print(f"Skipping a malformed record. Error: {e}")

    # Build HF dataset from simple flat string records
    formatted_dataset = Dataset.from_list(formatted_records)
    
    # Train/Test Split
    split_dataset = formatted_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    print(f"Train split size: {len(split_dataset['train'])}")
    print(f"Val split size: {len(split_dataset['test'])}")

    # Load model
    print(f"Loading model: {args.model_id}")
    
    # GTX 1650 lacks full FP16 support and can suffer from NaN gradient overflows, so we train in FP32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="eager"
    )

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
    grad_accum = max(1, 4 // args.batch_size)
    print(f"Using batch size: {args.batch_size}, gradient accumulation steps: {grad_accum}")
    training_kwargs = {
        "output_dir": args.output_dir,
        "packing": False,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": grad_accum,
        "gradient_checkpointing": True,  # Critical to save activation VRAM
        "optim": "adamw_torch_fused" if device == "cuda" else "adamw_torch",
        "logging_steps": 5,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": args.lr,
        "fp16": False,
        "bf16": False,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "report_to": "none"
    }

    sft_extra_kwargs = {}
    if HAS_SFT_CONFIG:
        training_kwargs["max_length"] = 512
        training_kwargs["dataset_text_field"] = "text"
        trainer_args = SFTConfig(**training_kwargs)
    else:
        # Standard TrainingArguments don't accept max_seq_length, SFTTrainer accepts it directly
        trainer_args = TrainingArguments(**training_kwargs)
        sft_extra_kwargs["max_seq_length"] = 512
        sft_extra_kwargs["dataset_text_field"] = "text"

    # 6. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        processing_class=tokenizer,
        **sft_extra_kwargs
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
