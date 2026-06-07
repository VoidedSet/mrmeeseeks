import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter weights into base model.")
    parser.add_argument("--base-model", default="google/functiongemma-270m-it", help="Base model ID or path")
    parser.add_argument("--adapter-dir", default="temp/meeseeks-functiongemma-ft", help="LoRA adapter checkpoint directory")
    parser.add_argument("--output-dir", default="temp/meeseeks-gemma-merged", help="Output directory for merged model")
    args = parser.parse_args()

    base_model_id = args.base_model
    adapter_dir = args.adapter_dir
    output_dir = args.output_dir

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    print("Loading adapter weights...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Merging weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)

    print("Model successfully merged and saved!")

if __name__ == "__main__":
    main()
