import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    base_model_id = "google/functiongemma-270m-it"
    adapter_dir = "/media/kshayik/New Volume/meeseeks_training_out"
    output_dir = "/media/kshayik/New Volume/meeseeks_gemma_merged"

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
