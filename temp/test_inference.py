import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Test inference on SFT merged model.")
    parser.add_argument("--model-dir", default="temp/meeseeks-gemma-merged", help="Merged model directory")
    parser.add_argument("--prompt", default="write a linux command to list files sorted by size", help="Prompt to test")
    args = parser.parse_args()

    print(f"Loading model and tokenizer from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # GTX 1650 fallback to FP32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )

    # Match the SFT training dataset formatting
    messages = [
        {"role": "user", "content": f"---\nUSER REQUEST: {args.prompt}"}
    ]
    
    print("Applying chat template...")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("Prompt:", repr(prompt))

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print("Generating response...")
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("Response:")
    print(response)

if __name__ == "__main__":
    main()
