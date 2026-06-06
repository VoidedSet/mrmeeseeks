import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_dir = "/media/kshayik/New Volume/meeseeks_gemma_merged"
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    messages = [
        {"role": "user", "content": "write a linux command to list files sorted by size"}
    ]
    
    print("Applying chat template...")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("Prompt:", repr(prompt))

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print("Generating response...")
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("Response:")
    print(response)

if __name__ == "__main__":
    main()
