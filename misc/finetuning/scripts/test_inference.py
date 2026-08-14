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
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.schema_registry import TOOL_SCHEMAS, REQUIRED_ARGS

    tool_defs = []
    for t_name, t_schema in TOOL_SCHEMAS.items():
        reqs = REQUIRED_ARGS.get(t_name, [])
        params_desc = []
        for p_name, p_desc in t_schema.get("args", {}).items():
            req_suffix = " (required)" if p_name in reqs else ""
            params_desc.append(f"    {p_name}: {p_desc}{req_suffix}")
        
        params_str = "\n" + "\n".join(params_desc) if params_desc else " None"
        tool_defs.append(
            f"- name: {t_name}\n"
            f"  description: {t_schema.get('description', '')}\n"
            f"  params:{params_str}"
        )
    
    dev_content = (
        "You are a model that can do function calling with the following functions\n"
        + "\n".join(tool_defs)
    )

    messages = [
        {"role": "developer", "content": dev_content},
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
