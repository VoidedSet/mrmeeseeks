#!/usr/bin/env python3
"""
format_dataset_v2.py — Post-processing script to convert flat dataset entries into FunctionGemma native format.
Uses a two-pass lookup to resolve and inject the original tool arguments into finalize entries.
"""

import os
import sys
import json

SYSTEM_PROMPT = """You are Meeseeks, a desktop assistant. Call tools to help the user.
Tools: get_ui_elements, list_at_spi_apps, read_element_text, find_element_by_label, click_at, double_click_at, type_text, key_press, scroll, run_bg_cmd, check_battery, get_active_window, list_open_windows, read_notifications, open_visible_terminal, list_memory_keys, update_memory, fetch_memory, simple_scrape, gui_research, speak, done"""

def main():
    input_path = "finetuning/data/meeseeks_yaml_dataset.jsonl"
    output_path = "finetuning/data/meeseeks_functiongemma_dataset_v2.jsonl"

    if not os.path.exists(input_path):
        print(f"[-] Input file {input_path} does not exist yet. Please wait for the generator to complete.")
        sys.exit(1)

    print(f"[*] Pass 1: Scanning {input_path} to build arguments lookup table...")
    args_lookup = {}
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "initiate":
                user_content = entry.get("user_content")
                tool_args = entry.get("tool_arguments")
                if user_content and tool_args:
                    args_lookup[user_content] = tool_args

    print(f"[+] Loaded {len(args_lookup)} query-arguments mappings.")

    print(f"[*] Pass 2: Formatting entries into FunctionGemma native format...")
    formatted_entries = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            entry_type = row.get("type")
            user_content = row.get("user_content")
            
            prompt = f"<bos><start_of_turn>developer\n{SYSTEM_PROMPT}<end_of_turn>\n<start_of_turn>user\n---\nUSER REQUEST: {user_content}<end_of_turn>\n<start_of_turn>model\n"
            
            if entry_type == "initiate":
                tool_name = row.get("tool_name")
                tool_args = row.get("tool_arguments", {})
                args_json = json.dumps({k: v for k, v in tool_args.items() if v is not None})
                target = f"<start_function_call>call:{tool_name}{{args:<escape>{args_json}<escape>}}<end_function_call>"
            elif entry_type == "finalize":
                initial_tool_name = row.get("initial_tool_name")
                
                # Check for arguments in the entry, fallback to lookup using the user_content key
                initial_args = row.get("initial_tool_arguments")
                if not initial_args:
                    initial_args = args_lookup.get(user_content, {})
                    
                args_json = json.dumps({k: v for k, v in initial_args.items() if v is not None})
                tool_resp = json.dumps(row.get("tool_output", ""))
                speech = row.get("speech", "")
                
                target = (
                    f"<start_function_call>call:{initial_tool_name}{{args:<escape>{args_json}<escape>}}<end_function_call>"
                    f"<start_function_response>response:{initial_tool_name}{{value:<escape>{tool_resp}<escape>}}<end_function_response>"
                    f"<start_function_call>call:done{{speech:<escape>{speech}<escape>}}<end_function_call>"
                )
            else:  # no_call / conversational
                speech = row.get("speech", "") or row.get("final_speech", "")
                target = f"{speech}<end_of_turn>"
                
            formatted_entries.append({"text": prompt + target})

    print(f"[*] Writing {len(formatted_entries)} pre-formatted entries to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in formatted_entries:
            f.write(json.dumps(entry) + "\n")

    print("[+] Done! Dataset formatted successfully.")
    
    # Locate and print a preview of the first formatted "finalize" entry to confirm args lookup worked
    print(f"\n[+] Preview of the first formatted 'finalize' entry with restored arguments:")
    for entry in formatted_entries:
        if "<start_function_response>" in entry["text"]:
            # Splitting to make the output easier to read in logs
            parts = entry["text"].split("<start_of_turn>model\n")
            print(f"--- PROMPT ---\n{parts[0]}<start_of_turn>model\n")
            print(f"--- TARGET TARGET ---\n{parts[1]}")
            break

if __name__ == "__main__":
    main()
