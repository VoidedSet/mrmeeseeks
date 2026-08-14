#!/usr/bin/env python3
"""
format_dataset.py — Post-processing script to convert flat dataset entries into FunctionGemma native format.
"""

import os
import sys
import json

SYSTEM_PROMPT = """You are Meeseeks, a desktop assistant. Call tools to help the user.
Tools: get_ui_elements, list_at_spi_apps, read_element_text, find_element_by_label, click_at, double_click_at, type_text, key_press, scroll, run_bg_cmd, check_battery, get_active_window, list_open_windows, read_notifications, open_visible_terminal, list_memory_keys, update_memory, fetch_memory, simple_scrape, gui_research, speak, done"""

def format_record(row):
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
        initial_args = row.get("initial_tool_arguments", {})
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
        
    return {"text": prompt + target}

def main():
    input_path = "finetuning/data/meeseeks_yaml_dataset.jsonl"
    output_path = "finetuning/data/meeseeks_functiongemma_dataset.jsonl"

    if not os.path.exists(input_path):
        print(f"[-] Input file {input_path} does not exist yet. Please wait for the generator to complete.")
        sys.exit(1)

    print(f"[*] Reading flat entries from {input_path}...")
    formatted_entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            formatted_entry = format_record(entry)
            formatted_entries.append(formatted_entry)

    print(f"[*] Writing {len(formatted_entries)} pre-formatted entries to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in formatted_entries:
            f.write(json.dumps(entry) + "\n")

    print("[+] Done! Dataset formatted successfully.")
    print(f"[+] Preview of the first entry:")
    if formatted_entries:
        print(formatted_entries[0]["text"])

if __name__ == "__main__":
    main()
