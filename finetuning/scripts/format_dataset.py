#!/usr/bin/env python3
"""
format_dataset.py — Post-processing script to convert flat dataset entries into ChatML format.
"""

import os
import sys
import json

# Add parent directory to sys.path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temp.generate_dataset_yaml import convert_flat_to_chatml

def main():
    input_path = "temp/meeseeks_yaml_dataset.jsonl"
    output_path = "temp/meeseeks_yaml_dataset_chatml.jsonl"

    if not os.path.exists(input_path):
        print(f"[-] Input file {input_path} does not exist yet. Please wait for the generator to complete.")
        sys.exit(1)

    print(f"[*] Reading flat entries from {input_path}...")
    chatml_entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            chatml_entry = convert_flat_to_chatml(entry)
            chatml_entries.append(chatml_entry)

    print(f"[*] Writing {len(chatml_entries)} ChatML entries to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in chatml_entries:
            f.write(json.dumps(entry) + "\n")

    print("[+] Done! Dataset formatted successfully.")
    print(f"[+] Preview of the first entry:")
    if chatml_entries:
        print(json.dumps(chatml_entries[0], indent=2))

if __name__ == "__main__":
    main()
