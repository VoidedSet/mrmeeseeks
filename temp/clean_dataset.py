import json
import re

def clean_text(text: str) -> str:
    if not text:
        return ""
    
    # Remove markdown bold/italic asterisks
    text = text.replace("**", "").replace("*", "")
    
    # Remove markdown header markers (e.g. ### Header)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove common emojis and miscellaneous symbols (Unicode ranges)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)  # Emojis outside BMP
    text = re.sub(r'[\u2600-\u27BF]', '', text)          # Miscellaneous symbols / Dingbats (like batteries, lightning bolts)
    
    # Remove smart Chinese/custom quotes and brackets
    text = text.replace("】", "").replace("【", "").replace("”", "").replace("“", "")
    
    # Clean up trailing braces/brackets that might have leaked from malformed parses
    text = text.strip('] }')
    
    # Normalize multiple consecutive spaces
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def clean_file(fpath: str):
    cleaned_records = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            # Clean assistant content
            for msg in record.get("messages", []):
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg["content"] = clean_text(msg["content"])
            cleaned_records.append(record)
            
    with open(fpath, "w", encoding="utf-8") as f:
        for record in cleaned_records:
            f.write(json.dumps(record) + "\n")
            
    print(f"Cleaned {fpath} successfully.")

if __name__ == "__main__":
    clean_file("temp/meeseeks-finetune-dataset.jsonl")
    clean_file("temp/meeseeks-finetune-dataset-extended.jsonl")
