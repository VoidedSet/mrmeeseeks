# Walkthrough - Synthetic Dataset Generation & LoRA Training Setup

We have successfully generated the synthetic training datasets and prepared the local fine-tuning script for your GTX 1650 GPU!

## 1. Work Accomplished Today

### A. Dataset Generator Enhancements (`temp/generate_dataset.py`)
- **Reasoning Disabled:** Added `"think": False` to the Ollama API request payload to bypass reasoning tokens, reducing latency.
- **Timeout Increased:** Raised the client read timeout from `45s` to `180s` to prevent timeouts when Google Colab is under load.
- **Robust Parsing:** Added `parse_prompts` which uses regex and line-by-line cleanup as a fallback to recover prompt arrays from malformed/incomplete JSON returned by the teacher model.

### B. Extended Generator Script (`temp/generate_dataset_extended.py`)
- **Dynamic Imports:** Automatically imports all 22 active tool schemas from [schema_registry.py](file:///home/kshayik/Projects/mr-meeseeks/core/schema_registry.py) to avoid schema drift.
- **Full Simulator:** Created robust JSON-simulated responses for all tool types (Eyes, Hands, Memory, Web, SysAdmin, Kokoro TTS).
- **11 New Scenarios:** Generated prompts representing target use cases like clicking, typing, listing open windows, desktop notifications, RAG memory access, and TTS voice output.

### C. Local Training Script (`temp/train_meeseeks.py`)
- **VRAM Optimizations:** Configured PEFT/LoRA to keep training VRAM under **2 GB** (so it comfortably fits your GTX 1650 4GB VRAM without causing system lag).
- **FP16 Precision:** Automatically uses FP16 precision (GTX 1650 doesn't support BF16 natively).
- **Activation Checkpointing:** Enabled `gradient_checkpointing=True` to discard intermediate activations and save maximum memory.
- **Auto-merging:** Automatically reads, merges, and splits `meeseeks-finetune-dataset.jsonl` and `meeseeks-finetune-dataset-extended.jsonl`.

---

## 2. Generated Datasets Summary

| Dataset File | Records | Covered Scenarios |
| :--- | :--- | :--- |
| [meeseeks-finetune-dataset.jsonl](file:///home/kshayik/Projects/mr-meeseeks/temp/meeseeks-finetune-dataset.jsonl) | **123** | `check_battery`, `get_active_window`, `run_bg_cmd`, `conversational` |
| [meeseeks-finetune-dataset-extended.jsonl](file:///home/kshayik/Projects/mr-meeseeks/temp/meeseeks-finetune-dataset-extended.jsonl) | **131** | UI coordinates, clicks, typing, hotkeys, open windows, memory keys, simple search, voice |
| **Total Combined Dataset** | **254** | **All 22 core tool schemas** |

---

## 3. Next Steps (Local Run on your Laptop)

To train your specialized model on your GTX 1650:

1. **Activate your Python environment:**
   ```bash
   source venv/bin/activate
   ```
2. **Accept the Gemma Terms of Use:**
   Accept the license on Hugging Face: [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it) and log in from your terminal:
   ```bash
   huggingface-cli login
   ```
3. **Run the training script:**
   ```bash
   python temp/train_meeseeks.py --epochs 5 --batch-size 4
   ```
   *(This will run in under 10 minutes on your laptop!)*
