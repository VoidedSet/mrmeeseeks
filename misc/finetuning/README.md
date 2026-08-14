# Mr Meeseeks - Model Fine-Tuning

This directory contains resources, helper scripts, datasets, and Modelfiles for training and fine-tuning custom local LLMs (like FunctionGemma or Qwen) to handle Mr Meeseeks tool-calling schemas and resolve selection ambiguities.

---

## Folder Layout

```
finetuning/
├── data/                  # ChatML/JSONL training and validation datasets
├── Modelfiles/            # Modelfiles for importing custom GGUF weights into Ollama
├── scripts/               # SFT (Supervised Fine-Tuning) and dataset helper scripts
└── finetuning-guide-by-google.md  # Official guide reference for training FunctionGemma
```

---

## Datasets

The fine-tuning process relies on custom ChatML datasets structured with tool definitions:
-   **[meeseeks_yaml_dataset_chatml.jsonl](data/meeseeks_yaml_dataset_chatml.jsonl)**: Training records mapped to the standard OpenAI/ChatML conversation scheme.
-   **[meeseeks_yaml_dataset.jsonl](data/meeseeks_yaml_dataset.jsonl)**: Original YAML-formatted dataset for raw text sequence training.

---

## Modelfiles

To load your fine-tuned model into **Ollama**, configure one of the Modelfiles:
-   **[Modelfile.functiongemma](Modelfiles/Modelfile.functiongemma)**: Templates and parameters optimized for FunctionGemma models.
-   **[Modelfile.qwen3.5](Modelfiles/Modelfile.qwen3.5)**: Templates optimized for Qwen-based architectures.

Load them via:
```bash
ollama create meeseeks-gemma -f finetuning/Modelfiles/Modelfile.gguf
```

---

## Helper Scripts

All Python scripts should be run from the repository root:
-   **[train_meeseeks.py](scripts/train_meeseeks.py)**: Initiates the PEFT/LoRA training loop using Hugging Face's SFTTrainer.
-   **[merge_lora.py](scripts/merge_lora.py)**: Merges trained adapter weights back into base model configurations.
-   **[convert_hf_to_gguf.py](scripts/convert_hf_to_gguf.py)**: Converts PyTorch/Safetensors configurations to GGUF format for Ollama compatibility.

---

## Local CUDA Compatibility Symlinks (Optional)

If your local GPU packages require linking older CUDA runtime library names (e.g. cu11 equivalents) to your virtual environment's site-packages, you can create local symlinks (these are machine-specific and excluded from Git tracking):

```bash
mkdir -p finetuning/cuda_compat
ln -sf $(pwd)/venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcufft.so.12 finetuning/cuda_compat/libcufft.so.11
ln -sf $(pwd)/venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcusolver.so.12 finetuning/cuda_compat/libcusolver.so.11
```
