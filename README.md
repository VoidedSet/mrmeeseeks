# Mr Meeseeks - Local AI OS Companion

Mr Meeseeks is a persistent, proactive OS-level AI companion that lives alongside you on Ubuntu. It listens for your voice commands, monitors system events (filesystem, processes, network, UI accessibility DOM), speaks answers, and can interact directly with your system using keyboard, mouse, or terminal execution.

It is built to feel collaborative, showing its own cursor overlay (overlay UI) to work alongside you rather than stealing your cursor.

## Tech Stack
- **AI Brain**: Qwen2.5:3b (via Ollama) or Groq API (ReAct routing loop).
- **Voice (I/O)**: Kokoro TTS (local text-to-speech) and faster-whisper (local speech-to-text) with Vosk or Porcupine wake-word detection.
- **Perception**: AT-SPI2 Linux Accessibility DOM (no heavy screenshot/CV overhead).
- **Interaction**: PyAutoGUI & xdotool (keyboard/mouse simulation) + transparent Qt6 overlays.
- **Listeners**: evdev, inotify, process, and network monitors mapping system event triggers directly to the brain.

---

## Directory Structure
```
mr-meeseeks/
├── core/                  # Main loop, LLM provider registry, state machine, D-Bus/IPC message bus
├── agents/                # Coordinator, Memory, SysAdmin, Eyes, Hands, Web, and Voice agents
├── kernel/                # OS kernel event listener hooks & app state trackers
├── safety/                # Command safety gates and destructive commands blacklist
├── memory/                # Lightweight persistent RAG store (JSON files)
├── logs/                  # Logs directory (outputs/ and raw/ are gitignored, finetune/ is tracked)
├── tests/                 # Diagnostic and verification tests for developers
├── finetuning/            # Model fine-tuning datasets, Modelfiles, and SFT scripts
├── main.py                # Entrypoint. Boots agents, GUI panel / CLI REPL, and kernel listeners
└── requirements.txt       # Core project dependencies
```

---

## Getting Started

### 1. Installation & Setup

Ensure you are running **Ubuntu** (tested on Ubuntu 22.04+). Install dependencies:

```bash
# System dependencies
sudo apt install xdotool at-spi2-core python3-atspi libx11-dev libcairo2-dev libxcomposite-dev

# Virtual Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and fill in local or cloud providers:
```bash
cp .env.example .env
```

### 3. Run Mr Meeseeks
To run the standard **GUI version** (status overlays, system tray, hotkeys):
```bash
python main.py
```

To run the **interactive CLI REPL** with local voice TTS/STT:
```bash
python main.py --cli
```

---

## Verification & Testing
Check out [tests/README.md](tests/README.md) for running diagnostic test scripts.

## Model Fine-Tuning
Check out [finetuning/README.md](finetuning/README.md) for details on training custom models with your own tools.
