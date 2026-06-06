# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "rich",
# ]
# ///
"""
Synthetic dataset generation script for training Mr Meeseeks.
Queries a large teacher model (e.g. Qwen 9B) running on Google Colab or locally.

Usage:
  python temp/generate_dataset.py --api-url "https://your-colab-tunnel.ngrok-free.app" --model "qwen3.5:9b-q4_K_M"
"""

import os
import sys
import json
import random
import argparse
import httpx
from rich.console import Console
from rich.progress import track
from rich.panel import Panel

console = Console()

# ─── Tool Definitions & Schemas ───────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_battery",
            "description": "Check the laptop's battery percentage and charging status.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_window",
            "description": "Return the title of the application window that is currently in focus on screen.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_bg_cmd",
            "description": "Execute a safe, read-only Linux shell command (ls, cat, pwd, grep, etc.) and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "A valid Linux shell command string using only cat, ls, grep, pwd, echo, df, free, uname."
                    }
                },
                "required": ["cmd"]
            }
        }
    }
]

# ─── Randomized Tool Output Generators (Prevents Memorization) ─────────────────
def generate_random_battery() -> str:
    level = random.randint(5, 100)
    status = random.choice(["Charging", "Discharging", "Full"]) if level < 100 else "Full"
    return json.dumps({"level": f"{level}%", "status": status})

def generate_random_window() -> str:
    windows = [
        "Visual Studio Code", "Mozilla Firefox", "Terminal", "Spotify", 
        "Slack", "Discord", "DocumentViewer", "Files (Nautilus)"
    ]
    return json.dumps({"window": random.choice(windows)})

def generate_random_bg_cmd(cmd: str) -> str:
    cmd = cmd.strip()
    if "ls" in cmd:
        return json.dumps({"output": "agents\ncore\nkernel\nmain.py\ntemp\nvenv", "exit_code": 0})
    if "pwd" in cmd:
        paths = ["/home/kshayik/Projects/mr-meeseeks", "/home/kshayik", "/etc"]
        return json.dumps({"output": random.choice(paths), "exit_code": 0})
    if "uname" in cmd:
        return json.dumps({"output": "Linux hp-victus 6.8.0-generic x86_64", "exit_code": 0})
    return json.dumps({"output": "Command ran successfully.", "exit_code": 0})

def get_simulated_result(tool_name: str, args: dict) -> str:
    if tool_name == "check_battery":
        return generate_random_battery()
    elif tool_name == "get_active_window":
        return generate_random_window()
    elif tool_name == "run_bg_cmd":
        return generate_random_bg_cmd(args.get("cmd", "ls"))
    return json.dumps({"error": "Unknown tool"})

# ─── Robust Prompt Parser ─────────────────────────────────────────────────────
def parse_prompts(content: str) -> list[str]:
    import re
    content = content.strip()
    # Try standard JSON parsing first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [str(x) for x in data if x]
    except Exception:
        pass
        
    # Fallback 1: extract all double-quoted strings
    # Matches double quotes, handles escaped quotes \"
    matches = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', content)
    if matches:
        cleaned = [m.replace('\\"', '"').strip() for m in matches]
        # Filter out empty/whitespace strings
        return [c for c in cleaned if c]
        
    # Fallback 2: line-by-line fallback cleaning
    lines = []
    for line in content.splitlines():
        line = line.strip().strip('[],"“”\'\'【】').strip()
        if line:
            lines.append(line)
    return lines

# ─── API Helper ───────────────────────────────────────────────────────────────
def call_ollama(api_url: str, payload: dict) -> dict:
    url = f"{api_url.rstrip('/')}/api/chat"
    headers = {
        "ngrok-skip-browser-warning": "true",
        "User-Agent": "python-httpx"
    }
    
    # Force disable thinking to reduce latency and prevent timeouts
    payload = payload.copy()
    payload["think"] = False
    
    try:
        with httpx.Client(timeout=180.0, headers=headers) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        console.print(f"[bold red]API HTTP Error ({e.response.status_code}):[/bold red] {e.response.text}")
        return {}
    except Exception as e:
        console.print(f"[bold red]API Error:[/bold red] {e}")
        return {}

# ─── Dataset Generator ────────────────────────────────────────────────────────
def generate_dataset(api_url: str, model_name: str, prompts_count: int):
    console.print(Panel(f"Generating synthetic training dataset using [cyan]{model_name}[/cyan]\nTarget API: [green]{api_url}[/green]", title="Dataset Generator"))
    
    dataset_records = []
    
    # We will generate data for our 3 target tools + conversational/routing queries
    scenarios = [
        {"name": "check_battery", "desc": "Checking laptop battery level and status"},
        {"name": "get_active_window", "desc": "Getting focused window title"},
        {"name": "run_bg_cmd", "desc": "Running filesystem read commands like ls, pwd, cat"},
        {"name": "conversational", "desc": "Greetings, small talk, meta-questions about capabilities"}
    ]
    
    for scenario in scenarios:
        name = scenario["name"]
        desc = scenario["desc"]
        
        console.print(f"\n[bold yellow]Step 1: Generating prompts for {name}...[/bold yellow]")
        
        system_gen_prompt = (
            f"You are a dataset generator assistant. Generate exactly {prompts_count} diverse user prompts in English "
            f"that match the scenario: '{desc}'.\n"
            f"Provide varying tones, sentence structures, and levels of detail (short, long, conversational greetings mixed with the request).\n"
            f"Respond ONLY with a valid JSON array of strings: [\"prompt 1\", \"prompt 2\", ...]. No explanations, no markdown formatting."
        )
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": system_gen_prompt}
            ],
            "stream": False,
            "format": "json"
        }
        
        res = call_ollama(api_url, payload)
        content = res.get("message", {}).get("content", "").strip()
        
        prompts = parse_prompts(content)
        if not prompts:
            console.print(f"[red]Failed to parse generated prompts for {name}.[/red]")
            console.print(f"[dim]Raw content: {content}[/dim]")
            continue
        # Limit to requested count
        prompts = prompts[:prompts_count]
            
        console.print(f"Generated [green]{len(prompts)}[/green] prompts for {name}.")
        
        # Step 2: Loop through prompts and build Chat template steps
        for p in track(prompts, description=f"Processing prompts for {name}..."):
            if name == "conversational":
                # For conversational prompts, we only need User -> Assistant (No Tool Call)
                payload_conv = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are Mr Meeseeks, a local AI OS companion. Do not output any thought blocks or JSON. Respond in plain conversational text."},
                        {"role": "user", "content": p}
                    ],
                    "stream": False
                }
                res_conv = call_ollama(api_url, payload_conv)
                assistant_response = res_conv.get("message", {}).get("content", "").strip()
                
                # Conversational template
                record = {
                    "messages": [
                        {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
                        {"role": "user", "content": p},
                        {"role": "assistant", "content": assistant_response}
                    ],
                    "tools": TOOLS
                }
                dataset_records.append(record)
            else:
                # For tool calling prompts, we need:
                # 1. User prompt -> Tool Call
                payload_tool = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": p}
                    ],
                    "tools": TOOLS,
                    "stream": False
                }
                res_tool = call_ollama(api_url, payload_tool)
                message = res_tool.get("message", {})
                
                if not message.get("tool_calls"):
                    # If model didn't call tool natively, skip or force it (we skip to ensure quality)
                    continue
                
                tc = message["tool_calls"][0]
                tool_call_name = tc.get("function", {}).get("name")
                tool_call_args = tc.get("function", {}).get("arguments", {})
                
                # Generate randomized tool response
                tool_response = get_simulated_result(tool_call_name, tool_call_args)
                
                # Get the final model response after receiving tool output
                messages_history = [
                    {"role": "user", "content": p},
                    message,
                    {"role": "tool", "name": tool_call_name, "content": tool_response}
                ]
                
                payload_final = {
                    "model": model_name,
                    "messages": messages_history,
                    "stream": False
                }
                res_final = call_ollama(api_url, payload_final)
                final_content = res_final.get("message", {}).get("content", "").strip()
                
                # Full structured tool-calling conversation record
                record = {
                    "messages": [
                        {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
                        {"role": "user", "content": p},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": tool_call_name,
                                        "arguments": tool_call_args
                                    }
                                }
                            ]
                        },
                        {"role": "tool", "name": tool_call_name, "content": tool_response},
                        {"role": "assistant", "content": final_content}
                    ],
                    "tools": TOOLS
                }
                dataset_records.append(record)

    # Save to JSONL file
    output_path = "temp/meeseeks-finetune-dataset.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for record in dataset_records:
            f.write(json.dumps(record) + "\n")
            
    console.print(f"\n[bold green]Success![/bold green] Generated dataset with [cyan]{len(dataset_records)}[/cyan] records.")
    console.print(f"Saved to: [yellow]{output_path}[/yellow]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for finetuning tool calling.")
    parser.add_argument("--api-url", default="http://localhost:11434", help="Ollama API base URL of the teacher model")
    parser.add_argument("--model", default="qwen3.5:9b-q4_K_M", help="Teacher model name registered in Ollama")
    parser.add_argument("--count", type=int, default=50, help="Number of prompts to generate per scenario")
    args = parser.parse_args()
    
    generate_dataset(args.api_url, args.model, args.count)
