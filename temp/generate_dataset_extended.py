# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "rich",
# ]
# ///
"""
Extended dataset generation script for training Mr Meeseeks.
Dynamically imports all schemas from the core registry and covers the complete tool suite.

Usage:
  python temp/generate_dataset_extended.py --api-url "https://your-colab-tunnel.ngrok-free.app" --model "qwen3.5:9b-q4_K_M"
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

# Add repository root to system path for core imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.schema_registry import get_openai_tools

console = Console()
ALL_TOOLS = get_openai_tools()

# ─── Simulated Tool Output Generator (Covers 100% of tools) ───────────────────
def get_simulated_result(tool_name: str, args: dict) -> str:
    # ── Eyes ──
    if tool_name == "get_ui_elements":
        return json.dumps({
            "buttons": [{"id": "btn_login", "label": "Login", "x": 120, "y": 180}],
            "inputs": [{"id": "input_user", "label": "Username", "x": 120, "y": 240}],
            "tabs": [{"id": "tab_home", "label": "Home", "x": 50, "y": 40}],
            "content": []
        })
    elif tool_name == "list_at_spi_apps":
        return json.dumps({"apps": ["firefox", "code", "gnome-terminal-server", "spotify"]})
    elif tool_name == "read_element_text":
        return json.dumps({"text": "User successfully logged in."})
    elif tool_name == "find_element_by_label":
        return json.dumps({"id": "btn_submit", "label": "Submit", "x": 350, "y": 500})
        
    # ── Hands ──
    elif tool_name in ["click_at", "double_click_at", "type_text", "key_press", "scroll"]:
        return json.dumps({"status": "success"})
        
    # ── SysAdmin (Silent) ──
    elif tool_name == "run_bg_cmd":
        cmd = args.get("cmd", "ls").strip()
        if "ls" in cmd:
            return json.dumps({"output": "documents\ndownloads\nprojects\nnotes.txt", "exit_code": 0})
        if "pwd" in cmd:
            return json.dumps({"output": "/home/kshayik/Projects/mr-meeseeks", "exit_code": 0})
        if "free" in cmd:
            return json.dumps({"output": "               total        used        free\nMem:        15939228     8432104     7507124", "exit_code": 0})
        return json.dumps({"output": "Command ran successfully.", "exit_code": 0})
    elif tool_name == "check_battery":
        return json.dumps({"level": "87%", "status": "Discharging"})
    elif tool_name == "get_active_window":
        return json.dumps({"title": "Firefox - YouTube", "class": "firefox"})
    elif tool_name == "list_open_windows":
        return json.dumps({"windows": ["Visual Studio Code", "Firefox", "Slack", "Spotify"]})
    elif tool_name == "read_notifications":
        return json.dumps({"notifications": [{"app": "Slack", "title": "New message", "body": "Hey, are you free for a call?"}]})
    elif tool_name == "open_visible_terminal":
        return json.dumps({"status": "success", "pid": 14205})
        
    # ── Memory ──
    elif tool_name == "list_memory_keys":
        return json.dumps({"keys": ["user_name", "preferred_editor", "theme"]})
    elif tool_name == "update_memory":
        return json.dumps({"status": "success"})
    elif tool_name == "fetch_memory":
        keys = args.get("keys", [])
        return json.dumps({k: f"Stored preference for {k}" for k in keys})
        
    # ── Web ──
    elif tool_name == "simple_scrape":
        return json.dumps({"results": [f"Web search results page content for query: {args.get('query', '')}"]})
    elif tool_name == "gui_research":
        return json.dumps({"status": "completed", "summary": "Found the requested data on page."})
        
    # ── Voice & done ──
    elif tool_name == "speak":
        return json.dumps({"status": "success"})
    elif tool_name == "done":
        return json.dumps({"status": "completed"})
        
    return json.dumps({"error": f"Unknown tool {tool_name}"})

# ─── Robust Prompt Parser ─────────────────────────────────────────────────────
def parse_prompts(content: str) -> list[str]:
    import re
    content = content.strip()
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [str(x) for x in data if x]
    except Exception:
        pass
        
    matches = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', content)
    if matches:
        cleaned = [m.replace('\\"', '"').strip() for m in matches]
        return [c for c in cleaned if c]
        
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
    payload = payload.copy()
    payload["think"] = False
    
    try:
        with httpx.Client(timeout=180.0, headers=headers) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        console.print(f"[bold red]API Error:[/bold red] {e}")
        return {}

# ─── Dataset Generator ────────────────────────────────────────────────────────
def generate_dataset(api_url: str, model_name: str, prompts_count: int):
    console.print(Panel(f"Generating EXTENDED training dataset using [cyan]{model_name}[/cyan]\nTarget API: [green]{api_url}[/green]", title="Extended Dataset Generator"))
    
    dataset_records = []
    
    # 11 diverse scenario targets representing different categories in core/schema_registry
    scenarios = [
        {"name": "get_ui_elements", "desc": "Inspecting or locating elements, buttons, and input fields on screen"},
        {"name": "click_at", "desc": "Clicking or double-clicking screen positions, buttons, or links"},
        {"name": "type_text", "desc": "Typing text, writing input, or typing passwords"},
        {"name": "key_press", "desc": "Pressing hotkeys or keyboard shortcuts like ctrl+c, Return, or alt+tab"},
        {"name": "list_open_windows", "desc": "Checking what windows are currently open, listing window titles"},
        {"name": "read_notifications", "desc": "Reading desktop notifications or checking system alerts"},
        {"name": "open_visible_terminal", "desc": "Launching a visible terminal window for running bash scripts or GUI apps"},
        {"name": "memory", "desc": "Storing user preferences, updating memory, or retrieving facts and keys"},
        {"name": "simple_scrape", "desc": "Searching the web silently for current news, facts, or scrape results"},
        {"name": "gui_research", "desc": "Opening a browser to interactively research, read, and find information"},
        {"name": "speak", "desc": "Speaking text out loud using text-to-speech voice output"}
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
        prompts = prompts[:prompts_count]
        
        console.print(f"Generated [green]{len(prompts)}[/green] prompts for {name}.")
        
        # Step 2: Loop through prompts and build Chat template steps
        for p in track(prompts, description=f"Processing prompts for {name}..."):
            # Determine target tool call name based on scenario
            target_tool = name
            if name == "memory":
                target_tool = random.choice(["list_memory_keys", "update_memory", "fetch_memory"])
            
            # 1. User prompt -> Tool Call
            payload_tool = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": p}
                ],
                "tools": ALL_TOOLS,
                "stream": False
            }
            res_tool = call_ollama(api_url, payload_tool)
            message = res_tool.get("message", {})
            
            # Check if model correctly outputted a tool call
            if not message.get("tool_calls"):
                # Proactively generate a simulated tool call if the model failed to output one natively
                # (ensures dataset dataset maintains consistent training target quality)
                sim_args = {}
                if target_tool == "click_at" or target_tool == "double_click_at":
                    sim_args = {"x": random.randint(100, 800), "y": random.randint(100, 600)}
                elif target_tool == "type_text":
                    sim_args = {"text": "hello"}
                elif target_tool == "key_press":
                    sim_args = {"keys": "Return"}
                elif target_tool == "update_memory":
                    sim_args = {"key": "user_name", "data": "Kshayik"}
                elif target_tool == "fetch_memory":
                    sim_args = {"keys": ["user_name"]}
                elif target_tool == "simple_scrape":
                    sim_args = {"query": p}
                elif target_tool == "gui_research":
                    sim_args = {"task": p}
                elif target_tool == "speak":
                    sim_args = {"text": "Process completed."}
                
                tool_call_name = target_tool
                tool_call_args = sim_args
            else:
                tc = message["tool_calls"][0]
                tool_call_name = tc.get("function", {}).get("name")
                tool_call_args = tc.get("function", {}).get("arguments", {})
            
            # Generate simulated tool response
            tool_response = get_simulated_result(tool_call_name, tool_call_args)
            
            # Get final response
            messages_history = [
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
                {"role": "tool", "name": tool_call_name, "content": tool_response}
            ]
            
            payload_final = {
                "model": model_name,
                "messages": messages_history,
                "stream": False
            }
            res_final = call_ollama(api_url, payload_final)
            final_content = res_final.get("message", {}).get("content", "").strip()
            
            # Construct dataset record
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
                "tools": ALL_TOOLS
            }
            dataset_records.append(record)
            
    # Save to JSONL file
    output_path = "temp/meeseeks-finetune-dataset-extended.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for record in dataset_records:
            f.write(json.dumps(record) + "\n")
            
    console.print(f"\n[bold green]Success![/bold green] Generated extended dataset with [cyan]{len(dataset_records)}[/cyan] records.")
    console.print(f"Saved to: [yellow]{output_path}[/yellow]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate extended dataset for finetuning tool calling.")
    parser.add_argument("--api-url", default="http://localhost:11434", help="Ollama API base URL of the teacher model")
    parser.add_argument("--model", default="qwen3.5:9b-q4_K_M", help="Teacher model name registered in Ollama")
    parser.add_argument("--count", type=int, default=20, help="Number of prompts to generate per scenario")
    args = parser.parse_args()
    
    generate_dataset(args.api_url, args.model, args.count)
