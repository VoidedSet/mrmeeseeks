#!/usr/bin/env python3
"""
generate_dataset_yaml.py — YAML Tool Fine-tuning Dataset Generator
Generates a flat JSONL dataset with initiate, finalize, and conversational entries.
Uses async HTTP calls to query any OpenAI-compatible API (Ollama, Gemini, Groq, etc.).
"""

import os
import sys
import json
import asyncio
import argparse
import time
import httpx
from typing import Any, Dict, List

# Add parent directory to sys.path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.schema_registry import TOOL_SCHEMAS, REQUIRED_ARGS

# Default conversational prompts for "No Call" general conversational category
CONVERSATIONAL_PROMPTS = [
    "Tell me a joke.",
    "Who created you?",
    "What is the capital of France?",
    "Can you write a python function to check if a number is prime?",
    "How do I install python on Ubuntu?",
    "Hello there! How's it going?",
    "Are you a robot?",
    "What can you do for me?",
    "Tell me a fun fact about space.",
    "How do I exit a vim session?",
]

SEMAPHORE = asyncio.Semaphore(2)

import traceback

async def generate_scenarios_for_tool(
    client: httpx.AsyncClient,
    api_url: str,
    api_key: str,
    model: str,
    tool_name: str,
    tool_schema: dict,
    required_args: List[str],
    batch_size: int = 5,
    batch_idx: int = 1,
    total_batches: int = 1
) -> List[dict]:
    """Generates a batch of user scenarios, arguments, simulated outputs, and final speech responses for a tool."""
    
    prompt = f"""You are an expert training data generator for AI agents.
Your task is to generate {batch_size} highly realistic and diverse user queries (scenarios) that would require calling the following tool:

Tool Name: {tool_name}
Description: {tool_schema.get('description', '')}
Parameters Schema: {json.dumps(tool_schema.get('args', {}), indent=2)}
Required Parameters: {json.dumps(required_args)}

For each of the {batch_size} scenarios, generate:
1. "user_query": A natural user query (in English, occasionally casual or conversational) that explicitly or implicitly requests this tool. Make the queries diverse: varying phrasing, details, and contexts.
2. "arguments": A flat dictionary of parameter-value pairs matching the Parameters Schema. Ensure all required parameters are provided.
3. "simulated_output": A realistic simulated stdout/output or result of running this command/action on a Linux desktop system. Keep it brief but realistic.
4. "final_speech": The final conversational response the agent should speak to the user, summarizing or explaining the result based on the simulated_output.

Return your response in a valid JSON object format matching this JSON schema:
{{
  "scenarios": [
    {{
      "user_query": "string",
      "arguments": {{}},
      "simulated_output": "string",
      "final_speech": "string"
    }}
  ]
}}
"""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    start_time = time.time()
    print(f"[*] [{tool_name}] Queueing batch {batch_idx}/{total_batches}...")
    
    async with SEMAPHORE:
        wait_time = time.time() - start_time
        print(f"[*] [{tool_name}] Sending batch {batch_idx}/{total_batches} to API (queued for {wait_time:.1f}s)...")
        api_start = time.time()
        try:
            resp = await client.post(
                f"{api_url.rstrip('/')}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON. Do NOT output any thinking process, <think> tags, or reasoning. Start your response directly with the JSON object."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 4096
                },
                timeout=300.0
            )
            resp.raise_for_status()
            res_json = resp.json()
            content = res_json["choices"][0]["message"]["content"].strip()
            data = json.loads(content)
            scenarios = data.get("scenarios", [])
            api_elapsed = time.time() - api_start
            total_elapsed = time.time() - start_time
            print(f"[+] [{tool_name}] Batch {batch_idx}/{total_batches} completed: API took {api_elapsed:.1f}s, Total took {total_elapsed:.1f}s (generated {len(scenarios)} scenarios)")
            return scenarios
        except Exception as e:
            api_elapsed = time.time() - api_start
            total_elapsed = time.time() - start_time
            print(f"[-] [{tool_name}] Batch {batch_idx}/{total_batches} failed after {total_elapsed:.1f}s (API took {api_elapsed:.1f}s):")
            if 'content' in locals():
                print(f"[-] Raw content returned: {repr(content)}")
            traceback.print_exc()
            return []

async def generate_conversational_entries(
    client: httpx.AsyncClient,
    api_url: str,
    api_key: str,
    model: str,
    count: int = 50,
    batch_idx: int = 1,
    total_batches: int = 1
) -> List[dict]:
    """Generates general conversation queries and responses (no-call entries)."""
    prompt = f"""You are an expert training data generator for AI agents.
Generate {count} diverse conversational queries that do NOT require using any system or desktop tools (e.g. casual small talk, basic questions, greetings, jokes).

For each scenario, generate:
1. "user_query": The user's input/greeting/question.
2. "final_speech": The natural conversational response the agent should output.

Return your response in a valid JSON object format matching this JSON schema:
{{
  "scenarios": [
    {{
      "user_query": "string",
      "final_speech": "string"
    }}
  ]
}}
"""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    start_time = time.time()
    print(f"[*] [conversational] Queueing batch {batch_idx}/{total_batches}...")
    
    async with SEMAPHORE:
        wait_time = time.time() - start_time
        print(f"[*] [conversational] Sending batch {batch_idx}/{total_batches} to API (queued for {wait_time:.1f}s)...")
        api_start = time.time()
        try:
            resp = await client.post(
                f"{api_url.rstrip('/')}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON. Do NOT output any thinking process, <think> tags, or reasoning. Start your response directly with the JSON object."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 4096
                },
                timeout=300.0
            )
            resp.raise_for_status()
            res_json = resp.json()
            content = res_json["choices"][0]["message"]["content"].strip()
            data = json.loads(content)
            scenarios = data.get("scenarios", [])
            api_elapsed = time.time() - api_start
            total_elapsed = time.time() - start_time
            print(f"[+] [conversational] Batch {batch_idx}/{total_batches} completed: API took {api_elapsed:.1f}s, Total took {total_elapsed:.1f}s (generated {len(scenarios)} scenarios)")
            return scenarios
        except Exception as e:
            api_elapsed = time.time() - api_start
            total_elapsed = time.time() - start_time
            print(f"[-] [conversational] Batch {batch_idx}/{total_batches} failed after {total_elapsed:.1f}s (API took {api_elapsed:.1f}s): {e}")
            return []

def format_as_yaml(tool_name: str, args: dict = None, speech: str = None) -> str:
    """Format a response into the custom YAML format."""
    lines = [f"call: {tool_name}"]
    if args:
        lines.append("args:")
        for k, v in args.items():
            # In YAML, if a string has spaces or colons, we quote it, otherwise keep it clean
            v_str = str(v)
            if ":" in v_str or "#" in v_str or not v_str.strip():
                # Escaping double quotes
                escaped = v_str.replace('"', '\\"')
                lines.append(f"  {k}: \"{escaped}\"")
            else:
                lines.append(f"  {k}: {v_str}")
    if speech:
        # Quote the speech since it can contain arbitrary characters
        escaped_speech = speech.replace('"', '\\"')
        lines.append(f"speech: \"{escaped_speech}\"")
    return "\n".join(lines)

def convert_flat_to_chatml(entry: dict) -> dict:
    """
    Converts a flat dataset entry into standard ChatML messages format.
    Used for local formatting test and in fine-tuning notebooks.
    """
    entry_type = entry.get("type")
    user_content = entry.get("user_content")
    
    # 1. Build developer message listing the tools in YAML-like format
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
        {"role": "developer", "content": dev_content}
    ]
    
    if entry_type == "initiate":
        messages.extend([
            {"role": "user", "content": f"---\nUSER REQUEST: {user_content}"},
            {"role": "assistant", "content": format_as_yaml(entry["tool_name"], entry["tool_arguments"])}
        ])
    elif entry_type == "finalize":
        # Multi-turn representation for the finalize step
        messages.extend([
            {"role": "user", "content": f"---\nUSER REQUEST: {user_content}"},
            # Model first did the initiate call
            {"role": "assistant", "content": f"call: {entry.get('initial_tool_name', 'tool')}\nargs: ..."},
            # System provides the observation
            {"role": "tool", "name": entry.get("initial_tool_name", "tool"), "content": str(entry["tool_output"]) if entry["tool_output"] is not None else ""},
            # Assistant outputs the final Speech
            {"role": "assistant", "content": format_as_yaml("none", speech=entry["speech"])}
        ])
    elif entry_type == "no_call":
        messages.extend([
            {"role": "user", "content": f"---\nUSER REQUEST: {user_content}"},
            {"role": "assistant", "content": format_as_yaml("none", speech=entry["speech"])}
        ])
        
    return {"messages": messages}

async def main():
    global SEMAPHORE
    parser = argparse.ArgumentParser(description="Generate YAML fine-tuning dataset.")
    parser.add_argument("--api-url", default="http://localhost:11434/v1", help="OpenAI-compatible API URL")
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY", ""), help="API Key (optional)")
    parser.add_argument("--model", default="gemini-1.5-flash", help="Teacher LLM Model Name")
    parser.add_argument("--count", type=int, default=50, help="Target scenarios/queries per tool (default 50)")
    parser.add_argument("--dry-run", action="store_true", help="Only generate 2 examples of each type to verify format")
    parser.add_argument("--output", default="temp/meeseeks_yaml_dataset.jsonl", help="Output file path")
    parser.add_argument("--tools", default="", help="Comma-separated list of specific tools to generate (e.g., check_battery,run_bg_cmd)")
    parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrent API requests (default 2)")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of scenarios to generate per API call (default 5)")
    
    args = parser.parse_args()
    
    SEMAPHORE = asyncio.Semaphore(args.concurrency)
    
    if args.dry_run:
        print("[*] Running DRY-RUN mode (generating 2 scenarios per tool).")
        args.count = 2
        
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"[*] Starting dataset generation using teacher model: {args.model}")
    print(f"[*] API Base URL: {args.api_url}")
    print(f"[*] Target count per tool: {args.count}")
    print(f"[*] Concurrency limit: {args.concurrency}")
    
    # 50 per tool in batches of 5 means 10 batches
    batch_size = args.batch_size
    if args.count < batch_size:
        batch_size = args.count
        
    num_batches = (args.count + batch_size - 1) // batch_size
    
    completed_tools = set()
    existing_entries_count = 0
    if os.path.exists(args.output):
        print(f"[*] Found existing output file {args.output}. Checking progress...")
        try:
            tool_counts = {}
            with open(args.output, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        existing_entries_count += 1
                        t_name = entry.get("tool_name")
                        if not t_name and "initial_tool_name" in entry:
                            t_name = entry["initial_tool_name"]
                        if t_name and t_name != "none":
                            tool_counts[t_name] = tool_counts.get(t_name, 0) + 1
            
            # Since each scenario generates 2 entries (initiate and finalize),
            # a tool is complete if it has at least count * 2 * 0.8 entries.
            threshold = int(args.count * 2 * 0.8)
            for t_name, num_entries in tool_counts.items():
                if num_entries >= threshold:
                    completed_tools.add(t_name)
            
            print(f"[+] Found {len(completed_tools)} tools already fully generated (>= {threshold} entries): {list(completed_tools)}")
        except Exception as e:
            print(f"[-] Error parsing existing progress: {e}")
            
    # Filter tools if requested
    target_tools = list(TOOL_SCHEMAS.keys())
    if args.tools:
        selected = [t.strip() for t in args.tools.split(",")]
        target_tools = [t for t in target_tools if t in selected]
    
    async with httpx.AsyncClient(timeout=600.0) as client:
        # Generate tool-specific scenarios
        for tool_name in target_tools:
            if tool_name in completed_tools:
                print(f"[*] Skipping already completed tool: {tool_name}")
                continue
                
            tool_schema = TOOL_SCHEMAS[tool_name]
            print(f"[*] Generating scenarios for tool: {tool_name} ({num_batches} batches of {batch_size})...")
            req_args = REQUIRED_ARGS.get(tool_name, [])
            
            # Run batches concurrently for the tool
            tasks = [
                generate_scenarios_for_tool(
                    client, args.api_url, args.api_key, args.model, 
                    tool_name, tool_schema, req_args, batch_size,
                    batch_idx=i+1, total_batches=num_batches
                )
                for i in range(num_batches)
            ]
            
            batches_results = await asyncio.gather(*tasks)
            
            tool_scenarios = []
            for batch in batches_results:
                tool_scenarios.extend(batch)
                
            # Truncate to exact target count if we generated extra
            tool_scenarios = tool_scenarios[:args.count]
            print(f"[+] Successfully generated {len(tool_scenarios)} scenarios for {tool_name}.")
            
            tool_entries = []
            for s in tool_scenarios:
                if not isinstance(s, dict):
                    print(f"[-] Warning: scenario is not a dict: {repr(s)}")
                    continue
                user_q = s.get("user_query")
                tool_args = s.get("arguments", {})
                sim_out = s.get("simulated_output")
                speech = s.get("final_speech")
                
                if not user_q:
                    continue
                    
                # 1. Add Initiate entry
                tool_entries.append({
                    "type": "initiate",
                    "user_content": user_q,
                    "tool_name": tool_name,
                    "tool_arguments": tool_args
                })
                
                # 2. Add Finalize entry
                tool_entries.append({
                    "type": "finalize",
                    "user_content": user_q,
                    "initial_tool_name": tool_name,
                    "tool_output": sim_out,
                    "speech": speech
                })
            
            if tool_entries:
                print(f"[*] Appending {len(tool_entries)} entries for {tool_name} to {args.output}...")
                with open(args.output, "a", encoding="utf-8") as f:
                    for entry in tool_entries:
                        f.write(json.dumps(entry) + "\n")
                        
        # Generate general conversational entries (No Call)
        # Recalculate total tool entries and conversational entries in the file
        total_tool_entries = 0
        existing_conv_count = 0
        if os.path.exists(args.output):
            with open(args.output, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        if entry.get("type") in ("initiate", "finalize"):
                            total_tool_entries += 1
                        elif entry.get("type") == "no_call":
                            existing_conv_count += 1
                            
        conv_target_count = int(total_tool_entries * 0.1) - existing_conv_count
        if args.dry_run:
            conv_target_count = 2 - existing_conv_count
            
        print(f"[*] Generating {conv_target_count} general conversational entries (need {int(total_tool_entries * 0.1)}, already have {existing_conv_count})...")
        
        if conv_target_count > 0:
            # Batch conversational queries
            conv_batch_size = min(20, conv_target_count)
            conv_num_batches = (conv_target_count + conv_batch_size - 1) // conv_batch_size
            
            tasks = [
                generate_conversational_entries(
                    client, args.api_url, args.api_key, args.model, conv_batch_size,
                    batch_idx=i+1, total_batches=conv_num_batches
                )
                for i in range(conv_num_batches)
            ]
            
            conv_batches_results = await asyncio.gather(*tasks)
            conv_scenarios = []
            for batch in conv_batches_results:
                conv_scenarios.extend(batch)
                
            conv_scenarios = conv_scenarios[:conv_target_count]
            print(f"[+] Successfully generated {len(conv_scenarios)} conversational scenarios.")
            
            conv_entries = []
            for s in conv_scenarios:
                if not isinstance(s, dict):
                    print(f"[-] Warning: scenario is not a dict: {repr(s)}")
                    continue
                user_q = s.get("user_query")
                speech = s.get("final_speech")
                
                if not user_q:
                    continue
                    
                conv_entries.append({
                    "type": "no_call",
                    "user_content": user_q,
                    "tool_name": "none",
                    "tool_arguments": {},
                    "speech": speech
                })
            
            if conv_entries:
                print(f"[*] Appending {len(conv_entries)} conversational entries to {args.output}...")
                with open(args.output, "a", encoding="utf-8") as f:
                    for entry in conv_entries:
                        f.write(json.dumps(entry) + "\n")
        else:
            print("[*] No additional conversational entries needed.")
            
    print("[+] Done! Dataset generated successfully.")
    
    # Show formatting output test
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    print("\n[*] Formatting Preview of First Generated Entry:")
                    preview_chatml = convert_flat_to_chatml(json.loads(first_line))
                    print(json.dumps(preview_chatml, indent=2))
        except Exception as e:
            pass

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
