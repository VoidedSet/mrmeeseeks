#!/usr/bin/env python3
"""
run_colab_generation.py
Polls the Colab ngrok endpoint until a model is fully downloaded and loaded,
then automatically kicks off the full YAML dataset generation.
"""

import time
import httpx
import subprocess
import sys

API_URL = "https://db38-34-21-177-185.ngrok-free.app"
POLL_INTERVAL = 30  # seconds

def get_loaded_model(api_url: str) -> str:
    try:
        resp = httpx.get(f"{api_url}/v1/models", timeout=10.0)
        if resp.status_code == 200:
            data = resp.json().get("data")
            if data and len(data) > 0:
                # Return the ID of the first loaded model
                return data[0].get("id")
    except Exception as e:
        pass
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Poll Colab ngrok tunnel and run generator.")
    parser.add_argument("--url", default=API_URL, help="Ngrok API URL")
    parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrent API requests (default 2)")
    parser.add_argument("--tools", default="", help="Comma-separated list of tools to generate")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of scenarios to generate per API call (default 5)")
    parser.add_argument("--count", type=int, default=50, help="Number of scenarios to generate per tool (default 50)")
    args = parser.parse_args()
    
    api_url = args.url.rstrip("/")
    print(f"[*] Starting Colab api poll at: {api_url}")
    print("[*] Waiting for model download and load to complete...")
    
    start_time = time.time()
    model_name = None
    
    while True:
        model_name = get_loaded_model(api_url)
        if model_name:
            elapsed = int(time.time() - start_time)
            print(f"\n[+] Model '{model_name}' is loaded and ready! (Elapsed: {elapsed}s)")
            break
        
        # Print a simple spinner/dot progress indicator
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(POLL_INTERVAL)
        
    # Launch the actual generation script
    cmd = [
        "venv/bin/python",
        "-u",
        "temp/generate_dataset_yaml.py",
        "--api-url", f"{api_url}/v1",
        "--model", model_name,
        "--count", str(args.count),
        "--output", "temp/meeseeks_yaml_dataset.jsonl",
        "--concurrency", str(args.concurrency),
        "--batch-size", str(args.batch_size)
    ]
    if args.tools:
        cmd.extend(["--tools", args.tools])
    
    print(f"[*] Launching dataset generation: {' '.join(cmd)}")
    try:
        # Run command and pipe output live to console
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if process.returncode == 0:
            print("[+] Dataset generation completed successfully!")
        else:
            print(f"[-] Dataset generation failed with exit code: {process.returncode}")
    except KeyboardInterrupt:
        print("\n[-] Cancelled by user.")
        process.terminate()

if __name__ == "__main__":
    main()
