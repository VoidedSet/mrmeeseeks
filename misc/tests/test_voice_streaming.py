import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import queue
import threading
import argparse
import asyncio
import json
import httpx
import numpy as np

# Load .env variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Timing variables
start_time = 0
first_token_time = None
first_audio_synthesized_time = None
first_audio_played_time = None
total_tokens_received = 0

def preload_cuda_libs():
    """
    Programmatically preload CUDA and cuDNN libraries from venv or system
    to make sure onnxruntime-gpu can load CUDAExecutionProvider correctly.
    Supports both CUDA 12 and CUDA 13 libraries.
    """
    import ctypes
    import glob
    
    if not sys.platform.startswith("linux"):
        return
        
    project_root = os.path.dirname(os.path.abspath(__file__))
    site_packages = os.path.join(project_root, "venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    nvidia_dir = os.path.join(site_packages, "nvidia")
    
    # Standard search paths
    search_dirs = []
    if os.path.exists(nvidia_dir):
        search_dirs.append(nvidia_dir)
    # Add system ollama paths as fallback
    search_dirs.append("/usr/local/lib/ollama/mlx_cuda_v13")
    search_dirs.append("/usr/local/lib/ollama/cuda_v13")
    search_dirs.append("/usr/local/lib/ollama/cuda_v12")
    
    print(f"[CUDA Loader] Scanning search paths: {search_dirs}")
    
    # Try loading CUDA 13 libraries first (system standard here), then CUDA 12
    libs_cuda13 = [
        ("cuda_runtime", "libcudart.so.13"),
        ("nvjitlink", "libnvjitlink.so.13"),
        ("cublas", "libcublasLt.so.13"),
        ("cublas", "libcublas.so.13"),
        ("cufft", "libcufft.so.12"),
        ("curand", "libcurand.so.10"),
        ("cusparse", "libcusparse.so.12"),
        ("cusolver", "libcusolver.so.12"),
        ("cudnn", "libcudnn.so.9"),
    ]
    
    libs_cuda12 = [
        ("cuda_runtime", "libcudart.so.12"),
        ("nvjitlink", "libnvjitlink.so.12"),
        ("cublas", "libcublasLt.so.12"),
        ("cublas", "libcublas.so.12"),
        ("cufft", "libcufft.so.11"),
        ("curand", "libcurand.so.10"),
        ("cusparse", "libcusparse.so.12"),
        ("cusolver", "libcusolver.so.11"),
        ("cudnn", "libcudnn.so.9"),
    ]
    
    # Detect which version to load based on what's available
    # We check if libcublasLt.so.13 is available in search_dirs
    use_cuda13 = False
    for search_dir in search_dirs:
        if glob.glob(os.path.join(search_dir, "**", "libcublasLt.so.13"), recursive=True) or \
           glob.glob(os.path.join(search_dir, "libcublasLt.so.13")):
            use_cuda13 = True
            break
            
    libs_to_load = libs_cuda13 if use_cuda13 else libs_cuda12
    print(f"[CUDA Loader] Attempting to load {'CUDA 13' if use_cuda13 else 'CUDA 12'} runtime libraries...")
    
    loaded_count = 0
    for pkg, libname in libs_to_load:
        lib_path = None
        # Try search directories
        for search_dir in search_dirs:
            # Check direct match
            path1 = os.path.join(search_dir, libname)
            if os.path.exists(path1):
                lib_path = path1
                break
            # Check venv package match: search_dir/pkg/lib/libname
            path2 = os.path.join(search_dir, pkg, "lib", libname)
            if os.path.exists(path2):
                lib_path = path2
                break
            # Fallback glob in search_dir
            pattern = os.path.join(search_dir, "**", libname)
            found = glob.glob(pattern, recursive=True)
            if found:
                lib_path = found[0]
                break
                
        if lib_path and os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                print(f"[CUDA Loader] Preloaded {libname} from {lib_path} ✓")
                loaded_count += 1
            except Exception as ex:
                print(f"[CUDA Loader] Warning: Could not preload {libname} from {lib_path}: {ex}")
        else:
            print(f"[CUDA Loader] Warning: Library {libname} not found in search paths.")
            
    print(f"[CUDA Loader] Preloaded {loaded_count} CUDA libraries.")

# ── Queues ───────────────────────────────────────────────────────────────────
text_queue = queue.Queue()
audio_queue = queue.Queue()

# ── Workers ───────────────────────────────────────────────────────────────────
def playback_worker():
    global first_audio_played_time
    import sounddevice as sd
    print("[Playback] Playback worker thread started.")
    
    while True:
        item = audio_queue.get()
        if item is None:
            break
            
        samples, sample_rate = item
        if first_audio_played_time is None:
            first_audio_played_time = time.time()
            print(f"\n⚡ LATENCY METRIC: Time to first audio playing: {first_audio_played_time - start_time:.3f} seconds from user request! ⚡\n")
            
        sd.play(samples, sample_rate)
        sd.wait()
        audio_queue.task_done()
        
    print("[Playback] Playback worker thread stopped.")

def synthesis_worker(kokoro_engine, voice, speed, text_queue, audio_queue):
    global first_audio_synthesized_time
    print("[Synthesis] Synthesis worker thread started.")
    
    while True:
        chunk = text_queue.get()
        if chunk is None:
            audio_queue.put(None)  # Signal playback thread to stop
            break
            
        if not chunk.strip():
            text_queue.task_done()
            continue
            
        s_start = time.time()
        try:
            samples, sample_rate = kokoro_engine.create(
                chunk,
                voice=voice,
                speed=speed,
                lang="en-us"
            )
            s_duration = time.time() - s_start
            print(f"\n[Synthesis] Synthesized chunk in {s_duration:.3f}s: '{chunk}'")
            
            if first_audio_synthesized_time is None:
                first_audio_synthesized_time = time.time()
                
            audio_queue.put((samples, sample_rate))
        except Exception as e:
            print(f"\n[Synthesis] Error synthesizing chunk '{chunk}': {e}")
            
        text_queue.task_done()
        
    print("[Synthesis] Synthesis worker thread stopped.")

# ── Groq Client ───────────────────────────────────────────────────────────────
async def stream_groq(prompt, api_key, model):
    global first_token_time, total_tokens_received
    
    base_url = "https://api.groq.com/openai/v1"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful, brief assistant. Keep your responses highly conversational, short, and punchy."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", f"{base_url}/chat/completions", json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        content = data["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.time()
                                print(f"⚡ LATENCY METRIC: Time to first token: {first_token_time - start_time:.3f} seconds ⚡")
                            total_tokens_received += 1
                            yield content
                    except Exception:
                        pass

# ── Text Chunking Loop ────────────────────────────────────────────────────────
async def run_streaming_pipeline(prompt, api_key, model, mode, chunk_size):
    buffer = ""
    punctuation_marks = {".", ",", "!", "?", ";", ":", "\n"}
    
    print("\n--- LLM Response ---")
    
    async for text in stream_groq(prompt, api_key, model):
        print(text, end="", flush=True)
        buffer += text
        
        if mode == "word":
            # Word-based splitting
            words = buffer.split(" ")
            if len(words) > chunk_size:
                # We have enough words to synthesize a chunk
                chunk_to_send = " ".join(words[:chunk_size]).strip()
                if chunk_to_send:
                    text_queue.put(chunk_to_send)
                buffer = " ".join(words[chunk_size:])
                
        elif mode == "punctuation":
            # Punctuation-based splitting
            split_idx = -1
            for i, char in enumerate(buffer):
                if char in punctuation_marks:
                    split_idx = i
                    break
                    
            if split_idx != -1:
                chunk_to_send = buffer[:split_idx + 1].strip()
                if chunk_to_send:
                    text_queue.put(chunk_to_send)
                buffer = buffer[split_idx + 1:]
            elif len(buffer) > 80:  # Fallback: chunk anyway if length exceeds 80 characters without punctuation
                # Find last space
                space_idx = buffer.rfind(" ")
                if space_idx != -1:
                    chunk_to_send = buffer[:space_idx].strip()
                    if chunk_to_send:
                        text_queue.put(chunk_to_send)
                    buffer = buffer[space_idx + 1:]
                    
        elif mode == "sentence":
            # Strict sentence-based splitting (only split on . ! ? \n)
            sentence_ends = {".", "!", "?", "\n"}
            split_idx = -1
            for i, char in enumerate(buffer):
                if char in sentence_ends:
                    split_idx = i
                    break
                    
            if split_idx != -1:
                chunk_to_send = buffer[:split_idx + 1].strip()
                if chunk_to_send:
                    text_queue.put(chunk_to_send)
                buffer = buffer[split_idx + 1:]
            elif len(buffer) > 120:  # Fallback
                space_idx = buffer.rfind(" ")
                if space_idx != -1:
                    chunk_to_send = buffer[:space_idx].strip()
                    if chunk_to_send:
                        text_queue.put(chunk_to_send)
                    buffer = buffer[space_idx + 1:]
                    
        elif mode == "full":
            # Accumulate the entire response
            pass
            
    print("\n--------------------\n")
    
    # Process anything remaining in the buffer
    if buffer.strip():
        text_queue.put(buffer.strip())
        
    # Signal synthesis thread that text stream is done
    text_queue.put(None)

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    global start_time, first_token_time, first_audio_synthesized_time, first_audio_played_time
    
    parser = argparse.ArgumentParser(description="Test voice streaming pipeline with GPU-accelerated Kokoro TTS")
    parser.add_argument("--mode", choices=["word", "punctuation", "sentence", "full"], default="punctuation",
                        help="Text chunking strategy: word, punctuation, sentence, full (default: punctuation)")
    parser.add_argument("--chunk-size", type=int, default=3,
                        help="Number of words per chunk in 'word' mode (default: 3)")
    parser.add_argument("--voice", default="af_sarah",
                        help="Kokoro voice to use (default: af_sarah)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed multiplier (default: 1.0)")
    parser.add_argument("--model", default="llama-3.1-8b-instant",
                        help="Groq LLM model to use (default: llama-3.1-8b-instant)")
    parser.add_argument("--prompt", default="Explain why space is black in 3 sentences, keeping it simple and poetic.",
                        help="User prompt to send to Groq")
    
    args = parser.parse_args()
    
    print(f"=== Mr Meeseeks Voice Streaming Test ===")
    print(f"Chunking Mode: {args.mode}")
    if args.mode == "word":
        print(f"Chunk Size   : {args.chunk_size} words")
    print(f"Voice        : {args.voice}")
    print(f"Speed        : {args.speed}")
    print(f"Groq Model   : {args.model}")
    print(f"Prompt       : '{args.prompt}'")
    print(f"========================================")
    
    # 1. Preload CUDA libraries
    preload_cuda_libs()
    
    # 2. Check Groq API key
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in environment or .env file.")
        sys.exit(1)
        
    # 3. Initialize Kokoro ONNX on GPU if available
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_root, "models", "kokoro-v1.0.fp16.onnx")
    voices_path = os.path.join(project_root, "models", "voices-v1.0.bin")
    
    if not os.path.exists(model_path) or not os.path.exists(voices_path):
        print(f"ERROR: Kokoro files not found at {model_path} or {voices_path}")
        sys.exit(1)
        
    print("\n[Init] Initializing Kokoro TTS Session...")
    try:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        
        # Check providers
        available_providers = ort.get_available_providers()
        print(f"[Init] Available ONNX providers: {available_providers}")
        
        selected_providers = []
        if "CUDAExecutionProvider" in available_providers:
            # Configure CUDA provider options to disable slow conv algorithm searches
            cuda_options = {
                "device_id": 0,
                "cudnn_conv_algo_search": "HEURISTIC"
            }
            selected_providers.append(("CUDAExecutionProvider", cuda_options))
            print("[Init] GPU acceleration (CUDAExecutionProvider) selected with HEURISTIC cuDNN search ✓")
        else:
            print("[Init] CUDAExecutionProvider not available, using CPUExecutionProvider.")
        selected_providers.append("CPUExecutionProvider")
        
        # Create InferenceSession
        session = ort.InferenceSession(model_path, providers=selected_providers)
        print(f"[Init] Active InferenceSession providers: {session.get_providers()}")
        
        kokoro_engine = Kokoro.from_session(session, voices_path)
        print("[Init] Kokoro TTS Engine ready ✓")
    except Exception as e:
        print(f"ERROR: Failed to initialize Kokoro engine: {e}")
        sys.exit(1)
        
    # 4. Start worker threads
    s_thread = threading.Thread(target=synthesis_worker, args=(kokoro_engine, args.voice, args.speed, text_queue, audio_queue), daemon=True)
    p_thread = threading.Thread(target=playback_worker, daemon=True)
    
    s_thread.start()
    p_thread.start()
    
    # 5. Run pipeline
    print("\n[Pipeline] Starting streaming voice pipeline...")
    start_time = time.time()
    
    await run_streaming_pipeline(args.prompt, api_key, args.model, args.mode, args.chunk_size)
    
    # Wait for processing to finish
    print("[Pipeline] Text stream completed. Waiting for synthesis and playback...")
    
    # Wait for threads to finish
    s_thread.join()
    p_thread.join()
    
    total_time = time.time() - start_time
    
    # 6. Report metrics
    print("\n================ METRICS SUMMARY ================")
    print(f"Total tokens received      : {total_tokens_received}")
    print(f"Total pipeline execution    : {total_time:.3f} seconds")
    if first_token_time:
        print(f"Time to first token (LLM)   : {first_token_time - start_time:.3f} seconds")
    if first_audio_synthesized_time:
        print(f"Time to first synthesized   : {first_audio_synthesized_time - start_time:.3f} seconds")
    if first_audio_played_time:
        print(f"Time to first played audio  : {first_audio_played_time - start_time:.3f} seconds")
        print(f"TTS latency (synth + play)  : {first_audio_played_time - first_token_time:.3f} seconds from first token")
    print("=================================================")

if __name__ == "__main__":
    asyncio.run(main())
