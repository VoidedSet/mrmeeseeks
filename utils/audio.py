import sys
import os
import re
import time
import select
import queue
import threading
import concurrent.futures
import numpy as np
import logging

_input_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

def _blocking_input(prompt: str) -> str:
    return input(prompt)

async def async_input(prompt: str) -> str:
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_input_executor, _blocking_input, prompt)

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

def preload_cuda_libs():
    """
    Programmatically preload CUDA and cuDNN libraries from venv or system
    to make sure onnxruntime-gpu loads CUDAExecutionProvider on GPU correctly.
    """
    import ctypes
    import glob
    
    if not sys.platform.startswith("linux"):
        return
        
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    site_packages = os.path.join(project_root, "venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    nvidia_dir = os.path.join(site_packages, "nvidia")
    
    search_dirs = [
        "/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
    ]
    if os.path.exists(nvidia_dir):
        search_dirs.append(nvidia_dir)
        for root, dirs, _ in os.walk(nvidia_dir):
            for d in dirs:
                if d == "lib":
                    search_dirs.append(os.path.join(root, d))
                    
    search_dirs.extend([
        "/usr/local/lib/ollama/mlx_cuda_v13",
        "/usr/local/lib/ollama/cuda_v13",
        "/usr/local/lib/ollama/cuda_v12",
    ])

    # Load ALL .so files in site-packages/nvidia directory into RTLD_GLOBAL
    if os.path.exists(nvidia_dir):
        for root, _, files in os.walk(nvidia_dir):
            for file in files:
                if file.endswith(".so") or ".so." in file:
                    path = os.path.join(root, file)
                    try:
                        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass

    # Symlink CUDA 12 system libraries to CUDA 13 names inside onnxruntime/capi if missing
    ort_capi = os.path.join(site_packages, "onnxruntime", "capi")
    if os.path.exists(ort_capi):
        symlink_map = {
            "libcublasLt.so.13": "/lib/x86_64-linux-gnu/libcublasLt.so.12",
            "libcublas.so.13": "/lib/x86_64-linux-gnu/libcublas.so.12",
            "libcudart.so.13": "/lib/x86_64-linux-gnu/libcudart.so.12",
            "libcudnn.so.9": "/lib/x86_64-linux-gnu/libcudnn.so.8",
        }
        for dst_name, src_path in symlink_map.items():
            dst_path = os.path.join(ort_capi, dst_name)
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                try:
                    os.symlink(src_path, dst_path)
                except Exception:
                    pass

from datetime import datetime, timedelta

_MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]

def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def _replace_iso_dates(match) -> str:
    year_str, month_str, day_str = match.group(1), match.group(2), match.group(3)
    try:
        year, month, day = int(year_str), int(month_str), int(day_str)
        today = datetime.now().date()
        target_date = datetime(year, month, day).date()
        
        if target_date == today:
            return "today"
        elif target_date == today - timedelta(days=1):
            return "yesterday"
        elif target_date == today + timedelta(days=1):
            return "tomorrow"
        else:
            month_name = _MONTH_NAMES[month - 1]
            day_ord = _ordinal(day)
            if year == today.year:
                return f"{day_ord} {month_name}"
            else:
                return f"{day_ord} {month_name} {year}"
    except Exception:
        return match.group(0)

def clean_text_for_tts(text: str) -> str:
    """
    Cleans text formatting, modifiers, dates, URLs, and pronunciations for speech synthesis.
    - Converts 2026-08-09 -> "today", "yesterday", "5th August"
    - Strips markdown links [text](url) -> text
    - Strips HTTP/HTTPS URLs
    - Strips markdown headers, bold, italics, backticks
    - Replaces x/y with "x or y" or "x on y"
    """
    # 1. Convert ISO dates YYYY-MM-DD or YYYY/MM/DD to natural speech before slash removal
    text = re.sub(r"\b(\d{4})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b", _replace_iso_dates, text)

    # 2. Strip markdown links [link text](url) -> link text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # 3. Strip standalone URLs
    text = re.sub(r"https?://\S+", "", text)

    # 4. Remove markdown headers (# Title)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

    # 5. Remove markdown styling (**, *, __, _, `, etc.)
    text = re.sub(r"\*\*|__|\*|_|`", "", text)

    # 6. Replace / between two numbers (with or without spaces) with "on"
    text = re.sub(r"(\d+)\s*/\s*(\d+)", r"\1 on \2", text)

    # 7. Replace / between two words (with or without spaces) with "or"
    text = re.sub(r"([a-zA-Z]+)\s*/\s*([a-zA-Z]+)", r"\1 or \2", text)

    # 8. Replace any remaining forward slashes or dashes between words with spaces
    text = text.replace("/", " ")

    # 9. Collapse multiple spaces and newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text

def playback_worker(audio_queue, playback_stream_ref, timing_ref=None):
    import sounddevice as sd
    
    stream = None
    try:
        while True:
            item = audio_queue.get()
            if item is None:
                break
                
            samples, sample_rate = item
            
            if stream is None:
                # Open output stream once for the session using correct sample rate
                stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32')
                stream.start()
                playback_stream_ref[0] = stream
                
            if timing_ref is not None:
                if timing_ref.get('first_audio_played_time') is None:
                    timing_ref['first_audio_played_time'] = time.time()
                    start_time = timing_ref.get('start_time', timing_ref['first_audio_played_time'])
                    print(f"\n[Playback] Playing audio (latency: {timing_ref['first_audio_played_time'] - start_time:.3f}s from LLM generation start)")
                
            # Reshape mono 1D array to 2D (frames, 1) for sounddevice OutputStream
            if samples.ndim == 1:
                samples_2d = samples.reshape(-1, 1)
            else:
                samples_2d = samples
                
            stream.write(samples_2d)
            audio_queue.task_done()
    except Exception:
        pass
    finally:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            playback_stream_ref[0] = None

def synthesis_worker(kokoro_gpu, kokoro_cpu, use_dynamic_cap, voice, speed, text_queue, audio_queue, sentence_count_ref):
    while True:
        chunk = text_queue.get()
        if chunk is None:
            audio_queue.put(None)  # Signal playback thread to stop
            break
            
        cleaned_chunk = clean_text_for_tts(chunk)
        if not cleaned_chunk.strip():
            text_queue.task_done()
            continue
            
        words = cleaned_chunk.split()
        is_long_sentence = len(words) >= 5
        
        # Always use GPU execution engine for speech synthesis (GPU ONLY)
        engine = kokoro_gpu
        engine_name = "GPU (CUDA)"
            
        log.debug(f"[Synthesis] Generating sentence on {engine_name} (Queue size: {audio_queue.qsize()}/3): '{cleaned_chunk}'")
        
        try:
            samples, sample_rate = engine.create(
                cleaned_chunk,
                voice=voice,
                speed=speed,
                lang="en-us"
            )
            
            # Put synthesized audio into the queue (will block if queue is full at maxsize=3)
            audio_queue.put((samples, sample_rate))
            
            if is_long_sentence:
                sentence_count_ref[0] += 1
                
        except Exception as e:
            print(f"\n[Synthesis] Error: {e}")
            
        text_queue.task_done()

def voice_interrupt_worker(whisper_model, voice_interrupt_flag, stop_listening_event, shared_audio_data, audio_lock):
    """
    Background worker that listens to the microphone while Mr Meeseeks is speaking/thinking,
    appends audio to shared_audio_data, and sets voice_interrupt_flag if it hears the user speak 3+ words.
    """
    import sounddevice as sd
    
    def callback(indata, frames, time_info, status):
        with audio_lock:
            shared_audio_data.append(indata.copy())
        
    sample_rate = 16000
    # Keep last 1.5 seconds of audio for VAD checking (1.5 * 16000 = 24000 samples)
    max_samples = int(sample_rate * 1.5)
    
    try:
        stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=callback)
        with stream:
            # First phase: check for user speaking while AI is responding
            while not stop_listening_event.is_set() and not voice_interrupt_flag.is_set():
                time.sleep(0.4)
                
                # Make a thread-safe copy of the shared list
                with audio_lock:
                    if not shared_audio_data:
                        continue
                    shared_audio_data_copy = list(shared_audio_data)
                    
                # Concatenate buffer contents
                current_audio = np.concatenate(shared_audio_data_copy, axis=0).flatten()
                
                # Slice to retain only the last 1.5s
                if len(current_audio) > max_samples:
                    current_audio = current_audio[-max_samples:]
                    
                # Run transcription if we have at least 0.5s of audio
                if len(current_audio) >= sample_rate * 0.5:
                    segments, _ = whisper_model.transcribe(current_audio, beam_size=1)
                    text = " ".join([seg.text for seg in segments]).strip()
                    
                    # Clean transcription and check for 3+ words
                    words = text.split()
                    if len(words) >= 3:
                        # Ignore common Whisper tiny hallucinations on silence
                        hallucinations = {"thank you", "thanks for", "you for watching", "subtitles by", "please subscribe", "watching"}
                        is_hallucination = any(h in text.lower() for h in hallucinations)
                        if not is_hallucination:
                            print(f"\n🗣️  [Voice Interrupt] Heard: '{text}'")
                            voice_interrupt_flag.set()
                            break
                            
            # Second phase: if we were interrupted by voice, we keep the stream open
            # and just append audio until stop_listening_event is set by the main thread.
            while not stop_listening_event.is_set():
                time.sleep(0.1)
    except Exception as e:
        print(f"\n[Error in Voice Interrupt Worker] {e}")

def check_stdin_interrupt() -> bool:
    """
    Checks if the user has pressed ENTER in the terminal to request an interrupt.
    Returns True if an input is pending, otherwise False.
    """
    if not sys.platform.startswith("linux"):
        return False
    # Use select to check if stdin has data ready to read (non-blocking)
    rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
    if rlist:
        # Consume the typed line so it doesn't affect subsequent prompt inputs
        sys.stdin.readline()
        return True
    return False
