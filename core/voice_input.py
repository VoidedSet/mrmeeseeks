"""
voice_input.py — Mr Meeseeks Voice Input Manager
Handles recording audio via sounddevice and transcribing it using faster-whisper on CPU.
"""

import asyncio
import logging
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

log = logging.getLogger("voice_input")

class VoiceInputManager:
    def __init__(self):
        self.model = None

    def load_model(self):
        """Lazy load the Whisper model on CPU with int8 quantization."""
        if self.model is None:
            log.info("Loading Whisper model (tiny.en, CPU, int8)...")
            try:
                self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8", local_files_only=True)
            except Exception:
                log.info("Local Whisper model not found. Downloading from Hugging Face...")
                self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            log.info("Whisper model loaded successfully ✓")

    def record_and_transcribe(self) -> str:
        """
        Record audio from the microphone until Enter is pressed,
        then transcribe it using Whisper.
        """
        try:
            self.load_model()
        except Exception as e:
            log.error(f"Failed to load Whisper model: {e}")
            print("\n[Meeseeks] Error: Could not initialize Speech-to-Text engine.")
            return ""

        audio_data = []

        def callback(indata, frames, time, status):
            if status:
                log.warning(f"Audio stream status: {status}")
            audio_data.append(indata.copy())

        sample_rate = 16000
        try:
            stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=callback)
            print("\n[Meeseeks] 🎙️  Recording... Press Enter to stop.")
            with stream:
                input()  # Blocks until user presses Enter again
        except Exception as e:
            log.exception(f"Audio recording failed: {e}")
            print("\n[Meeseeks] Error: Failed to access microphone.")
            return ""

        if not audio_data:
            return ""

        audio = np.concatenate(audio_data, axis=0).flatten()
        
        # Check if the recording is too short (e.g. accidental press)
        if len(audio) < sample_rate * 0.5:
            log.info("Audio recording too short, ignoring.")
            return ""

        log.info("Transcribing recorded audio...")
        try:
            segments, info = self.model.transcribe(audio, beam_size=1)
            transcription = []
            for segment in segments:
                transcription.append(segment.text)
            
            text = " ".join(transcription).strip()
            return text
        except Exception as e:
            log.error(f"Transcription failed: {e}")
            print("\n[Meeseeks] Error: Transcription failed.")
            return ""

    async def record_and_transcribe_async(self, stop_event: asyncio.Event) -> str:
        """
        Record audio from the microphone in a non-blocking way until stop_event is set,
        then transcribe it using Whisper.
        """
        try:
            self.load_model()
        except Exception as e:
            log.error(f"Failed to load Whisper model: {e}")
            return ""

        audio_data = []

        def callback(indata, frames, time, status):
            if status:
                log.warning(f"Audio stream status: {status}")
            audio_data.append(indata.copy())

        sample_rate = 16000
        try:
            stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=callback)
            log.info("🎙️ Async Recording started...")
            with stream:
                while not stop_event.is_set():
                    await asyncio.sleep(0.05)
            log.info("🎙️ Async Recording stopped.")
        except Exception as e:
            log.exception(f"Audio recording failed: {e}")
            return ""

        if not audio_data:
            return ""

        audio = np.concatenate(audio_data, axis=0).flatten()
        
        # Check if the recording is too short (e.g. accidental press)
        if len(audio) < sample_rate * 0.5:
            log.info("Audio recording too short, ignoring.")
            return ""

        log.info("Transcribing recorded audio...")
        try:
            loop = asyncio.get_running_loop()
            
            def do_transcribe():
                segments, info = self.model.transcribe(audio, beam_size=1)
                transcription = []
                for segment in segments:
                    transcription.append(segment.text)
                return " ".join(transcription).strip()

            text = await loop.run_in_executor(None, do_transcribe)
            return text
        except Exception as e:
            log.error(f"Transcription failed: {e}")
            return ""

