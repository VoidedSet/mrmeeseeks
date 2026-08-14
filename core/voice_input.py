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
        """Lazy load the Whisper model on GPU (CUDA) with float16 precision and fallback."""
        if self.model is None:
            log.info("Loading Whisper model (base.en, CUDA, float16)...")
            try:
                self.model = WhisperModel("base.en", device="cuda", compute_type="float16", local_files_only=True)
            except Exception:
                log.info("Local Whisper base.en CUDA model not found or CUDA unavailable. Trying download/fallback...")
                try:
                    self.model = WhisperModel("base.en", device="cuda", compute_type="float16")
                except Exception as e:
                    log.warning(f"Failed to load base.en on CUDA ({e}). Falling back to CPU with int8...")
                    self.model = WhisperModel("base.en", device="cpu", compute_type="int8")
            log.info("Whisper model loaded successfully ✓")

    def _post_process_text(self, text: str) -> str:
        """Apply spelling corrections for user's name (Kshayik)."""
        if not text:
            return text
        import re
        # Catch common phonetical mishearings: 'shik', 'shike', 'shyke', 'kshayeek', etc.
        misspellings = [r"\bshik\b", r"\bshike\b", r"\bshyke\b", r"\bkshayeek\b", r"\bksayeek\b"]
        for pattern in misspellings:
            text = re.sub(pattern, "Kshayik", text, flags=re.IGNORECASE)
        return text

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
            # We pass initial_prompt to bias recognition toward user's name "Kshayik"
            segments, info = self.model.transcribe(audio, beam_size=1, initial_prompt="Kshayik")
            transcription = []
            for segment in segments:
                transcription.append(segment.text)
            
            text = " ".join(transcription).strip()
            return self._post_process_text(text)
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
                # Pass initial_prompt to bias recognition toward user's name "Kshayik"
                segments, info = self.model.transcribe(audio, beam_size=1, initial_prompt="Kshayik")
                transcription = []
                for segment in segments:
                    transcription.append(segment.text)
                text = " ".join(transcription).strip()
                return self._post_process_text(text)

            text = await loop.run_in_executor(None, do_transcribe)
            return text
        except Exception as e:
            log.error(f"Transcription failed: {e}")
            return ""

