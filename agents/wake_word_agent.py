"""
wake_word_agent.py — Wake word activation for Mr Meeseeks.

Recognizes wake phrases via faster-whisper ('mr meeseeks', 'mr me6', 'mister meeseeks').
Also handles GNOME extension notch click triggers.
"""
import asyncio
import logging
import threading
import time
from typing import Optional, Callable

log = logging.getLogger("wake_word")

WAKE_PHRASES = [
    "mr meeseeks",
    "mr me6",
    "mister meeseeks",
    "mr. meeseeks",
    "meeseeks",
]


class WakeWordAgent:
    """
    Wake word detector using faster-whisper rolling window or manual trigger.
    """

    def __init__(self, on_wake: Optional[Callable] = None):
        self.on_wake = on_wake
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self, loop: asyncio.AbstractEventLoop):
        """Start background wake word listening thread."""
        if self._running:
            return
        self._running = True
        self._loop = loop
        self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="wake-word")
        self._thread.start()
        log.info("[WakeWord] Listener active (listening for 'mr meeseeks' / 'mr me6') ✓")

    def stop(self):
        self._running = False

    def trigger(self):
        """Manually trigger activation (e.g. from GNOME extension notch click)."""
        log.info("[WakeWord] Manually triggered (GNOME extension / notch click)")
        if self.on_wake and self._loop:
            asyncio.run_coroutine_threadsafe(self.on_wake(), self._loop)

    def _listen_loop(self):
        try:
            import sounddevice as sd
            import numpy as np
            from faster_whisper import WhisperModel

            model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            sample_rate = 16000
            window_samples = int(sample_rate * 1.5)
            buffer = []
            lock = threading.Lock()

            def callback(indata, frames, time_info, status):
                with lock:
                    buffer.append(indata.copy())

            with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
                while self._running:
                    time.sleep(0.8)
                    try:
                        from core.state_machine import State
                        from core.brain import brain
                        if brain.state_machine.current != State.IDLE:
                            with lock:
                                buffer.clear()
                            continue
                    except Exception:
                        pass

                    with lock:
                        if not buffer:
                            continue
                        audio_data = np.concatenate(buffer, axis=0).flatten()
                        buffer.clear()

                    if len(audio_data) > window_samples:
                        audio_data = audio_data[-window_samples:]

                    if np.abs(audio_data).mean() < 0.002:
                        continue

                    segments, _ = model.transcribe(audio_data, beam_size=1)
                    text = " ".join(seg.text for seg in segments).strip().lower()

                    if any(phrase in text for phrase in WAKE_PHRASES):
                        log.info(f"[WakeWord] Phrase detected in speech: '{text}'")
                        if self.on_wake and self._loop:
                            asyncio.run_coroutine_threadsafe(self.on_wake(), self._loop)

        except Exception as e:
            log.warning(f"[WakeWord] Passive listener paused ({e}). Notch click trigger remains ready.")


_agent: Optional[WakeWordAgent] = None


def get_wake_word_agent() -> Optional[WakeWordAgent]:
    return _agent


def init_wake_word_agent(on_wake_callback: Callable) -> WakeWordAgent:
    global _agent
    _agent = WakeWordAgent(on_wake=on_wake_callback)
    return _agent
