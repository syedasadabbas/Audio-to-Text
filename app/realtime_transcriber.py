"""
Real-time transcription engine for the GUI application.

Captures microphone audio in 100ms chunks, detects speech via RMS energy,
and flushes completed utterances to a transcription thread that calls
Transcriber.transcribe_array() — the same pipeline used for file transcription.

Thread model:
  Audio callback  →  _segment_queue  →  transcription thread  →  on_segment callback
  (sounddevice)                         (faster-whisper)          (.after(0,...) in UI)
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).parent.parent))
from transcriber import Transcriber

SAMPLE_RATE = 16_000
CHUNK_MS = 100
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)

SILENCE_RMS_THRESH = 0.015   # float32 scale; raise for noisy environments
MIN_SPEECH_MS = 300          # discard clips shorter than this
MIN_SILENCE_MS = 600         # silence this long marks end of an utterance
MAX_SEGMENT_MS = 12_000      # force-flush if a speaker never pauses


class RealtimeTranscriber:
    """
    Records from the default microphone and streams transcribed segments
    to a callback as they complete.

    Args:
        on_segment: Called with (start_sec, end_sec, text, is_final) from the
                    transcription thread. Use .after(0, ...) to update Tk widgets.
        model_size: Whisper model size — "tiny" or "base" recommended for
                    real-time use.
        language:   ISO-639-1 code or None for auto-detect.
    """

    def __init__(
        self,
        on_segment: Callable[[float, float, str, bool], None],
        model_size: str = "base",
        language: str | None = None,
    ):
        self._on_segment = on_segment
        self._language = language
        self._transcriber = Transcriber(model_size=model_size)

        self._speech_buffer: list[np.ndarray] = []
        self._silence_chunks = 0
        self._speech_chunks = 0
        self._in_speech = False
        self._session_start = 0.0
        self._segment_start = 0.0
        self._last_rms: float = 0.0

        self._segment_queue: deque = deque()
        self._stop_event = threading.Event()
        self._stream: sd.InputStream | None = None
        self._transcribe_thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._session_start = time.time()
        self._segment_start = 0.0

        self._transcribe_thread = threading.Thread(
            target=self._transcription_loop, daemon=True
        )
        self._transcribe_thread.start()

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._speech_buffer:
            self._flush_segment(final=True)
        self._stop_event.set()
        if self._transcribe_thread:
            self._transcribe_thread.join(timeout=10)

    def _audio_callback(self, indata: np.ndarray, frames: int, _ti, _st) -> None:
        if self._stop_event.is_set():
            return
        chunk = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        self._last_rms = rms
        is_speech = rms >= SILENCE_RMS_THRESH

        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                self._segment_start = time.time() - self._session_start
                self._silence_chunks = 0
            self._speech_buffer.append(chunk)
            self._speech_chunks += 1
            self._silence_chunks = 0
            if self._speech_chunks * CHUNK_MS >= MAX_SEGMENT_MS:
                self._flush_segment(final=False)
        else:
            if self._in_speech:
                self._speech_buffer.append(chunk)
                self._silence_chunks += 1
                if self._silence_chunks * CHUNK_MS >= MIN_SILENCE_MS:
                    if self._speech_chunks * CHUNK_MS >= MIN_SPEECH_MS:
                        self._flush_segment(final=True)
                    else:
                        self._reset_buffer()

    def _flush_segment(self, final: bool) -> None:
        if not self._speech_buffer:
            return
        audio = np.concatenate(self._speech_buffer)
        start = self._segment_start
        end = time.time() - self._session_start
        self._segment_queue.append((audio, start, end, final))
        self._reset_buffer()

    def _reset_buffer(self) -> None:
        self._speech_buffer.clear()
        self._speech_chunks = 0
        self._silence_chunks = 0
        self._in_speech = False

    def _transcription_loop(self) -> None:
        while not self._stop_event.is_set() or self._segment_queue:
            if not self._segment_queue:
                time.sleep(0.05)
                continue
            audio, start, end, final = self._segment_queue.popleft()
            text = self._transcribe(audio)
            if text:
                self._on_segment(start, end, text, final)

    def _transcribe(self, audio: np.ndarray) -> str:
        try:
            segments, _ = self._transcriber.transcribe_array(
                audio,
                language=self._language,
                beam_size=3,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception:
            return ""
