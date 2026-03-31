"""
Classical (non-AI) transcription engine using the SpeechRecognition library.

Two backend options:
  1. Google Web Speech API (default) — free, requires internet, no API key needed
  2. PocketSphinx (offline=True) — fully offline, lower accuracy, English only
     Install with: pip install pocketsphinx

Timestamps come from silence detection (pydub RMS energy), not from the
speech recognizer — each non-silent region's position in the file is used
as its timestamp.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import speech_recognition as sr

sys.path.insert(0, str(Path(__file__).parent))

from audio_processor_classic import (
    cleanup,
    export_to_temp_wav,
    load_and_normalize,
    merge_short_segments,
    split_into_segments,
)


@dataclasses.dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclasses.dataclass
class TranscriptionResult:
    file_path: str
    language: str
    duration: float
    segments: list[Segment]
    backend: str

    def as_dict(self) -> dict:
        return {
            "file": self.file_path,
            "language": self.language,
            "duration_seconds": round(self.duration, 2),
            "backend": self.backend,
            "segments": [
                {
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "text": s.text.strip(),
                }
                for s in self.segments
            ],
        }


class ClassicTranscriber:
    def __init__(self, offline: bool = False, language: str = "en-US"):
        self._recognizer = sr.Recognizer()
        self._offline = offline
        self._language = language
        self._backend = "sphinx" if offline else "google"

        self._recognizer.dynamic_energy_threshold = True
        self._recognizer.pause_threshold = 0.5

    def transcribe(self, file_path: str) -> TranscriptionResult:
        audio = load_and_normalize(file_path)
        duration_seconds = len(audio) / 1000.0

        raw_segments = split_into_segments(audio)
        segments_to_transcribe = merge_short_segments(raw_segments)

        results: list[Segment] = []
        for chunk, start_sec, end_sec in segments_to_transcribe:
            text = self._transcribe_chunk(chunk)
            if text:
                results.append(Segment(start=start_sec, end=end_sec, text=text))

        return TranscriptionResult(
            file_path=file_path,
            language=self._language,
            duration=duration_seconds,
            segments=results,
            backend=self._backend,
        )

    def _transcribe_chunk(self, chunk) -> str:
        """Transcribe a single AudioSegment. Returns empty string on failure."""
        tmp_path = export_to_temp_wav(chunk)
        try:
            with sr.AudioFile(tmp_path) as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio_data = self._recognizer.record(source)

            if self._offline:
                return self._recognizer.recognize_sphinx(audio_data)
            else:
                return self._recognizer.recognize_google(
                    audio_data,
                    language=self._language,
                )

        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            raise RuntimeError(f"Google Speech API error: {e}") from e
        finally:
            cleanup(tmp_path)
