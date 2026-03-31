"""
Core transcription engine built on faster-whisper.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np   # only needed for the transcribe_array type hint

from faster_whisper import WhisperModel

from audio_processor import (
    cleanup,
    export_to_temp_wav,
    iter_chunks,
    load_and_normalize,
    should_chunk,
    OVERLAP_MS,
)


@dataclasses.dataclass
class Word:
    start: float
    end: float
    text: str


@dataclasses.dataclass
class Segment:
    start: float   # seconds from the beginning of the original file
    end: float
    text: str
    confidence: float  # average log-prob converted to 0-1 range
    words: list[Word] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TranscriptionResult:
    file_path: str
    language: str
    duration: float
    segments: list[Segment]

    def as_dict(self) -> dict:
        return {
            "file": self.file_path,
            "language": self.language,
            "duration_seconds": round(self.duration, 2),
            "segments": [
                {
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "text": s.text.strip(),
                    "confidence": round(s.confidence, 3),
                    "words": [
                        {"start": round(w.start, 3), "end": round(w.end, 3), "text": w.text}
                        for w in s.words
                    ],
                }
                for s in self.segments
            ],
        }


class Transcriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default",
    ):
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        self.model_size = model_size

    def transcribe(self, file_path: str, language: Optional[str] = None) -> TranscriptionResult:
        """
        Transcribe an audio file. Handles format conversion and chunking
        transparently — callers only see a single merged result.

        Args:
            file_path: Path to any ffmpeg-readable audio file.
            language:  ISO-639-1 code (e.g. "en"). None = auto-detect.
        """
        audio = load_and_normalize(file_path)
        duration_seconds = len(audio) / 1000.0

        if should_chunk(audio):
            segments, detected_lang = self._transcribe_chunked(audio, language)
        else:
            segments, detected_lang = self._transcribe_audio(audio, offset=0.0, language=language)

        # Sort by start time — sequential chunks should already be ordered,
        # but an explicit sort is cheap and safe.
        segments.sort(key=lambda s: s.start)

        return TranscriptionResult(
            file_path=file_path,
            language=detected_lang,
            duration=duration_seconds,
            segments=segments,
        )

    def transcribe_array(
        self,
        audio: "np.ndarray",
        language: Optional[str] = None,
        beam_size: int = 3,
    ) -> tuple[list[Segment], str]:
        """
        Transcribe a float32 numpy array directly (no file I/O, no chunking).
        Used by the real-time pipeline where audio is already in memory.

        Returns (segments, detected_language).
        """
        raw_segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            vad_filter=False,
            word_timestamps=False,
        )
        segments = [
            Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                confidence=self._avg_confidence(seg),
            )
            for seg in raw_segments
        ]
        return segments, info.language

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transcribe_audio(
        self,
        audio,
        offset: float,
        language: Optional[str],
    ) -> tuple[list[Segment], str]:
        """Run Whisper on a single AudioSegment and return segments + language."""
        tmp_path = export_to_temp_wav(audio)
        try:
            raw_segments, info = self._model.transcribe(
                tmp_path,
                language=language,
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
            )
            detected_lang = info.language
            segments = []
            for seg in raw_segments:
                words = [
                    Word(start=w.start + offset, end=w.end + offset, text=w.word)
                    for w in (seg.words or [])
                ]
                segments.append(Segment(
                    start=seg.start + offset,
                    end=seg.end + offset,
                    text=seg.text,
                    confidence=self._avg_confidence(seg),
                    words=words,
                ))
        finally:
            cleanup(tmp_path)

        return segments, detected_lang

    def _transcribe_chunked(
        self,
        audio,
        language: Optional[str],
    ) -> tuple[list[Segment], str]:
        """
        Split audio into overlapping chunks, transcribe each, then merge.

        Segments whose start time falls within the overlap window of the
        previous chunk are dropped — that region is already covered with
        better context from the prior chunk.
        """
        all_segments: list[Segment] = []
        detected_lang = "en"
        overlap_seconds = OVERLAP_MS / 1000.0
        prev_chunk_end = 0.0

        for chunk, offset in iter_chunks(audio):
            segs, lang = self._transcribe_audio(chunk, offset=offset, language=language)
            detected_lang = lang

            for seg in segs:
                if all_segments and seg.start < prev_chunk_end - overlap_seconds:
                    continue
                all_segments.append(seg)

            if segs:
                prev_chunk_end = segs[-1].end

        return all_segments, detected_lang

    @staticmethod
    def _avg_confidence(segment) -> float:
        """Map faster-whisper's avg_logprob (negative) to a 0–1 score."""
        log_prob = getattr(segment, "avg_logprob", -0.5)
        clamped = max(-2.0, min(0.0, log_prob))
        return 1.0 + clamped / 2.0
