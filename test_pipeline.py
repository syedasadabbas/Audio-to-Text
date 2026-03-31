"""
Tests for the transcription pipeline.

These tests are split into two tiers:
  - Unit tests that run without a GPU or Whisper model (mock the model).
  - Integration tests (marked slow) that exercise the real Whisper path
    and require a real audio file.

Run unit tests only:
  pytest test_pipeline.py -m "not slow"

Run everything (needs ffmpeg + internet for model download on first run):
  pytest test_pipeline.py
"""

import dataclasses
import json
import math
import os
import struct
import tempfile
import types
import wave
from unittest.mock import MagicMock, patch

import pytest

from audio_processor import (
    CHUNK_LENGTH_MS,
    OVERLAP_MS,
    TARGET_SAMPLE_RATE,
    export_to_temp_wav,
    iter_chunks,
    load_and_normalize,
    should_chunk,
)
from transcriber import Segment, Transcriber, TranscriptionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_wav(duration_seconds: float, sample_rate: int = 16_000) -> str:
    """Write a silent WAV file to a temp path and return the path."""
    num_frames = int(duration_seconds * sample_rate)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    tmp.close()
    return tmp.name


def fake_whisper_segment(start, end, text, avg_logprob=-0.3):
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    seg.avg_logprob = avg_logprob
    return seg


# ---------------------------------------------------------------------------
# audio_processor tests
# ---------------------------------------------------------------------------

class TestLoadAndNormalize:
    def test_loads_wav_and_normalises_to_16khz_mono(self):
        path = make_wav(2.0, sample_rate=44_100)
        try:
            audio = load_and_normalize(path)
            assert audio.channels == 1
            assert audio.frame_rate == TARGET_SAMPLE_RATE
        finally:
            os.unlink(path)

    def test_raises_if_file_missing(self):
        with pytest.raises(FileNotFoundError):
            load_and_normalize("/nonexistent/path/audio.wav")

    def test_duration_is_approximately_correct(self):
        path = make_wav(5.0)
        try:
            audio = load_and_normalize(path)
            assert abs(len(audio) / 1000.0 - 5.0) < 0.1
        finally:
            os.unlink(path)


class TestChunking:
    def _make_audio(self, duration_ms: int):
        from pydub import AudioSegment
        return AudioSegment.silent(duration=duration_ms, frame_rate=TARGET_SAMPLE_RATE)

    def test_short_audio_does_not_require_chunking(self):
        audio = self._make_audio(5 * 60 * 1000)  # 5 min
        assert not should_chunk(audio)

    def test_long_audio_requires_chunking(self):
        audio = self._make_audio(CHUNK_LENGTH_MS + 1)
        assert should_chunk(audio)

    def test_chunks_cover_full_duration(self):
        duration_ms = int(2.5 * CHUNK_LENGTH_MS)
        audio = self._make_audio(duration_ms)
        chunks = list(iter_chunks(audio))
        assert len(chunks) >= 2
        # Last chunk's end must reach or exceed total duration
        last_chunk, last_offset = chunks[-1]
        last_end_ms = last_offset * 1000 + len(last_chunk)
        assert last_end_ms >= duration_ms

    def test_chunk_offsets_are_monotonically_increasing(self):
        audio = self._make_audio(int(2.2 * CHUNK_LENGTH_MS))
        offsets = [offset for _, offset in iter_chunks(audio)]
        assert offsets == sorted(offsets)
        assert offsets[0] == 0.0

    def test_consecutive_chunks_overlap(self):
        audio = self._make_audio(int(1.5 * CHUNK_LENGTH_MS))
        chunks = list(iter_chunks(audio))
        assert len(chunks) == 2
        _, offset1 = chunks[0]
        _, offset2 = chunks[1]
        # Gap between chunk starts should be less than CHUNK_LENGTH_MS
        gap_ms = (offset2 - offset1) * 1000
        assert gap_ms < CHUNK_LENGTH_MS


# ---------------------------------------------------------------------------
# transcriber unit tests (model is mocked)
# ---------------------------------------------------------------------------

class TestTranscriptionResult:
    def test_as_dict_shape(self):
        result = TranscriptionResult(
            file_path="test.wav",
            language="en",
            duration=10.5,
            segments=[
                Segment(start=0.0, end=2.1, text="Hello world", confidence=0.9),
                Segment(start=2.5, end=5.0, text="How are you", confidence=0.85),
            ],
        )
        d = result.as_dict()
        assert d["language"] == "en"
        assert d["duration_seconds"] == 10.5
        assert len(d["segments"]) == 2
        assert d["segments"][0]["start"] == 0.0
        assert d["segments"][0]["text"] == "Hello world"
        assert "confidence" in d["segments"][0]

    def test_as_dict_strips_whitespace_from_text(self):
        result = TranscriptionResult(
            file_path="x.wav",
            language="en",
            duration=5.0,
            segments=[Segment(start=0, end=1, text="  padded  ", confidence=1.0)],
        )
        assert result.as_dict()["segments"][0]["text"] == "padded"


class TestTranscriberWithMock:
    def _make_mock_model(self, segments):
        """Return a WhisperModel mock that yields the given segments."""
        info = MagicMock()
        info.language = "en"
        model = MagicMock()
        model.transcribe.return_value = (iter(segments), info)
        return model

    @patch("transcriber.WhisperModel")
    def test_short_file_transcribed_without_chunking(self, MockWhisper):
        wav_path = make_wav(3.0)
        fake_segs = [
            fake_whisper_segment(0.0, 1.5, " Hello"),
            fake_whisper_segment(1.6, 3.0, " world"),
        ]
        MockWhisper.return_value = self._make_mock_model(fake_segs)

        try:
            t = Transcriber()
            result = t.transcribe(wav_path)
        finally:
            os.unlink(wav_path)

        assert result.language == "en"
        assert len(result.segments) == 2
        assert result.segments[0].text == " Hello"
        assert result.segments[0].start == pytest.approx(0.0)

    @patch("transcriber.WhisperModel")
    def test_segments_sorted_by_start_time(self, MockWhisper):
        wav_path = make_wav(3.0)
        # Return segments out of order
        fake_segs = [
            fake_whisper_segment(2.0, 3.0, " second"),
            fake_whisper_segment(0.0, 1.5, " first"),
        ]
        MockWhisper.return_value = self._make_mock_model(fake_segs)

        try:
            t = Transcriber()
            result = t.transcribe(wav_path)
        finally:
            os.unlink(wav_path)

        starts = [s.start for s in result.segments]
        assert starts == sorted(starts)

    @patch("transcriber.WhisperModel")
    def test_confidence_is_within_zero_one(self, MockWhisper):
        wav_path = make_wav(2.0)
        fake_segs = [fake_whisper_segment(0.0, 2.0, " test", avg_logprob=-0.8)]
        MockWhisper.return_value = self._make_mock_model(fake_segs)

        try:
            t = Transcriber()
            result = t.transcribe(wav_path)
        finally:
            os.unlink(wav_path)

        c = result.segments[0].confidence
        assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# Integration tests (real Whisper, require ffmpeg)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestIntegration:
    """
    Skipped in unit test runs. Invoke with: pytest -m slow

    These require ffmpeg installed and will download the 'tiny' model on
    first run (~75MB).
    """

    def test_transcribes_generated_wav(self):
        """
        A silent WAV won't produce meaningful text, but the pipeline should
        complete without errors and return an empty or near-empty transcript.
        """
        wav_path = make_wav(3.0)
        try:
            t = Transcriber(model_size="tiny")
            result = t.transcribe(wav_path)
            assert isinstance(result, TranscriptionResult)
            assert result.duration == pytest.approx(3.0, abs=0.2)
            # Silence may or may not yield segments — either is acceptable
            assert isinstance(result.segments, list)
        finally:
            os.unlink(wav_path)
