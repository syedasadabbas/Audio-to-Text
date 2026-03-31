"""
Audio preprocessing for the classical (non-AI) pipeline.
Segment boundaries are found by energy-based silence detection via pydub.
"""

import os
import tempfile
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

TARGET_SAMPLE_RATE = 16_000

SILENCE_THRESH_DBFS = -40   # works for clean recordings; raise for noisy audio
MIN_SILENCE_MS = 500
KEEP_SILENCE_MS = 200       # padding so edge words aren't clipped


def load_and_normalize(file_path: str) -> AudioSegment:
    """Load any ffmpeg-supported format and normalize to 16kHz mono."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio = AudioSegment.from_file(str(path))

    if audio.channels > 1:
        audio = audio.set_channels(1)
    if audio.frame_rate != TARGET_SAMPLE_RATE:
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)

    return audio


def split_into_segments(audio: AudioSegment) -> list[tuple[AudioSegment, float, float]]:
    """
    Split audio on silence and return (chunk, start_sec, end_sec) triples.
    Uses energy threshold (dBFS) to find speech regions — no ML involved.
    """
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=SILENCE_THRESH_DBFS,
        seek_step=10,   # ms granularity — smaller is more accurate but slower
    )

    if not nonsilent_ranges:
        return []

    segments = []
    for start_ms, end_ms in nonsilent_ranges:
        padded_start = max(0, start_ms - KEEP_SILENCE_MS)
        padded_end = min(len(audio), end_ms + KEEP_SILENCE_MS)
        chunk = audio[padded_start:padded_end]
        segments.append((chunk, padded_start / 1000.0, padded_end / 1000.0))

    return segments


def merge_short_segments(
    segments: list[tuple[AudioSegment, float, float]],
    min_duration_sec: float = 1.5,
) -> list[tuple[AudioSegment, float, float]]:
    """
    Merge segments shorter than min_duration_sec into the next one.
    The Google Speech API performs poorly on very short clips — merging
    short segments gives it more acoustic context.
    """
    if not segments:
        return []

    merged = []
    buffer_audio, buffer_start, buffer_end = segments[0]

    for chunk, start, end in segments[1:]:
        if buffer_end - buffer_start < min_duration_sec:
            buffer_audio = buffer_audio + chunk
            buffer_end = end
        else:
            merged.append((buffer_audio, buffer_start, buffer_end))
            buffer_audio, buffer_start, buffer_end = chunk, start, end

    merged.append((buffer_audio, buffer_start, buffer_end))
    return merged


def export_to_temp_wav(audio: AudioSegment) -> str:
    """Write AudioSegment to a temp WAV file, return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp.name, format="wav")
    tmp.close()
    return tmp.name


def cleanup(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
