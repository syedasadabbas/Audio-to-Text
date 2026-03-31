"""
Audio preprocessing: format normalization and chunking for long files.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

from pydub import AudioSegment

TARGET_SAMPLE_RATE = 16_000

# 10-minute chunks keep peak memory flat regardless of file length
CHUNK_LENGTH_MS = 10 * 60 * 1000

# Overlap between consecutive chunks to avoid losing words at boundaries
OVERLAP_MS = 5 * 1000

CHUNK_THRESHOLD_MS = CHUNK_LENGTH_MS


def load_and_normalize(file_path: str) -> AudioSegment:
    """
    Load any ffmpeg-supported audio file and convert it to 16kHz mono PCM.

    Raises FileNotFoundError if the path doesn't exist.
    Raises pydub.exceptions.CouldntDecodeError if the file can't be decoded.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio = AudioSegment.from_file(str(path))

    if audio.channels > 1:
        audio = audio.set_channels(1)
    if audio.frame_rate != TARGET_SAMPLE_RATE:
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)

    return audio


def should_chunk(audio: AudioSegment) -> bool:
    return len(audio) > CHUNK_THRESHOLD_MS


def iter_chunks(audio: AudioSegment) -> Generator[tuple[AudioSegment, float], None, None]:
    """
    Yield (chunk, start_offset_seconds) pairs.

    Chunks overlap by OVERLAP_MS so words at boundaries aren't lost.
    The caller deduplicates segments that fall inside the overlap window.
    """
    total_ms = len(audio)
    start_ms = 0

    while start_ms < total_ms:
        end_ms = min(start_ms + CHUNK_LENGTH_MS, total_ms)
        chunk = audio[start_ms:end_ms]
        yield chunk, start_ms / 1000.0

        if end_ms == total_ms:
            break
        # Step forward by chunk length minus overlap so the next chunk
        # re-covers the tail of this one.
        start_ms += CHUNK_LENGTH_MS - OVERLAP_MS


def export_to_temp_wav(audio: AudioSegment) -> str:
    """Write an AudioSegment to a temporary WAV file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp.name, format="wav")
    tmp.close()
    return tmp.name


def cleanup(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
