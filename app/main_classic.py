"""
CLI entry point for the classical (non-AI) transcription pipeline.

Usage:
  python main_classic.py sample.wav
  python main_classic.py interview.mp3 --language en-US --output result.json
  python main_classic.py lecture.wav --offline
  python main_classic.py podcast.m4a --plain
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from transcriber_classic import ClassicTranscriber


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Transcribe audio using classical speech recognition (no AI model).",
    )
    p.add_argument("audio_file", help="Path to audio file (WAV, MP3, M4A, FLAC, …)")
    p.add_argument(
        "--language",
        default="en-US",
        metavar="LANG",
        help="BCP-47 language code for Google backend (default: en-US).",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Use PocketSphinx (offline) instead of Google Web Speech API.",
    )
    p.add_argument(
        "--output",
        default="-",
        metavar="FILE",
        help="Write JSON output to FILE. Use '-' for stdout (default).",
    )
    p.add_argument(
        "--plain",
        action="store_true",
        help="Print plain text instead of JSON.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    backend = "PocketSphinx (offline)" if args.offline else "Google Web Speech API"
    print(f"[pipeline] Backend: {backend}", file=sys.stderr)
    print(f"[pipeline] Transcribing: {args.audio_file}", file=sys.stderr)

    transcriber = ClassicTranscriber(offline=args.offline, language=args.language)
    result = transcriber.transcribe(args.audio_file)

    print(
        f"[pipeline] Done. Duration={result.duration:.1f}s, Segments={len(result.segments)}",
        file=sys.stderr,
    )

    if args.plain:
        output = " ".join(s.text.strip() for s in result.segments)
    else:
        output = json.dumps(result.as_dict(), ensure_ascii=False, indent=2)

    if args.output == "-":
        print(output)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"[pipeline] Output written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
