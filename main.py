"""
CLI entry point for the transcription pipeline.

Usage examples:
  python main.py audio/interview.mp3
  python main.py audio/lecture.wav --model small --language en --output result.json
  python main.py audio/podcast.m4a --model medium --output -
"""

import argparse
import json
import sys

from transcriber import Transcriber


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Transcribe an audio file and return timestamped segments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("audio_file", help="Path to audio file (WAV, MP3, M4A, FLAC, …)")
    p.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: base). Larger = more accurate, slower.",
    )
    p.add_argument(
        "--language",
        default=None,
        metavar="LANG",
        help="Force source language (ISO 639-1, e.g. 'en'). Auto-detected if omitted.",
    )
    p.add_argument(
        "--output",
        default="-",
        metavar="FILE",
        help="Write JSON output to FILE. Use '-' for stdout (default).",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device (default: auto).",
    )
    p.add_argument(
        "--plain",
        action="store_true",
        help="Print plain text transcript instead of JSON.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    print(f"[pipeline] Loading model '{args.model}' on device '{args.device}' …", file=sys.stderr)
    transcriber = Transcriber(model_size=args.model, device=args.device)

    print(f"[pipeline] Transcribing: {args.audio_file}", file=sys.stderr)
    result = transcriber.transcribe(args.audio_file, language=args.language)

    print(
        f"[pipeline] Done. Language={result.language}, "
        f"Duration={result.duration:.1f}s, Segments={len(result.segments)}",
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
