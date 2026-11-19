"""Command-line interface for AI Podcast Studio."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

from .pipeline import quality_check, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a layered cartoon podcast from an audio master.")
    parser.add_argument("mode", choices=("quality-check", "preview", "full-render"))
    parser.add_argument("--audio", type=Path, default=Path("input/audio/episode.wav"))
    parser.add_argument("--project", type=Path, default=Path("input/project.json"))
    parser.add_argument("--characters", type=Path, default=Path("assets/characters"))
    parser.add_argument("--backgrounds", type=Path, default=Path("assets/backgrounds"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--schema", type=Path, default=Path("config/project.schema.json"))
    parser.add_argument("--width", type=int, default=None, help="Override project resolution width")
    parser.add_argument("--height", type=int, default=None, help="Override project resolution height")
    parser.add_argument("--fps", type=int, default=None, help="Override project frame rate")
    parser.add_argument("--preview-scale", type=float, default=0.25)
    parser.add_argument("--no-resume", action="store_true", help="Rebuild cached audio tracks and cues")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.mode == "quality-check":
            report = quality_check(audio_path=args.audio, project_path=args.project,
                                   character_root=args.characters, background_root=args.backgrounds,
                                   schema_path=args.schema)
        else:
            report = run_pipeline(audio_path=args.audio, project_path=args.project,
                                  character_root=args.characters, background_root=args.backgrounds,
                                  output_dir=args.output, schema_path=args.schema,
                                  preview=args.mode == "preview", resume=not args.no_resume,
                                  width=args.width, height=args.height, fps=args.fps,
                                  preview_scale=args.preview_scale)
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        error_report = {"status": "error", "type": type(exc).__name__, "message": str(exc),
                        "traceback": traceback.format_exc()}
        (args.output / "error_report.json").write_text(json.dumps(error_report, indent=2), encoding="utf-8")
        logging.exception("Pipeline failed; error report written to %s", args.output / "error_report.json")
        return 2


if __name__ == "__main__":
    sys.exit(main())
