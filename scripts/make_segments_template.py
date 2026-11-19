#!/usr/bin/env python
"""Create an editable speaker-segment template for a whole episode.

Runs voice-activity detection (VAD) on the audio master and writes real speech
timings. VAD only finds WHEN someone speaks, never WHO: every segment gets the
placeholder speaker "TODO" unless you opt in to ECAPA clustering (--diarize
ecapa), whose Speaker_N labels are temporary acoustic clusters you must review.

Workflow (three speakers):
  1. python scripts/make_segments_template.py input/audio/episode.wav --project input/project.json
  2. Listen, then edit input/episode.segments.tsv: replace TODO with a speaker_id
     from the project (or a character id such as char_1). Fix any boundary by hand.
  3. python scripts/make_segments_template.py input/audio/episode.wav --project input/project.json \
         --from-tsv input/episode.segments.tsv --merge-output input/project.json

Limits: VAD cannot see overlapping speech, so interruptions inside another
person's turn are NOT separate segments. Add those by hand if you need them
(note: overlapping intervals are withheld from character audio tracks).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ai_podcast_studio.audio import AudioError, inspect_audio, read_audio, validate_segments
from ai_podcast_studio.config import load_project
from ai_podcast_studio.vad import OptionalBackendError, run_vad

PLACEHOLDER = "TODO"
TSV_HEADER = (
    "# start\tend\tspeaker_id\n"
    "# Replace TODO with a speaker_id from your project (e.g. Speaker_0) or a character id (e.g. char_1).\n"
    "# Fix start/end by hand if needed. Blank lines and text after '#' are ignored.\n"
)


def split_long_regions(regions: list[tuple[float, float]], audio: np.ndarray, rate: int,
                       max_seconds: float, frame_ms: int = 30,
                       min_piece_seconds: float = 0.3) -> list[tuple[float, float]]:
    """Cut regions longer than ``max_seconds`` at the quietest frame near the limit.

    Long unbroken stretches are usually several turns with tiny pauses, so a
    shorter piece is easier to label correctly.
    """
    if max_seconds <= 0:
        return list(regions)
    frame = max(1, round(rate * frame_ms / 1000))
    out: list[tuple[float, float]] = []
    for start, end in regions:
        while end - start > max_seconds:
            lo = int((start + 0.5 * max_seconds) * rate)
            hi = min(len(audio), int((start + max_seconds) * rate))
            usable = (hi - lo) // frame
            if usable < 1:
                split = start + max_seconds
            else:
                blocks = audio[lo:lo + usable * frame].reshape(usable, frame)
                quiet = int(np.argmin(np.sqrt(np.mean(np.square(blocks, dtype=np.float64), axis=1))))
                split = (lo + quiet * frame + frame / 2) / rate
            out.append((start, split))
            start = split
        if end - start >= min_piece_seconds or not out:
            out.append((start, end))
        elif out:  # tiny remainder: attach to the previous piece
            out[-1] = (out[-1][0], end)
    return out


def build_segments(regions: list[tuple[float, float]], label: str = PLACEHOLDER) -> list[dict[str, Any]]:
    return [{"start": round(s, 3), "end": round(e, 3), "speaker_id": label} for s, e in regions if e > s]


def assign_labels(segments: list[dict[str, Any]], turns: list[dict[str, Any]],
                  min_fraction: float = 0.3) -> tuple[list[dict[str, Any]], int]:
    """Label each VAD segment with the diarization turn that overlaps it most."""
    labelled, unresolved = [], 0
    for seg in segments:
        length = seg["end"] - seg["start"]
        overlap: dict[str, float] = {}
        for turn in turns:
            shared = min(seg["end"], float(turn["end"])) - max(seg["start"], float(turn["start"]))
            if shared > 0:
                overlap[str(turn["speaker_id"])] = overlap.get(str(turn["speaker_id"]), 0.0) + shared
        best = max(overlap, key=overlap.get) if overlap else None
        if best is not None and overlap[best] >= min_fraction * length:
            labelled.append({**seg, "speaker_id": best})
        else:
            labelled.append({**seg, "speaker_id": PLACEHOLDER})
            unresolved += 1
    return labelled, unresolved


def write_tsv(path: Path, segments: list[dict[str, Any]]) -> None:
    lines = [TSV_HEADER]
    for seg in segments:
        lines.append(f"{seg['start']:.3f}\t{seg['end']:.3f}\t{seg['speaker_id']}\t# {seg['end'] - seg['start']:.2f}s\n")
    path.write_text("".join(lines), encoding="utf-8")


def resolve_label(label: str, project: dict[str, Any] | None) -> str:
    """Accept a speaker_id directly, or a character id mapped through the project."""
    characters = (project or {}).get("characters", {})
    if label in characters:
        speaker = characters[label].get("speaker_id")
        if not speaker:
            raise ValueError(f"Character {label!r} has no speaker_id in the project; set one first")
        return str(speaker)
    return label


def read_tsv(path: Path, project: dict[str, Any] | None) -> list[dict[str, Any]]:
    segments = []
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"{path}:{number}: expected 'start end speaker_id', got {raw!r}")
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: start/end must be numbers (seconds)") from exc
        segments.append({"start": start, "end": end, "speaker_id": resolve_label(parts[2], project)})
    if not segments:
        raise ValueError(f"{path} contains no segments")
    return segments


def merge_into_project(project_path: Path, segments: list[dict[str, Any]], output: Path) -> None:
    project = load_project(project_path)
    project["segments"] = segments
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(project, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    load_project(output)  # re-validate what was written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a speaker-segment template from real VAD timings.")
    p.add_argument("audio", type=Path, help="Audio master (WAV/FLAC)")
    p.add_argument("--project", type=Path, help="Project JSON (for character ids and merging)")
    p.add_argument("--out-dir", type=Path, default=Path("input"), help="Where to write the template files")
    p.add_argument("--vad", choices=("energy", "silero", "auto"), default="energy")
    p.add_argument("--split-gap", type=float, default=0.3, help="Silence (s) that separates two segments")
    p.add_argument("--min-speech", type=float, default=0.25, help="Shortest segment to keep (s)")
    p.add_argument("--max-segment", type=float, default=12.0, help="Split longer segments (s); 0 disables")
    p.add_argument("--diarize", choices=("none", "ecapa"), default="none",
                   help="Pre-label with ECAPA clusters (needs the neural extra; review the result)")
    p.add_argument("--num-speakers", type=int, default=3)
    p.add_argument("--cache-dir", type=Path, default=Path("models/cache/spkrec-ecapa-voxceleb"))
    p.add_argument("--from-tsv", type=Path, help="Convert an edited TSV back to JSON (and merge)")
    p.add_argument("--merge-output", type=Path, help="Write the full project with these segments (needs --project)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.merge_output and not args.project:
        print("--merge-output requires --project", file=sys.stderr)
        return 2
    try:
        duration = inspect_audio(args.audio).duration
        project = load_project(args.project) if args.project else None
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.audio.stem
        json_path = args.out_dir / f"{stem}.segments.json"
        tsv_path = args.out_dir / f"{stem}.segments.tsv"
        notes: list[str] = []
        if args.from_tsv:
            segments = validate_segments(read_tsv(args.from_tsv, project), duration)
            remaining = sum(1 for s in segments if s["speaker_id"] == PLACEHOLDER)
            if remaining:
                notes.append(f"{remaining} segment(s) still labelled {PLACEHOLDER}; they stay unmapped and silent.")
        else:
            audio, rate = read_audio(args.audio, mono=True)
            vad = run_vad(audio, rate, backend=args.vad,
                          min_speech_duration_ms=int(args.min_speech * 1000),
                          min_silence_duration_ms=int(args.split_gap * 1000))
            notes += vad.get("warnings", [])
            regions = [(float(r["start"]), float(r["end"])) for r in vad["regions"]]
            regions = split_long_regions(regions, audio, rate, args.max_segment)
            segments = validate_segments(build_segments(regions), duration)
            if not segments:
                print("No speech detected; check the audio level or try --vad silero.", file=sys.stderr)
                return 1
            if args.diarize == "ecapa":
                from ai_podcast_studio.diarization import diarize_audio
                result = diarize_audio(audio, rate, vad["regions"], expected_speakers=args.num_speakers,
                                       cache_dir=args.cache_dir)
                notes += result.get("warnings", [])
                segments, unresolved = assign_labels(segments, result["segments"])
                if unresolved:
                    notes.append(f"{unresolved} segment(s) had no clear cluster and remain {PLACEHOLDER}.")
            write_tsv(tsv_path, segments)
            print(f"Wrote editable template: {tsv_path}")
        json_path.write_text(json.dumps({"segments": segments}, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote segments JSON:    {json_path}")
        if args.merge_output:
            merge_into_project(args.project, segments, args.merge_output)
            print(f"Wrote full project:     {args.merge_output}")
        speech = sum(s["end"] - s["start"] for s in segments)
        print(f"{len(segments)} segments, {speech:.1f}s of speech in {duration:.1f}s audio "
              f"({100 * speech / duration:.0f}%).")
        for note in notes:
            print("NOTE:", note)
        if not args.from_tsv:
            print("VAD finds when people speak, not who. Edit the TSV, then re-run with --from-tsv.")
        return 0
    except (AudioError, OptionalBackendError, ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
