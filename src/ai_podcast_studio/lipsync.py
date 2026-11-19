"""Audio-derived mouth timing with optional Rhubarb Lip Sync integration."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

MOUTH_SHAPES = set("ABCDEFGHX")


def sanitize_cues(cues: Any, duration: float) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep only finite, in-range cues and resolve overlaps in stable time order."""
    if not isinstance(cues, list):
        raise ValueError("Lip-sync cues must be a list")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Audio duration must be finite and positive")
    valid: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, cue in enumerate(cues):
        try:
            start, end = float(cue["start"]), float(cue["end"])
            value = str(cue["value"]).upper()
        except (KeyError, TypeError, ValueError, OverflowError):
            warnings.append(f"Cue {index} is missing valid start/end/value fields and was ignored.")
            continue
        if (not math.isfinite(start) or not math.isfinite(end) or start < 0 or start >= duration or
                end <= start or end > duration + 0.02 or value not in MOUTH_SHAPES):
            warnings.append(f"Cue {index} has invalid bounds or mouth shape and was ignored.")
            continue
        valid.append({"start": start, "end": min(end, duration), "value": value})
    valid.sort(key=lambda item: (item["start"], item["end"]))
    clean: list[dict[str, Any]] = []
    for cue in valid:
        if clean and cue["start"] < clean[-1]["end"]:
            cue["start"] = clean[-1]["end"]
            if cue["end"] <= cue["start"]:
                warnings.append("An overlapping cue was fully covered by an earlier cue and was ignored.")
                continue
        clean.append(cue)
    return clean, warnings


# Relative "openness" of each Rhubarb mouth shape, used for audio-driven body motion.
MOUTH_INTENSITY = {"X": 0.0, "A": 0.3, "B": 0.5, "C": 0.7, "D": 1.0,
                   "E": 0.75, "F": 0.45, "G": 0.4, "H": 0.8}


def _merge_short_runs(values: list[str], min_bins: int) -> list[str]:
    """Absorb runs shorter than ``min_bins`` into the preceding run to stop flicker."""
    if min_bins <= 1 or not values:
        return values
    runs: list[list[Any]] = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    for index, run in enumerate(runs):
        if run[1] < min_bins and index > 0:
            run[0] = runs[index - 1][0]
    merged: list[str] = []
    for value, count in runs:
        merged.extend([value] * count)
    return merged


def envelope_cues(audio_path: Path, *, bin_seconds: float = 0.04, threshold: float = 0.012,
                  min_hold_seconds: float = 0.08) -> list[dict[str, Any]]:
    """Create graded mouth-opening cues from the smoothed waveform envelope.

    Quiet -> X; audible speech is graded A (barely open) .. D (wide open) relative
    to the recording's own loud passages. This follows loudness only; it is NOT
    phoneme recognition. Use Rhubarb for phonetic mouth-shape estimates.
    """
    samples, rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if not len(mono):
        return []
    width = max(1, round(rate * bin_seconds))
    bins = math.ceil(len(mono) / width)
    padded = np.zeros(bins * width, dtype=np.float64)
    padded[:len(mono)] = mono
    sums = np.square(padded).reshape(bins, width).sum(axis=1)
    counts = np.full(bins, width, dtype=np.float64)
    counts[-1] = len(mono) - (bins - 1) * width
    rms = np.sqrt(sums / counts + 1e-12)
    if bins >= 3:
        rms = np.convolve(np.pad(rms, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    active = rms >= threshold
    reference = float(np.percentile(rms[active], 90)) if active.any() else 1.0
    level = rms / max(reference, 1e-9)
    values = ["X" if not active[i] else
              "A" if level[i] < 0.30 else "B" if level[i] < 0.55 else "C" if level[i] < 0.80 else "D"
              for i in range(bins)]
    values = _merge_short_runs(values, max(1, round(min_hold_seconds / bin_seconds)))
    cues: list[dict[str, Any]] = []
    duration = len(mono) / rate
    for index, value in enumerate(values):
        begin = index * width / rate
        end = min(duration, (index + 1) * width / rate)
        if cues and cues[-1]["value"] == value:
            cues[-1]["end"] = end
        else:
            cues.append({"start": begin, "end": end, "value": value})
    return cues


def extract_cues(audio_path: Path, output_path: Path, *, rhubarb_binary: str = "rhubarb") -> dict[str, Any]:
    """Use Rhubarb when installed; otherwise save clearly labelled envelope cues."""
    audio_path, output_path = Path(audio_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary = shutil.which(rhubarb_binary)
    fallback_reason = "Rhubarb is not installed."
    if binary:
        raw_path = output_path.with_suffix(".rhubarb.json")
        try:
            result = subprocess.run([binary, "-f", "json", "-o", str(raw_path), str(audio_path)],
                                    capture_output=True, text=True, timeout=900, check=False)
        except subprocess.TimeoutExpired:
            fallback_reason = "Rhubarb timed out after 900 seconds."
        else:
            if result.returncode != 0:
                fallback_reason = f"Rhubarb exited with code {result.returncode}: {result.stderr.strip()[:500]}"
            elif not raw_path.is_file():
                fallback_reason = "Rhubarb reported success but did not create JSON output."
            else:
                fallback_reason = "Rhubarb returned malformed JSON; using envelope fallback."
        if 'result' in locals() and result.returncode == 0 and raw_path.is_file():
            try:
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                cues = raw.get("mouthCues", [])
                normalized = [{"start": float(c["start"]), "end": float(c["end"]),
                               "value": str(c["value"])} for c in cues]
                duration = float(sf.info(str(audio_path)).duration)
                normalized, cue_warnings = sanitize_cues(normalized, duration)
                payload = {"method": "rhubarb", "phonetic": True, "cues": normalized}
                if cue_warnings:
                    payload["warnings"] = cue_warnings
                output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                return payload
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                fallback_reason = "Rhubarb output could not be parsed; using envelope fallback."
    cues = envelope_cues(audio_path)
    payload = {"method": "waveform_envelope_fallback", "phonetic": False,
               "warning": f"Timing-only open/closed cues; not phoneme-level lip sync. {fallback_reason}",
               "cues": cues}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def mouth_at(cues: list[dict[str, Any]], time_seconds: float) -> str:
    """Return the cue active at a given time, defaulting safely to closed mouth."""
    for cue in cues:
        if not isinstance(cue, dict):
            continue
        try:
            start, end = float(cue.get("start")), float(cue.get("end"))
        except (TypeError, ValueError, OverflowError):
            continue
        value = str(cue.get("value", "X")).upper()
        if math.isfinite(start) and math.isfinite(end) and value in MOUTH_SHAPES and start <= time_seconds < end:
            return value
    return "X"


def build_cue_index(cues: Any) -> dict[str, list]:
    """Sorted lookup arrays so per-frame mouth queries are O(log n), not O(n)."""
    rows: list[tuple[float, float, str]] = []
    for cue in cues if isinstance(cues, list) else []:
        if not isinstance(cue, dict):
            continue
        try:
            start, end = float(cue.get("start")), float(cue.get("end"))
        except (TypeError, ValueError, OverflowError):
            continue
        value = str(cue.get("value", "X")).upper()
        if math.isfinite(start) and math.isfinite(end) and end > start and value in MOUTH_SHAPES:
            rows.append((start, end, value))
    rows.sort()
    return {"starts": [r[0] for r in rows], "ends": [r[1] for r in rows], "values": [r[2] for r in rows]}


def indexed_mouth_at(index: dict[str, list], time_seconds: float) -> str:
    """Same result as ``mouth_at`` for non-overlapping cues, using a built index."""
    position = bisect_right(index["starts"], time_seconds) - 1
    if position >= 0 and time_seconds < index["ends"][position]:
        return index["values"][position]
    return "X"


def speech_energy(index: dict[str, list], time_seconds: float, *,
                  before: float = 0.25, after: float = 0.15, samples: int = 8) -> float:
    """Mean mouth openness (0..1) around a time; drives nods and emphasis.

    Looking slightly ahead lets head motion anticipate syllables instead of lagging them.
    """
    if not index["starts"]:
        return 0.0
    total = 0.0
    for step in range(samples):
        probe = time_seconds - before + (before + after) * step / max(1, samples - 1)
        total += MOUTH_INTENSITY.get(indexed_mouth_at(index, probe), 0.0)
    return total / samples
