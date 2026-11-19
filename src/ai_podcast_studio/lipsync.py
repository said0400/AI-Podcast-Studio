"""Audio-derived mouth timing with optional Rhubarb Lip Sync integration."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
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


def envelope_cues(audio_path: Path, *, bin_seconds: float = 0.08, threshold: float = 0.012) -> list[dict[str, Any]]:
    """Create open/closed cues from the waveform envelope, not from transcript text.

    This fallback is timing-only (A while audible, X while quiet), not phoneme
    recognition. Use Rhubarb for phonetic mouth-shape estimates.
    """
    samples, rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if not len(mono):
        return []
    width = max(1, round(rate * bin_seconds))
    cues: list[dict[str, Any]] = []
    last = None
    for offset in range(0, len(mono), width):
        segment = mono[offset:offset + width]
        rms = float(np.sqrt(np.mean(segment * segment) + 1e-12))
        value = "A" if rms >= threshold else "X"
        begin = offset / rate
        end = min(len(mono) / rate, (offset + len(segment)) / rate)
        if last == value and cues:
            cues[-1]["end"] = end
        else:
            cues.append({"start": begin, "end": end, "value": value})
            last = value
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
