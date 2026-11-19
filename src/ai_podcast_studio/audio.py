"""Audio inspection, conservative energy VAD and full-length character tracks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


class AudioError(ValueError):
    """Raised when an audio input cannot be safely processed."""


@dataclass(frozen=True)
class AudioInfo:
    path: str
    sample_rate: int
    channels: int
    frames: int
    duration: float
    format: str
    subtype: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_audio(path: Path) -> AudioInfo:
    """Read container metadata without decoding the complete media file."""
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"Audio file not found: {path}")
    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError) as exc:
        raise AudioError(f"Unsupported or damaged audio file {path}: {exc}") from exc
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise AudioError(f"Audio has invalid dimensions: {path}")
    return AudioInfo(str(path), int(info.samplerate), int(info.channels), int(info.frames),
                     float(info.duration), str(info.format), str(info.subtype))


def read_audio(path: Path, *, mono: bool = False) -> tuple[np.ndarray, int]:
    """Decode to float32; input duration and absolute timing are preserved."""
    inspect_audio(path)
    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (RuntimeError, OSError) as exc:
        raise AudioError(f"Could not decode audio {path}: {exc}") from exc
    if data.size == 0 or not np.isfinite(data).all():
        raise AudioError("Audio is empty or contains NaN/Inf samples")
    if mono:
        data = data.mean(axis=1, dtype=np.float32)
    return data, int(sample_rate)


def energy_vad(audio: np.ndarray, sample_rate: int, *, frame_ms: int = 30,
               min_speech_ms: int = 120, max_gap_ms: int = 180,
               absolute_floor: float = 0.008) -> list[dict[str, Any]]:
    """Conservative RMS VAD. It is not a neural speech recognizer or diarizer."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if sample_rate <= 0 or frame_ms <= 0:
        raise ValueError("sample_rate and frame_ms must be positive")
    frame = max(1, round(sample_rate * frame_ms / 1000))
    if samples.size == 0:
        return []
    usable = (samples.size // frame) * frame
    chunks = samples[:usable].reshape(-1, frame) if usable else samples.reshape(1, -1)
    rms = np.sqrt(np.mean(np.square(chunks, dtype=np.float32), axis=1) + 1e-12)
    floor = float(np.quantile(rms, 0.2)) if rms.size > 4 else float(rms.min())
    threshold = max(float(absolute_floor), floor * 3.0)
    active = rms >= threshold
    gap_frames = max(0, round(max_gap_ms / frame_ms))
    # Bridge only short internal gaps; no speaker labels are inferred here.
    idx = 0
    while idx < len(active):
        if active[idx]:
            idx += 1
            continue
        end = idx
        while end < len(active) and not active[end]:
            end += 1
        if idx > 0 and end < len(active) and end - idx <= gap_frames:
            active[idx:end] = True
        idx = end
    min_frames = max(1, round(min_speech_ms / frame_ms))
    result: list[dict[str, Any]] = []
    idx = 0
    while idx < len(active):
        if not active[idx]:
            idx += 1
            continue
        end = idx + 1
        while end < len(active) and active[end]:
            end += 1
        if end - idx >= min_frames:
            result.append({"start": idx * frame_ms / 1000.0,
                           "end": min(samples.size / sample_rate, end * frame_ms / 1000.0),
                           "method": "energy_vad", "confidence": None})
        idx = end
    return result


def validate_segments(segments: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    """Validate manual speaker intervals; overlapping intervals remain intact."""
    if not np.isfinite(duration) or duration <= 0:
        raise AudioError("Audio duration must be a finite positive value")
    cleaned: list[dict[str, Any]] = []
    for i, raw in enumerate(segments):
        try:
            start, end = float(raw["start"]), float(raw["end"])
            speaker = str(raw["speaker_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AudioError(f"Invalid segment #{i}: expected start, end and speaker_id") from exc
        if (not np.isfinite([start, end]).all() or start < 0 or start >= duration or
                end <= start or end > duration + 0.02 or not speaker):
            raise AudioError(f"Invalid segment #{i}: interval must be within audio duration")
        clipped_end = min(duration, end)
        if clipped_end <= start:
            raise AudioError(f"Invalid segment #{i}: interval becomes empty after clipping to audio duration")
        cleaned.append({**raw, "start": start, "end": clipped_end, "speaker_id": speaker})
    return sorted(cleaned, key=lambda item: (item["start"], item["end"], item["speaker_id"]))


def make_character_tracks(audio: np.ndarray, sample_rate: int,
                          project: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Write duration-aligned tracks for confidently assigned non-overlap intervals.

    This is interval masking, not source separation. Samples during overlapping
    speakers are withheld from character tracks unless a real separator is added;
    the original master remains the authoritative soundtrack.
    """
    if audio.ndim == 1:
        audio = audio[:, None]
    audio = np.asarray(audio, dtype=np.float32)
    duration = len(audio) / sample_rate
    segments = validate_segments(project.get("segments", []), duration)
    characters = project.get("characters", {})
    if not characters:
        raise AudioError("Project must define at least one character")
    char_by_speaker: dict[str, str] = {}
    for char_id, spec in characters.items():
        speaker = spec.get("speaker_id")
        if speaker:
            if speaker in char_by_speaker:
                raise AudioError(f"Duplicate speaker mapping: {speaker}")
            char_by_speaker[str(speaker)] = str(char_id)
    mapped = set(char_by_speaker)
    masks = {char_id: np.zeros(len(audio), dtype=bool) for char_id in characters}
    overlap_mask = np.zeros(len(audio), dtype=bool)
    known_mask = np.zeros(len(audio), dtype=bool)
    sample_segments = []
    for segment in segments:
        start = max(0, int(round(segment["start"] * sample_rate)))
        end = min(len(audio), int(round(segment["end"] * sample_rate)))
        if end <= start:
            continue
        speaker = segment["speaker_id"]
        sample_segments.append((start, end, speaker))
        if speaker in mapped:
            masks[char_by_speaker[speaker]][start:end] = True
    # Any sample attributed to more than one distinct diarized speaker is ambiguous.
    all_speakers = {segment["speaker_id"] for segment in segments}
    speaker_masks = {speaker: np.zeros(len(audio), dtype=bool) for speaker in all_speakers}
    for start, end, speaker in sample_segments:
        speaker_masks[speaker][start:end] = True
        if speaker in mapped:
            known_mask[start:end] = True
    active_once = np.zeros(len(audio), dtype=bool)
    overlap_mask = np.zeros(len(audio), dtype=bool)
    for speaker_mask in speaker_masks.values():
        overlap_mask |= active_once & speaker_mask
        active_once |= speaker_mask
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for char_id, mask in masks.items():
        safe_mask = mask & ~overlap_mask
        track = np.zeros_like(audio)
        track[safe_mask] = audio[safe_mask]
        out = output_dir / f"{char_id}.wav"
        sf.write(str(out), track, sample_rate, subtype="PCM_16")
        written[char_id] = {"path": str(out), "frames": len(track), "sample_rate": sample_rate,
                            "channels": track.shape[1]}
    overlap_seconds = float(overlap_mask.sum() / sample_rate)
    return {"tracks": written, "master_audio_required": True,
            "method": "manual_interval_masking_not_source_separation",
            "unmapped_speakers": sorted({s["speaker_id"] for s in segments} - mapped),
            "overlap_seconds_excluded_from_tracks": overlap_seconds,
            "unassigned_seconds": float((~known_mask).sum() / sample_rate),
            "warnings": (["Source separation was not performed; use the original audio master."] +
                         (["Overlapping speech samples were left out of character tracks for manual review."] if overlap_seconds else []))}
