"""Audio track validation and cautious separated-source reconstruction."""
from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


class MixError(RuntimeError):
    """Raised when stems cannot be safely aligned and mixed."""


PCM_PEAK_LIMIT = 0.999


def _validate_pcm_stem(samples: np.ndarray, label: str) -> None:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > PCM_PEAK_LIMIT:
        raise MixError(f"{label} peak {peak:.4f} exceeds PCM stem limit {PCM_PEAK_LIMIT}; refusing silent clipping")


def write_source_stems(sources: np.ndarray, sample_rate: int, output_dir: Path) -> dict[str, str]:
    """Save separated mono sources with identical timeline length and sample rate."""
    sources = np.asarray(sources, dtype=np.float32)
    if (sources.ndim != 2 or sources.shape[0] == 0 or sources.shape[1] == 0 or
            sample_rate <= 0 or not np.isfinite(sources).all()):
        raise MixError("Expected finite [samples, sources] separation output")
    for source_index in range(sources.shape[1]):
        _validate_pcm_stem(sources[:, source_index], f"source_{source_index}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for source_index in range(sources.shape[1]):
        path = output_dir / f"source_{source_index}.wav"
        sf.write(str(path), sources[:, source_index], sample_rate, subtype="PCM_16")
        paths[f"source_{source_index}"] = str(path)
    return paths


def write_character_stems(sources: np.ndarray, source_rate: int, source_to_character: dict[str, str],
                          character_ids: set[str], output_dir: Path, target_rate: int) -> dict[str, Any]:
    """Map inspected source indices to characters only via explicit project JSON."""
    sources = np.asarray(sources, dtype=np.float32)
    if sources.ndim != 2:
        raise MixError("Expected [samples, sources] separated audio")
    seen_characters: set[str] = set()
    source_paths: dict[str, str] = {}
    output_dir = Path(output_dir)
    valid_sources = {f"source_{idx}" for idx in range(sources.shape[1])}
    for source_id, character_id in source_to_character.items():
        if source_id not in valid_sources:
            raise MixError(f"Unknown separated source identifier: {source_id}")
        if character_id not in character_ids:
            raise MixError(f"Unknown character identifier: {character_id}")
        if character_id in seen_characters:
            raise MixError(f"Multiple separated sources map to the same character: {character_id}")
        seen_characters.add(character_id)
    seen_characters.clear()
    prepared: dict[str, np.ndarray] = {}
    for source_id, character_id in source_to_character.items():
        seen_characters.add(character_id)
        samples = sources[:, int(source_id.split("_")[1])]
        if source_rate != target_rate:
            try:
                from scipy.signal import resample_poly
            except ImportError as exc:
                raise MixError("Resampling separated character stems requires scipy") from exc
            divisor = gcd(int(source_rate), int(target_rate))
            samples = resample_poly(samples, target_rate // divisor, source_rate // divisor).astype(np.float32)
        _validate_pcm_stem(samples, f"{source_id} mapped to {character_id}")
        prepared[character_id] = np.asarray(samples, dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    for character_id, samples in prepared.items():
        path = output_dir / f"{character_id}.wav"
        sf.write(str(path), samples, target_rate, subtype="PCM_16")
        source_paths[character_id] = str(path)
    return {"mapped_characters": source_paths,
            "unmapped_sources": sorted({f"source_{i}" for i in range(sources.shape[1])} - set(source_to_character)),
            "unassigned_characters": sorted(character_ids - seen_characters),
            "mapping_method": "explicit_project_json_source_to_character"}


def reconstruct_master(sources: np.ndarray, source_rate: int, target_rate: int,
                       output_path: Path, *, peak_limit: float = 0.999) -> dict[str, Any]:
    """Sum all separated stems only if aligned and unclipped after rate conversion.

    A rejected mix must not replace the original master. No loudness normalization
    or hidden clipping is applied.
    """
    sources = np.asarray(sources, dtype=np.float32)
    if sources.ndim != 2 or sources.shape[1] < 2 or not np.isfinite(sources).all():
        raise MixError("At least two finite separated stems are required for reconstruction")
    mixture = np.sum(sources.astype(np.float64), axis=1).astype(np.float32)
    if source_rate != target_rate:
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise MixError("Reconstructing at the input sample rate requires scipy") from exc
        divisor = gcd(int(source_rate), int(target_rate))
        mixture = resample_poly(mixture, target_rate // divisor, source_rate // divisor).astype(np.float32)
    peak = float(np.max(np.abs(mixture))) if mixture.size else 0.0
    if peak > peak_limit:
        raise MixError(f"Separated stem sum peak {peak:.4f} exceeds {peak_limit}; original master retained")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), mixture, target_rate, subtype="PCM_16")
    return {"path": str(output_path), "sample_rate": target_rate, "frames": len(mixture),
            "peak": peak, "method": "sum_of_sepformer_sources_without_normalization"}
