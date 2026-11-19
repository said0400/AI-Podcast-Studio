"""Optional SpeechBrain SepFormer separation with overlapped chunk stitching.

The 2Mix and 3Mix checkpoints target their respective WSJ0 benchmarks. This
module validates tensor outputs and timeline continuity, but cannot guarantee
quality on arbitrary recordings. Source indices are not speaker identities.
"""
from __future__ import annotations

import itertools
import logging
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np

from .vad import OptionalBackendError

LOGGER = logging.getLogger(__name__)


MODEL_BY_SOURCES = {
    2: "speechbrain/sepformer-wsj02mix",
    3: "speechbrain/sepformer-wsj03mix",
}


def plan_chunks(sample_count: int, chunk_size: int, overlap_size: int) -> list[tuple[int, int]]:
    """Create complete, non-duplicated timeline ranges with requested overlap."""
    if sample_count < 1 or chunk_size < 2 or overlap_size < 0 or overlap_size >= chunk_size:
        raise ValueError("Invalid audio/chunk/overlap sizes")
    if sample_count <= chunk_size:
        return [(0, sample_count)]
    step = chunk_size - overlap_size
    starts = [0]
    while starts[-1] + chunk_size < sample_count:
        starts.append(starts[-1] + step)
    return [(start, min(sample_count, start + chunk_size)) for start in starts]


def _normalize_audio(audio: np.ndarray, sample_rate: int, target_rate: int = 8000) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("SepFormer input must be a finite non-empty mono-convertible waveform")
    if sample_rate == target_rate:
        return np.ascontiguousarray(samples)
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise OptionalBackendError("SepFormer requires the optional scipy dependency to resample to 8 kHz") from exc
    divisor = gcd(int(sample_rate), int(target_rate))
    return np.asarray(resample_poly(samples, target_rate // divisor, sample_rate // divisor), dtype=np.float32)


def _align_sources(previous: np.ndarray, current: np.ndarray, overlap_count: int,
                   minimum_mean_correlation: float = 0.08) -> tuple[np.ndarray, dict[str, Any]]:
    """Permute current chunk sources to best match the preceding overlap."""
    count = previous.shape[0]
    overlap_count = min(overlap_count, previous.shape[1], current.shape[1])
    if overlap_count <= 8:
        return current, {"aligned": False, "reason": "overlap too short"}
    left = previous[:, -overlap_count:].astype(np.float64)
    right = current[:, :overlap_count].astype(np.float64)
    correlations = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        a = left[i] - left[i].mean()
        for j in range(count):
            b = right[j] - right[j].mean()
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            correlations[i, j] = abs(float(np.dot(a, b) / denominator)) if denominator > 1e-12 else 0.0
    best_perm = max(itertools.permutations(range(count)),
                    key=lambda perm: sum(correlations[i, perm[i]] for i in range(count)))
    scores = [float(correlations[i, best_perm[i]]) for i in range(count)]
    mean_score = float(np.mean(scores)) if scores else 0.0
    if mean_score < minimum_mean_correlation:
        return current, {"aligned": False, "mean_overlap_correlation": mean_score,
                         "reason": "ambiguous source order; original model order retained"}
    return current[list(best_perm), :], {"aligned": True, "mean_overlap_correlation": mean_score,
                                         "permutation": list(best_perm)}


def _chunk_weights(length: int, overlap: int, has_left: bool, has_right: bool) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)
    ramp = min(overlap, length // 2)
    if ramp > 0:
        phase = np.linspace(0.0, np.pi / 2.0, ramp, endpoint=True, dtype=np.float32)
        if has_left:
            weights[:ramp] = np.sin(phase) ** 2
        if has_right:
            weights[-ramp:] = np.sin(phase[::-1]) ** 2
    return weights


def separate_audio(audio: np.ndarray, sample_rate: int, *, num_speakers: int,
                   cache_dir: Path = Path("models/cache"), chunk_seconds: float = 12.0,
                   overlap_seconds: float = 1.0, device: str = "cpu",
                   model_source: str | None = None, model: Any | None = None) -> dict[str, Any]:
    """Separate mono-converted audio into 2 or 3 source estimates on an 8 kHz timeline.

    A pretrained SpeechBrain SepFormer is loaded when no model object is passed.
    Chunk outputs are aligned by correlation over overlaps and overlap-added.
    Source ordering is arbitrary; map sources to characters explicitly after
    listening to the saved source WAVs.
    """
    if num_speakers not in MODEL_BY_SOURCES:
        raise ValueError("SepFormer backend supports only num_speakers=2 or 3")
    selected_model = model_source or MODEL_BY_SOURCES[num_speakers]
    if selected_model not in MODEL_BY_SOURCES.values():
        raise ValueError(f"Unverified SepFormer model source: {selected_model}")
    if selected_model != MODEL_BY_SOURCES[num_speakers]:
        raise ValueError(f"Model {selected_model} is not the verified {num_speakers}-source checkpoint")
    if chunk_seconds <= overlap_seconds or overlap_seconds < 0:
        raise ValueError("chunk_seconds must be greater than non-negative overlap_seconds")
    mono = _normalize_audio(audio, sample_rate, 8000)
    chunk_size = max(2, round(chunk_seconds * 8000))
    overlap_size = max(0, round(overlap_seconds * 8000))
    ranges = plan_chunks(len(mono), chunk_size, overlap_size)
    if model is None:
        try:
            import torch
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError as exc:
            raise OptionalBackendError("Install the optional neural extra to run SpeechBrain SepFormer") from exc
        try:
            torch.set_num_threads(1)
            model = SepformerSeparation.from_hparams(
                source=selected_model, savedir=str(Path(cache_dir) / selected_model.rsplit("/", 1)[-1]),
                run_opts={"device": device},
            )
        except Exception as exc:
            raise OptionalBackendError(f"Could not load SepFormer weights {selected_model}: {type(exc).__name__}: {exc}") from exc
    try:
        import torch
    except ImportError as exc:
        raise OptionalBackendError("PyTorch is required by the SpeechBrain SepFormer backend") from exc
    source_sums = np.zeros((num_speakers, len(mono)), dtype=np.float64)
    weight_sums = np.zeros(len(mono), dtype=np.float64)
    previous_output: np.ndarray | None = None
    previous_range: tuple[int, int] | None = None
    alignment_log: list[dict[str, Any]] = []
    for chunk_index, (start, end) in enumerate(ranges):
        segment = np.ascontiguousarray(mono[start:end], dtype=np.float32)
        actual_length = len(segment)
        if actual_length < chunk_size:
            # Keep the model input shape stable while ignoring all padded output.
            pad_mode = "reflect" if actual_length > 1 else "edge"
            segment = np.pad(segment, (0, chunk_size - actual_length), mode=pad_mode)
        tensor = torch.from_numpy(segment).unsqueeze(0)
        try:
            with torch.no_grad():
                estimated = model.separate_batch(tensor)
        except Exception as exc:
            raise OptionalBackendError(f"SepFormer inference failed on chunk {chunk_index}: {type(exc).__name__}: {exc}") from exc
        if not hasattr(estimated, "detach"):
            raise OptionalBackendError("SepFormer returned a non-tensor result; no stems were written")
        current = estimated.detach().cpu().numpy()
        if current.ndim != 3 or current.shape[0] != 1 or current.shape[2] != num_speakers:
            raise OptionalBackendError(f"SepFormer output must have shape [1, time, {num_speakers}], got {current.shape}")
        current = np.asarray(current[0].T, dtype=np.float32)
        if not np.isfinite(current).all():
            raise OptionalBackendError("SepFormer produced NaN/Inf samples; refusing to save stems")
        if current.shape[1] != chunk_size:
            if abs(current.shape[1] - chunk_size) > max(16, round(0.01 * 8000)):
                raise OptionalBackendError("SepFormer output duration differs materially from input chunk")
            current = current[:, :chunk_size] if current.shape[1] > chunk_size else np.pad(current, ((0, 0), (0, chunk_size - current.shape[1])))
        current = current[:, :actual_length]
        if previous_output is not None and previous_range is not None:
            overlap_count = max(0, previous_range[1] - start)
            current, alignment = _align_sources(previous_output, current, overlap_count)
            alignment["chunk_index"] = chunk_index
            alignment_log.append(alignment)
        weights = _chunk_weights(end - start, overlap_size, start > 0, end < len(mono))
        source_sums[:, start:end] += current.astype(np.float64) * weights[None, :]
        weight_sums[start:end] += weights
        previous_output = current
        previous_range = (start, end)
        LOGGER.info("SepFormer chunk %d/%d processed", chunk_index + 1, len(ranges))
    if np.any(weight_sums <= 0):
        raise OptionalBackendError("Chunk stitching left uncovered samples in the timeline")
    sources = (source_sums / weight_sums[None, :]).T.astype(np.float32)
    rms = np.sqrt(np.mean(np.square(sources, dtype=np.float64), axis=0))
    warnings = []
    silent = [f"source_{idx}" for idx, value in enumerate(rms) if value < 1e-7]
    if silent:
        warnings.append("One or more estimated sources are near-silent and require manual review: " + ", ".join(silent))
    uncertain_alignments = sum(not item.get("aligned", False) for item in alignment_log)
    if uncertain_alignments:
        warnings.append(f"Source order could not be matched confidently in {uncertain_alignments} chunk boundary/boundaries.")
    metadata = {"backend": "speechbrain_sepformer", "model_source": selected_model,
                "expected_sources": num_speakers, "sample_rate": 8000,
                "duration_seconds": len(sources) / 8000, "chunk_count": len(ranges),
                "chunk_seconds": chunk_seconds, "overlap_seconds": overlap_seconds,
                "source_order_is_identity": False,
                "source_rms": {f"source_{idx}": float(value) for idx, value in enumerate(rms)},
                "alignment": alignment_log, "warnings": warnings,
                "benchmark_scope": "WSJ0-2Mix" if num_speakers == 2 else "WSJ0-3Mix"}
    return {"sources": sources, "sample_rate": 8000, "metadata": metadata}
