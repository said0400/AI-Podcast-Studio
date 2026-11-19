"""Optional SpeechBrain ECAPA embedding diarization for non-overlap speech windows.

This is embedding extraction plus centroid-agglomerative clustering, not a
production overlap detector. Speaker labels are ephemeral cluster IDs and must
be explicitly mapped to characters by the user.
"""
from __future__ import annotations

import logging
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np

from .vad import OptionalBackendError

LOGGER = logging.getLogger(__name__)


def _resample_mono(audio: np.ndarray, sample_rate: int, target_rate: int = 16000) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("Diarization requires finite mono-convertible audio")
    if sample_rate == target_rate:
        return np.ascontiguousarray(samples)
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise OptionalBackendError("SpeechBrain diarization at this sample rate requires scipy") from exc
    divisor = gcd(int(sample_rate), int(target_rate))
    return np.asarray(resample_poly(samples, target_rate // divisor, sample_rate // divisor), dtype=np.float32)


def _make_windows(regions: list[dict[str, Any]], sample_rate: int, sample_count: int,
                  window_seconds: float, hop_seconds: float, minimum_seconds: float) -> list[tuple[int, int]]:
    """Tile VAD speech regions with overlapping fixed-length embedding windows."""
    window = max(1, round(window_seconds * sample_rate))
    hop = max(1, round(hop_seconds * sample_rate))
    minimum = max(1, round(minimum_seconds * sample_rate))
    intervals: list[tuple[int, int]] = []
    for region in regions:
        start = max(0, int(float(region["start"]) * sample_rate))
        end = min(sample_count, int(float(region["end"]) * sample_rate))
        if end - start < minimum:
            continue
        if end - start <= window:
            intervals.append((start, end))
            continue
        positions = list(range(start, end - window + 1, hop))
        final_start = end - window
        if not positions or positions[-1] != final_start:
            positions.append(final_start)
        intervals.extend((pos, pos + window) for pos in positions)
    return intervals


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise ValueError("Speaker model returned an invalid/zero embedding")
    return np.asarray(vector / norm, dtype=np.float32)


def _cluster_embeddings(embeddings: np.ndarray, expected_speakers: int | None,
                        max_cosine_distance: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """Agglomerate normalized ECAPA vectors by nearest centroid cosine distance."""
    clusters: list[list[int]] = [[idx] for idx in range(len(embeddings))]
    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")
        for left in range(len(clusters)):
            left_center = _normalize(embeddings[clusters[left]].mean(axis=0))
            for right in range(left + 1, len(clusters)):
                right_center = _normalize(embeddings[clusters[right]].mean(axis=0))
                distance = 1.0 - float(np.dot(left_center, right_center))
                if distance < best_distance:
                    best_pair, best_distance = (left, right), distance
        if best_pair is None:
            break
        if expected_speakers is not None:
            if len(clusters) <= max(1, expected_speakers):
                break
        elif best_distance > max_cosine_distance:
            break
        left, right = best_pair
        clusters[left] = clusters[left] + clusters[right]
        del clusters[right]
    clusters.sort(key=lambda members: min(members))
    labels = np.empty(len(embeddings), dtype=np.int32)
    centers = []
    for label, members in enumerate(clusters):
        center = _normalize(embeddings[members].mean(axis=0))
        centers.append(center)
        labels[members] = label
    return labels, centers


def _segments_from_labeled_windows(windows: list[tuple[int, int]], labels: np.ndarray,
                                   embeddings: np.ndarray, centers: list[np.ndarray],
                                   min_embedding_similarity: float,
                                   sample_rate: int = 16000) -> tuple[list[dict[str, Any]], int]:
    """Turn overlapped embedding windows into approximate non-overlapping turns.

    Adjacent differing labels are split at the midpoint between window centers.
    This is a timeline convention, not a detector of true simultaneous speech.
    """
    candidates = []
    uncertain = 0
    for index, ((start, end), label) in enumerate(zip(windows, labels)):
        similarity = float(np.dot(embeddings[index], centers[int(label)]))
        if similarity < min_embedding_similarity:
            uncertain += 1
            continue
        candidates.append({"start_sample": start, "end_sample": end,
                           "center_sample": (start + end) / 2.0, "speaker_id": f"Speaker_{int(label)}",
                           "similarity": similarity})
    turns: list[dict[str, Any]] = []
    for item in candidates:
        start, end = item["start_sample"], item["end_sample"]
        if turns and start < turns[-1]["end_sample"]:
            previous = turns[-1]
            boundary = int(round((previous["center_sample"] + item["center_sample"]) / 2.0))
            boundary = max(previous["start_sample"] + 1, min(end - 1, boundary))
            previous["end_sample"] = boundary
            start = boundary
        if turns and turns[-1]["speaker_id"] == item["speaker_id"] and start <= turns[-1]["end_sample"]:
            turns[-1]["end_sample"] = max(turns[-1]["end_sample"], end)
            turns[-1]["similarities"].append(item["similarity"])
            turns[-1]["center_sample"] = item["center_sample"]
            continue
        if start >= end:
            continue
        turns.append({"start_sample": start, "end_sample": end, "center_sample": item["center_sample"],
                      "speaker_id": item["speaker_id"], "similarities": [item["similarity"]]})
    segments = [{"start": turn["start_sample"] / sample_rate,
                 "end": turn["end_sample"] / sample_rate,
                 "speaker_id": turn["speaker_id"],
                 "embedding_similarity": float(np.mean(turn["similarities"])),
                 "score_is_calibrated_probability": False,
                 "method": "speechbrain_ecapa_window_cluster_midpoint_boundary"}
                for turn in turns if turn["end_sample"] > turn["start_sample"]]
    return segments, uncertain


def diarize_audio(audio: np.ndarray, sample_rate: int, speech_regions: list[dict[str, Any]], *,
                  expected_speakers: int | None = None, model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
                  cache_dir: Path = Path("models/cache/spkrec-ecapa-voxceleb"),
                  window_seconds: float = 2.0, hop_seconds: float = 1.0,
                  min_window_seconds: float = 0.8, max_cosine_distance: float = 0.28,
                  min_embedding_similarity: float = 0.20) -> dict[str, Any]:
    """Run real ECAPA embeddings on VAD windows and cluster likely speakers.

    Window scores are cosine similarities, not calibrated probabilities. The
    method does not detect simultaneous speakers inside one mixed window.
    """
    if expected_speakers is not None and expected_speakers < 1:
        raise ValueError("expected_speakers must be positive when specified")
    samples = _resample_mono(audio, sample_rate, 16000)
    windows = _make_windows(speech_regions, 16000, len(samples), window_seconds,
                            hop_seconds, min_window_seconds)
    if not windows:
        return {"backend": "speechbrain_ecapa", "segments": [], "warnings": [
            "No VAD region was long enough for speaker embeddings."], "window_count": 0}
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as exc:
        raise OptionalBackendError("Install the optional neural extra to use SpeechBrain diarization") from exc
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        torch.set_num_threads(1)
        encoder = EncoderClassifier.from_hparams(source=model_source, savedir=str(cache_dir),
                                                  run_opts={"device": "cpu"})
        vectors = []
        for start, end in windows:
            chunk = samples[start:end]
            minimum = round(min_window_seconds * 16000)
            if len(chunk) < minimum:
                chunk = np.pad(chunk, (0, minimum - len(chunk)))
            tensor = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                result = encoder.encode_batch(tensor)
            vector = result.detach().cpu().numpy().reshape(-1)
            vectors.append(_normalize(vector))
    except OptionalBackendError:
        raise
    except Exception as exc:
        raise OptionalBackendError(f"SpeechBrain speaker embedding inference failed: {type(exc).__name__}: {exc}") from exc
    embeddings = np.stack(vectors)
    labels, centers = _cluster_embeddings(embeddings, expected_speakers, max_cosine_distance)
    segments, uncertain = _segments_from_labeled_windows(
        windows, labels, embeddings, centers, min_embedding_similarity)
    warnings = ["This embedding-clustering backend does not identify multiple speakers inside one overlapping mixture window.",
                "Overlapping embedding windows are converted to approximate turns using midpoint boundaries; these are not overlap detections.",
                "Speaker_N labels are temporary acoustic clusters; map them to characters explicitly in project JSON.",
                "Cosine similarities are not calibrated confidence probabilities."]
    if uncertain:
        warnings.append(f"{uncertain} low-similarity windows were excluded for manual review.")
    if expected_speakers is not None and len(centers) != expected_speakers:
        warnings.append(f"Observed {len(centers)} clusters for {expected_speakers} requested speakers; inspect manually.")
    # Profiles stay in memory for optional internal matching and are not written to JSON reports.
    return {"backend": "speechbrain_ecapa", "segments": segments, "window_count": len(windows),
            "speaker_cluster_count": len(centers), "warnings": warnings,
            "_profiles": {f"Speaker_{idx}": center for idx, center in enumerate(centers)}}
