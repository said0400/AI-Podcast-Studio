"""Selectable VAD backends: deterministic energy fallback or Silero neural VAD."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .audio import energy_vad

LOGGER = logging.getLogger(__name__)


class OptionalBackendError(RuntimeError):
    """An explicitly requested optional model backend is unavailable or failed."""


def _mono_at_rate(audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    """Convert to mono and polyphase-resample without changing timeline seconds."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1, dtype=np.float32)
    if samples.ndim != 1 or not samples.size:
        raise ValueError("VAD input must be a non-empty one-dimensional waveform")
    if sample_rate == target_rate:
        return np.ascontiguousarray(samples)
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise OptionalBackendError("Silero VAD at this sample rate requires the optional scipy dependency") from exc
    from math import gcd
    divisor = gcd(int(sample_rate), int(target_rate))
    return np.asarray(resample_poly(samples, target_rate // divisor, sample_rate // divisor), dtype=np.float32)


def run_vad(audio: np.ndarray, sample_rate: int, *, backend: str = "energy",
            threshold: float = 0.5, min_speech_duration_ms: int = 120,
            min_silence_duration_ms: int = 180) -> dict[str, Any]:
    """Run energy VAD or Silero VAD; never silently relabel energy as neural output."""
    selected = backend.lower().strip()
    if selected == "energy":
        regions = energy_vad(audio, sample_rate, min_speech_ms=min_speech_duration_ms,
                             max_gap_ms=min_silence_duration_ms)
        return {"backend": "energy", "regions": regions, "warnings": []}
    if selected not in {"silero", "auto"}:
        raise ValueError(f"Unknown VAD backend: {backend}")
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:
        if selected == "auto":
            LOGGER.warning("Silero VAD is not installed; using the explicitly labelled energy fallback")
            regions = energy_vad(audio, sample_rate, min_speech_ms=min_speech_duration_ms,
                                 max_gap_ms=min_silence_duration_ms)
            return {"backend": "energy", "regions": regions,
                    "warnings": ["Silero VAD unavailable; energy VAD fallback used."]}
        raise OptionalBackendError("Install the optional neural extra and Silero VAD to use this backend") from exc
    target_rate = 16000 if sample_rate != 8000 else 8000
    samples = _mono_at_rate(audio, sample_rate, target_rate)
    try:
        torch.set_num_threads(1)
        model = load_silero_vad()
        stamps = get_speech_timestamps(
            torch.from_numpy(samples), model, sampling_rate=target_rate,
            threshold=float(threshold), min_speech_duration_ms=int(min_speech_duration_ms),
            min_silence_duration_ms=int(min_silence_duration_ms), return_seconds=True,
        )
    except Exception as exc:
        raise OptionalBackendError(f"Silero VAD inference failed: {type(exc).__name__}: {exc}") from exc
    regions = [{"start": float(stamp["start"]), "end": float(stamp["end"]),
                "method": "silero_vad", "confidence": None} for stamp in stamps]
    return {"backend": "silero", "regions": regions, "warnings": []}
