"""Create a clearly synthetic three-voice dialogue (tones, NOT speech) and its matching project.

The segment times below match the generated audio exactly, so the example
project can be rendered end to end, and the template script can be tried on
the same WAV to compare its detected timings with these known ones.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RATE = 16000
DURATION = 26.8
PITCH = {"Speaker_0": 120.0, "Speaker_1": 190.0, "Speaker_2": 260.0}
SEGMENTS = [
    (0.6, 4.0, "Speaker_0"), (4.6, 8.2, "Speaker_1"), (8.8, 12.2, "Speaker_2"),
    (12.8, 15.4, "Speaker_0"), (16.0, 19.2, "Speaker_2"), (19.8, 22.4, "Speaker_1"),
    (23.0, 26.0, "Speaker_0"),
]


def build_audio() -> np.ndarray:
    signal = np.zeros(round(RATE * DURATION), dtype=np.float32)
    for start, end, speaker in SEGMENTS:
        a, b = round(start * RATE), round(end * RATE)
        t = np.arange(b - a, dtype=np.float64) / RATE
        f0 = PITCH[speaker] * (1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t))
        phase = 2 * np.pi * np.cumsum(f0) / RATE
        tone = sum(np.sin(k * phase) / k for k in range(1, 6))
        syllables = 0.35 + 0.65 * np.abs(np.sin(np.pi * 3.5 * t))  # never fully silent inside a turn
        fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.05)
        signal[a:b] = (0.12 * tone * syllables * fade).astype(np.float32)
    return signal


def build_project() -> dict:
    return {
        "project": {"name": "Three-character synthetic dialogue (tones, not speech)",
                    "resolution": "1920x1080", "fps": 30},
        "characters": {
            "char_1": {"name": "Mina", "speaker_id": "Speaker_0", "position": "left"},
            "char_2": {"name": "Omar", "speaker_id": "Speaker_1", "position": "center"},
            "char_3": {"name": "Nour", "speaker_id": "Speaker_2", "position": "right"},
        },
        "segments": [{"start": s, "end": e, "speaker_id": spk, "confidence": 1.0} for s, e, spk in SEGMENTS],
        "models": {"vad": "energy", "diarization": "manual", "separation": "none",
                   "cache_dir": "models/cache", "source_to_character": {}},
        "audio": {"final_mix": "original"},
        "director": {"mode": "hybrid", "automatic_closeups": True, "reaction_shots": True,
                     "min_shot_duration": 2.0, "cue_hold_seconds": 3.0},
        "cues": [
            {"time": 5.0, "character": "char_2", "expression": "happy"},
            {"time": 13.0, "camera": "wide_shot", "duration": 1.5, "transition": "cut"},
            {"time": 16.0, "character": "char_3", "expression": "sad"},
            {"time": 19.5, "character": "char_3", "expression": "neutral"},
            {"time": 20.0, "camera": "close_up", "character": "char_2", "duration": 2.0,
             "transition": "dissolve"},
        ],
    }


def main() -> None:
    folder = ROOT / "examples" / "dialogue"
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(folder / "dialogue.wav", build_audio(), RATE)
    (folder / "project.json").write_text(json.dumps(build_project(), indent=2) + "\n", encoding="utf-8")
    print(f"Created synthetic three-voice fixture in {folder}")


if __name__ == "__main__":
    main()
