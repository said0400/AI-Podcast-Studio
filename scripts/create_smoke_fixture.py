"""Create an obviously synthetic non-speech WAV and tiny project for smoke tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    folder = ROOT / "examples" / "smoke"
    folder.mkdir(parents=True, exist_ok=True)
    rate = 16000
    duration = 2.0
    t = np.arange(round(rate * duration), dtype=np.float32) / rate
    # A quiet 440 Hz test tone, not speech, deliberately signals synthetic data.
    signal = np.zeros_like(t)
    active = (t >= .35) & (t <= 1.3)
    signal[active] = .12 * np.sin(2 * np.pi * 440 * t[active])
    sf.write(folder / "synthetic_tone.wav", signal, rate)
    project = {
        "project": {"name": "Synthetic smoke-test tone (not speech)", "resolution": "640x360", "fps": 10},
        "characters": {"char_1": {"name": "Demo", "speaker_id": "Speaker_0", "position": "center"}},
        "segments": [{"start": .35, "end": 1.3, "speaker_id": "Speaker_0"}],
        "director": {"mode": "hybrid", "automatic_closeups": True, "min_shot_duration": .5}
    }
    (folder / "project.json").write_text(json.dumps(project, indent=2), encoding="utf-8")
    print(f"Created non-speech test fixture in {folder}")

if __name__ == "__main__": main()
