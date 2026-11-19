from pathlib import Path

import numpy as np
import soundfile as sf

from ai_podcast_studio.director import Director, active_at
from ai_podcast_studio.lipsync import extract_cues, mouth_at
from ai_podcast_studio.speaker_mapping import map_speakers


def test_speaker_map_never_depends_on_first_appearance():
    project = {"characters": {"char_1": {"speaker_id": "S2"}, "char_2": {"speaker_id": "S0"}}}
    result = map_speakers(project, {"S0", "S1", "S2"})
    assert result["mapping"] == {"S2": "char_1", "S0": "char_2"}
    assert result["unmapped_speakers"] == ["S1"]


def test_director_respects_minimum_shot_and_overlap():
    director = Director(min_shot_duration=2.0, hysteresis=0.5)
    assert director.update(0.0, []).kind == "wide"
    assert director.update(0.6, ["char_1", "char_2"]).kind == "wide"
    assert director.update(2.1, ["char_1", "char_2"]).kind == "two_shot"
    assert director.update(2.2, ["char_1"]).kind == "two_shot"
    assert director.update(4.2, ["char_1"]).focus == "char_1"
    active = active_at([{"start": 1.0, "end": 3.0, "speaker_id": "S0"},
                        {"start": 2.0, "end": 4.0, "speaker_id": "S1"}],
                       2.5, {"S0": "char_1", "S1": "char_2"})
    assert active == ["char_1", "char_2"]


def test_envelope_lipsync_is_audio_derived_and_fallback_labeled(tmp_path: Path):
    rate = 8000
    signal = np.zeros(rate, dtype=np.float32)
    signal[2000:6000] = .2 * np.sin(2 * np.pi * 180 * np.arange(4000) / rate)
    wav = tmp_path / "voice.wav"
    sf.write(wav, signal, rate)
    cues = extract_cues(wav, tmp_path / "cues.json", rhubarb_binary="rhubarb-definitely-not-installed")
    assert cues["method"] == "waveform_envelope_fallback"
    assert cues["phonetic"] is False
    assert mouth_at(cues["cues"], 0.4) in {"A", "B", "C", "D"}
    assert mouth_at(cues["cues"], 0.05) == "X"



def test_mouth_at_ignores_malformed_cues_without_crashing():
    cues = [None, {}, {"start": "bad", "end": 1, "value": "A"},
            {"start": 0.0, "end": 1.0, "value": "A"}]
    assert mouth_at(cues, 0.5) == "A"
    assert mouth_at(cues[:3], 0.5) == "X"


def test_sanitize_cues_clips_small_overrun_and_resolves_overlap():
    from ai_podcast_studio.lipsync import sanitize_cues

    cues, warnings = sanitize_cues([
        {"start": 0.0, "end": 0.6, "value": "a"},
        {"start": 0.4, "end": 0.8, "value": "B"},
        {"start": float("nan"), "end": 0.9, "value": "C"},
        {"start": 0.9, "end": 1.01, "value": "X"},
        {"start": 0.95, "end": 1.02, "value": "INVALID"},
    ], 1.0)
    assert cues == [
        {"start": 0.0, "end": 0.6, "value": "A"},
        {"start": 0.6, "end": 0.8, "value": "B"},
        {"start": 0.9, "end": 1.0, "value": "X"},
    ]
    assert len(warnings) == 2
