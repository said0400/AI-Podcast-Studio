from pathlib import Path

import numpy as np
import soundfile as sf

from ai_podcast_studio.audio import energy_vad, inspect_audio, make_character_tracks, read_audio


def test_audio_metadata_and_read(tmp_path: Path):
    path = tmp_path / "input.wav"
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
    info = inspect_audio(path)
    audio, rate = read_audio(path, mono=True)
    assert info.duration == 1.0
    assert rate == 16000 and audio.shape == (16000,)


def test_energy_vad_detects_synthetic_tone_only_as_activity():
    rate = 16000
    x = np.zeros(rate, dtype=np.float32)
    x[3200:9600] = 0.15 * np.sin(2 * np.pi * 220 * np.arange(6400) / rate)
    regions = energy_vad(x, rate)
    assert regions
    assert regions[0]["start"] <= 0.23
    assert regions[0]["end"] >= 0.57
    assert regions[0]["method"] == "energy_vad"


def test_tracks_preserve_timing_and_exclude_overlap(tmp_path: Path):
    rate = 1000
    audio = np.ones((1000, 1), dtype=np.float32) * 0.2
    project = {"characters": {
        "char_1": {"speaker_id": "S0"}, "char_2": {"speaker_id": "S1"}},
        "segments": [
            {"start": 0.0, "end": 0.6, "speaker_id": "S0"},
            {"start": 0.4, "end": 0.9, "speaker_id": "S1"}]}
    report = make_character_tracks(audio, rate, project, tmp_path)
    a, ar = sf.read(report["tracks"]["char_1"]["path"], dtype="float32", always_2d=True)
    b, br = sf.read(report["tracks"]["char_2"]["path"], dtype="float32", always_2d=True)
    assert len(a) == len(b) == len(audio)
    assert ar == br == rate
    assert np.max(np.abs(a[450:550])) == 0
    assert np.max(np.abs(b[450:550])) == 0
    assert report["overlap_seconds_excluded_from_tracks"] > 0.15


def test_unmapped_overlapping_speaker_is_not_copied_into_mapped_track(tmp_path: Path):
    audio = np.full((1000, 1), 0.2, dtype=np.float32)
    project = {"characters": {"char_1": {"speaker_id": "S0"}},
               "segments": [{"start": 0.0, "end": 0.7, "speaker_id": "S0"},
                            {"start": 0.4, "end": 0.8, "speaker_id": "unmapped"}]}
    report = make_character_tracks(audio, 1000, project, tmp_path)
    track, _ = sf.read(report["tracks"]["char_1"]["path"], dtype="float32")
    assert np.max(np.abs(track[450:550])) == 0
    assert report["unmapped_speakers"] == ["unmapped"]



def test_segments_clip_small_end_overrun_but_reject_starts_after_audio_end():
    from ai_podcast_studio.audio import AudioError, validate_segments

    accepted = validate_segments([{"start": 0.99, "end": 1.01, "speaker_id": "S0"}], 1.0)
    assert accepted[0]["start"] == 0.99
    assert accepted[0]["end"] == 1.0
    try:
        validate_segments([{"start": 1.01, "end": 1.02, "speaker_id": "S0"}], 1.0)
    except AudioError:
        pass
    else:
        raise AssertionError("Segment wholly beyond audio duration should be rejected")


def test_overlap_mask_does_not_overflow_with_many_speakers(tmp_path: Path):
    audio = np.full((100, 1), 0.2, dtype=np.float32)
    segments = [{"start": 0.0, "end": 0.05, "speaker_id": f"S{i}"} for i in range(256)]
    project = {"characters": {"char_1": {"speaker_id": "S0"}}, "segments": segments}
    report = make_character_tracks(audio, 1000, project, tmp_path)
    track, _ = sf.read(report["tracks"]["char_1"]["path"], dtype="float32")
    assert np.max(np.abs(track[:50])) == 0
    assert report["overlap_seconds_excluded_from_tracks"] == 0.05
