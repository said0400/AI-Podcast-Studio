import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ai_podcast_studio.cli import build_parser, main
from ai_podcast_studio.director import ManualCameraCues
from ai_podcast_studio.lipsync import (build_cue_index, envelope_cues, indexed_mouth_at,
                                       mouth_at, speech_energy)
from ai_podcast_studio.rendering import CameraRig, _apply_camera
from PIL import Image


def _speechlike(path: Path, rate: int = 8000) -> None:
    t = np.arange(rate * 2) / rate
    envelope = np.clip(np.sin(2 * np.pi * 2.5 * t), 0, None) * (0.05 + 0.15 * (t > 1.0))
    sf.write(path, (envelope * np.sin(2 * np.pi * 180 * t)).astype(np.float32), rate)


def test_envelope_cues_use_graded_shapes_and_quiet_is_closed(tmp_path: Path):
    wav = tmp_path / "v.wav"
    _speechlike(wav)
    cues = envelope_cues(wav)
    shapes = {c["value"] for c in cues}
    assert "X" in shapes and len(shapes & {"A", "B", "C", "D"}) >= 2
    assert all(c["end"] > c["start"] for c in cues)


def test_indexed_lookup_matches_linear_lookup(tmp_path: Path):
    wav = tmp_path / "v.wav"
    _speechlike(wav)
    cues = envelope_cues(wav)
    index = build_cue_index(cues)
    for t in np.linspace(0, 1.99, 97):
        assert indexed_mouth_at(index, float(t)) == mouth_at(cues, float(t))
    assert speech_energy(build_cue_index([]), 1.0) == 0.0
    assert 0.0 < speech_energy(index, 1.2) <= 1.0


def test_hybrid_cue_expires_and_manual_holds_and_automatic_ignores():
    cues = [{"time": 2.0, "camera": "close_up", "character": "char_1", "transition": "cut"}]
    hybrid = ManualCameraCues(cues, "hybrid", 1.5, ["char_1"])
    assert hybrid.step(1.9) == (None, None)
    shot, transition = hybrid.step(2.0)
    assert shot.kind == "speaker_focus" and shot.focus == "char_1" and transition == "cut"
    assert hybrid.step(3.0)[0] is not None
    assert hybrid.step(3.6)[0] is None
    assert ManualCameraCues(cues, "manual", 1.5, ["char_1"]).step(99.0)[0] is not None
    assert ManualCameraCues(cues, "automatic", 1.5, ["char_1"]).step(99.0)[0] is None
    assert ManualCameraCues([{"time": 0, "camera": "nonsense"}], "hybrid", 4, []).step(1.0)[0] is None


def test_camera_glides_toward_focus_and_stays_inside_frame():
    from ai_podcast_studio.director import Shot
    rig = CameraRig((640, 360))
    positions = {"char_1": 109, "char_2": 531}
    rig.update(Shot(), positions, 0.0, 245)
    boxes = [rig.update(Shot("speaker_focus", "char_1", 0.0), positions, i / 10, 245) for i in range(1, 40)]
    assert boxes[0][2] - boxes[0][0] > boxes[-1][2] - boxes[-1][0] > 640 / 1.4  # zooming gradually
    assert boxes[-1][0] < boxes[0][0] or boxes[0][0] == 0.0  # drifting toward the left character
    for left, top, right, bottom in boxes:
        assert 0 <= left < right <= 640 and 0 <= top < bottom <= 360
    assert _apply_camera(Image.new("RGBA", (640, 360)), boxes[-1]).size == (640, 360)


def test_cli_schema_defaults_to_packaged_and_works_from_any_directory(tmp_path: Path, monkeypatch, capsys):
    assert build_parser().parse_args(["quality-check"]).schema is None
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(8000, dtype=np.float32), 8000)
    project = tmp_path / "p.json"
    project.write_text(json.dumps({"project": {"name": "t", "resolution": "320x180", "fps": 5},
                                   "characters": {"char_1": {"speaker_id": None}}}))
    monkeypatch.chdir(tmp_path)
    assert main(["quality-check", "--audio", str(wav), "--project", str(project)]) == 0
    assert '"status": "ok"' in capsys.readouterr().out


def test_unused_source_mapping_is_reported(tmp_path: Path, monkeypatch):
    from ai_podcast_studio import pipeline
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(800, dtype=np.float32), 8000)
    project = tmp_path / "p.json"
    project.write_text(json.dumps({
        "project": {"name": "t", "resolution": "320x180", "fps": 5},
        "characters": {"char_1": {"speaker_id": None}},
        "models": {"separation": "none", "source_to_character": {"source_0": "char_1"}}}))
    monkeypatch.setattr(pipeline, "render_video", lambda **k: (k["output_path"].write_bytes(b"v"), {"path": "x"})[1])
    monkeypatch.setattr(pipeline, "validate_video", lambda path: {"streams": []})
    report = pipeline.run_pipeline(audio_path=wav, project_path=project, character_root=tmp_path / "c",
                                   background_root=tmp_path / "b", output_dir=tmp_path / "out",
                                   resume=False, width=320, height=180, fps=5)
    assert any("ignored" in w for w in report["separation"]["warnings"])
