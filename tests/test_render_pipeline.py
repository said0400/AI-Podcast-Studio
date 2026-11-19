import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PIL import Image, ImageDraw

from ai_podcast_studio.pipeline import quality_check, run_pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools unavailable")
def test_short_end_to_end_render_has_video_and_audio(tmp_path: Path):
    rate = 8000
    audio_path = tmp_path / "voice.wav"
    signal = np.zeros(int(rate * .7), dtype=np.float32)
    signal[1000:4500] = .15 * np.sin(2 * np.pi * 220 * np.arange(3500) / rate)
    sf.write(audio_path, signal, rate)
    project_path = tmp_path / "project.json"
    project_path.write_text(json.dumps({
        "project": {"name": "Test", "resolution": "320x180", "fps": 5},
        "characters": {"char_1": {"speaker_id": "S0", "position": "center"}},
        "segments": [{"start": .1, "end": .5, "speaker_id": "S0"}],
        "director": {"min_shot_duration": 1.0},
        "cues": [{"time": .2, "camera": "speaker_focus", "character": "char_1",
                   "expression": "happy", "transition": "dissolve"}]
    }))
    char_root = tmp_path / "characters" / "char_1"
    char_root.mkdir(parents=True)
    body = Image.new("RGBA", (512, 768), (0, 0, 0, 0))
    ImageDraw.Draw(body).ellipse((80, 70, 430, 740), fill=(220, 120, 80, 255))
    body.save(char_root / "body.png")
    report = quality_check(audio_path=audio_path, project_path=project_path,
                           character_root=tmp_path / "characters", background_root=tmp_path / "backgrounds",
                           schema_path=ROOT / "config/project.schema.json")
    assert report["status"] == "ok"
    output_dir = tmp_path / "out"
    rendered = run_pipeline(audio_path=audio_path, project_path=project_path,
                            character_root=tmp_path / "characters", background_root=tmp_path / "backgrounds",
                            output_dir=output_dir, schema_path=ROOT / "config/project.schema.json",
                            width=320, height=180, fps=5)
    assert Path(rendered["video"]["path"]).is_file()
    assert any(s.get("codec_type") == "audio" for s in rendered["validation"]["streams"])
    assert rendered["separation_status"].startswith("not performed")
    assert (output_dir / "reports/run_report.json").is_file()


def test_render_size_and_fps_limits_are_enforced(tmp_path: Path):
    from ai_podcast_studio.rendering import MAX_RENDER_FPS, validate_render_size

    with pytest.raises(ValueError, match="safe limits"):
        validate_render_size(20_000, 12_000)
    with pytest.raises(ValueError, match="at least 2"):
        validate_render_size(1, 200)
    assert MAX_RENDER_FPS == 60

    from ai_podcast_studio import rendering
    audio = tmp_path / "short.wav"
    sf.write(audio, np.zeros(800, dtype=np.float32), 8000)
    with pytest.raises(ValueError, match="FPS"):
        rendering.render_video(
            audio_path=audio,
            project={"characters": {}}, character_root=ROOT / "assets/characters",
            background_root=ROOT / "assets/backgrounds", cues_by_character={},
            output_path=ROOT / "output/invalid-fps.mp4", width=320, height=180, fps=61)


def test_ffmpeg_process_and_partial_output_are_cleaned_on_unexpected_error(tmp_path: Path, monkeypatch):
    import io
    from ai_podcast_studio import rendering

    audio = tmp_path / "short.wav"
    sf.write(audio, np.zeros(800, dtype=np.float32), 8000)
    output = tmp_path / "partial.mp4"
    output.write_bytes(b"old or partial data")
    state = {"killed": False, "waited": False}

    class FakeProcess:
        def __init__(self):
            self.stdin = io.BytesIO()

        def poll(self):
            return None if not state["killed"] else -9

        def kill(self):
            state["killed"] = True

        def wait(self, timeout=None):
            state["waited"] = True
            return -9

    class BrokenRig:
        warnings = []

        def __init__(self, folder):
            pass

        def render(self, *args, **kwargs):
            raise LookupError("synthetic drawing failure")

    monkeypatch.setattr(rendering.shutil, "which", lambda binary: "/fake/ffmpeg")
    monkeypatch.setattr(rendering.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(rendering, "CharacterRig", BrokenRig)
    with pytest.raises(LookupError, match="synthetic drawing failure"):
        rendering.render_video(
            audio_path=audio,
            project={"characters": {"char_1": {"position": "center"}}},
            character_root=tmp_path / "characters", background_root=tmp_path / "backgrounds",
            cues_by_character={}, output_path=output, width=320, height=180, fps=5)
    assert state == {"killed": True, "waited": True}
    assert not output.exists()
