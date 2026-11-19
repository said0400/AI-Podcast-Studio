import json
from pathlib import Path

from PIL import Image

from ai_podcast_studio.animation import CharacterRig
from ai_podcast_studio.checkpoint import read_checkpoint, save_checkpoint
from ai_podcast_studio.config import load_project

ROOT = Path(__file__).resolve().parents[1]


def test_example_project_validates():
    project = load_project(ROOT / "input/project.example.json", ROOT / "config/project.schema.json")
    packaged_default = load_project(ROOT / "input/project.example.json")
    assert len(project["characters"]) == 3
    assert packaged_default == project
    assert (ROOT / "src/ai_podcast_studio/project.schema.json").read_bytes() == (ROOT / "config/project.schema.json").read_bytes()


def test_invalid_project_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"project": {"fps": 0}, "characters": {}}))
    try:
        load_project(path, ROOT / "config/project.schema.json")
    except ValueError as exc:
        assert "schema" in str(exc)
    else:
        raise AssertionError("Invalid project unexpectedly passed schema validation")


def test_sepformer_rejects_unsupported_single_source_configuration(tmp_path: Path):
    path = tmp_path / "one_source.json"
    path.write_text(json.dumps({
        "project": {"name": "Test", "resolution": "320x180", "fps": 5},
        "characters": {"char_1": {"speaker_id": "S0"}},
        "models": {"separation": "sepformer", "num_speakers": 1},
    }))
    try:
        load_project(path, ROOT / "config/project.schema.json")
    except ValueError as exc:
        assert "schema" in str(exc)
    else:
        raise AssertionError("SepFormer accepted an unsupported one-source setting")


def test_missing_character_layers_fall_back_without_crashing(tmp_path: Path):
    rig = CharacterRig(tmp_path)
    frame = rig.render(1.0, mouth="A", speaking=True)
    assert frame.mode == "RGBA" and frame.size == (512, 768)
    assert rig.warnings


def test_checkpoint_round_trip_and_corrupt_recovery(tmp_path: Path):
    path = tmp_path / "checkpoint.json"
    save_checkpoint(path, {"stage": "tracks", "done": True})
    assert read_checkpoint(path)["stage"] == "tracks"
    path.write_bytes(b"\xff\xfe")
    assert read_checkpoint(path) is None



def test_layer_motion_clips_at_edges_instead_of_wrapping():
    image = Image.new("RGBA", (4, 3), (0, 0, 0, 0))
    image.putpixel((0, 1), (255, 0, 0, 255))
    moved_right = CharacterRig._move(image, 1, 0)
    assert moved_right.getpixel((1, 1)) == (255, 0, 0, 255)
    assert moved_right.getpixel((3, 1))[3] == 0

    image.putpixel((3, 1), (0, 255, 0, 255))
    moved_left = CharacterRig._move(image, -1, 0)
    assert moved_left.getpixel((2, 1)) == (0, 255, 0, 255)
    assert moved_left.getpixel((0, 1))[3] == 0
