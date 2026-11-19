import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ai_podcast_studio.audio import energy_vad
from ai_podcast_studio.config import load_project

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixture = _load("create_dialogue_fixture")
template = _load("make_segments_template")


@pytest.fixture()
def dialogue(tmp_path: Path):
    wav = tmp_path / "dialogue.wav"
    sf.write(wav, fixture.build_audio(), fixture.RATE)
    project = tmp_path / "project.json"
    project.write_text(json.dumps(fixture.build_project()))
    return wav, project


def test_example_project_matches_fixture_and_validates():
    example = json.loads((ROOT / "input/project.example.json").read_text())
    assert example == fixture.build_project()
    project = load_project(ROOT / "input/project.example.json")
    assert len(project["characters"]) == 3
    assert max(s["end"] for s in project["segments"]) <= fixture.DURATION


def test_energy_vad_finds_dialogue_that_is_mostly_speech():
    # Regression: the noise floor used to come from the 20th percentile, which lies
    # inside speech when speech is >80% of the recording, hiding most of it.
    regions = energy_vad(fixture.build_audio(), fixture.RATE, max_gap_ms=300)
    assert len(regions) == len(fixture.SEGMENTS)
    for region, (start, end, _speaker) in zip(regions, fixture.SEGMENTS):
        assert abs(region["start"] - start) < 0.08 and abs(region["end"] - end) < 0.08


def test_template_to_project_round_trip(dialogue, tmp_path: Path):
    wav, project = dialogue
    out = tmp_path / "out"
    assert template.main([str(wav), "--project", str(project), "--out-dir", str(out)]) == 0
    tsv = out / "dialogue.segments.tsv"
    rows = [line for line in tsv.read_text().splitlines() if line and not line.startswith("#")]
    assert len(rows) == 7 and all("TODO" in row for row in rows)
    labels = ["char_1", "Speaker_1", "char_3", "char_1", "Speaker_2", "char_2", "Speaker_0"]
    edited = []
    for row, label in zip(rows, labels):
        start, end, _ = row.split("\t")[:3]
        edited.append(f"{start}\t{end}\t{label}")
    tsv.write_text("\n".join(edited) + "\n")
    merged = tmp_path / "merged.json"
    assert template.main([str(wav), "--project", str(project), "--out-dir", str(out),
                          "--from-tsv", str(tsv), "--merge-output", str(merged)]) == 0
    result = load_project(merged)
    assert [s["speaker_id"] for s in result["segments"]] == [
        "Speaker_0", "Speaker_1", "Speaker_2", "Speaker_0", "Speaker_2", "Speaker_1", "Speaker_0"]


def test_unresolvable_labels_and_bad_rows_fail_clearly(dialogue, tmp_path: Path, capsys):
    wav, project = dialogue
    no_speaker = json.loads(project.read_text())
    no_speaker["characters"]["char_1"]["speaker_id"] = None
    project.write_text(json.dumps(no_speaker))
    tsv = tmp_path / "bad.tsv"
    tsv.write_text("1.0\t2.0\tchar_1\n")
    assert template.main([str(wav), "--project", str(project), "--out-dir", str(tmp_path),
                          "--from-tsv", str(tsv)]) == 2
    assert "no speaker_id" in capsys.readouterr().err
    tsv.write_text("one two Speaker_0\n")
    assert template.main([str(wav), "--out-dir", str(tmp_path), "--from-tsv", str(tsv)]) == 2
    tsv.write_text("1.0\t999.0\tSpeaker_0\n")
    assert template.main([str(wav), "--out-dir", str(tmp_path), "--from-tsv", str(tsv)]) == 2


def test_long_regions_split_at_quiet_frame():
    rate = 1000
    audio = np.full(20 * rate, 0.2, dtype=np.float32)
    audio[int(9.0 * rate):int(9.1 * rate)] = 0.0  # a brief dip inside a long run
    pieces = template.split_long_regions([(0.0, 20.0)], audio, rate, max_seconds=12.0, frame_ms=30)
    assert len(pieces) == 2 and 9.0 <= pieces[0][1] <= 9.1
    assert pieces[0][1] == pieces[1][0] and pieces[-1][1] == 20.0


def test_assign_labels_uses_largest_overlap_and_leaves_unclear_as_todo():
    segments = [{"start": 0.0, "end": 4.0, "speaker_id": "TODO"},
                {"start": 5.0, "end": 9.0, "speaker_id": "TODO"}]
    turns = [{"start": 0.0, "end": 3.0, "speaker_id": "Speaker_1"},
             {"start": 3.0, "end": 4.0, "speaker_id": "Speaker_0"},
             {"start": 8.5, "end": 9.0, "speaker_id": "Speaker_0"}]
    labelled, unresolved = template.assign_labels(segments, turns)
    assert [s["speaker_id"] for s in labelled] == ["Speaker_1", "TODO"] and unresolved == 1
