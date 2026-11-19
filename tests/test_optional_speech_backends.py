import sys
from pathlib import Path

import numpy as np
import pytest

from ai_podcast_studio.audio_mixer import MixError, reconstruct_master, write_character_stems
from ai_podcast_studio.speech_separation import _align_sources, plan_chunks
from ai_podcast_studio.vad import OptionalBackendError, run_vad


def test_chunk_plan_covers_timeline_with_bounded_requested_overlap():
    chunk_size = 100
    overlap_size = 20
    for sample_count in (1, 100, 101, 180, 201, 250, 1000):
        chunks = plan_chunks(sample_count, chunk_size, overlap_size)
        assert chunks[0][0] == 0
        assert chunks[-1][1] == sample_count
        assert all(start < end for start, end in chunks)
        coverage = np.zeros(sample_count, dtype=np.int8)
        for start, end in chunks:
            coverage[start:end] += 1
        assert np.all(coverage > 0)
        assert all(chunks[index - 1][1] - chunks[index][0] <= overlap_size
                   for index in range(1, len(chunks)))
        assert all(chunks[index - 1][0] < chunks[index][0] for index in range(1, len(chunks)))


def test_align_sources_reorders_source_axis_for_swap():
    time = np.arange(200, dtype=np.float32)
    previous = np.vstack([np.sin(time / 9.0), np.cos(time / 13.0)])
    current = previous[[1, 0], :]
    aligned, info = _align_sources(previous, current, overlap_count=200)
    assert info["aligned"] is True
    assert np.allclose(aligned, previous)
    assert info["permutation"] == [1, 0]


def test_explicit_character_mapping_writes_only_selected_stems(tmp_path: Path):
    stems = np.column_stack([
        np.full(64, 0.1, dtype=np.float32),
        np.full(64, 0.2, dtype=np.float32),
        np.full(64, 0.3, dtype=np.float32),
    ])
    result = write_character_stems(
        stems, 8000, {"source_2": "char_3"}, {"char_1", "char_2", "char_3"}, tmp_path, 8000)
    assert Path(result["mapped_characters"]["char_3"]).is_file()
    assert result["unmapped_sources"] == ["source_0", "source_1"]
    assert result["unassigned_characters"] == ["char_1", "char_2"]
    with pytest.raises(MixError, match="Unknown character"):
        write_character_stems(stems, 8000, {"source_0": "char_missing"},
                              {"char_1"}, tmp_path, 8000)


def test_reconstruct_master_sums_stems_without_clipping(tmp_path: Path):
    stems = np.column_stack([
        np.full(32, 0.1, dtype=np.float32),
        np.full(32, 0.2, dtype=np.float32),
    ])
    output = tmp_path / "master.wav"
    result = reconstruct_master(stems, 8000, 8000, output)
    assert result["method"] == "sum_of_sepformer_sources_without_normalization"
    assert result["peak"] == pytest.approx(0.3)
    assert output.is_file()


def test_reconstruct_master_refuses_clipping(tmp_path: Path):
    stems = np.column_stack([
        np.full(32, 0.7, dtype=np.float32),
        np.full(32, 0.6, dtype=np.float32),
    ])
    with pytest.raises(MixError, match="exceeds"):
        reconstruct_master(stems, 8000, 8000, tmp_path / "should_not_exist.wav")
    assert not (tmp_path / "should_not_exist.wav").exists()


def test_explicit_silero_failure_is_not_mislabeled_as_energy(monkeypatch):
    monkeypatch.setitem(sys.modules, "silero_vad", None)
    with pytest.raises(OptionalBackendError, match="optional neural extra"):
        run_vad(np.zeros(1600, dtype=np.float32), 16000, backend="silero")



def test_pipeline_keeps_original_master_when_separator_fails(tmp_path: Path, monkeypatch):
    import json
    import soundfile as sf

    from ai_podcast_studio import pipeline
    from ai_podcast_studio import speech_separation

    root = Path(__file__).resolve().parents[1]
    audio_path = tmp_path / "input.wav"
    sf.write(audio_path, np.zeros(800, dtype=np.float32), 8000)
    project_path = tmp_path / "project.json"
    project_path.write_text(json.dumps({
        "project": {"name": "fallback", "resolution": "320x180", "fps": 5},
        "characters": {"char_1": {"speaker_id": None, "position": "center"}},
        "segments": [],
        "models": {"vad": "energy", "diarization": "manual", "separation": "sepformer",
                   "num_speakers": 2},
        "audio": {"final_mix": "original"},
    }))
    captured = {}

    def fail_separator(*args, **kwargs):
        raise RuntimeError("simulated model load failure")

    def fake_render_video(**kwargs):
        captured["audio_path"] = str(kwargs["audio_path"])
        kwargs["output_path"].write_bytes(b"synthetic-video-placeholder")
        return {"path": str(kwargs["output_path"]), "audio_source": str(kwargs["audio_path"])}

    monkeypatch.setattr(speech_separation, "separate_audio", fail_separator)
    monkeypatch.setattr(pipeline, "render_video", fake_render_video)
    monkeypatch.setattr(pipeline, "validate_video", lambda path: {"streams": []})
    report = pipeline.run_pipeline(
        audio_path=audio_path, project_path=project_path,
        character_root=tmp_path / "characters", background_root=tmp_path / "backgrounds",
        output_dir=tmp_path / "out", schema_path=root / "config/project.schema.json",
        resume=False, width=320, height=180, fps=5)
    assert report["separation"]["status"] == "failed_original_master_retained"
    assert report["render_audio_source"] == str(audio_path)
    assert captured["audio_path"] == str(audio_path)



def test_separation_chunk_stitching_uses_consistent_padding_and_timeline(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from ai_podcast_studio.speech_separation import separate_audio

    class FakeTensor:
        def __init__(self, array):
            self.array = np.asarray(array, dtype=np.float32)

        def unsqueeze(self, axis):
            return FakeTensor(np.expand_dims(self.array, axis))

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    fake_torch = SimpleNamespace(from_numpy=FakeTensor, no_grad=nullcontext, set_num_threads=lambda count: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeSeparator:
        def separate_batch(self, batch):
            waveform = batch.array[0]
            estimates = np.stack((waveform * 0.6, waveform * 0.4), axis=-1)[None, :, :]
            return FakeTensor(estimates)

    audio = np.linspace(-0.2, 0.2, 1001, dtype=np.float32)
    result = separate_audio(audio, 8000, num_speakers=2, model=FakeSeparator(),
                            chunk_seconds=0.1, overlap_seconds=0.02)
    assert result["sources"].shape == (len(audio), 2)
    assert result["sample_rate"] == 8000
    assert result["metadata"]["chunk_count"] == 2
    assert np.isfinite(result["sources"]).all()



def test_ecapa_overlapping_windows_become_approximate_nonoverlapping_turns():
    from ai_podcast_studio.diarization import _segments_from_labeled_windows

    windows = [(0, 20), (10, 30), (20, 40)]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    centers = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)]
    segments, uncertain = _segments_from_labeled_windows(
        windows, np.array([0, 1, 1]), embeddings, centers, 0.5, sample_rate=10)
    assert uncertain == 0
    assert len(segments) == 2
    assert segments[0]["end"] == pytest.approx(1.5)
    assert segments[1]["start"] == pytest.approx(1.5)
    assert segments[1]["end"] == pytest.approx(4.0)
    assert segments[0]["speaker_id"] == "Speaker_0"
    assert segments[1]["speaker_id"] == "Speaker_1"



def test_pipeline_applies_explicit_source_mapping_after_base_tracks(tmp_path: Path, monkeypatch):
    import json
    import soundfile as sf

    from ai_podcast_studio import pipeline
    from ai_podcast_studio import speech_separation

    root = Path(__file__).resolve().parents[1]
    audio_path = tmp_path / "input.wav"
    sf.write(audio_path, np.zeros(800, dtype=np.float32), 8000)
    project_path = tmp_path / "project.json"
    project_path.write_text(json.dumps({
        "project": {"name": "mapping", "resolution": "320x180", "fps": 5},
        "characters": {"char_1": {"speaker_id": None, "position": "center"}},
        "segments": [],
        "models": {"vad": "energy", "diarization": "manual", "separation": "sepformer",
                   "num_speakers": 2, "source_to_character": {"source_1": "char_1"}},
        "audio": {"final_mix": "separated_sources"},
    }))
    captured = {}

    def fake_separator(audio, sample_rate, **kwargs):
        sources = np.column_stack([
            np.full(len(audio), 0.1, dtype=np.float32),
            np.full(len(audio), 0.2, dtype=np.float32),
        ])
        return {"sources": sources, "sample_rate": 8000,
                "metadata": {"backend": "fake_test", "expected_sources": 2}}

    def fake_render_video(**kwargs):
        captured["audio_path"] = str(kwargs["audio_path"])
        kwargs["output_path"].write_bytes(b"synthetic-video-placeholder")
        return {"path": str(kwargs["output_path"]), "audio_source": str(kwargs["audio_path"])}

    monkeypatch.setattr(speech_separation, "separate_audio", fake_separator)
    monkeypatch.setattr(pipeline, "render_video", fake_render_video)
    monkeypatch.setattr(pipeline, "validate_video", lambda path: {"streams": []})
    output_dir = tmp_path / "out"
    report = pipeline.run_pipeline(
        audio_path=audio_path, project_path=project_path,
        character_root=tmp_path / "characters", background_root=tmp_path / "backgrounds",
        output_dir=output_dir, schema_path=root / "config/project.schema.json",
        resume=False, width=320, height=180, fps=5)
    mapped_path = Path(report["separation"]["character_stem_mapping"]["mapped_characters"]["char_1"])
    assert mapped_path.parent.parent.parent.name == "separated_sources"
    assert (output_dir / "work/char_1.wav").is_file()
    mapped, rate = sf.read(mapped_path)
    assert rate == 8000
    assert float(np.mean(mapped)) == pytest.approx(0.2, abs=2e-5)
    assert report["separation"]["character_stem_mapping"]["mapped_characters"]["char_1"] == str(mapped_path)
    assert Path(report["render_audio_source"]).name == "separated_master.wav"
    assert captured["audio_path"] == report["render_audio_source"]



def test_stem_writers_refuse_pcm16_clipping(tmp_path: Path):
    from ai_podcast_studio.audio_mixer import write_source_stems

    over_peak = np.column_stack([np.array([0.1, 1.02, -0.1], dtype=np.float32),
                                 np.array([0.1, 0.2, 0.1], dtype=np.float32)])
    with pytest.raises(MixError, match="silent clipping"):
        write_source_stems(over_peak, 8000, tmp_path / "sources")
    with pytest.raises(MixError, match="silent clipping"):
        write_character_stems(over_peak, 8000, {"source_0": "char_1"},
                              {"char_1"}, tmp_path / "characters", 8000)
    assert not (tmp_path / "sources/source_0.wav").exists()
    assert not (tmp_path / "characters/char_1.wav").exists()



def test_cache_fingerprint_includes_lipsync_backend_signature(tmp_path: Path):
    from ai_podcast_studio.pipeline import _fingerprint

    audio = tmp_path / "audio.wav"
    project = tmp_path / "project.json"
    audio.write_bytes(b"audio")
    project.write_bytes(b"project")
    assert _fingerprint(audio, project, "fallback:v1") != _fingerprint(audio, project, "rhubarb:v1")


def test_corrupt_track_and_lipsync_cache_are_rebuilt(tmp_path: Path, monkeypatch):
    import json
    import soundfile as sf

    from ai_podcast_studio import pipeline

    root = Path(__file__).resolve().parents[1]
    audio_path = tmp_path / "input.wav"
    sf.write(audio_path, np.zeros(800, dtype=np.float32), 8000)
    project_path = tmp_path / "project.json"
    project_path.write_text(json.dumps({
        "project": {"name": "cache recovery", "resolution": "320x180", "fps": 5},
        "characters": {"char_1": {"speaker_id": None, "position": "center"}},
        "segments": [],
        "models": {"vad": "energy", "diarization": "manual", "separation": "none"},
        "audio": {"final_mix": "original"},
    }))
    output_dir = tmp_path / "out"
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True)
    (work_dir / "checkpoint.json").write_text(json.dumps({"fingerprint": "fixed"}))
    (work_dir / "track_report.json").write_bytes(b"\xff\xfe")
    (work_dir / "char_1.lipsync.json").write_text(json.dumps({
        "method": "rhubarb", "phonetic": True,
        "cues": [{"start": "not-a-time", "end": 2, "value": "A"}],
    }))
    rebuilt = {"called": False}

    def fake_render_video(**kwargs):
        kwargs["output_path"].write_bytes(b"video")
        return {"path": str(kwargs["output_path"]), "audio_source": str(kwargs["audio_path"])}

    real_make_tracks = pipeline.make_character_tracks

    def track_builder(*args, **kwargs):
        rebuilt["called"] = True
        return real_make_tracks(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_fingerprint", lambda *args: "fixed")
    monkeypatch.setattr(pipeline, "make_character_tracks", track_builder)
    monkeypatch.setattr(pipeline, "render_video", fake_render_video)
    monkeypatch.setattr(pipeline, "validate_video", lambda path: {"streams": []})
    report = pipeline.run_pipeline(
        audio_path=audio_path, project_path=project_path,
        character_root=tmp_path / "characters", background_root=tmp_path / "backgrounds",
        output_dir=output_dir, schema_path=root / "config/project.schema.json",
        resume=True, width=320, height=180, fps=5)

    assert rebuilt["called"] is True
    assert report["resume_cache_used"] is True
    assert report["lipsync"]["char_1"]["method"] == "waveform_envelope_fallback"
    assert json.loads((work_dir / "track_report.json").read_text())["method"] == \
        "manual_interval_masking_not_source_separation"
    cached_cues = json.loads((work_dir / "char_1.lipsync.json").read_text())
    assert cached_cues["cues"] and all(isinstance(cue["start"], (int, float)) for cue in cached_cues["cues"])
