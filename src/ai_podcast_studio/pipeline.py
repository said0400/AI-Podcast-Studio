"""End-to-end orchestration for offline analysis, preview and full renders."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from .audio import inspect_audio, make_character_tracks, read_audio
from .checkpoint import read_checkpoint, resource_snapshot, save_checkpoint
from .config import load_project
from .lipsync import extract_cues, sanitize_cues
from .rendering import MAX_RENDER_FPS, render_video, validate_render_size
from .speaker_mapping import map_speakers
from .vad import run_vad

LOGGER = logging.getLogger(__name__)


def _fingerprint(audio_path: Path, project_path: Path, cue_backend_signature: str = "") -> str:
    """Hash inputs and cue backend identity for resumable intermediate stages."""
    digest = hashlib.sha256()
    digest.update(cue_backend_signature.encode("utf-8"))
    for path in (audio_path, project_path):
        digest.update(str(path.resolve()).encode())
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()


def _cached_track_report(path: Path, track_paths: list[Path]) -> dict[str, Any] | None:
    """Return a structurally valid track report or signal that it must be rebuilt."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(report, dict) or report.get("method") != "manual_interval_masking_not_source_separation":
        return None
    tracks = report.get("tracks")
    expected_ids = {track_path.stem for track_path in track_paths}
    if not isinstance(tracks, dict) or set(tracks) != expected_ids:
        return None
    for track_path in track_paths:
        entry = tracks.get(track_path.stem)
        if not isinstance(entry, dict) or entry.get("path") != str(track_path) or not track_path.is_file():
            return None
    return report


def _verify_assets(root: Path, project: dict[str, Any]) -> list[str]:
    warnings = []
    for character_id in project.get("characters", {}):
        folder = root / character_id
        if not folder.is_dir():
            warnings.append(f"Character folder missing: {folder}; placeholder will render")
    return warnings


def _run_vad_safely(audio: Any, sample_rate: int, models: dict[str, Any]) -> dict[str, Any]:
    backend = str(models.get("vad", "energy"))
    try:
        return run_vad(audio, sample_rate, backend=backend,
                       threshold=float(models.get("vad_threshold", 0.5)),
                       min_speech_duration_ms=int(models.get("min_speech_duration_ms", 120)),
                       min_silence_duration_ms=int(models.get("min_silence_duration_ms", 180)))
    except Exception as exc:
        warning = f"Requested VAD backend {backend!r} unavailable; no VAD regions used: {type(exc).__name__}: {exc}"
        LOGGER.warning(warning)
        return {"backend": "unavailable", "regions": [], "warnings": [warning]}


def _json_safe_diarization(result: dict[str, Any]) -> dict[str, Any]:
    """Remove private in-memory embedding vectors before writing reports."""
    return {key: value for key, value in result.items() if not key.startswith("_")}


def quality_check(*, audio_path: Path, project_path: Path, character_root: Path,
                  background_root: Path, schema_path: Path | None = None) -> dict[str, Any]:
    """Validate project/audio/artwork and run only the configured VAD stage."""
    project = load_project(project_path, schema_path)
    audio_info = inspect_audio(audio_path)
    audio, rate = read_audio(audio_path, mono=True)
    warnings = _verify_assets(Path(character_root), project)
    for message in warnings:
        LOGGER.warning(message)
    duration = len(audio) / rate
    from .audio import validate_segments
    segments = validate_segments(project.get("segments", []), duration)
    observed = {item["speaker_id"] for item in segments}
    mapping = map_speakers(project, observed)
    vad = _run_vad_safely(audio, rate, project.get("models", {}))
    return {"status": "ok", "audio": audio_info.to_dict(), "duration_seconds": duration,
            "manual_segments": len(segments), "vad_backend": vad["backend"],
            "vad_regions": len(vad["regions"]), "vad_warnings": vad.get("warnings", []),
            "speaker_mapping": mapping, "assets_warnings": warnings,
            "background_directory_exists": Path(background_root).is_dir(),
            "diarization_or_separation": "not run by quality-check"}


def run_pipeline(*, audio_path: Path, project_path: Path, character_root: Path,
                 background_root: Path, output_dir: Path, schema_path: Path | None = None,
                 preview: bool = False, resume: bool = True, width: int | None = None,
                 height: int | None = None, fps: int | None = None, preview_scale: float = 0.25) -> dict[str, Any]:
    """Run optional local backends; always retain the original master on failures."""
    audio_path, project_path = Path(audio_path), Path(project_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"
    report_dir = output_dir / "reports"
    work_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    project = load_project(project_path, schema_path)
    models = project.get("models", {})
    audio_settings = project.get("audio", {})
    project_width, project_height = (int(part) for part in project["project"]["resolution"].lower().split("x", 1))
    width = project_width if width is None else int(width)
    height = project_height if height is None else int(height)
    fps = int(project["project"]["fps"]) if fps is None else int(fps)
    validate_render_size(width, height, preview_scale if preview else 1.0)
    if fps < 1 or fps > MAX_RENDER_FPS:
        raise ValueError(f"Render FPS must be between 1 and {MAX_RENDER_FPS}")
    audio_info = inspect_audio(audio_path)
    audio, sample_rate = read_audio(audio_path)
    mono_audio = audio.mean(axis=1, dtype="float32") if audio.ndim == 2 else audio
    duration = len(audio) / sample_rate
    from .audio import validate_segments
    rhubarb_path = shutil.which("rhubarb")
    if rhubarb_path:
        try:
            rhubarb_stat = Path(rhubarb_path).stat()
            cue_backend_signature = f"rhubarb:{rhubarb_path}:{rhubarb_stat.st_size}:{rhubarb_stat.st_mtime_ns}"
        except OSError:
            cue_backend_signature = f"rhubarb:{rhubarb_path}"
    else:
        cue_backend_signature = "waveform_envelope_fallback:v1"
    fingerprint = _fingerprint(audio_path, project_path, cue_backend_signature)

    vad = _run_vad_safely(mono_audio, sample_rate, models)
    (report_dir / "vad.json").write_text(json.dumps(vad, indent=2), encoding="utf-8")
    vad_warnings = list(vad.get("warnings", []))

    diarization: dict[str, Any] = {"backend": "manual", "segments": [], "warnings": []}
    original_manual_segments = project.get("segments", [])
    configured_diarization = str(models.get("diarization", "manual"))
    if original_manual_segments:
        diarization = {"backend": "manual", "segments": original_manual_segments,
                       "warnings": ["Manual project segments take precedence; automatic diarization was not run."]
                       if configured_diarization != "manual" else []}
    elif configured_diarization == "speechbrain_ecapa":
        try:
            from .diarization import diarize_audio
            diarization = diarize_audio(
                mono_audio, sample_rate, vad.get("regions", []),
                expected_speakers=(int(models["num_speakers"]) if models.get("num_speakers") is not None else None),
                cache_dir=Path(models.get("cache_dir", "models/cache")) / "spkrec-ecapa-voxceleb",
                window_seconds=float(models.get("diarization_window_seconds", 2.0)),
                hop_seconds=float(models.get("diarization_hop_seconds", 1.0)),
                max_cosine_distance=float(models.get("max_cosine_distance", 0.28)))
        except Exception as exc:
            warning = f"SpeechBrain diarization failed; use manual segments/review: {type(exc).__name__}: {exc}"
            LOGGER.warning(warning)
            diarization = {"backend": "unavailable", "segments": [], "warnings": [warning]}
    elif configured_diarization != "manual":
        warning = f"Unknown diarization backend {configured_diarization!r}; manual segments only."
        diarization = {"backend": "unavailable", "segments": [], "warnings": [warning]}
    if not original_manual_segments and diarization.get("segments"):
        project["segments"] = diarization["segments"]
    project["segments"] = validate_segments(project.get("segments", []), duration)
    (report_dir / "diarization.json").write_text(
        json.dumps(_json_safe_diarization(diarization), indent=2), encoding="utf-8")
    diarization_warnings = list(diarization.get("warnings", []))
    observed = {segment["speaker_id"] for segment in project["segments"]}
    mapping = map_speakers(project, observed)

    separation_status: dict[str, Any] = {"backend": str(models.get("separation", "none")),
                                         "status": "not_requested", "original_master_retained": True,
                                         "source_to_character_is_explicit": True}
    separated_sources = None
    source_sample_rate = None
    source_paths: dict[str, str] = {}
    mapped_character_paths: dict[str, str] = {}
    separation_run_dir = work_dir / "separated_sources" / fingerprint[:16]
    source_map = models.get("source_to_character", {})
    separation_requested = str(models.get("separation", "none")) == "sepformer"
    if separation_requested:
        try:
            from .audio_mixer import write_source_stems
            from .speech_separation import separate_audio
            result = separate_audio(
                mono_audio, sample_rate, num_speakers=int(models.get("num_speakers", 2)),
                cache_dir=Path(models.get("cache_dir", "models/cache")),
                chunk_seconds=float(models.get("separation_chunk_seconds", 12.0)),
                overlap_seconds=float(models.get("separation_overlap_seconds", 1.0)))
            separated_sources = result["sources"]
            source_sample_rate = int(result["sample_rate"])
            source_paths = write_source_stems(separated_sources, source_sample_rate, separation_run_dir)
            separation_status = {"backend": "speechbrain_sepformer", "status": "completed_requires_listening_review",
                                 "metadata": result["metadata"], "source_files": source_paths,
                                 "original_master_retained": True,
                                 "warning": "Source_N numbers are arbitrary outputs, not speaker identities."}
        except Exception as exc:
            warning = f"SepFormer separation failed; no stems used and original master retained: {type(exc).__name__}: {exc}"
            LOGGER.warning(warning)
            separation_status = {"backend": "speechbrain_sepformer", "status": "failed_original_master_retained",
                                 "error": warning, "source_files": {}, "original_master_retained": True}
            separated_sources = None
            source_sample_rate = None
            source_paths = {}

    checkpoint_path = work_dir / "checkpoint.json"
    prior = read_checkpoint(checkpoint_path) if resume else None
    cached = prior is not None and prior.get("fingerprint") == fingerprint
    track_report_path = work_dir / "track_report.json"
    track_paths = [work_dir / f"{char_id}.wav" for char_id in project["characters"]]
    # Rebuild base tracks whenever a separator was requested, so stale mapped
    # stems from a prior run can never survive a backend failure.
    can_reuse_tracks = cached and not separation_requested
    track_report = (_cached_track_report(track_report_path, track_paths)
                    if can_reuse_tracks and track_report_path.is_file() else None)
    if track_report is not None:
        LOGGER.info("Resuming from validated audio-track checkpoint")
    else:
        track_report = make_character_tracks(audio, sample_rate, project, work_dir)
        track_report_path.write_text(json.dumps(track_report, indent=2), encoding="utf-8")
        save_checkpoint(checkpoint_path, {"stage": "audio_tracks", "fingerprint": fingerprint,
                                          "track_report": str(track_report_path)})

    if separated_sources is not None and source_map:
        try:
            from .audio_mixer import write_character_stems
            map_result = write_character_stems(
                separated_sources, source_sample_rate, source_map,
                set(project.get("characters", {})), separation_run_dir / "characters", sample_rate)
            separation_status["character_stem_mapping"] = map_result
            separation_status["status"] = "completed_mapped_stems_require_listening_review"
            mapped_character_paths = map_result["mapped_characters"]
        except Exception as exc:
            warning = f"Source-to-character mapping failed; conservative base tracks retained: {type(exc).__name__}: {exc}"
            LOGGER.warning(warning)
            separation_status["character_stem_mapping_error"] = warning

    cues_by_character: dict[str, list[dict[str, Any]]] = {}
    lipsync_status = {}
    for index, (char_id, base_track_path) in enumerate(zip(project["characters"], track_paths), start=1):
        track_path = Path(mapped_character_paths.get(char_id, str(base_track_path)))
        cue_path = work_dir / f"{char_id}.lipsync.json"
        if cached and cue_path.is_file() and not separation_requested:
            try:
                payload = json.loads(cue_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or not isinstance(payload.get("method"), str):
                    raise ValueError("Cached lip-sync payload is missing method metadata")
                clean_cues, cue_warnings = sanitize_cues(payload.get("cues"), duration)
                if cue_warnings:
                    raise ValueError("Cached lip-sync cues failed validation")
                payload["cues"] = clean_cues
            except (OSError, json.JSONDecodeError, ValueError, TypeError):
                payload = extract_cues(track_path, cue_path)
        else:
            payload = extract_cues(track_path, cue_path)
        cues_by_character[char_id] = payload.get("cues", [])
        lipsync_status[char_id] = {"method": payload.get("method"), "phonetic": payload.get("phonetic", False),
                                   "cue_count": len(payload.get("cues", []))}
        LOGGER.info("Lip-sync cues prepared for %s (%d/%d)", char_id, index, len(track_paths))

    render_audio_path = audio_path
    final_mix = str(audio_settings.get("final_mix", "original"))
    if final_mix == "separated_sources":
        if separated_sources is None or source_sample_rate is None:
            separation_status["final_mix"] = "requested_but_unavailable_original_used"
        else:
            try:
                from .audio_mixer import reconstruct_master
                mix_path = work_dir / "separated_master.wav"
                mix_report = reconstruct_master(separated_sources, source_sample_rate, sample_rate, mix_path)
                render_audio_path = mix_path
                separation_status["final_mix"] = mix_report
            except Exception as exc:
                warning = f"Separated stems were not safely mixable; original master retained: {type(exc).__name__}: {exc}"
                LOGGER.warning(warning)
                separation_status["final_mix"] = {"status": "rejected_original_used", "warning": warning}
    elif final_mix != "original":
        warning = f"Unknown final_mix {final_mix!r}; original master retained."
        separation_status["final_mix"] = {"status": "unknown_option_original_used", "warning": warning}

    scale = preview_scale if preview else 1.0
    render_name = "preview.mp4" if preview else "episode.mp4"
    video_path = output_dir / render_name
    render_report = render_video(audio_path=render_audio_path, project=project, character_root=Path(character_root),
                                 background_root=Path(background_root), cues_by_character=cues_by_character,
                                 output_path=video_path, width=width, height=height, fps=fps,
                                 preview_scale=scale)
    verification = validate_video(video_path)
    report = {"status": "ok", "video": render_report, "validation": verification,
              "audio": audio_info.to_dict(), "speaker_mapping": mapping,
              "track_report": track_report, "lipsync": lipsync_status,
              "vad": {"backend": vad.get("backend"), "region_count": len(vad.get("regions", [])),
                      "warnings": vad_warnings},
              "diarization": _json_safe_diarization(diarization),
              "diarization_mapping": mapping,
              "diarization_warnings": diarization_warnings,
              "separation": separation_status,
              "separation_status": ("not performed; original master retained" if not separation_requested
                                    else str(separation_status.get("status", "unknown"))),
              "render_audio_source": str(render_audio_path),
              "resource_snapshot": resource_snapshot(), "resume_cache_used": bool(cached)}
    report_path = report_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_checkpoint(checkpoint_path, {"stage": "render_complete", "fingerprint": fingerprint,
                                      "report": str(report_path), "video": str(video_path)})
    return report


def validate_video(path: Path) -> dict[str, Any]:
    """Use ffprobe to verify container, video dimensions and audio stream."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required to validate the generated MP4")
    import subprocess
    result = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                             "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate",
                             "-of", "json", str(path)], capture_output=True, text=True, timeout=60, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe rejected output: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not any(stream.get("codec_type") == "video" for stream in streams):
        raise RuntimeError("Output has no video stream")
    if not any(stream.get("codec_type") == "audio" for stream in streams):
        raise RuntimeError("Output has no audio stream")
    return {"format_duration": float(data.get("format", {}).get("duration", 0)), "streams": streams}
