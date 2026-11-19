"""Streaming Pillow compositor and FFmpeg H.264 encoder."""
from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .animation import CharacterRig, resize_character
from .director import Director, Shot, active_at
from .lipsync import mouth_at

LOGGER = logging.getLogger(__name__)
MAX_RENDER_DIMENSION = 8192
MAX_RENDER_PIXELS = 16_777_216
MAX_RENDER_FPS = 60


def validate_render_size(width: int, height: int, scale: float = 1.0) -> tuple[int, int]:
    """Clamp preview scale and reject frame sizes with unsafe memory demands."""
    if width < 2 or height < 2:
        raise ValueError("Render width and height must each be at least 2 pixels")
    scale = min(1.0, max(0.1, float(scale)))
    width = max(2, int(round(width * scale / 2) * 2))
    height = max(2, int(round(height * scale / 2) * 2))
    if width > MAX_RENDER_DIMENSION or height > MAX_RENDER_DIMENSION or width * height > MAX_RENDER_PIXELS:
        raise ValueError(f"Render size {width}x{height} exceeds safe limits ({MAX_RENDER_DIMENSION}px per side, "
                         f"{MAX_RENDER_PIXELS} pixels total)")
    return width, height


def _background_layer(folder: Path, name: str, size: tuple[int, int]) -> Image.Image | None:
    path = folder / f"{name}.png"
    if not path.is_file():
        return None
    try:
        return Image.open(path).convert("RGBA").resize(size, Image.Resampling.LANCZOS)
    except OSError as exc:
        LOGGER.warning("Ignoring unreadable background layer %s: %s", path, exc)
        return None


def compose_background(folder: Path, size: tuple[int, int], time_seconds: float) -> Image.Image:
    """Compose far/mid/front/lighting assets, with a generated fallback set."""
    width, height = size
    frame = Image.new("RGBA", size, (27, 42, 68, 255))
    far = _background_layer(folder, "bg_far", size)
    if far:
        frame.alpha_composite(far)
    else:
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, width, height), fill=(34, 53, 82, 255))
        draw.ellipse((width * .25, height * .02, width * .75, height * .95), fill=(49, 73, 106, 255))
    for name in ("bg_mid", "bg_front", "lighting", "logo"):
        layer = _background_layer(folder, name, size)
        if layer:
            frame.alpha_composite(layer)
    return frame


def _camera_crop(frame: Image.Image, shot: Any, positions: dict[str, int], time_seconds: float) -> Image.Image:
    """Apply a gently eased zoom and focus crop for the selected shot."""
    w, h = frame.size
    kind = getattr(shot, "kind", "wide")
    target_zoom = 1.16 if kind == "speaker_focus" else 1.035 if kind == "two_shot" else 1.0
    started = float(getattr(shot, "started_at", 0.0))
    eased = min(1.0, max(0.0, (time_seconds - started) / 0.7))
    eased = eased * eased * (3.0 - 2.0 * eased)
    zoom = 1.0 + (target_zoom - 1.0) * eased
    focus = getattr(shot, "focus", None)
    center_x = positions.get(focus, w // 2) if focus else w // 2
    crop_w, crop_h = w / zoom, h / zoom
    left = max(0.0, min(w - crop_w, center_x - crop_w / 2))
    top = max(0.0, min(h - crop_h, h - crop_h / 2))
    return frame.crop((int(left), int(top), int(left + crop_w), int(top + crop_h))).resize((w, h), Image.Resampling.LANCZOS)


def render_video(*, audio_path: Path, project: dict[str, Any], character_root: Path,
                 background_root: Path, cues_by_character: dict[str, list[dict[str, Any]]],
                 output_path: Path, width: int, height: int, fps: int,
                 preview_scale: float = 1.0, ffmpeg: str = "ffmpeg") -> dict[str, Any]:
    """Stream one RGB frame at a time into FFmpeg; frames are not retained in RAM."""
    ffmpeg_path = shutil.which(ffmpeg)
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg is required; install it and ensure it is on PATH")
    import soundfile as sf
    audio_info = sf.info(str(audio_path))
    duration = float(audio_info.duration)
    if duration <= 0:
        raise ValueError("Audio duration must be positive")
    width, height = validate_render_size(width, height, preview_scale)
    fps = int(fps)
    if fps < 1 or fps > MAX_RENDER_FPS:
        raise ValueError(f"Render FPS must be between 1 and {MAX_RENDER_FPS}")
    frame_count = math.ceil(duration * fps)
    char_specs = project.get("characters", {})
    rigs = {char_id: CharacterRig(Path(character_root) / char_id) for char_id in char_specs}
    for char_id, rig in rigs.items():
        for warning in rig.warnings:
            LOGGER.warning("%s: %s", char_id, warning)
    center_fraction = {"left": .17, "center": .50, "right": .83}
    positions = {char_id: int(width * center_fraction.get(spec.get("position", "center"), .5))
                 for char_id, spec in char_specs.items()}
    speaker_to_character = {str(spec["speaker_id"]): char_id for char_id, spec in char_specs.items() if spec.get("speaker_id")}
    segments = list(project.get("segments", []))
    director_cfg = project.get("director", {})
    director = Director(float(director_cfg.get("min_shot_duration", 3.0)),
                       float(director_cfg.get("camera_hysteresis", 1.0)),
                       bool(director_cfg.get("automatic_closeups", True)))
    manual_cues = sorted(project.get("cues", []), key=lambda item: float(item.get("time", 0)))
    camera_cues = [cue for cue in manual_cues if cue.get("camera")]
    expression_cues = [cue for cue in manual_cues if cue.get("expression") and cue.get("character") in char_specs]
    camera_index = 0
    expression_index = 0
    current_camera_cue: dict[str, Any] | None = None
    expressions = {char_id: "neutral" for char_id in char_specs}
    last_shot_key: tuple[str, str | None] | None = None
    last_camera_frame: Image.Image | None = None
    transition_from: Image.Image | None = None
    transition_started = 0.0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
               "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
               "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
               "-t", f"{duration:.6f}", "-movflags", "+faststart", str(output_path)]
    log_path = output_path.with_suffix(".ffmpeg.log")
    with log_path.open("w", encoding="utf-8") as error_log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_log)
        assert process.stdin is not None
        return_code = None
        completed = False
        try:
            character_height = int(height * .68)
            for frame_index in range(frame_count):
                t = frame_index / fps
                active = active_at(segments, t, speaker_to_character)
                shot = director.update(t, active)
                while camera_index < len(camera_cues) and float(camera_cues[camera_index].get("time", 0)) <= t:
                    current_camera_cue = camera_cues[camera_index]
                    camera_index += 1
                if current_camera_cue:
                    camera_name = str(current_camera_cue.get("camera", "")).lower().replace("-", "_").replace(" ", "_")
                    shot_aliases = {"wide_shot": "wide", "three_shot": "wide", "medium": "two_shot",
                                    "close_up": "speaker_focus", "extreme_close_up": "speaker_focus",
                                    "reaction_shot": "speaker_focus", "over_the_shoulder": "two_shot"}
                    kind = shot_aliases.get(camera_name, camera_name)
                    if kind in {"wide", "two_shot", "speaker_focus"}:
                        focus = current_camera_cue.get("character") if kind == "speaker_focus" else None
                        shot = Shot(kind, focus if focus in char_specs else None,
                                    float(current_camera_cue.get("time", 0)),
                                    int(current_camera_cue.get("priority", 5)))
                while expression_index < len(expression_cues) and float(expression_cues[expression_index].get("time", 0)) <= t:
                    cue = expression_cues[expression_index]
                    expressions[str(cue["character"])] = str(cue["expression"]).lower()
                    expression_index += 1
                shot_key = (shot.kind, shot.focus)
                if last_shot_key is not None and shot_key != last_shot_key and last_camera_frame is not None:
                    transition = str((current_camera_cue or {}).get("transition", "dissolve")).lower()
                    transition_from = last_camera_frame.copy() if transition in {"dissolve", "fade"} else None
                    transition_started = t
                last_shot_key = shot_key
                frame = compose_background(Path(background_root), (width, height), t)
                for char_id, spec in char_specs.items():
                    char_cues = cues_by_character.get(char_id, [])
                    mouth = mouth_at(char_cues, t)
                    speaking = mouth not in {"X", "x", ""}
                    sprite = resize_character(rigs[char_id].render(t, mouth=mouth, speaking=speaking,
                                                                      expression=expressions[char_id]), character_height)
                    x = positions[char_id] - sprite.width // 2
                    y = height - sprite.height
                    # Soft ground shadow, deliberately beneath each character.
                    shadow = Image.new("RGBA", (sprite.width + 60, 36), (0, 0, 0, 0))
                    ImageDraw.Draw(shadow).ellipse((0, 5, shadow.width, 30), fill=(0, 0, 0, 75))
                    frame.alpha_composite(shadow, (x - 30, height - 34))
                    frame.alpha_composite(sprite, (x, y))
                frame = _camera_crop(frame, shot, positions, t)
                if transition_from is not None:
                    alpha = min(1.0, max(0.0, (t - transition_started) / 0.45))
                    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                    frame = Image.blend(transition_from, frame, alpha)
                    if alpha >= 1.0:
                        transition_from = None
                last_camera_frame = frame.copy()
                rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8)
                process.stdin.write(rgb.tobytes())
                if frame_index % max(1, fps * 10) == 0:
                    LOGGER.info("Rendering %.1f%%", 100.0 * frame_index / max(1, frame_count))
            process.stdin.close()
            return_code = process.wait(timeout=max(120, duration * 20))
            completed = (return_code == 0 and output_path.is_file() and output_path.stat().st_size > 0)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"FFmpeg render failed; see {log_path}: {exc}") from exc
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"FFmpeg render failed; see {log_path}: {exc}") from exc
        finally:
            try:
                if process.poll() is None:
                    try:
                        process.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
                    process.kill()
                process.wait()
            except OSError:
                LOGGER.exception("Failed to clean up FFmpeg process after render failure")
            if not completed:
                try:
                    output_path.unlink(missing_ok=True)
                except OSError:
                    LOGGER.warning("Could not remove incomplete render file %s", output_path)
    if return_code != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg exited with code {return_code}; see {log_path}")
    return {"path": str(output_path), "duration_seconds": duration, "fps": fps,
            "width": width, "height": height, "frames_sent": frame_count,
            "audio_source": str(audio_path), "ffmpeg_log": str(log_path)}
