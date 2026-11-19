"""Lightweight layered 2D character rig with deterministic idle motion."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

LAYER_NAMES = ["body", "head", "eyes_open", "eyes_closed", "pupils",
               "eyebrows_neutral", "eyebrows_happy", "eyebrows_angry", "eyebrows_sad",
               "mouth_A", "mouth_B", "mouth_C", "mouth_D", "mouth_E", "mouth_F",
               "mouth_G", "mouth_H", "mouth_X", "hand_idle", "hand_point", "hand_think", "hand_wave"]


class CharacterRig:
    """Load a transparent PNG layer set and compose one animated character frame."""
    def __init__(self, asset_dir: Path, *, canvas_size: tuple[int, int] = (512, 768)) -> None:
        self.asset_dir = Path(asset_dir)
        self.canvas_size = canvas_size
        self.layers: dict[str, Image.Image] = {}
        self.warnings: list[str] = []
        for name in LAYER_NAMES:
            path = self.asset_dir / f"{name}.png"
            if not path.exists():
                self.warnings.append(f"Missing optional layer: {path}")
                continue
            try:
                image = Image.open(path).convert("RGBA")
                if image.size != canvas_size:
                    raise ValueError(f"expected {canvas_size}, got {image.size}")
                self.layers[name] = image
            except (OSError, ValueError) as exc:
                self.warnings.append(f"Ignored invalid layer {path}: {exc}")
        if "body" not in self.layers and "head" not in self.layers:
            self.warnings.append("No body/head artwork found; rendering a safe placeholder silhouette.")

    @staticmethod
    def _move(image: Image.Image, dx: int, dy: int) -> Image.Image:
        moved = Image.new("RGBA", image.size, (0, 0, 0, 0))
        dst_x, dst_y = max(0, dx), max(0, dy)
        src_x, src_y = max(0, -dx), max(0, -dy)
        copy_width = min(image.width - src_x, image.width - dst_x)
        copy_height = min(image.height - src_y, image.height - dst_y)
        if copy_width > 0 and copy_height > 0:
            crop = image.crop((src_x, src_y, src_x + copy_width, src_y + copy_height))
            moved.alpha_composite(crop, (dst_x, dst_y))
        return moved

    def render(self, time_seconds: float, *, mouth: str = "X", speaking: bool = False,
               expression: str = "neutral") -> Image.Image:
        """Compose a frame; idle movement is subtle and deterministic."""
        w, h = self.canvas_size
        frame = Image.new("RGBA", self.canvas_size, (0, 0, 0, 0))
        bob = round(2.0 * math.sin(time_seconds * 2.0))
        for name in ("body", "hand_idle", "hand_wave", "head"):
            layer = self.layers.get(name)
            if layer:
                frame.alpha_composite(self._move(layer, 0, bob if name == "body" else 0))
        eye_cycle = time_seconds % 4.7
        blink = 0.12 < eye_cycle < 0.24
        eyes = self.layers.get("eyes_closed" if blink else "eyes_open")
        if eyes:
            frame.alpha_composite(eyes)
        for name in ("pupils", "eyebrows_" + (expression if expression in {"happy", "angry", "sad"} else "neutral")):
            if name in self.layers:
                frame.alpha_composite(self.layers[name])
        shape = str(mouth).upper()
        if shape not in "ABCDEFGHX":
            shape = "A" if speaking else "X"
        mouth_layer = self.layers.get(f"mouth_{shape}")
        if mouth_layer:
            frame.alpha_composite(mouth_layer)
        elif speaking:
            # Visible, deterministic fallback at the conventional lower-face location.
            draw = ImageDraw.Draw(frame)
            draw.ellipse((w * 0.40, h * 0.49, w * 0.60, h * 0.54), fill=(75, 20, 35, 255))
        if "body" not in self.layers and "head" not in self.layers:
            draw = ImageDraw.Draw(frame)
            draw.ellipse((w * .2, h * .1, w * .8, h * .78), fill=(100, 150, 200, 255), outline=(30, 40, 60, 255), width=5)
        return frame


def resize_character(frame: Image.Image, target_height: int) -> Image.Image:
    """Resize a character while preserving aspect ratio."""
    target_width = max(1, round(frame.width * target_height / frame.height))
    return frame.resize((target_width, target_height), Image.Resampling.LANCZOS)
