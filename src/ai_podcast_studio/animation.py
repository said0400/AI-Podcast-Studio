"""Lightweight layered 2D character rig with deterministic idle motion."""
from __future__ import annotations

import math
import zlib
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
        # Per-character phase so several characters never blink or sway in unison.
        seed = zlib.crc32(self.asset_dir.name.encode("utf-8"))
        self._phase = (seed % 1000) / 1000.0 * 2.0 * math.pi
        self._blink_offset = (seed % 1000) / 1000.0 * 4.7
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

    @staticmethod
    def _transform(image: Image.Image, angle_degrees: float, dx: float, dy: float,
                   pivot: tuple[float, float]) -> Image.Image:
        """Rotate about a pivot and translate with sub-pixel precision (premultiplied alpha)."""
        theta = math.radians(angle_degrees)
        a, b = math.cos(theta), math.sin(theta)
        cx, cy = pivot
        coeffs = (a, b, -a * (cx + dx) - b * (cy + dy) + cx,
                  -b, a, b * (cx + dx) - a * (cy + dy) + cy)
        premultiplied = image.convert("RGBa")
        moved = premultiplied.transform(image.size, Image.Transform.AFFINE, coeffs,
                                        resample=Image.Resampling.BILINEAR)
        return moved.convert("RGBA")

    def render(self, time_seconds: float, *, mouth: str = "X", speaking: bool = False,
               expression: str = "neutral", energy: float = 0.0, look: float = 0.0,
               listening: bool = False) -> Image.Image:
        """Compose a frame with deterministic motion.

        ``energy`` (0..1) is how strongly the character is talking right now: it adds
        head nods, tilt and eyebrow emphasis. ``look`` (-1..1) turns eyes and head toward
        whoever is speaking; ``listening`` adds slow acknowledging nods.
        """
        w, h = self.canvas_size
        t = float(time_seconds)
        phase = self._phase
        energy = min(1.0, max(0.0, float(energy)))
        look = min(1.0, max(-1.0, float(look)))
        frame = Image.new("RGBA", self.canvas_size, (0, 0, 0, 0))
        bob = 2.0 * math.sin(t * 2.0 + phase)
        for name in ("body", "hand_idle", "hand_wave"):
            layer = self.layers.get(name)
            if layer:
                frame.alpha_composite(self._move(layer, 0, round(bob) if name == "body" else 0))

        head = Image.new("RGBA", self.canvas_size, (0, 0, 0, 0))
        if "head" in self.layers:
            head.alpha_composite(self.layers["head"])
        blink = 0.12 < (t + self._blink_offset) % 4.7 < 0.24
        eyes = self.layers.get("eyes_closed" if blink else "eyes_open")
        if eyes:
            head.alpha_composite(eyes)
        if "pupils" in self.layers:
            head.alpha_composite(self._move(self.layers["pupils"], round(look * 7), 0))
        emphasis = max(0.0, energy - 0.65) / 0.35
        brow_name = "eyebrows_" + (expression if expression in {"happy", "angry", "sad"} else "neutral")
        if brow_name in self.layers:
            head.alpha_composite(self._move(self.layers[brow_name], 0, -round(5 * emphasis)))
        shape = str(mouth).upper()
        if shape not in "ABCDEFGHX" or len(shape) != 1:
            shape = "A" if speaking else "X"
        mouth_layer = self.layers.get(f"mouth_{shape}")
        if mouth_layer:
            head.alpha_composite(mouth_layer)
        elif speaking:
            # Visible, deterministic fallback at the conventional lower-face location.
            ImageDraw.Draw(head).ellipse((w * 0.40, h * 0.49, w * 0.60, h * 0.54), fill=(75, 20, 35, 255))

        # Head motion: slow idle sway + speech-driven nods/tilt + listening acknowledgements.
        tilt = 1.0 * math.sin(t * 0.6 + phase) + energy * 1.6 * math.sin(t * 5.7 + phase)
        dy = 1.2 * math.sin(t * 1.1 + 2 * phase) + energy * 3.5 * math.sin(t * 12.6 + phase)
        dx = 0.8 * math.sin(t * 0.45 + phase) + look * 4.0
        if listening:
            pulse = max(0.0, math.sin(2 * math.pi * t / 3.8 + phase)) ** 6
            dy += 3.0 * pulse
            tilt += -1.5 * look + 0.8 * pulse
        head = self._transform(head, tilt, dx, dy, (w * 0.5, h * 0.56))
        frame.alpha_composite(head)
        if "body" not in self.layers and "head" not in self.layers:
            draw = ImageDraw.Draw(frame)
            draw.ellipse((w * .2, h * .1, w * .8, h * .78), fill=(100, 150, 200, 255), outline=(30, 40, 60, 255), width=5)
        return frame


def resize_character(frame: Image.Image, target_height: int) -> Image.Image:
    """Resize a character while preserving aspect ratio."""
    target_width = max(1, round(frame.width * target_height / frame.height))
    return frame.resize((target_width, target_height), Image.Resampling.LANCZOS)
