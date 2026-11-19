"""Rule-based TV director with minimum shot duration and hysteresis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class Shot:
    kind: str = "wide"
    focus: str | None = None
    started_at: float = 0.0
    priority: int = 1


class Director:
    """Select wide, focus and group shots without reacting to every pause."""
    def __init__(self, min_shot_duration: float = 3.0, hysteresis: float = 1.0,
                 automatic_closeups: bool = True) -> None:
        self.min_shot_duration = max(0.0, min_shot_duration)
        self.hysteresis = max(0.0, hysteresis)
        self.automatic_closeups = automatic_closeups
        self.current = Shot()

    @staticmethod
    def desired(active_characters: Iterable[str], automatic_closeups: bool = True) -> Shot:
        active = sorted(set(active_characters))
        if len(active) == 1 and automatic_closeups:
            return Shot("speaker_focus", active[0], priority=3)
        if len(active) == 2:
            return Shot("two_shot", None, priority=2)
        return Shot("wide", None, priority=1)

    def update(self, time_seconds: float, active_characters: Iterable[str]) -> Shot:
        """Switch only after minimum hold, unless a materially higher priority arrives."""
        wanted = self.desired(active_characters, self.automatic_closeups)
        same = wanted.kind == self.current.kind and wanted.focus == self.current.focus
        if same:
            return self.current
        held = time_seconds - self.current.started_at
        priority_jump = wanted.priority >= self.current.priority + 2
        if held >= self.min_shot_duration or (priority_jump and held >= self.hysteresis):
            wanted.started_at = time_seconds
            self.current = wanted
        return self.current


def active_at(segments: list[dict], time_seconds: float, speaker_to_character: dict[str, str]) -> list[str]:
    """Find all mapped speakers active at a time; preserve overlap as a set."""
    return sorted({speaker_to_character[item["speaker_id"]] for item in segments
                   if item["start"] <= time_seconds < item["end"]
                   and item["speaker_id"] in speaker_to_character})
