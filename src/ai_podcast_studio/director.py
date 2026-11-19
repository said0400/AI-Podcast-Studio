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


SHOT_ALIASES = {"wide_shot": "wide", "three_shot": "wide", "medium": "two_shot",
                "close_up": "speaker_focus", "extreme_close_up": "speaker_focus",
                "reaction_shot": "speaker_focus", "over_the_shoulder": "two_shot"}


def cue_to_shot(cue: dict, character_ids: Iterable[str]) -> Shot | None:
    """Translate a manual camera cue to a Shot, or None if its camera name is unknown."""
    name = str(cue.get("camera", "")).lower().replace("-", "_").replace(" ", "_")
    kind = SHOT_ALIASES.get(name, name)
    if kind not in {"wide", "two_shot", "speaker_focus"}:
        return None
    focus = cue.get("character") if kind == "speaker_focus" else None
    return Shot(kind, focus if focus in set(character_ids) else None,
                float(cue.get("time", 0)), int(cue.get("priority", 5)))


class ManualCameraCues:
    """Apply project ``cues`` camera instructions according to ``director.mode``.

    * ``automatic``: camera cues are ignored; the rule-based Director decides everything.
    * ``manual``: a cue holds until the next cue; before the first cue the shot is wide.
    * ``hybrid`` (default): a cue overrides the Director for ``duration`` seconds
      (cue field, else ``hold_seconds``), then control returns to the Director.
    """

    def __init__(self, cues: list[dict], mode: str, hold_seconds: float,
                 character_ids: Iterable[str]) -> None:
        self.mode = mode if mode in {"automatic", "manual", "hybrid"} else "hybrid"
        self.hold_seconds = max(0.0, hold_seconds)
        self.character_ids = list(character_ids)
        self._cues = sorted((c for c in cues if c.get("camera")), key=lambda c: float(c.get("time", 0)))
        self._index = 0
        self._shot: Shot | None = None
        self._expires = float("inf")

    def step(self, t: float) -> tuple[Shot | None, str | None]:
        """Return (active manual shot or None, transition name if a new cue just began)."""
        new_transition: str | None = None
        while self._index < len(self._cues) and float(self._cues[self._index].get("time", 0)) <= t:
            cue = self._cues[self._index]
            self._index += 1
            if self.mode == "automatic":
                continue
            shot = cue_to_shot(cue, self.character_ids)
            if shot is None:
                continue
            self._shot = shot
            if self.mode == "hybrid":
                try:
                    hold = float(cue.get("duration", self.hold_seconds))
                except (TypeError, ValueError):
                    hold = self.hold_seconds
                self._expires = float(cue.get("time", 0)) + max(0.0, hold)
            else:
                self._expires = float("inf")
            new_transition = str(cue.get("transition", "dissolve")).lower()
        if self._shot is not None and t >= self._expires:
            self._shot = None
        if self._shot is None:
            new_transition = None
        return self._shot, new_transition
