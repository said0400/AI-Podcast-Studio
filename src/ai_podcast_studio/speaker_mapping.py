"""Manual, explicit speaker-to-character mapping utilities."""
from __future__ import annotations

from typing import Any


def map_speakers(project: dict[str, Any], observed_speakers: set[str]) -> dict[str, Any]:
    """Return explicit mappings and unresolved speaker IDs; never infer by order."""
    mapping: dict[str, str] = {}
    for character_id, specification in project.get("characters", {}).items():
        speaker_id = specification.get("speaker_id")
        if speaker_id:
            if speaker_id in mapping:
                raise ValueError(f"Speaker {speaker_id!r} is assigned to multiple characters")
            mapping[str(speaker_id)] = str(character_id)
    return {"mapping": mapping,
            "unmapped_speakers": sorted(observed_speakers - set(mapping)),
            "unassigned_characters": sorted(set(project.get("characters", {})) - set(mapping.values())),
            "method": "explicit_project_json"}
