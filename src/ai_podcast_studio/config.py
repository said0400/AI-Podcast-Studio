"""Project configuration and schema validation."""
from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


class ProjectConfigError(ValueError):
    """Raised when project JSON is invalid or incomplete."""


def load_project(path: Path, schema_path: Path | None = None) -> dict[str, Any]:
    """Load project JSON and validate it against the shipped JSON Schema."""
    path = Path(path)
    if not path.is_file():
        raise ProjectConfigError(f"Project JSON not found: {path}")
    try:
        project = json.loads(path.read_text(encoding="utf-8"))
        if schema_path is None:
            schema_resource = files("ai_podcast_studio").joinpath("project.schema.json")
            if schema_resource.is_file():
                schema = json.loads(schema_resource.read_text(encoding="utf-8"))
            else:
                # Source-tree fallback supports older editable checkouts.
                source_schema = Path(__file__).resolve().parents[2] / "config" / "project.schema.json"
                schema = json.loads(source_schema.read_text(encoding="utf-8"))
        else:
            schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ProjectConfigError(f"Could not read project/schema JSON: {exc}") from exc
    errors = sorted(Draft202012Validator(schema).iter_errors(project), key=lambda e: list(e.path))
    if errors:
        detail = "; ".join(f"{'.'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:8])
        raise ProjectConfigError(f"Project JSON failed schema validation: {detail}")
    return project
