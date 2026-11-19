"""Atomic progress checkpoints and optional host resource readings."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically persist a JSON checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def read_checkpoint(path: Path) -> dict[str, Any] | None:
    """Return a valid checkpoint, or None when absent/corrupt."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return state if isinstance(state, dict) else None


def resource_snapshot() -> dict[str, Any]:
    """Report memory use when psutil is installed; never required for rendering."""
    try:
        import psutil  # optional dependency
        process = psutil.Process()
        return {"rss_bytes": process.memory_info().rss,
                "available_bytes": psutil.virtual_memory().available}
    except ImportError:
        return {"monitor": "psutil not installed"}
