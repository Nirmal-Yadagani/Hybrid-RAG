"""Typed JSON helpers for reading JSON files produced by pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T")


def _resolve(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    if isinstance(path, str):
        path = Path(path)
    if not path.is_absolute():
        package_dir = Path(__file__).resolve().parent
        repo_root = package_dir.parent.parent  # src.data -> src -> repo root
        if (repo_root / path).exists():
            return (repo_root / path).resolve()
    return path.resolve()


def load_json[T: BaseModel](path, *, model: type[T]) -> list[T]:
    """Load a JSON array of dict-like records into a list of typed models.

    Each record may be a dict or already-constructed model (both tolerated).
    """
    try:
        data = json.loads(_resolve(path).read_text(encoding="utf-8"))
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Expected file not found: {path}") from err

    if not isinstance(data, list):
        raise ValueError(f"{path} expected a JSON array, got {type(data).__name__}")

    out: list[T] = []
    for record in data:
        if isinstance(record, dict):
            out.append(model.model_validate(record))
        else:
            out.append(record)  # type: ignore[assignment]
    return out
