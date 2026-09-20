"""Typed JSON helpers for writing pipeline stage outputs to disk.

These are the write-side companions to :mod:`src.data.load`. They serialize typed
models and write pretty-printed JSON, creating any parent directories needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T")


def _resolve(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""

    if isinstance(path, str):
        path = Path(path)
    if not path.is_absolute():
        package_dir = Path(__file__).resolve().parent
        repo_root = package_dir.parent.parent  # src.data -> src -> repo root
        return (repo_root / path).resolve() if not (repo_root / path).exists() else path.resolve()
    return path


def write_json_array[T: Union[BaseModel, dict]](path: str | Path, records: Sequence[T]) -> None:
    """Serialize a sequence of models or ``dict``s to a JSON array."""

    resolved = _resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    payload = [r.model_dump() if isinstance(r, BaseModel) else r for r in records]
    resolved.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")
