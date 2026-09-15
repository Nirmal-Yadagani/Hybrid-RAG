"""Structured, per-stage logging for the Hybrid-RAG pipeline.

Every stage writes JSON Lines (one JSON object per line) to ``logs/<stage>.jsonl``
so runs can be grepped / diffed / loaded into a dashboard. A terse readable line is
mirrored to stderr so the terminal stays usable.

Usage::

    from logger import StageLogger

    log = StageLogger("chunk")
    log.info("chunk.start", "Chunking started", chunker="docling")
    log.info("chunk.done", "Chunking completed", chunk_count=1234, elapsed_s=12.3)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

# All stage logs land under a single directory, relative to the CWD (repo root).
LOG_DIR = "logs"


class _JsonLineFormatter(logging.Formatter):
    """Serialize each LogRecord to a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "stage": getattr(record, "stage", "unknown"),
            "event": getattr(record, "event", None),
            "message": record.getMessage(),
        }

        data = getattr(record, "data", None)
        if data:
            payload["data"] = data

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


class StageLogger:
    """A small, self-describing logger scoped to one pipeline stage.

    Instances are cached per-stage name so repeated calls reuse the same
    handlers (no duplicate log lines, no file-handle leaks).
    """

    _instances: dict[str, "StageLogger"] = {}

    def __new__(cls, stage: str) -> "StageLogger":
        if stage not in cls._instances:
            cls._instances[stage] = super().__new__(cls)
        return cls._instances[stage]

    def __init__(self, stage: str) -> None:
        if getattr(self, "_configured", False):
            return
        self.stage = stage
        self._configure()
        self._configured = True

    # -- internals -------------------------------------------------------------
    def _configure(self) -> None:
        os.makedirs(LOG_DIR, exist_ok=True)
        file_path = os.path.join(LOG_DIR, f"{self.stage}.jsonl")

        self._logger = logging.getLogger(f"hybrid_rag.{self.stage}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.handlers.clear()

        file_handler = RotatingFileHandler(
            file_path, maxBytes=5_000_000, backupCount=2, encoding="utf-8"
        )
        file_handler.setFormatter(_JsonLineFormatter())
        self._logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(f"[{self.stage}] %(levelname)s: %(message)s")
        )
        self._logger.addHandler(console_handler)

    def _emit(self, level: int, event: str, message: str | None = None, **data) -> None:
        self._logger.log(
            level,
            message or event,
            extra={"stage": self.stage, "event": event, "data": data},
        )

    # -- public API ------------------------------------------------------------
    def info(self, event: str, message: str | None = None, **data) -> None:
        self._emit(logging.INFO, event, message, **data)

    def warning(self, event: str, message: str | None = None, **data) -> None:
        self._emit(logging.WARNING, event, message, **data)

    def error(self, event: str, message: str | None = None, **data) -> None:
        self._emit(logging.ERROR, event, message, **data)
