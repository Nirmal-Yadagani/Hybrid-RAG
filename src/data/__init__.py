"""Typed data layer for the Hybrid-RAG pipeline.

Exposes the ``Config`` (Single Source of Truth over ``params.yaml``) and the
data models that flow between stages, plus the JSON load/write helpers that read
and write those models from disk with schema validation on the way in.

Usage::

    from src.data import load_config, RawPage, load_json, write_json_array

    cfg = load_config()
    raws = load_json("data/raw/ai_ml_raw_html.json", model=RawPage)
"""

from .config import Config, DEFAULT_PROJECT, default_config, load_config
from .load import T, load_json
from .models import ChunkRecord, DocumentMetadata, GoldenRecord, RawPage
from .save import write_json_array

__all__ = [
    "Config",
    "DEFAULT_PROJECT",
    "default_config",
    "load_config",
    "load_json",
    "T",
    "ChunkRecord",
    "DocumentMetadata",
    "GoldenRecord",
    "RawPage",
    "write_json_array",
]
