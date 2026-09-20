"""Single source of truth for pipeline configuration.

Reads ``params.yaml`` once and exposes it as a typed, validated ``Config`` object.
Stages take a ``Config`` in their constructor instead of calling ``yaml.safe_load``
on every file.

Parameter ownership stays in ``params.yaml`` (DVC tracks the keys there); this
module only *models* the file as typed Python so a bad value fails fast with a
readable ``ValidationError`` instead of a vague ``KeyError`` deep in a LangChain
call.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_PROJECT = "hybrid-rag"


class Config(BaseModel):
    """Top-level configuration object handed to every stage.

    Fields mirror ``params.yaml`` directly. A ``before`` validator coerces the raw
    flat dict into these named attrs, so subclasses (and tools) get typed access.
    """

    model_config = ConfigDict(populate_by_name=True)

    # Environment / device
    device: str = "cuda"

    # Data ingestion
    urls_path: str

    # Chunking
    chunk_max_tokens: int = 350
    chunk_merge_peers: bool = True

    # Embeddings
    dense_embedding_model: str = "BAAI/bge-base-en-v1.5"
    dense_embd_dim: int = 768
    sparse_embedding_model: str = "Qdrant/bm25"
    normalize_embeddings: bool = True

    # Vector store
    collection_name: str = "simple-wikipedia"
    persist_directory: str = "data/embeddings"

    # Retrieval & reranking
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_k: int = 60
    rerank_score_threshold: float = -2.5
    top_n: int = 5

    # Chat / answer generation
    chat_model: str = "gemma4:31b"
    chat_temperature: float = 0.0

    # Synthetic golden-set generation
    synthesizer_model: str = "qwen3.8:27b"
    synthesizer_temperature: float = 0.7

    # Golden quality controls
    quality_threshold: float = 0.6
    max_quality_retries: int = 3
    min_golden_quality_to_keep: float = 0.5
    num_evolutions: int = 1

    # Validation / LLM-as-a-judge
    validator_model: str = "gemma4:31b"
    validator_temperature: float = 0.0

    # Weights & Biases (params.yaml keys: wandb_project / wandb_group)
    wandb_project: str = "hybrid-rag"
    wandb_group: str | None = None

    @property
    def hf_token(self) -> str | None:
        return os.getenv("HF_TOKEN")

    @property
    def google_api_key(self) -> str | None:
        return os.getenv("GOOGLE_API_KEY")

    def wandb_common_kwargs(self) -> dict:
        """Shared ``wandb.init(...)`` kwargs (project/group) with env precedence.

        Resolution order (highest first): ``WANDB_PROJECT`` / ``WANDB_GROUP`` env
        vars, then ``params.yaml`` keys, then the repo defaults. ``None`` values
        are omitted so wandb's own defaults/env resolution still apply.
        """

        project = os.getenv("WANDB_PROJECT") or self.wandb_project
        kwargs: dict = {"project": project}
        if self.wandb_group:
            kwargs["group"] = self.wandb_group
        return kwargs

    @model_validator(mode="before")
    @classmethod
    def _coerce_flat(cls: type["Config"], data: Any) -> Any:
        """Coerce the flat ``params.yaml`` dict into typed fields.

        Handles stray/None values and the unusual ``w&b`` YAML key (parsed as a
        null anchor) gracefully. Unknown keys are accepted (params.yaml is the
        source of truth and may gain entries) but only known keys are stored.
        """

        if isinstance(data, Config):
            return data.model_dump()
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")
        return {k: v for k, v in data.items() if k in cls.model_fields and v is not None}


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load ``params.yaml`` from the repository root and validate it.

    Path resolves relative to the ``src`` package directory so it works from any
    working directory.
    """

    if path is None:
        package_dir = Path(__file__).resolve().parent
        yaml_path = package_dir.parent.parent / "params.yaml"
    else:
        yaml_path = Path(path)

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return Config(**raw)


@lru_cache(maxsize=1)
def default_config() -> Config:
    """Cached config - safe because it reads only params.yaml."""

    return load_config()


__all__ = ["Config", "load_config", "default_config", "DEFAULT_PROJECT"]
