"""Single source of truth for W&B project/group resolution.

Every stage's ``wandb.init(...)`` should pull these two keys through here so all
runs land under one parent project (and, when a group is set, one shared cluster
in the W&B UI).

Resolution order (highest precedence first):

1. Environment variable ``WANDB_PROJECT`` / ``WANDB_GROUP`` — ideal for
   ``dvc exp run`` or ad-hoc runs that want a specific parent per invocation.
2. ``params.yaml`` keys ``wandb_project`` / ``wandb_group`` — the repo default.
3. Sensible fallback (project ``hybrid-rag``; no group).

Per-stage run ``name`` is left to the caller — it stays distinct so that runs
inside a group are still individually labeled.
"""

from __future__ import annotations

import os

DEFAULT_PROJECT = "hybrid-rag"


def project(params: dict) -> str:
    return os.getenv("WANDB_PROJECT") or params.get("wandb_project") or DEFAULT_PROJECT


def group(params: dict) -> str | None:
    return os.getenv("WANDB_GROUP") or params.get("wandb_group")


def common_kwargs(params: dict) -> dict:
    """Return the shared ``wandb.init(...)`` kwargs (project/group).

    ``None`` values are omitted so wandb's own defaults/env resolution still
    apply if we don't have a value.
    """
    kwargs: dict = {"project": project(params)}
    g = group(params)
    if g:
        kwargs["group"] = g
    return kwargs
