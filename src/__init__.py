"""Typed package root for the Hybrid-RAG pipeline.

Importing this exposes the pipeline's typed entry points (``QAbot``) so stages and
external callers can avoid the ad-hoc ``from <mod> import ...`` split that stage
modules historically used.
"""

from .retriever import QAbot

__all__ = ["QAbot"]
