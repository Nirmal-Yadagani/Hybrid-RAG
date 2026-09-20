"""Typed data models / schemas for the Hybrid-RAG pipeline.

These describe the JSON payloads that flow between pipeline stages and are
validated as documents are read from disk. They replace the implicit, stringly
constrained dicts that each stage used to build inline, so schema drift shows up
as a clear ``pydantic.ValidationError`` at load time instead of a vague
``KeyError`` deep in a LangChain call.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata attached to a raw source document (stage 1) or its chunks (stage 2 + embed)."""

    source_url: str
    topic: str
    source_type: str
    extraction_date: str
    chunk_id: int | None = None
    headings: str | None = None

    model_config = {"extra": "forbid"}


class RawPage(BaseModel):
    """Output of the dataset stage - the cleaned HTML body for one Wikipedia topic."""

    metadata: DocumentMetadata
    raw_html: str

    model_config = {"extra": "forbid"}


class ChunkRecord(BaseModel):
    """Output of the chunk stage - one Docling chunk with metadata from the parent document."""

    page_content: str
    metadata: DocumentMetadata

    model_config = {"extra": "forbid"}


class GoldenRecord(BaseModel):
    """A synthetic/golden Q&A pair as produced by the generate stage and consumed by eval.

    Field names match DeepEval's ``EvaluationDataset`` JSON format so downstream
    stages can reuse it directly.
    """

    input: str
    expected_output: str
    context: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}
