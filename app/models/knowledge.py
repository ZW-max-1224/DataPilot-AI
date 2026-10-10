from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeDocument(BaseModel):
    """Metadata and extracted text for an explicitly approved knowledge source."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=255)
    source: str = Field(min_length=1, max_length=512)
    source_type: Literal["builtin", "upload"]
    knowledge_base_id: str = Field(min_length=1, max_length=64)
    dataset_id: str | None = Field(default=None, max_length=64)
    content_hash: str = Field(min_length=64, max_length=64)
    content: str = Field(min_length=1)
    status: Literal["pending", "indexing", "indexed", "failed"] = "pending"
    chunk_count: int = Field(default=0, ge=0)
    indexing_error: str | None = Field(default=None, max_length=500)
    embedding_provider: str | None = Field(default=None, max_length=64)
    embedding_model: str | None = Field(default=None, max_length=255)
    embedding_dimensions: int | None = Field(default=None, gt=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class KnowledgeChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1, max_length=64)
    document_id: str = Field(min_length=1, max_length=64)
    knowledge_base_id: str = Field(min_length=1, max_length=64)
    dataset_id: str | None = Field(default=None, max_length=64)
    title: str = Field(min_length=1, max_length=255)
    source: str = Field(min_length=1, max_length=512)
    section: str = Field(default="", max_length=500)
    chunk_index: int = Field(ge=0)
    content_hash: str = Field(min_length=64, max_length=64)
    language: Literal["zh", "en", "mixed", "unknown"]
    text: str = Field(min_length=1)


class KnowledgeSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    chunk_id: str
    title: str
    source: str
    content: str
    score: float
    metadata: dict[str, str | int | None] = Field(default_factory=dict)


class KnowledgeSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "no_results", "embedding_unavailable", "embedding_failed", "knowledge_base_unavailable"]
    results: list[KnowledgeSearchResult] = Field(default_factory=list)
    result_count: int = Field(default=0, ge=0)
    truncated: bool = False
    error_code: str | None = None


class KnowledgeIndexResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    status: Literal["indexed", "failed"]
    chunk_count: int = Field(default=0, ge=0)
    error_code: str | None = None
