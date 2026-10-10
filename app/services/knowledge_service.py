from __future__ import annotations

import logging
import hashlib
import time
from typing import Protocol

from app.models.knowledge import (
    KnowledgeDocument,
    KnowledgeIndexResult,
    KnowledgeSearchResponse,
    KnowledgeSearchResult,
)
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError
from app.services.tracing_service import TracingService
from app.vectorstore.qdrant_store import QdrantVectorStore, VectorStoreError

logger = logging.getLogger(__name__)
MAX_SEARCH_RESULTS = 5
MAX_CHUNK_CHARS = 1_000
MAX_TOTAL_CONTEXT_CHARS = 4_000
EMBEDDING_BATCH_SIZE = 128


class KnowledgeDocumentRepository(Protocol):
    def save_document(self, document: KnowledgeDocument) -> object: ...


class KnowledgeService:
    """Orchestrate document indexing and bounded knowledge retrieval."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        repository: KnowledgeDocumentRepository | None = None,
        tracing: TracingService | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.repository = repository
        self.tracing = tracing or TracingService()

    def index_document(self, document: KnowledgeDocument) -> KnowledgeIndexResult:
        """Index (or re-index) one knowledge document.

        Failure semantics: every point of this document is deleted before the
        replacement chunks are embedded and written, so a re-index can never
        leave the previous version searchable. If deletion succeeds but
        embedding or writing then fails, the document is reported as ``failed``
        (never ``indexed``) and the partial writes are removed again. If the
        deletion itself fails, indexing is also reported as ``failed`` rather
        than pretending the previous version was replaced.
        """
        from app.knowledge.chunking import chunk_document

        indexing = document.model_copy(update={"status": "indexing", "indexing_error": None})
        try:
            if self.repository is not None:
                self.repository.save_document(indexing)
            chunks = chunk_document(document)
            if not chunks:
                raise ValueError("The knowledge document did not produce any indexable chunks.")
            # Chunk ids are f"{document_id}:{index}", so writing the new version
            # only overwrites indices 0..len(chunks)-1: without this deletion the
            # higher indices of a longer previous version stay searchable.
            self.vector_store.delete_document(document.document_id)
            for offset in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch = chunks[offset:offset + EMBEDDING_BATCH_SIZE]
                vectors = self.embedding_service.embed_documents([chunk.text for chunk in batch])
                self.vector_store.upsert_chunks(batch, vectors)
            indexed = document.model_copy(update={
                "status": "indexed",
                "chunk_count": len(chunks),
                "embedding_provider": self.embedding_service.provider,
                "embedding_model": self.embedding_service.model,
                "embedding_dimensions": self.embedding_service.dimensions,
                "indexing_error": None,
            })
            if self.repository is not None:
                self.repository.save_document(indexed)
            return KnowledgeIndexResult(document_id=document.document_id, status="indexed", chunk_count=len(chunks))
        except EmbeddingServiceError as exc:
            self._delete_partial(document.document_id)
            return self._failed(document, exc.code)
        except VectorStoreError as exc:
            self._delete_partial(document.document_id)
            return self._failed(document, getattr(exc, "code", None) or "knowledge_base_unavailable")
        except Exception as exc:
            logger.warning("Knowledge indexing failed (%s)", type(exc).__name__)
            self._delete_partial(document.document_id)
            return self._failed(document, "indexing_failed")

    def search(
        self,
        query: str,
        *,
        knowledge_base_id: str,
        dataset_id: str | None = None,
        top_k: int = 5,
    ) -> KnowledgeSearchResponse:
        if not query or not query.strip() or len(query) > 2_000:
            raise ValueError("Knowledge query must contain between 1 and 2000 characters.")
        if not knowledge_base_id or len(knowledge_base_id) > 64:
            raise ValueError("Knowledge base selection is invalid.")
        if dataset_id is not None and len(dataset_id) > 64:
            raise ValueError("Dataset selection is invalid.")
        if not 1 <= top_k <= MAX_SEARCH_RESULTS:
            raise ValueError("Knowledge retrieval limit must be between 1 and 5.")
        started = time.perf_counter()
        query_text = query.strip()
        with self.tracing.span(
            "Knowledge Retrieval",
            run_type="retriever",
            inputs={
                "query_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
                "query_chars": len(query_text),
                "knowledge_base_id": knowledge_base_id,
                "dataset_id": dataset_id,
                "top_k": top_k,
            },
            metadata={"embedding_model": self.embedding_service.model,
                      "embedding_provider": self.embedding_service.provider},
        ) as trace:
            response = self._search_validated(
                query_text,
                knowledge_base_id=knowledge_base_id,
                dataset_id=dataset_id,
                top_k=top_k,
            )
            trace.end({
                "status": response.status,
                "error_code": response.error_code,
                "result_count": response.result_count,
                "truncated": response.truncated,
                "duration_ms": round((time.perf_counter() - started) * 1_000),
                "results": [
                    {"document_id": item.document_id, "chunk_id": item.chunk_id,
                     "score": round(item.score, 6)}
                    for item in response.results
                ],
            })
            return response

    def _search_validated(
        self,
        query: str,
        *,
        knowledge_base_id: str,
        dataset_id: str | None,
        top_k: int,
    ) -> KnowledgeSearchResponse:
        if not self.embedding_service.is_configured:
            return KnowledgeSearchResponse(status="embedding_unavailable", error_code="embedding_unavailable")
        try:
            query_vector = self.embedding_service.embed_query(query)
        except EmbeddingServiceError as exc:
            status = "embedding_unavailable" if exc.code == "embedding_unavailable" else "embedding_failed"
            return KnowledgeSearchResponse(status=status, error_code=exc.code)
        try:
            results = self.vector_store.search(
                query_vector,
                knowledge_base_id=knowledge_base_id,
                dataset_id=dataset_id,
                limit=top_k,
            )
        except VectorStoreError as exc:
            # The response status stays the generic knowledge_base_unavailable
            # (a Literal in the response model); the distinct, actionable reason
            # is carried in error_code, e.g. embedding_dimension_mismatch.
            return KnowledgeSearchResponse(
                status="knowledge_base_unavailable",
                error_code=getattr(exc, "code", None) or "knowledge_base_unavailable",
            )
        bounded, truncated = self._bound_results(results)
        if not bounded:
            return KnowledgeSearchResponse(status="no_results")
        return KnowledgeSearchResponse(
            status="ok", results=bounded, result_count=len(bounded), truncated=truncated,
        )

    def _failed(self, document: KnowledgeDocument, code: str) -> KnowledgeIndexResult:
        failed = document.model_copy(update={"status": "failed", "indexing_error": code})
        if self.repository is not None:
            try:
                self.repository.save_document(failed)
            except Exception as exc:
                logger.warning("Knowledge failure status persistence failed (%s)", type(exc).__name__)
        return KnowledgeIndexResult(document_id=document.document_id, status="failed", error_code=code)

    def _delete_partial(self, document_id: str) -> None:
        try:
            self.vector_store.delete_document(document_id)
        except VectorStoreError:
            logger.warning("Could not remove partial knowledge index for document %s", document_id)

    @staticmethod
    def _bound_results(
        results: list[KnowledgeSearchResult],
    ) -> tuple[list[KnowledgeSearchResult], bool]:
        bounded: list[KnowledgeSearchResult] = []
        total_chars = 0
        truncated = False
        for result in results[:MAX_SEARCH_RESULTS]:
            content = result.content
            if len(content) > MAX_CHUNK_CHARS:
                content = content[:MAX_CHUNK_CHARS]
                truncated = True
            chunk_truncated = len(content) < len(result.content)
            remaining = MAX_TOTAL_CONTEXT_CHARS - total_chars
            if remaining <= 0:
                truncated = True
                break
            if len(content) > remaining:
                content = content[:remaining]
                truncated = True
            if not content:
                continue
            bounded.append(result.model_copy(update={"content": content}))
            total_chars += len(content)
            if len(content) < MAX_CHUNK_CHARS and chunk_truncated:
                break
        if len(results) > MAX_SEARCH_RESULTS:
            truncated = True
        return bounded, truncated
