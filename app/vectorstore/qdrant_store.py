from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Sequence

from qdrant_client import QdrantClient, models

from app.models.knowledge import KnowledgeChunk, KnowledgeSearchResult

logger = logging.getLogger(__name__)
COLLECTION_NAME = "datapilot_knowledge"
# Distinct availability codes so callers can tell an action-required index
# problem (rebuild the collection) from a plain backend outage.
ERROR_STORE_UNAVAILABLE = "knowledge_base_unavailable"
ERROR_DIMENSION_MISMATCH = "embedding_dimension_mismatch"


class VectorStoreError(RuntimeError):
    """Safe Qdrant error without backend credentials or implementation details."""

    code = ERROR_STORE_UNAVAILABLE


class EmbeddingDimensionMismatchError(VectorStoreError):
    """The persisted collection was built with different vector dimensions.

    A distinct, actionable error: retrying the same operation cannot succeed.
    The caller should surface :data:`ERROR_DIMENSION_MISMATCH` and can offer
    :meth:`QdrantVectorStore.rebuild_collection` to recreate the index.
    """

    code = ERROR_DIMENSION_MISMATCH


class QdrantVectorStore:
    """Persistent local Qdrant adapter; one process should own a local path."""

    def __init__(
        self,
        *,
        path: str | Path = "data/qdrant",
        collection_name: str = COLLECTION_NAME,
        dimensions: int = 1024,
        client: Any | None = None,
    ) -> None:
        if not collection_name or len(collection_name) > 128:
            raise ValueError("Qdrant collection name is invalid.")
        if dimensions <= 0:
            raise ValueError("Qdrant vector dimensions must be positive.")
        self.path = Path(path)
        self.collection_name = collection_name
        self.dimensions = dimensions
        try:
            self._client = client or QdrantClient(path=str(self.path))
        except Exception as exc:
            logger.warning("Qdrant local client initialization failed (%s)", type(exc).__name__)
            raise VectorStoreError("Qdrant local storage is unavailable.") from exc

    @property
    def is_available(self) -> bool:
        """``True`` only when storage works and the collection dimensions match."""
        return self.availability()[0]

    @property
    def availability_error_code(self) -> str | None:
        """``None`` when available, otherwise the distinct reason code."""
        return self.availability()[1]

    def availability(self) -> tuple[bool, str | None]:
        """Probe storage and collection health without raising.

        Returns ``(available, error_code)``. A collection created with other
        vector dimensions reports :data:`ERROR_DIMENSION_MISMATCH` instead of
        the generic :data:`ERROR_STORE_UNAVAILABLE`, because the fix is to
        rebuild the index rather than to retry the connection.
        """
        try:
            self._client.get_collections()
        except Exception:
            return False, ERROR_STORE_UNAVAILABLE
        try:
            if not self._client.collection_exists(self.collection_name):
                return True, None
            self._verify_collection_dimensions()
        except EmbeddingDimensionMismatchError:
            return False, ERROR_DIMENSION_MISMATCH
        except Exception:
            return False, ERROR_STORE_UNAVAILABLE
        return True, None

    def _collection_dimensions(self) -> int:
        """Vector size of the persisted collection; raises when it is unreadable."""
        info = self._client.get_collection(self.collection_name)
        vector_config = info.config.params.vectors
        if isinstance(vector_config, dict):
            if not vector_config:
                raise VectorStoreError("Qdrant collection has no vector configuration.")
            config = vector_config.get("") or next(iter(vector_config.values()))
        else:
            config = vector_config
        return int(config.size)

    def _verify_collection_dimensions(self) -> None:
        """Raise :class:`EmbeddingDimensionMismatchError` when the sizes differ."""
        if not self._client.collection_exists(self.collection_name):
            return
        if self._collection_dimensions() != self.dimensions:
            raise EmbeddingDimensionMismatchError(
                "Existing collection dimensions do not match the configured embedding dimensions."
            )

    def create_collection(self) -> None:
        """Create the collection when missing; verify dimensions when it exists.

        Raises :class:`EmbeddingDimensionMismatchError` when the persisted
        collection was built with other dimensions; a caller can then offer
        :meth:`rebuild_collection` to delete and recreate it.
        """
        try:
            if self._client.collection_exists(self.collection_name):
                self._verify_collection_dimensions()
                return
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.dimensions, distance=models.Distance.COSINE),
            )
            # Local mode evaluates payload filters without server-side indexes.
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.warning("Qdrant collection initialization failed (%s)", type(exc).__name__)
            raise VectorStoreError("Qdrant collection is unavailable.") from exc

    def rebuild_collection(self) -> None:
        """Delete and recreate the collection so a caller can offer "rebuild index".

        Destructive by design: every indexed chunk in this collection is
        dropped. Use it to recover from :data:`ERROR_DIMENSION_MISMATCH`.
        """
        try:
            self._release_local_collection()
            if self._client.collection_exists(self.collection_name):
                self._client.delete_collection(self.collection_name)
            self.create_collection()
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.warning("Qdrant collection rebuild failed (%s)", type(exc).__name__)
            raise VectorStoreError("Qdrant collection could not be rebuilt.") from exc

    def _release_local_collection(self) -> None:
        """Close the local-mode collection handle so its directory can be removed.

        qdrant-client local mode deletes the collection directory with
        ``shutil.rmtree(..., ignore_errors=True)``. On Windows that silently
        fails while the collection's SQLite file is still open, and the next
        ``create_collection`` then rehydrates the previous points (with their
        old dimensions) into the new collection. Closing the local collection
        first makes the delete real. Remote clients have no such handle.
        """
        local = getattr(self._client, "_client", None)
        collections = getattr(local, "collections", None)
        collection = collections.get(self.collection_name) if isinstance(collections, dict) else None
        close = getattr(collection, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"datapilot:{chunk_id}"))

    def upsert_chunks(self, chunks: Sequence[KnowledgeChunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("The number of vectors must match the number of chunks.")
        if not chunks:
            return
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.dimensions:
                raise ValueError("Vector dimensions do not match the collection configuration.")
            payload = chunk.model_dump(mode="json", exclude={"text"})
            payload["text"] = chunk.text
            points.append(models.PointStruct(
                id=self._point_id(chunk.chunk_id), vector=list(vector), payload=payload,
            ))
        try:
            self.create_collection()
            self._client.upsert(collection_name=self.collection_name, points=points, wait=True)
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.warning("Qdrant upsert failed (%s)", type(exc).__name__)
            raise VectorStoreError("Qdrant failed to index knowledge chunks.") from exc

    def search(
        self,
        vector: Sequence[float],
        *,
        knowledge_base_id: str,
        dataset_id: str | None = None,
        limit: int = 5,
    ) -> list[KnowledgeSearchResult]:
        if not 1 <= limit <= 5:
            raise ValueError("Knowledge retrieval limit must be between 1 and 5.")
        if len(vector) != self.dimensions:
            raise ValueError("Query vector dimensions do not match the collection configuration.")
        should = [models.FieldCondition(key="dataset_id", is_null=True)]
        if dataset_id:
            should.append(models.FieldCondition(key="dataset_id", match=models.MatchValue(value=dataset_id)))
        query_filter = models.Filter(
            must=[models.FieldCondition(key="knowledge_base_id", match=models.MatchValue(value=knowledge_base_id))],
            should=should,
        )
        try:
            if not self._client.collection_exists(self.collection_name):
                return []
            # A stale collection built with other dimensions must fail as a
            # distinct, actionable error instead of a generic backend outage.
            self._verify_collection_dimensions()
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=list(vector),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            results = []
            for point in response.points:
                payload = point.payload or {}
                results.append(KnowledgeSearchResult(
                    document_id=str(payload.get("document_id", "")),
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    title=str(payload.get("title", "")),
                    source=str(payload.get("source", "")),
                    content=str(payload.get("text", "")),
                    score=float(point.score),
                    metadata={
                        "knowledge_base_id": payload.get("knowledge_base_id"),
                        "dataset_id": payload.get("dataset_id"),
                        "section": payload.get("section"),
                        "chunk_index": payload.get("chunk_index"),
                        "language": payload.get("language"),
                    },
                ))
            return results
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.warning("Qdrant search failed (%s)", type(exc).__name__)
            raise VectorStoreError("Qdrant knowledge search failed.") from exc

    def delete_document(self, document_id: str) -> None:
        try:
            if not self._client.collection_exists(self.collection_name):
                return
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id),
                    )]),
                ),
                wait=True,
            )
        except Exception as exc:
            logger.warning("Qdrant document deletion failed (%s)", type(exc).__name__)
            raise VectorStoreError("Qdrant failed to delete the knowledge document.") from exc

    def close(self) -> None:
        """Release the local storage lock so the same path can be reopened."""
        try:
            self._client.close()
        except Exception as exc:
            logger.warning("Qdrant client close failed (%s)", type(exc).__name__)
