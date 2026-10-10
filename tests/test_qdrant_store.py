from pathlib import Path

import pytest

from app.knowledge.loader import load_knowledge_document
from app.knowledge.chunking import chunk_document
from app.vectorstore.qdrant_store import (
    ERROR_DIMENSION_MISMATCH,
    EmbeddingDimensionMismatchError,
    QdrantVectorStore,
    VectorStoreError,
)


def _chunks(document_id: str, kb: str, dataset: str | None, text: str):
    document = load_knowledge_document(
        f"{document_id}.txt", text.encode(), document_id=document_id,
        knowledge_base_id=kb, dataset_id=dataset,
    )
    return chunk_document(document, target_tokens=100, overlap_tokens=0, max_tokens=120)


def test_local_qdrant_collection_upsert_search_filter_and_delete(tmp_path: Path):
    store = QdrantVectorStore(path=tmp_path / "qdrant", dimensions=2)
    try:
        assert store.is_available
        store.create_collection()
        target = _chunks("target", "kb-a", "ds-a", "Revenue is net sales after discounts.")[0]
        general = _chunks("general", "kb-a", None, "Margin means profit divided by revenue.")[0]
        other_dataset = _chunks("other-dataset", "kb-a", "ds-b", "Revenue includes taxes.")[0]
        other_kb = _chunks("other-kb", "kb-b", "ds-a", "Revenue is gross sales.")[0]
        store.upsert_chunks([target, general, other_dataset, other_kb],
                            [[1, 0], [0.95, 0.05], [0.9, 0.1], [0.8, 0.2]])

        scoped = store.search([1, 0], knowledge_base_id="kb-a", dataset_id="ds-a", limit=5)
        assert {item.document_id for item in scoped} == {"target", "general"}
        general_only = store.search([1, 0], knowledge_base_id="kb-a", dataset_id=None, limit=5)
        assert [item.document_id for item in general_only] == ["general"]
        missing_kb = store.search([1, 0], knowledge_base_id="kb-empty", limit=5)
        assert missing_kb == []
        store.delete_document("target")
        remaining = store.search([1, 0], knowledge_base_id="kb-a", dataset_id="ds-a")
        assert {item.document_id for item in remaining} == {"general"}
    finally:
        store.close()


def test_qdrant_rejects_wrong_vector_dimensions(tmp_path: Path):
    store = QdrantVectorStore(path=tmp_path / "qdrant", dimensions=2)
    with pytest.raises(ValueError):
        store.upsert_chunks(_chunks("target", "kb-a", None, "Some knowledge.") , [[1]])
    store.close()


def test_qdrant_detects_existing_collection_dimension_mismatch(tmp_path: Path):
    first = QdrantVectorStore(path=tmp_path / "qdrant", dimensions=2)
    first.create_collection()
    first.close()
    second = QdrantVectorStore(path=tmp_path / "qdrant", dimensions=3)
    with pytest.raises(VectorStoreError):
        second.create_collection()
    second.close()


def test_qdrant_availability_surfaces_dimension_mismatch_as_distinct_code(tmp_path: Path):
    first = QdrantVectorStore(path=tmp_path / "qdrant", dimensions=2)
    first.create_collection()
    assert first.availability() == (True, None)
    assert first.availability_error_code is None
    first.close()

    # A fresh store (e.g. a different embedding model) reusing the same path.
    second = QdrantVectorStore(path=tmp_path / "qdrant", dimensions=3)
    try:
        assert not second.is_available
        assert second.availability() == (False, ERROR_DIMENSION_MISMATCH)
        assert second.availability_error_code == ERROR_DIMENSION_MISMATCH
        with pytest.raises(EmbeddingDimensionMismatchError) as create_error:
            second.create_collection()
        assert create_error.value.code == ERROR_DIMENSION_MISMATCH
        with pytest.raises(EmbeddingDimensionMismatchError):
            second.search([1.0, 0.0, 0.0], knowledge_base_id="kb-a")
    finally:
        second.close()


def test_qdrant_rebuild_collection_recovers_from_dimension_mismatch(tmp_path: Path):
    path = tmp_path / "qdrant"
    first = QdrantVectorStore(path=path, dimensions=2)
    first.create_collection()
    first.upsert_chunks(_chunks("doc-1", "kb-a", None, "Revenue is net sales."), [[1.0, 0.0]])
    first.close()

    second = QdrantVectorStore(path=path, dimensions=3)
    try:
        assert second.availability() == (False, ERROR_DIMENSION_MISMATCH)
        second.rebuild_collection()
        assert second.availability() == (True, None)
        assert second.is_available
        chunk = _chunks("doc-2", "kb-a", None, "Margin is profit over revenue.")[0]
        second.upsert_chunks([chunk], [[1.0, 0.0, 0.0]])
        assert [item.document_id for item in second.search([1.0, 0.0, 0.0], knowledge_base_id="kb-a")] == ["doc-2"]
    finally:
        second.close()


def test_qdrant_close_releases_the_local_path_for_reopen(tmp_path: Path):
    path = tmp_path / "qdrant"
    store = QdrantVectorStore(path=path, dimensions=2)
    store.create_collection()
    chunk = _chunks("doc-1", "kb-a", None, "Revenue is net sales after discounts.")[0]
    store.upsert_chunks([chunk], [[1.0, 0.0]])
    store.close()

    reopened = QdrantVectorStore(path=path, dimensions=2)
    try:
        assert reopened.is_available
        assert reopened.availability() == (True, None)
        results = reopened.search([1.0, 0.0], knowledge_base_id="kb-a")
        assert [item.chunk_id for item in results] == [chunk.chunk_id]
        assert reopened.availability_error_code is None
    finally:
        reopened.close()
