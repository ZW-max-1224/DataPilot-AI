from pathlib import Path
from contextlib import contextmanager

from pydantic import SecretStr

from app.config.settings import EmbeddingSettings
from app.knowledge.loader import load_knowledge_document
from app.models.knowledge import KnowledgeSearchResult
from app.services.embedding_service import EmbeddingService
from app.services.knowledge_service import KnowledgeService
from app.vectorstore.qdrant_store import QdrantVectorStore, VectorStoreError


class FakeEmbeddingsClient:
    def __init__(self, vector=(1.0, 0.0), error=None):
        self.vector = list(vector)
        self.error = error

    def create(self, *, model, input):
        if self.error:
            raise self.error
        from types import SimpleNamespace
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=self.vector) for i, _ in enumerate(input)])


class FakeStore:
    def __init__(self, results=None, upsert_error=None, search_error=None):
        self.results = results or []
        self.upsert_error = upsert_error
        self.search_error = search_error
        self.upserts = []
        self.deleted = []

    def upsert_chunks(self, chunks, vectors):
        if self.upsert_error:
            raise self.upsert_error
        self.upserts.append((chunks, vectors))

    def search(self, vector, *, knowledge_base_id, dataset_id=None, limit=5):
        if self.search_error:
            raise self.search_error
        return self.results[:limit]

    def delete_document(self, document_id):
        self.deleted.append(document_id)


class FakeRepository:
    def __init__(self):
        self.documents = []

    def save_document(self, document):
        self.documents.append(document)


class TraceRecorder:
    def __init__(self):
        self.spans = []

    @contextmanager
    def span(self, name, **kwargs):
        record = {"name": name, **kwargs, "outputs": {}}
        self.spans.append(record)
        class Handle:
            def end(self, outputs=None):
                record["outputs"] = outputs or {}
        yield Handle()


def _embedding(client=None, enabled=True):
    settings = EmbeddingSettings(
        provider="openai_compatible" if enabled else "disabled",
        model="test", api_key=SecretStr("fake"), base_url="https://example.test/v1", dimensions=2,
    )
    return EmbeddingService(settings, client=client or FakeEmbeddingsClient())


def _doc():
    return load_knowledge_document("metric.md", "# 指标\n\n利润率定义为利润除以收入。".encode(),
                                   document_id="doc-a", knowledge_base_id="kb-a")


def test_indexing_runs_end_to_end_against_local_qdrant(tmp_path: Path):
    store = QdrantVectorStore(path=tmp_path / "qdrant", dimensions=2)
    repository = FakeRepository()
    service = KnowledgeService(_embedding(), store, repository=repository)
    result = service.index_document(_doc())
    assert result.status == "indexed"
    assert [document.status for document in repository.documents] == ["indexing", "indexed"]
    assert repository.documents[-1].embedding_provider == "openai_compatible"
    response = service.search("利润率定义", knowledge_base_id="kb-a")
    assert response.status == "ok"
    assert response.results[0].document_id == "doc-a"
    store.close()


def test_embedding_disabled_and_failure_do_not_index():
    store = FakeStore()
    disabled = KnowledgeService(_embedding(enabled=False), store).index_document(_doc())
    assert disabled.status == "failed"
    assert disabled.error_code == "embedding_unavailable"
    assert store.upserts == []

    failed = KnowledgeService(
        _embedding(FakeEmbeddingsClient(error=RuntimeError("network error"))), FakeStore()
    ).index_document(_doc())
    assert failed.status == "failed"
    assert failed.error_code == "embedding_failed"


def test_embedding_failure_persists_failed_status():
    repository = FakeRepository()
    result = KnowledgeService(
        _embedding(FakeEmbeddingsClient(error=RuntimeError("network error"))),
        FakeStore(), repository=repository,
    ).index_document(_doc())
    assert result.status == "failed"
    assert [document.status for document in repository.documents] == ["indexing", "failed"]
    assert repository.documents[-1].indexing_error == "embedding_failed"


def test_vector_store_failure_marks_indexing_failed_and_cleans_partial():
    store = FakeStore(upsert_error=VectorStoreError("unavailable"))
    result = KnowledgeService(_embedding(), store).index_document(_doc())
    assert result.status == "failed"
    assert result.error_code == "knowledge_base_unavailable"
    # The previous index is purged before writing, and the partial write is
    # purged again on failure, so no stale point can survive.
    assert store.deleted == ["doc-a", "doc-a"]


def test_dimension_mismatch_error_code_is_not_flattened():
    from app.vectorstore.qdrant_store import EmbeddingDimensionMismatchError

    store = FakeStore(upsert_error=EmbeddingDimensionMismatchError("dims"))
    result = KnowledgeService(_embedding(), store).index_document(_doc())
    assert result.status == "failed"
    assert result.error_code == "embedding_dimension_mismatch"

    search_store = FakeStore(search_error=EmbeddingDimensionMismatchError("dims"))
    response = KnowledgeService(_embedding(), search_store).search("query", knowledge_base_id="kb")
    assert response.status == "knowledge_base_unavailable"
    assert response.error_code == "embedding_dimension_mismatch"


def _paragraph_document(document_id: str = "doc-b", knowledge_base_id: str = "kb-a", *, paragraphs: int,
                       text: str | None = None):
    body = text if text is not None else "\n\n".join(
        f"Paragraph {index} " + " ".join(["knowledge"] * 30) + "." for index in range(paragraphs)
    )
    return load_knowledge_document(
        f"{document_id}.txt", body.encode(), document_id=document_id, knowledge_base_id=knowledge_base_id,
    )


def test_reindexing_replaces_the_previous_chunk_set(tmp_path: Path):
    store = QdrantVectorStore(path=tmp_path / "qdrant", dimensions=2)
    service = KnowledgeService(_embedding(), store)
    try:
        first = service.index_document(_paragraph_document(paragraphs=12))
        assert first.status == "indexed"
        assert first.chunk_count == 12
        before = store.search([1.0, 0.0], knowledge_base_id="kb-a")
        assert {item.document_id for item in before} == {"doc-b"}
        assert len(before) == 5  # the default retrieval limit, i.e. the index is populated

        second = service.index_document(
            _paragraph_document(paragraphs=0, text="Single replacement paragraph about revenue.")
        )
        assert second.status == "indexed"
        assert second.chunk_count == 1

        response = service.search("revenue", knowledge_base_id="kb-a")
        assert response.status == "ok"
        # Exactly the one new chunk: every stale point of the 12-chunk version is gone.
        assert response.result_count == 1
        assert {item.chunk_id for item in response.results} == {"doc-b:0"}
        assert response.results[0].content == "Single replacement paragraph about revenue."
        assert "Paragraph 11" not in response.results[0].content
    finally:
        store.close()


def test_reindexing_that_fails_to_embed_leaves_no_stale_points(tmp_path: Path):
    store = QdrantVectorStore(path=tmp_path / "qdrant", dimensions=2)
    try:
        healthy = KnowledgeService(_embedding(), store)
        assert healthy.index_document(_paragraph_document(paragraphs=12)).chunk_count == 12

        failing = KnowledgeService(
            _embedding(FakeEmbeddingsClient(error=RuntimeError("network error"))), store
        )
        result = failing.index_document(_paragraph_document(paragraphs=0, text="Replacement text."))
        assert result.status == "failed"
        assert result.error_code == "embedding_failed"
        # The replaced document must not stay searchable with its deleted text.
        assert store.search([1.0, 0.0], knowledge_base_id="kb-a") == []
    finally:
        store.close()


def test_search_status_and_context_bounds():
    disabled = KnowledgeService(_embedding(enabled=False), FakeStore()).search("query", knowledge_base_id="kb")
    assert disabled.status == "embedding_unavailable"

    store = FakeStore(results=[KnowledgeSearchResult(
        document_id=f"doc-{index}", chunk_id=f"chunk-{index}", title="Title", source="x.md",
        content="x" * 1_200, score=0.8,
    ) for index in range(5)])
    result = KnowledgeService(_embedding(), store).search("query", knowledge_base_id="kb")
    assert result.status == "ok"
    assert result.truncated
    assert result.result_count == 4
    assert all(len(item.content) <= 1_000 for item in result.results)
    assert sum(len(item.content) for item in result.results) <= 4_000


def test_search_maps_vector_failure_and_no_results():
    unavailable = KnowledgeService(_embedding(), FakeStore(search_error=VectorStoreError("down")))
    assert unavailable.search("query", knowledge_base_id="kb").status == "knowledge_base_unavailable"
    empty = KnowledgeService(_embedding(), FakeStore()).search("query", knowledge_base_id="kb")
    assert empty.status == "no_results"


def test_retrieval_trace_records_metadata_but_not_query_or_chunk_text():
    result = KnowledgeSearchResult(
        document_id="doc-1", chunk_id="doc-1:0", title="Metrics", source="metrics.md",
        content="Sensitive business details", score=0.87654321,
    )
    tracer = TraceRecorder()
    service = KnowledgeService(_embedding(), FakeStore(results=[result]), tracing=tracer)
    response = service.search("private customer revenue?", knowledge_base_id="kb-1", dataset_id="ds-1")
    assert response.status == "ok"
    span = tracer.spans[0]
    assert span["name"] == "Knowledge Retrieval"
    assert "query" not in span["inputs"]
    assert span["inputs"]["query_sha256"]
    assert span["outputs"]["results"] == [
        {"document_id": "doc-1", "chunk_id": "doc-1:0", "score": 0.876543}
    ]
    assert "Sensitive business details" not in repr(span)
    assert "private customer revenue" not in repr(span)
