import json

import pandas as pd
import pytest
from pydantic import SecretStr
from sqlalchemy import func, select
from sqlalchemy.exc import DataError, IntegrityError, InterfaceError, OperationalError, StatementError

from app.agent import AnalystAgent, AgentRequest
from app.config.settings import DatabaseSettings
from app.database.core import Database, DatabaseUnavailableError
from app.database.models import AnalysisRun, Dataset, DatasetColumn, KnowledgeDocumentRecord
from app.models.dataset_context import DatasetContext
from app.models.dataset_profile import ColumnProfile, DatasetProfile, DuplicateSummary, MissingSummary
from app.models.llm_models import LLMResponse
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.knowledge_repository import KnowledgeRepository
from app.knowledge.loader import load_knowledge_document
from app.services.dataset_service import load_dataset
from app.services.persistence_service import PersistenceService, classify_persistence_failure
from app.services.profiler import profile_dataframe
from app.tools.analysis_tools import find_top_n


def build_context(frame: pd.DataFrame, dataset_id: str = "dataset-test") -> DatasetContext:
    filename = "orders.csv"
    return DatasetContext(dataset_id, filename, frame, profile_dataframe(frame, filename))


def test_database_settings_are_optional_and_secret_safe():
    disabled = DatabaseSettings.from_env({})
    assert not disabled.database_configured
    configured = DatabaseSettings.from_env({
        "MYSQL_HOST": "db.internal", "MYSQL_PORT": "3307", "MYSQL_DATABASE": "datapilot",
        "MYSQL_USER": "analyst", "MYSQL_PASSWORD": "mysql-test-secret",
    })
    assert configured.database_configured and configured.port == 3307
    assert "mysql-test-secret" not in repr(configured)


def test_mysql_url_encodes_credentials_without_exposing_them():
    settings = DatabaseSettings(
        host="localhost", database="datapilot", username="analyst", password=SecretStr("p@ss:/word")
    )
    database = Database(settings)
    try:
        assert database.engine.url.drivername == "mysql+pymysql"
        assert database.engine.url.password == "p@ss:/word"
        assert "p@ss:/word" not in str(database.engine.url)
    finally:
        database.dispose()


def test_sqlite_memory_schema_is_idempotent_and_dataset_upsert():
    database = Database(url="sqlite:///:memory:")
    try:
        database.check_connection()
        database.initialize()
        database.initialize()
        repository = DatasetRepository(database)
        first_frame = pd.DataFrame({"category": ["A", "B"], "amount": [10, 20]})
        first_context = build_context(first_frame)
        repository.save_context(first_context, "upload")

        updated_frame = pd.DataFrame({"category": ["A", "B", "C"], "amount": [10, 20, 30]})
        repository.save_context(build_context(updated_frame), "demo")
        fetched = repository.get_by_dataset_id("dataset-test")
        assert fetched["source"] == "demo"
        assert fetched["row_count"] == 3
        assert {item["name"] for item in fetched["columns"]} == {"category", "amount"}
        with database.session() as session:
            assert session.scalar(select(func.count()).select_from(Dataset)) == 1
            assert session.scalar(select(func.count()).select_from(DatasetColumn)) == 2
    finally:
        database.dispose()


def test_csv_profile_context_repository_and_analysis_run_integration():
    csv_bytes = b"product,sales\nAlpha,10\nBeta,30\n"
    frame = load_dataset("sample.csv", csv_bytes)
    profile = profile_dataframe(frame, "sample.csv")
    context = DatasetContext("csv-digest", "sample.csv", frame, profile)
    database = Database(url="sqlite:///:memory:")
    persistence = PersistenceService(database=database)
    try:
        assert persistence.status == "connected"
        assert persistence.persist_dataset(context, "upload")
        assert find_top_n(context.dataframe, "sales", n=1).rows == [{"product": "Beta", "sales": 30}]

        class MockLLM:
            def __init__(self):
                self.calls = 0

            def generate(self, _request):
                self.calls += 1
                if self.calls == 1:
                    content = json.dumps({"action": "tool_call", "tool_name": "find_top_n",
                                          "tool_input": {"column": "sales", "n": 1}})
                else:
                    content = json.dumps({"action": "final", "final_answer": "Beta sales are highest at 30."})
                return LLMResponse(content=content, model="mock", mode="llm")

        result = AnalystAgent(MockLLM()).run(AgentRequest(user_query="Which product sold most?", dataset_context=context))
        assert result.success and result.tools_used == ["find_top_n"]
        assert persistence.record_analysis_run(
            dataset_id=context.dataset_id, question="Which product sold most?", mode="agent",
            status="success", iterations=len(result.steps), tools_used=result.tools_used, duration_ms=12,
        )
        runs = AnalysisRepository(database).list_for_dataset(context.dataset_id)
        assert len(runs) == 1
        assert runs[0]["question"] == "Which product sold most?"
        assert runs[0]["tools_used"] == ["find_top_n"]
        assert runs[0]["status"] == "success"
        assert DatasetRepository(database).get_by_dataset_id(context.dataset_id)["row_count"] == 2
    finally:
        database.dispose()


def test_persistence_disabled_does_not_affect_pandas():
    service = PersistenceService(DatabaseSettings.from_env({}))
    frame = pd.DataFrame({"name": ["A", "B"], "score": [3, 7]})
    assert service.status == "disabled"
    assert not service.persist_dataset(build_context(frame), "upload")
    assert find_top_n(frame, "score", n=1).rows == [{"name": "B", "score": 7}]
    service.close()


def test_database_failure_is_nonfatal_and_does_not_log_credentials(caplog, monkeypatch):
    database = Database(url="sqlite:///:memory:")
    monkeypatch.setattr(database, "check_connection", lambda: (_ for _ in ()).throw(
        RuntimeError("mysql password should not be logged")
    ))
    service = PersistenceService(database=database)
    frame = pd.DataFrame({"group": ["A", "B"], "value": [1, 2]})
    assert service.status == "unavailable"
    assert not service.persist_dataset(build_context(frame), "upload")
    assert "mysql password should not be logged" not in caplog.text
    assert "RuntimeError" in caplog.text
    assert find_top_n(frame, "value", n=1).rows[0]["value"] == 2
    service.close()


def test_repository_validates_bounds_before_writing():
    database = Database(url="sqlite:///:memory:")
    database.initialize()
    try:
        repository = AnalysisRepository(database)
        with pytest.raises(ValueError, match="4000"):
            repository.create(dataset_id="x", question="q" * 4_001, mode="agent", status="success",
                              iterations=1, tools_used=[], duration_ms=1)
        with pytest.raises(ValueError, match="mode or status"):
            repository.create(dataset_id="x", question="safe", mode="sql", status="success",
                              iterations=1, tools_used=[], duration_ms=1)
        with database.session() as session:
            assert session.scalar(select(func.count()).select_from(AnalysisRun)) == 0
    finally:
        database.dispose()


def test_knowledge_repository_persists_metadata_without_document_content():
    database = Database(url="sqlite:///:memory:")
    database.initialize()
    repository = KnowledgeRepository(database)
    document = load_knowledge_document(
        "metrics.md", b"# Private body\n\nRevenue is net of discounts.",
        document_id="doc-metrics", knowledge_base_id="kb-default", dataset_id="dataset-a",
    )
    try:
        repository.save_document(document.model_copy(update={"status": "indexing"}))
        indexed = document.model_copy(update={
            "status": "indexed", "chunk_count": 2, "embedding_provider": "openai_compatible",
            "embedding_model": "bge-m3", "embedding_dimensions": 1024,
        })
        repository.save_document(indexed)
        fetched = repository.get_document("doc-metrics")
        assert fetched["status"] == "indexed"
        assert fetched["chunk_count"] == 2
        assert fetched["embedding_provider"] == "openai_compatible"
        assert fetched["embedding_model"] == "bge-m3"
        assert "content" not in fetched
        assert repository.list_documents("kb-default", "dataset-a")[0]["document_id"] == "doc-metrics"
        with database.session() as session:
            assert session.scalar(select(func.count()).select_from(KnowledgeDocumentRecord)) == 1
            assert "content" not in KnowledgeDocumentRecord.__table__.columns
    finally:
        database.dispose()


def test_record_level_failure_keeps_persistence_connected():
    database = Database(url="sqlite:///:memory:")
    persistence = PersistenceService(database=database)
    frame = pd.DataFrame({"name": ["A", "B"], "score": [3, 7]})
    try:
        assert persistence.status == "connected"
        # analysis_runs.dataset_id is a foreign key: an unknown dataset_id is a
        # record-level IntegrityError, not a connection failure.
        assert not persistence.record_analysis_run(
            dataset_id="never-persisted", question="Which product sold most?", mode="agent",
            status="success", iterations=1, tools_used=[], duration_ms=1,
        )
        assert persistence.status == "connected"

        # A later valid write still works, and the failed record left no row.
        assert persistence.persist_dataset(build_context(frame), "upload")
        assert persistence.record_analysis_run(
            dataset_id="dataset-test", question="Which product sold most?", mode="agent",
            status="success", iterations=1, tools_used=[], duration_ms=1,
        )
        assert persistence.status == "connected"
        assert len(AnalysisRepository(database).list_for_dataset("dataset-test")) == 1
    finally:
        database.dispose()


def test_connection_level_failure_marks_persistence_unavailable(monkeypatch):
    database = Database(url="sqlite:///:memory:")
    persistence = PersistenceService(database=database)
    frame = pd.DataFrame({"name": ["A", "B"], "score": [3, 7]})
    attempts = []

    def unreachable(*args, **kwargs):
        attempts.append(1)
        raise OperationalError("INSERT INTO datasets", {}, Exception("MySQL server has gone away"))

    monkeypatch.setattr(persistence.dataset_repository, "save_context", unreachable)
    try:
        assert persistence.status == "connected"
        assert not persistence.persist_dataset(build_context(frame), "upload")
        assert persistence.status == "unavailable"
        # Once unavailable, persistence stops touching the database at all.
        assert not persistence.persist_dataset(build_context(frame), "upload")
        assert len(attempts) == 1
    finally:
        database.dispose()


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (IntegrityError("INSERT INTO datasets", {}, Exception("duplicate")), "record"),
        (DataError("INSERT INTO datasets", {}, Exception("out of range")), "record"),
        (StatementError("bad statement", "INSERT INTO datasets", {}, Exception("bad")), "record"),
        (OperationalError("SELECT 1", {}, Exception("gone away")), "connection"),
        (InterfaceError("SELECT 1", {}, Exception("interface error")), "connection"),
        (DatabaseUnavailableError("MySQL persistence is not configured."), "connection"),
        (ValueError("unexpected payload"), "record"),
    ],
)
def test_persistence_failure_classification(failure, expected):
    assert classify_persistence_failure(failure) == expected


def test_dataset_column_missing_rate_is_stored_as_a_fraction():
    database = Database(url="sqlite:///:memory:")
    try:
        database.initialize()
        repository = DatasetRepository(database)
        frame = pd.DataFrame({"with_gaps": [1.0, None, None, None], "complete": [1, 2, 3, 4]})
        repository.save_context(build_context(frame), "upload")
        fetched = repository.get_by_dataset_id("dataset-test")
        rates = {item["name"]: item["missing_rate"] for item in fetched["columns"]}
        assert rates["with_gaps"] == pytest.approx(0.75)
        assert rates["complete"] == 0.0
        assert all(0.0 <= rate <= 1.0 for rate in rates.values())
        counts = {item["name"]: item["missing_count"] for item in fetched["columns"]}
        assert counts["with_gaps"] == 3
    finally:
        database.dispose()


def test_dataset_column_missing_rate_handles_a_zero_row_profile():
    database = Database(url="sqlite:///:memory:")
    try:
        database.initialize()
        repository = DatasetRepository(database)
        profile = DatasetProfile(
            dataset_name="empty.csv", row_count=0, column_count=1, memory_usage_bytes=0,
            columns=[ColumnProfile(
                column_name="empty", pandas_dtype="float64", semantic_type="numeric",
                non_null_count=0, missing_count=0, missing_percentage=0.0, unique_count=0,
            )],
            missing=MissingSummary(total_missing_values=0, missing_rate=0.0),
            duplicates=DuplicateSummary(duplicate_rows=0, duplicate_rate=0.0),
            numeric_summary=[], categorical_summary=[], datetime_columns=[],
            quality_score=100, quality_issues=[], quality_score_note="No data rows.",
        )
        context = DatasetContext("empty-dataset", "empty.csv", pd.DataFrame({"empty": []}), profile)
        repository.save_context(context, "upload")
        fetched = repository.get_by_dataset_id("empty-dataset")
        assert fetched["row_count"] == 0
        assert {item["name"]: item["missing_rate"] for item in fetched["columns"]} == {"empty": 0.0}
    finally:
        database.dispose()
