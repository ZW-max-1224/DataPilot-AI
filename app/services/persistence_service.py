from __future__ import annotations

import logging

from sqlalchemy.exc import DataError, IntegrityError, InterfaceError, OperationalError, StatementError

from app.config.settings import DatabaseSettings
from app.database.core import Database, DatabaseUnavailableError
from app.models.dataset_context import DatasetContext
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)

# Connection-level failures mean the database itself cannot be reached, so
# persistence is disabled for the rest of the process.
CONNECTION_FAILURES = (OperationalError, InterfaceError, DatabaseUnavailableError)
# Record-level failures (bad/duplicate/out-of-range data, broken statements) are
# caused by one payload: log it, fail that write, and keep persisting.
RECORD_FAILURES = (IntegrityError, DataError, StatementError)


def classify_persistence_failure(exc: BaseException) -> str:
    """Return ``"connection"`` or ``"record"`` for a persistence exception.

    Connection failures are checked first because SQLAlchemy's
    ``OperationalError``/``InterfaceError`` are also ``StatementError``
    subclasses. Anything unrecognized is treated as record-level so a single
    unexpected payload cannot permanently disable persistence.
    """
    if isinstance(exc, CONNECTION_FAILURES):
        return "connection"
    if isinstance(exc, RECORD_FAILURES):
        return "record"
    # Unknown failures are treated as record-level: one unexpected payload must
    # not permanently disable persistence for the process.
    return "record"


class PersistenceService:
    """Best-effort metadata persistence that never blocks in-memory analysis."""

    def __init__(self, settings: DatabaseSettings | None = None, *, database: Database | None = None) -> None:
        self.settings = settings or DatabaseSettings.from_env()
        self._status = "disabled"
        self.dataset_repository: DatasetRepository | None = None
        self.analysis_repository: AnalysisRepository | None = None
        self.knowledge_repository: KnowledgeRepository | None = None
        try:
            self.database = database or Database(self.settings)
        except Exception as exc:
            self.database = Database(DatabaseSettings.from_env({}))
            self._status = "unavailable"
            logger.warning("Database engine could not be configured (%s)", type(exc).__name__)
            return
        if self.database.configured:
            try:
                self.database.check_connection()
                self.database.initialize()
                self.dataset_repository = DatasetRepository(self.database)
                self.analysis_repository = AnalysisRepository(self.database)
                self.knowledge_repository = KnowledgeRepository(self.database)
                self._status = "connected"
            except Exception as exc:
                self._status = "unavailable"
                logger.warning("Database persistence is unavailable (%s)", type(exc).__name__)

    @property
    def status(self) -> str:
        return self._status

    def persist_dataset(self, context: DatasetContext, source: str) -> bool:
        if self.status != "connected" or self.dataset_repository is None:
            return False
        try:
            self.dataset_repository.save_context(context, source)
            return True
        except Exception as exc:
            return self._write_failure("Dataset metadata persistence failed", exc)

    def _write_failure(self, message: str, exc: BaseException) -> bool:
        """Report one failed write without escaping; returns ``False`` always.

        Record-level failures keep ``status == "connected"`` so later valid
        writes still work. Only connection-level failures mark the service
        unavailable for the rest of the process. No automatic connection
        re-check is attempted: once a driver-level connection error is seen the
        service stops using the database, and a caller that wants to probe again
        can construct a new ``PersistenceService``.
        """
        if classify_persistence_failure(exc) == "connection":
            self._status = "unavailable"
            logger.warning("%s: connection lost (%s)", message, type(exc).__name__)
        else:
            logger.warning("%s (%s)", message, type(exc).__name__)
        return False

    def record_analysis_run(
        self,
        *,
        dataset_id: str,
        question: str,
        mode: str,
        status: str,
        iterations: int,
        tools_used: list[str],
        duration_ms: int,
    ) -> bool:
        if self.status != "connected" or self.analysis_repository is None:
            return False
        try:
            self.analysis_repository.create(
                dataset_id=dataset_id,
                question=question,
                mode=mode,
                status=status,
                iterations=iterations,
                tools_used=tools_used,
                duration_ms=duration_ms,
            )
            return True
        except Exception as exc:
            return self._write_failure("Analysis run persistence failed", exc)

    def close(self) -> None:
        self.database.dispose()
