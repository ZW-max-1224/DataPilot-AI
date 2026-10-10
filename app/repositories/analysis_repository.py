from __future__ import annotations

from sqlalchemy import select

from app.database.core import Database
from app.database.models import AnalysisRun


class AnalysisRepository:
    """Store bounded Agent run metadata; never stores prompts or completions."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def create(
        self,
        *,
        dataset_id: str,
        question: str,
        mode: str,
        status: str,
        iterations: int,
        tools_used: list[str],
        duration_ms: int,
    ) -> AnalysisRun:
        if not question or len(question) > 4_000:
            raise ValueError("Analysis question must contain between 1 and 4000 characters.")
        if mode not in {"agent", "manual"} or status not in {"success", "failed"}:
            raise ValueError("Analysis run mode or status is invalid.")
        if not 0 <= iterations <= 8 or not 0 <= duration_ms <= 2_147_483_647:
            raise ValueError("Analysis run metrics exceed supported limits.")
        if len(tools_used) > 20 or any(not isinstance(name, str) or not name or len(name) > 80 for name in tools_used):
            raise ValueError("Analysis tool list exceeds supported limits.")
        with self._database.session() as session:
            record = AnalysisRun(
                dataset_id=dataset_id,
                question=question,
                mode=mode,
                status=status,
                iterations=iterations,
                tools_used=list(tools_used),
                duration_ms=duration_ms,
            )
            session.add(record)
            session.flush()
            return record

    def list_for_dataset(self, dataset_id: str) -> list[dict]:
        with self._database.session() as session:
            records = session.scalars(
                select(AnalysisRun)
                .where(AnalysisRun.dataset_id == dataset_id)
                .order_by(AnalysisRun.created_at.desc(), AnalysisRun.id.desc())
            ).all()
            return [
                {"id": row.id, "question": row.question, "mode": row.mode, "status": row.status,
                 "iterations": row.iterations, "tools_used": row.tools_used, "duration_ms": row.duration_ms}
                for row in records
            ]
