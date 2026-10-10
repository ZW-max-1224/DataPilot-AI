from __future__ import annotations

from sqlalchemy import delete, select

from app.database.core import Database
from app.database.models import Dataset, DatasetColumn
from app.models.dataset_context import DatasetContext


class DatasetRepository:
    """Persist dataset identity and non-row profile metadata only."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def save_context(self, context: DatasetContext, source: str) -> Dataset:
        if len(context.dataset_id) > 64 or len(context.filename) > 255 or len(source) > 32:
            raise ValueError("Dataset metadata exceeds the supported schema limits.")
        profile = context.profile
        with self._database.session() as session:
            record = session.scalar(select(Dataset).where(Dataset.dataset_id == context.dataset_id))
            if record is None:
                record = Dataset(dataset_id=context.dataset_id, filename=context.filename, source=source,
                                 row_count=profile.row_count, column_count=profile.column_count,
                                 missing_count=profile.missing.total_missing_values,
                                 duplicate_count=profile.duplicates.duplicate_rows,
                                 quality_score=profile.quality_score)
                session.add(record)
            else:
                record.filename = context.filename
                record.source = source
                record.row_count = profile.row_count
                record.column_count = profile.column_count
                record.missing_count = profile.missing.total_missing_values
                record.duplicate_count = profile.duplicates.duplicate_rows
                record.quality_score = profile.quality_score
                session.execute(delete(DatasetColumn).where(DatasetColumn.dataset_id == context.dataset_id))

            for column in profile.columns:
                if len(column.column_name) > 255 or len(column.pandas_dtype) > 64:
                    raise ValueError("Dataset profile contains a field that exceeds the supported schema limits.")
                # dataset_columns.missing_rate is a 0-1 rate (the profile stores a
                # 0-100 percentage); guard the empty-dataset zero row count.
                missing_rate = min(1.0, column.missing_count / profile.row_count) if profile.row_count else 0.0
                session.add(DatasetColumn(
                    dataset_id=context.dataset_id,
                    column_name=column.column_name,
                    semantic_type=column.semantic_type,
                    pandas_dtype=column.pandas_dtype,
                    missing_count=column.missing_count,
                    missing_rate=missing_rate,
                    unique_count=column.unique_count,
                ))
            session.flush()
            return record

    def get_by_dataset_id(self, dataset_id: str) -> dict | None:
        with self._database.session() as session:
            record = session.scalar(select(Dataset).where(Dataset.dataset_id == dataset_id))
            if record is None:
                return None
            columns = session.scalars(
                select(DatasetColumn).where(DatasetColumn.dataset_id == dataset_id).order_by(DatasetColumn.id)
            ).all()
            return {
                "dataset_id": record.dataset_id,
                "filename": record.filename,
                "source": record.source,
                "row_count": record.row_count,
                "column_count": record.column_count,
                "missing_count": record.missing_count,
                "duplicate_count": record.duplicate_count,
                "quality_score": record.quality_score,
                "columns": [
                    {"name": column.column_name, "semantic_type": column.semantic_type,
                     "pandas_dtype": column.pandas_dtype, "missing_count": column.missing_count,
                     "missing_rate": column.missing_rate, "unique_count": column.unique_count}
                    for column in columns
                ],
            }
