from __future__ import annotations

from sqlalchemy import select

from app.database.core import Database
from app.database.models import KnowledgeDocumentRecord
from app.models.knowledge import KnowledgeDocument


class KnowledgeRepository:
    """Persist knowledge document metadata only; document and chunk text stay out of MySQL."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def save_document(self, document: KnowledgeDocument) -> dict:
        with self._database.session() as session:
            record = session.get(KnowledgeDocumentRecord, document.document_id)
            if record is None:
                record = KnowledgeDocumentRecord(document_id=document.document_id)
                session.add(record)
            record.title = document.title
            record.source = document.source
            record.source_type = document.source_type
            record.knowledge_base_id = document.knowledge_base_id
            record.dataset_id = document.dataset_id
            record.content_hash = document.content_hash
            record.status = document.status
            record.chunk_count = document.chunk_count
            record.embedding_provider = document.embedding_provider
            record.embedding_model = document.embedding_model
            record.embedding_dimensions = document.embedding_dimensions
            record.indexing_error = document.indexing_error
            session.flush()
            return self._to_dict(record)

    def get_document(self, document_id: str) -> dict | None:
        with self._database.session() as session:
            record = session.get(KnowledgeDocumentRecord, document_id)
            return self._to_dict(record) if record else None

    def list_documents(self, knowledge_base_id: str, dataset_id: str | None = None) -> list[dict]:
        statement = select(KnowledgeDocumentRecord).where(
            KnowledgeDocumentRecord.knowledge_base_id == knowledge_base_id,
        )
        if dataset_id is not None:
            statement = statement.where(KnowledgeDocumentRecord.dataset_id == dataset_id)
        with self._database.session() as session:
            records = session.scalars(statement.order_by(
                KnowledgeDocumentRecord.created_at.desc(), KnowledgeDocumentRecord.document_id,
            )).all()
            return [self._to_dict(record) for record in records]

    @staticmethod
    def _to_dict(record: KnowledgeDocumentRecord) -> dict:
        return {
            "document_id": record.document_id,
            "title": record.title,
            "source": record.source,
            "source_type": record.source_type,
            "knowledge_base_id": record.knowledge_base_id,
            "dataset_id": record.dataset_id,
            "content_hash": record.content_hash,
            "status": record.status,
            "chunk_count": record.chunk_count,
            "embedding_provider": record.embedding_provider,
            "embedding_model": record.embedding_model,
            "embedding_dimensions": record.embedding_dimensions,
            "indexing_error": record.indexing_error,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }
