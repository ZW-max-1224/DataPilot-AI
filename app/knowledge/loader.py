from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.models.knowledge import KnowledgeDocument

MAX_KNOWLEDGE_BYTES = 1 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".md", ".txt"}


class KnowledgeLoadError(ValueError):
    """A knowledge document is unsupported or cannot be decoded safely."""


def load_knowledge_document(
    filename: str,
    content: bytes,
    *,
    document_id: str,
    knowledge_base_id: str,
    dataset_id: str | None = None,
    source_type: str = "upload",
) -> KnowledgeDocument:
    """Decode a bounded Markdown/TXT upload into a metadata model."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise KnowledgeLoadError("Only .md and .txt knowledge documents are supported.")
    if not content:
        raise KnowledgeLoadError("The knowledge document is empty.")
    if len(content) > MAX_KNOWLEDGE_BYTES:
        raise KnowledgeLoadError("The knowledge document exceeds the 1 MiB limit.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise KnowledgeLoadError("Knowledge documents must be UTF-8 encoded.") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not text:
        raise KnowledgeLoadError("The knowledge document has no readable text.")
    return KnowledgeDocument(
        document_id=document_id,
        title=Path(filename).stem[:255] or "Untitled",
        source=Path(filename).name,
        source_type=source_type,  # validated by Pydantic
        knowledge_base_id=knowledge_base_id,
        dataset_id=dataset_id,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        content=text,
    )
