from __future__ import annotations

import logging
import math
from typing import Any, Protocol, Sequence

from openai import OpenAI

from app.config.settings import EmbeddingSettings

logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Safe provider-independent embedding error without remote response details."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class EmbeddingsClient(Protocol):
    def create(self, *, model: str, input: list[str]) -> Any: ...


class EmbeddingService:
    """Provider-agnostic boundary for configured OpenAI-compatible embeddings."""

    def __init__(self, settings: EmbeddingSettings | None = None, *, client: EmbeddingsClient | None = None) -> None:
        self.settings = settings or EmbeddingSettings.from_env()
        self._client = client
        if self.settings.configuration_problem:
            raise EmbeddingServiceError("embedding_unavailable", self.settings.configuration_problem)
        if self.settings.is_configured and self._client is None:
            self._client = OpenAI(
                api_key=self.settings.api_key.get_secret_value(),  # type: ignore[union-attr]
                base_url=self.settings.base_url,
                timeout=self.settings.timeout,
                max_retries=0,
            ).embeddings

    @property
    def is_configured(self) -> bool:
        return self.settings.is_configured and self._client is not None

    @property
    def model(self) -> str:
        return self.settings.model

    @property
    def dimensions(self) -> int:
        return self.settings.dimensions

    @property
    def provider(self) -> str:
        return self.settings.provider

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.is_configured or self._client is None:
            raise EmbeddingServiceError("embedding_unavailable", "Embedding service is not configured.")
        clean_texts = [text.strip() for text in texts]
        if not clean_texts or any(not text for text in clean_texts):
            raise ValueError("Embedding input must contain non-empty text values.")
        if len(clean_texts) > 128:
            raise ValueError("An embedding request may contain at most 128 texts.")
        try:
            response = self._client.create(model=self.settings.model, input=clean_texts)
            items = sorted(response.data, key=lambda item: item.index)
            vectors = [list(item.embedding) for item in items]
            if len(vectors) != len(clean_texts):
                raise ValueError("Embedding provider returned an unexpected vector count.")
            for vector in vectors:
                if len(vector) != self.settings.dimensions:
                    raise ValueError("Embedding provider returned an unexpected vector dimension.")
                if any(not math.isfinite(value) for value in vector):
                    raise ValueError("Embedding provider returned a non-finite vector value.")
            return vectors
        except EmbeddingServiceError:
            raise
        except Exception as exc:
            logger.warning("Embedding request failed (%s)", type(exc).__name__)
            raise EmbeddingServiceError("embedding_failed", "The embedding provider request failed.") from exc

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
