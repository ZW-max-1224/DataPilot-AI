from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.config.settings import EmbeddingSettings
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError


class FakeClient:
    def __init__(self, vectors=None, error=None):
        self.vectors = vectors or [[0.1, 0.2], [0.3, 0.4]]
        self.error = error
        self.calls = []

    def create(self, *, model, input):
        self.calls.append((model, input))
        if self.error:
            raise self.error
        return SimpleNamespace(data=[
            SimpleNamespace(index=index, embedding=vector) for index, vector in enumerate(self.vectors[:len(input)])
        ])


def _settings(**overrides):
    values = dict(provider="openai_compatible", model="test-embedding", api_key=SecretStr("secret"),
                  base_url="https://embedding.example/v1", dimensions=2)
    values.update(overrides)
    return EmbeddingSettings(**values)


def test_embedding_settings_default_to_disabled_and_hide_key():
    settings = EmbeddingSettings.from_env(environ={})
    assert settings.provider == "disabled"
    assert not settings.is_configured
    assert "secret" not in repr(_settings())


def test_embedding_settings_reject_incomplete_or_unsafe_url():
    with pytest.raises(EmbeddingServiceError) as exc:
        EmbeddingService(_settings(base_url=None), client=FakeClient())
    assert exc.value.code == "embedding_unavailable"
    with pytest.raises(EmbeddingServiceError):
        EmbeddingService(_settings(base_url="https://user:pass@example/v1"), client=FakeClient())


def test_embedding_documents_and_query_return_vectors():
    client = FakeClient()
    service = EmbeddingService(_settings(), client=client)
    assert service.embed_documents(["first", "second"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert service.embed_query("question") == [0.1, 0.2]
    assert client.calls[0] == ("test-embedding", ["first", "second"])


def test_embedding_provider_failure_is_safe_and_structured():
    service = EmbeddingService(_settings(), client=FakeClient(error=RuntimeError("secret response")))
    with pytest.raises(EmbeddingServiceError) as exc:
        service.embed_query("query")
    assert exc.value.code == "embedding_failed"
    assert "secret response" not in str(exc.value)


def test_embedding_rejects_dimension_mismatch_and_empty_text():
    service = EmbeddingService(_settings(), client=FakeClient(vectors=[[0.1]]))
    with pytest.raises(EmbeddingServiceError):
        service.embed_query("query")
    with pytest.raises(ValueError):
        service.embed_documents([" "])
