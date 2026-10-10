from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import SecretStr


def _positive_number(value: str | None, default: float) -> float:
    try:
        parsed = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _positive_integer(value: str | None, default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _non_negative_integer(value: str | None, default: int) -> int:
    """Retry counts may legitimately be 0 (retries disabled)."""
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


@dataclass(frozen=True)
class LLMSettings:
    """Immutable provider settings; the secret is redacted from repr output."""

    api_key: SecretStr | None = None
    base_url: str | None = None
    model: str | None = None
    timeout: float = 30.0
    max_tokens: int = 1024
    # Transport-level retries delegated to the OpenAI SDK (it applies exponential
    # backoff for timeouts, connection resets, 429 and 5xx). The SDK default is 2;
    # the project previously hardcoded 0, which turned a transient network blip into
    # a fatal agent failure.
    max_retries: int = 2

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        env_file: Path | None = None,
    ) -> LLMSettings:
        """Load .env then environment overrides, without mutating process globals.

        Supplying ``environ`` makes the source deterministic for tests; an env file
        is only read in that mode when explicitly passed.
        """
        values: dict[str, str | None] = {}
        if env_file is not None:
            values.update(dotenv_values(env_file))
        elif environ is None:
            project_env = Path(__file__).resolve().parents[2] / ".env"
            values.update(dotenv_values(project_env))

        values.update(dict(os.environ if environ is None else environ))
        raw_key = (values.get("LLM_API_KEY") or "").strip()
        raw_base_url = (values.get("LLM_BASE_URL") or "").strip()
        raw_model = (values.get("LLM_MODEL") or "").strip()
        return cls(
            api_key=SecretStr(raw_key) if raw_key else None,
            base_url=raw_base_url or None,
            model=raw_model or None,
            timeout=_positive_number(values.get("LLM_TIMEOUT"), 30.0),
            max_tokens=_positive_integer(values.get("LLM_MAX_TOKENS"), 1024),
            max_retries=_non_negative_integer(values.get("LLM_MAX_RETRIES"), 2),
        )

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.get_secret_value().strip())

    @property
    def is_configured(self) -> bool:
        return self.has_api_key and bool(self.model and self._valid_base_url())

    @property
    def configuration_problem(self) -> str | None:
        if not self.has_api_key:
            return None
        if not self.base_url:
            return "LLM_BASE_URL is required when an API key is configured."
        if not self._valid_base_url():
            return "LLM_BASE_URL must be an absolute HTTP or HTTPS URL without credentials, query, or fragment."
        if not self.model:
            return "LLM_MODEL is required when an API key is configured."
        return None

    def require_valid(self) -> None:
        problem = self.configuration_problem
        if problem:
            raise ValueError(problem)
        if not self.has_api_key:
            raise ValueError("LLM_API_KEY is not configured.")

    def _valid_base_url(self) -> bool:
        if not self.base_url:
            return False
        try:
            parsed = urlsplit(self.base_url)
            return (
                parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
                and not parsed.query
                and not parsed.fragment
            )
        except ValueError:
            return False


@dataclass(frozen=True)
class EmbeddingSettings:
    """Independent, secret-safe settings for an OpenAI-compatible embeddings API."""

    provider: str = "disabled"
    model: str = "BAAI/bge-m3"
    api_key: SecretStr | None = None
    base_url: str | None = None
    timeout: float = 20.0
    dimensions: int = 1024

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        env_file: Path | None = None,
    ) -> "EmbeddingSettings":
        values: dict[str, str | None] = {}
        if env_file is not None:
            values.update(dotenv_values(env_file))
        elif environ is None:
            values.update(dotenv_values(Path(__file__).resolve().parents[2] / ".env"))
        values.update(dict(os.environ if environ is None else environ))
        raw_key = (values.get("EMBEDDING_API_KEY") or "").strip()
        return cls(
            provider=(values.get("EMBEDDING_PROVIDER") or "disabled").strip().lower(),
            model=(values.get("EMBEDDING_MODEL") or "BAAI/bge-m3").strip(),
            api_key=SecretStr(raw_key) if raw_key else None,
            base_url=(values.get("EMBEDDING_BASE_URL") or "").strip() or None,
            timeout=_positive_number(values.get("EMBEDDING_TIMEOUT"), 20.0),
            dimensions=_positive_integer(values.get("EMBEDDING_DIMENSIONS"), 1024),
        )

    @property
    def is_configured(self) -> bool:
        return (
            self.provider == "openai_compatible"
            and bool(self.api_key and self.api_key.get_secret_value().strip())
            and bool(self.model)
            and self._valid_base_url()
            and self.dimensions > 0
        )

    @property
    def configuration_problem(self) -> str | None:
        if self.provider == "disabled":
            return None
        if self.provider != "openai_compatible":
            return "EMBEDDING_PROVIDER must be disabled or openai_compatible."
        if not self.api_key or not self.api_key.get_secret_value().strip():
            return "EMBEDDING_API_KEY is required when embeddings are enabled."
        if not self._valid_base_url():
            return "EMBEDDING_BASE_URL must be an absolute HTTP or HTTPS URL without credentials, query, or fragment."
        if not self.model:
            return "EMBEDDING_MODEL is required when embeddings are enabled."
        return None

    def _valid_base_url(self) -> bool:
        if not self.base_url:
            return False
        try:
            parsed = urlsplit(self.base_url)
            return (
                parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
                and not parsed.query
                and not parsed.fragment
            )
        except ValueError:
            return False


@dataclass(frozen=True)
class QdrantSettings:
    """Local Qdrant persistence settings for the single-process MVP."""

    path: Path = Path("data/qdrant")
    collection: str = "datapilot_knowledge"

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        env_file: Path | None = None,
    ) -> "QdrantSettings":
        values: dict[str, str | None] = {}
        if env_file is not None:
            values.update(dotenv_values(env_file))
        elif environ is None:
            values.update(dotenv_values(Path(__file__).resolve().parents[2] / ".env"))
        values.update(dict(os.environ if environ is None else environ))
        return cls(
            path=Path(values.get("QDRANT_PATH", "data/qdrant")).expanduser(),
            collection=(values.get("QDRANT_COLLECTION") or "datapilot_knowledge").strip(),
        )


@dataclass(frozen=True)
class LangSmithSettings:
    """Secret-safe LangSmith tracing configuration with legacy env compatibility."""

    tracing_enabled: bool = False
    api_key: SecretStr | None = None
    project: str = "datapilot-ai"
    endpoint: str | None = None

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        env_file: Path | None = None,
    ) -> "LangSmithSettings":
        values: dict[str, str | None] = {}
        if env_file is not None:
            values.update(dotenv_values(env_file))
        elif environ is None:
            values.update(dotenv_values(Path(__file__).resolve().parents[2] / ".env"))
        values.update(dict(os.environ if environ is None else environ))
        enabled_raw = values.get("LANGSMITH_TRACING", values.get("LANGCHAIN_TRACING_V2", "false"))
        raw_key = (values.get("LANGSMITH_API_KEY") or values.get("LANGCHAIN_API_KEY") or "").strip()
        raw_project = (values.get("LANGSMITH_PROJECT") or values.get("LANGCHAIN_PROJECT") or "datapilot-ai").strip()
        raw_endpoint = (values.get("LANGSMITH_ENDPOINT") or values.get("LANGCHAIN_ENDPOINT") or "").strip()
        return cls(
            tracing_enabled=str(enabled_raw).strip().lower() in {"1", "true", "yes", "on"},
            api_key=SecretStr(raw_key) if raw_key else None,
            project=raw_project or "datapilot-ai",
            endpoint=raw_endpoint or None,
        )

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.get_secret_value().strip())

    @property
    def status(self) -> str:
        if not self.tracing_enabled:
            return "disabled"
        if not self.has_api_key:
            return "configuration_incomplete"
        if self.endpoint and not self._valid_endpoint():
            return "configuration_incomplete"
        return "enabled"

    def _valid_endpoint(self) -> bool:
        try:
            parsed = urlsplit(self.endpoint or "")
            return (
                parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
                and not parsed.query
                and not parsed.fragment
            )
        except ValueError:
            return False


@dataclass(frozen=True)
class DatabaseSettings:
    """Optional MySQL connection settings; credentials are always redacted."""

    host: str = "localhost"
    port: int = 3306
    database: str = "datapilot"
    username: str | None = None
    password: SecretStr | None = None

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        env_file: Path | None = None,
    ) -> "DatabaseSettings":
        values: dict[str, str | None] = {}
        if env_file is not None:
            values.update(dotenv_values(env_file))
        elif environ is None:
            values.update(dotenv_values(Path(__file__).resolve().parents[2] / ".env"))
        values.update(dict(os.environ if environ is None else environ))
        raw_user = (values.get("MYSQL_USER") or "").strip()
        raw_password = (values.get("MYSQL_PASSWORD") or "").strip()
        return cls(
            host=(values.get("MYSQL_HOST") or "localhost").strip() or "localhost",
            port=_positive_integer(values.get("MYSQL_PORT"), 3306),
            database=(values.get("MYSQL_DATABASE") or "datapilot").strip() or "datapilot",
            username=raw_user or None,
            password=SecretStr(raw_password) if raw_password else None,
        )

    @property
    def database_configured(self) -> bool:
        return bool(self.host and self.database and self.username)
