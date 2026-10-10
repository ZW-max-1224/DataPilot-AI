from __future__ import annotations

from typing import Any, Callable

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from app.config.settings import LLMSettings
from app.models.llm_models import LLMRequest, LLMResponse, LLMUsage
from app.services.tracing_service import TracingService


class LLMConfigurationError(ValueError):
    """The provider settings are incomplete or invalid."""


class LLMServiceError(RuntimeError):
    """A safe, provider-independent model invocation error."""


class LLMTimeoutError(LLMServiceError):
    """The configured provider did not respond before the timeout."""


class LLMResponseError(LLMServiceError):
    """The provider response did not contain usable assistant text."""


ClientFactory = Callable[..., Any]


class LLMService:
    """One-shot prompt-to-response service with no tools, memory, or agent loop."""

    def __init__(
        self,
        settings: LLMSettings,
        client_factory: ClientFactory | None = None,
        tracing_service: TracingService | None = None,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory or OpenAI
        self.tracing = tracing_service or TracingService()

    @property
    def is_configured(self) -> bool:
        return self._settings.is_configured

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response, or return an explicitly marked Demo response.

        The OpenAI-compatible SDK response is normalized here; no SDK types escape
        this service boundary.
        """
        if not self._settings.has_api_key:
            return LLMResponse(
                content="当前未配置 LLM API Key，DataPilot AI 正处于 Demo Mode。未调用模型，也未生成 AI 分析结果。",
                model="demo",
                usage=None,
                finish_reason="demo",
                mode="demo",
            )

        try:
            self._settings.require_valid()
        except ValueError as exc:
            raise LLMConfigurationError(str(exc)) from exc

        safe_inputs = {
            "model": self._settings.model,
            "system_prompt_chars": len(request.system_prompt),
            "user_prompt_chars": len(request.user_prompt),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or self._settings.max_tokens,
        }
        with self.tracing.span("OpenAI-compatible LLM Call", run_type="llm", inputs=safe_inputs) as span:
            response = self._generate_configured(request)
            span.end({
                "model": response.model,
                "mode": response.mode,
                "finish_reason": response.finish_reason,
                "usage": response.usage.model_dump(mode="json") if response.usage else None,
            })
            return response

    def _generate_configured(self, request: LLMRequest) -> LLMResponse:
        try:
            client = self._client_factory(
                api_key=self._settings.api_key.get_secret_value(),
                base_url=self._settings.base_url,
                timeout=self._settings.timeout,
                max_retries=self._settings.max_retries,
            )
            result = client.chat.completions.create(
                model=self._settings.model,
                messages=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens or self._settings.max_tokens,
            )
        except APITimeoutError as exc:
            raise LLMTimeoutError("The LLM provider request timed out.") from exc
        except APIConnectionError as exc:
            raise LLMServiceError("Could not connect to the configured LLM provider.") from exc
        except APIError as exc:
            raise LLMServiceError("The LLM provider returned an API error.") from exc
        except Exception as exc:
            raise LLMServiceError("The LLM request failed unexpectedly.") from exc

        try:
            if not hasattr(result, "choices"):
                raise LLMResponseError("The LLM provider returned an unexpected response format.")
            if not result.choices:
                raise LLMResponseError("The LLM provider returned an empty response.")
            choice = result.choices[0]
            content = choice.message.content
            if not isinstance(content, str) or not content.strip():
                raise LLMResponseError("The LLM provider returned an empty response.")

            raw_usage = getattr(result, "usage", None)
            usage = None
            if raw_usage is not None:
                usage = LLMUsage(
                    prompt_tokens=getattr(raw_usage, "prompt_tokens", None),
                    completion_tokens=getattr(raw_usage, "completion_tokens", None),
                    total_tokens=getattr(raw_usage, "total_tokens", None),
                )
            return LLMResponse(
                content=content.strip(),
                model=getattr(result, "model", None) or self._settings.model,
                usage=usage,
                finish_reason=getattr(choice, "finish_reason", None),
                mode="llm",
            )
        except LLMResponseError:
            raise
        except Exception as exc:
            raise LLMResponseError("The LLM provider returned an unexpected response format.") from exc
