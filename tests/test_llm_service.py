from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError
from openai import APIError, APIConnectionError, APITimeoutError

from app.config.settings import LLMSettings
from app.models.llm_models import LLMRequest, LLMResponse
from app.services.llm_service import (
    LLMConfigurationError,
    LLMResponseError,
    LLMService,
    LLMServiceError,
    LLMTimeoutError,
)

TEST_SECRET = "unit-test-secret-do-not-log"
BASE_URL = "https://llm.example.test/v1"


def configured_settings(**overrides) -> LLMSettings:
    values = {"LLM_API_KEY": TEST_SECRET, "LLM_BASE_URL": BASE_URL, "LLM_MODEL": "test-model"}
    values.update(overrides)
    return LLMSettings.from_env(values)


def fake_completion(content="hello", model="test-model", usage=True, finish_reason="stop"):
    raw_usage = SimpleNamespace(prompt_tokens=11, completion_tokens=4, total_tokens=15) if usage else None
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model, usage=raw_usage)


def test_settings_load_configured_and_missing_without_exposing_secret(tmp_path):
    settings = LLMSettings.from_env({
        "LLM_API_KEY": TEST_SECRET,
        "LLM_BASE_URL": BASE_URL,
        "LLM_MODEL": "test-model",
        "LLM_TIMEOUT": "12.5",
        "LLM_MAX_TOKENS": "2048",
    })
    assert settings.is_configured
    assert settings.timeout == 12.5
    assert settings.max_tokens == 2048
    assert TEST_SECRET not in repr(settings)

    missing = LLMSettings.from_env({})
    assert not missing.is_configured
    assert not missing.has_api_key
    assert missing.configuration_problem is None

    env_file = tmp_path / ".env"
    env_file.write_text(f"LLM_API_KEY={TEST_SECRET}\nLLM_BASE_URL={BASE_URL}\nLLM_MODEL=file-model\n", encoding="utf-8")
    loaded = LLMSettings.from_env({"LLM_MODEL": "env-model"}, env_file=env_file)
    assert loaded.model == "env-model"
    assert loaded.is_configured


def test_settings_honor_retry_configuration_and_allow_explicit_zero():
    assert LLMSettings.from_env({}).max_retries == 2
    assert LLMSettings.from_env({"LLM_MAX_RETRIES": "5"}).max_retries == 5
    # 0 must survive: it means "retries disabled", not "use the default".
    assert LLMSettings.from_env({"LLM_MAX_RETRIES": "0"}).max_retries == 0
    # Garbage falls back to the default instead of crashing the app.
    assert LLMSettings.from_env({"LLM_MAX_RETRIES": "abc"}).max_retries == 2
    assert LLMSettings.from_env({"LLM_MAX_RETRIES": "-3"}).max_retries == 2


def test_service_passes_configured_retry_count_to_the_client():
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: fake_completion("ok")))

    LLMService(configured_settings(LLM_MAX_RETRIES="4"), client_factory=FakeClient).generate(
        LLMRequest(user_prompt="question")
    )
    assert captured["max_retries"] == 4


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"LLM_BASE_URL": ""}, "LLM_BASE_URL is required"),
        ({"LLM_MODEL": ""}, "LLM_MODEL is required"),
        ({"LLM_BASE_URL": "file:///tmp"}, "absolute HTTP or HTTPS URL"),
    ],
)
def test_partial_or_invalid_settings_raise_safe_configuration_error(overrides, expected):
    service = LLMService(configured_settings(**overrides), client_factory=lambda **_: pytest.fail("client must not be created"))
    with pytest.raises(LLMConfigurationError, match=expected):
        service.generate(LLMRequest(user_prompt="hello"))


def test_request_and_response_models_are_validated_and_serializable():
    request = LLMRequest(user_prompt="  say hello  ", temperature=0.4, max_tokens=20)
    assert request.user_prompt == "say hello"
    assert LLMRequest.model_validate_json(request.model_dump_json()) == request
    with pytest.raises(ValidationError):
        LLMRequest(user_prompt=" ", temperature=2.1)
    with pytest.raises(ValidationError):
        LLMRequest(user_prompt="hello", exec="print(1)")

    response = LLMResponse(
        content="hello", model="test-model",
        usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        finish_reason="stop", mode="llm",
    )
    assert LLMResponse.model_validate_json(response.model_dump_json()) == response
    assert response.usage.total_tokens == 3


def test_service_normalizes_sdk_success_and_uses_request_parameters():
    calls = {}

    class FakeClient:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls["request"] = kwargs
            return fake_completion("  normalized answer  ")

    result = LLMService(configured_settings(), client_factory=FakeClient).generate(
        LLMRequest(system_prompt="system", user_prompt="question", temperature=0.7, max_tokens=64)
    )
    assert result == LLMResponse(
        content="normalized answer", model="test-model",
        usage={"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        finish_reason="stop", mode="llm",
    )
    assert calls["client"]["api_key"] == TEST_SECRET
    # Transport retries are delegated to the SDK. The previous hardcoded 0 turned a
    # transient connection failure into a fatal agent failure; the default is now 2.
    assert calls["client"]["max_retries"] == 2
    assert calls["request"]["temperature"] == 0.7
    assert calls["request"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]
    assert TEST_SECRET not in result.model_dump_json()


@pytest.mark.parametrize(
    "error,expected",
    [
        (lambda request: APITimeoutError(request=request), LLMTimeoutError),
        (lambda request: APIConnectionError(request=request), LLMServiceError),
        (lambda request: APIError("provider failure", request=request, body=None), LLMServiceError),
    ],
)
def test_service_maps_sdk_errors_without_leaking_secret(error, expected):
    request = httpx.Request("POST", BASE_URL + "/chat/completions")

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(error(request)))
            )

    with pytest.raises(expected) as caught:
        LLMService(configured_settings(), client_factory=FakeClient).generate(LLMRequest(user_prompt="hello"))
    assert TEST_SECRET not in str(caught.value)


@pytest.mark.parametrize("completion", [SimpleNamespace(choices=[]), fake_completion(content="   ")])
def test_service_rejects_empty_response(completion):
    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: completion))

    with pytest.raises(LLMResponseError, match="empty response"):
        LLMService(configured_settings(), client_factory=FakeClient).generate(LLMRequest(user_prompt="hello"))


def test_demo_mode_is_explicit_and_never_creates_a_client():
    def forbidden_factory(**kwargs):
        pytest.fail("Demo Mode must not instantiate an SDK client")

    response = LLMService(LLMSettings.from_env({}), client_factory=forbidden_factory).generate(
        LLMRequest(user_prompt="analyze this dataset")
    )
    assert response.mode == "demo"
    assert response.model == "demo"
    assert "Demo Mode" in response.content
    assert "未调用模型" in response.content


def test_sdk_is_not_called_until_generate():
    created = []

    def factory(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: fake_completion())))

    service = LLMService(configured_settings(), client_factory=factory)
    assert created == []
    service.generate(LLMRequest(user_prompt="hello"))
    assert len(created) == 1
