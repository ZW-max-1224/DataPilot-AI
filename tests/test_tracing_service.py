from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest

from app.agent import AnalystAgent, AgentRequest
from app.config.settings import LLMSettings, LangSmithSettings
from app.models.dataset_context import DatasetContext
from app.models.llm_models import LLMRequest, LLMResponse
from app.services.llm_service import LLMService
from app.services.profiler import profile_dataframe
from app.services.tracing_service import TracingService


class FakeRun:
    def __init__(self):
        self.outputs = None
        self.error = None

    def end(self, *, outputs=None, error=None):
        self.outputs = outputs
        self.error = error


class FakeTraceContext:
    def __init__(self, run):
        self.run = run

    def __enter__(self):
        return self.run

    def __exit__(self, *_):
        return False


class FakeTraceFactory:
    def __init__(self):
        self.calls = []
        self.runs = []

    def __call__(self, name, run_type, **kwargs):
        run = FakeRun()
        self.calls.append({"name": name, "run_type": run_type, **kwargs})
        self.runs.append(run)
        return FakeTraceContext(run)


def enabled_settings(**overrides):
    values = {"LANGCHAIN_TRACING_V2": "true", "LANGCHAIN_API_KEY": "test-langsmith-secret"}
    values.update(overrides)
    return LangSmithSettings.from_env(values)


def test_langsmith_settings_support_legacy_and_current_environment_names():
    legacy = LangSmithSettings.from_env({"LANGCHAIN_TRACING_V2": "true", "LANGCHAIN_API_KEY": "secret"})
    assert legacy.status == "enabled"
    assert legacy.project == "datapilot-ai"
    assert "secret" not in repr(legacy)
    current = LangSmithSettings.from_env({
        "LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "new-secret",
        "LANGSMITH_PROJECT": "custom-project", "LANGSMITH_ENDPOINT": "https://smith.example.test",
    })
    assert current.status == "enabled"
    assert current.project == "custom-project"
    assert current.endpoint == "https://smith.example.test"
    assert LangSmithSettings.from_env({}).status == "disabled"


def test_enabled_without_key_is_incomplete_and_never_creates_trace():
    factory = FakeTraceFactory()
    settings = LangSmithSettings.from_env({"LANGCHAIN_TRACING_V2": "true"})
    service = TracingService(settings, trace_factory=factory, activation_factory=lambda **_: nullcontext())
    assert service.status == "configuration_incomplete"
    with service.span("should-not-run") as span:
        span.end({"ok": True})
    assert factory.calls == []


def test_llm_still_works_when_tracing_is_enabled_but_key_is_missing():
    factory = FakeTraceFactory()
    tracing = TracingService(
        LangSmithSettings.from_env({"LANGCHAIN_TRACING_V2": "true"}),
        trace_factory=factory,
        activation_factory=lambda **_: nullcontext(),
    )
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="still works"), finish_reason="stop")],
        model="mock-model", usage=None,
    )
    llm = LLMService(
        LLMSettings.from_env({"LLM_API_KEY": "llm-key", "LLM_BASE_URL": "https://llm.example.test/v1", "LLM_MODEL": "mock-model"}),
        client_factory=lambda **_: SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **__: completion)
        )),
        tracing_service=tracing,
    )
    assert llm.generate(LLMRequest(user_prompt="question")).content == "still works"
    assert factory.calls == []


def test_disabled_llm_service_remains_functional_without_langsmith_key():
    settings = LangSmithSettings.from_env({"LANGCHAIN_TRACING_V2": "false"})
    factory = FakeTraceFactory()
    tracing = TracingService(settings, trace_factory=factory, activation_factory=lambda **_: nullcontext())
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="mock answer"), finish_reason="stop")],
        model="mock-model", usage=None,
    )
    def client(**_):
        return SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **__: completion)
        ))
    llm = LLMService(
        LLMSettings.from_env({"LLM_API_KEY": "llm-key", "LLM_BASE_URL": "https://llm.example.test/v1", "LLM_MODEL": "mock-model"}),
        client_factory=client,
        tracing_service=tracing,
    )
    response = llm.generate(LLMRequest(user_prompt="private prompt"))
    assert response.content == "mock answer"
    assert factory.calls == []


def test_enabled_mock_tracing_only_receives_sanitized_llm_metadata():
    factory = FakeTraceFactory()
    tracing = TracingService(enabled_settings(), client=object(), trace_factory=factory,
                             activation_factory=lambda **_: nullcontext())
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="private output"), finish_reason="stop")],
        model="mock-model", usage=SimpleNamespace(prompt_tokens=12, completion_tokens=2, total_tokens=14),
    )
    def client(**_):
        return SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **__: completion)
        ))
    llm = LLMService(
        LLMSettings.from_env({"LLM_API_KEY": "llm-key", "LLM_BASE_URL": "https://llm.example.test/v1", "LLM_MODEL": "mock-model"}),
        client_factory=client,
        tracing_service=tracing,
    )
    response = llm.generate(LLMRequest(
        user_prompt="private prompt", system_prompt="private system prompt"
    ))
    assert response.content == "private output"
    assert factory.calls[0]["run_type"] == "llm"
    assert "private prompt" not in repr(factory.calls[0])
    assert "private output" not in repr(factory.runs[0].outputs)
    assert factory.runs[0].outputs["usage"]["total_tokens"] == 14


def test_agent_tracing_preserves_result_and_redacts_tool_data():
    factory = FakeTraceFactory()
    tracing = TracingService(enabled_settings(), client=object(), trace_factory=factory,
                             activation_factory=lambda **_: nullcontext())
    frame = pd.DataFrame({"private_customer": ["secret-a", "secret-b"], "sales": [1, 9]})
    dataset = DatasetContext("id", "private.csv", frame, profile_dataframe(frame, "private.csv"))

    class MockLLM:
        def __init__(self):
            self.tracing = tracing
            self.calls = 0

        @property
        def is_configured(self):
            return True

        def generate(self, request):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(content='{"action":"tool_call","tool_name":"find_top_n","tool_input":{"column":"private_customer","n":1}}', model="mock", mode="llm")
            return LLMResponse(content='{"action":"final","final_answer":"A result was computed."}', model="mock", mode="llm")

    llm = MockLLM()
    result = AnalystAgent(llm).run(AgentRequest(user_query="Find top customer", dataset_context=dataset))
    assert result.success and result.tools_used == ["find_top_n"]
    assert [call["name"] for call in factory.calls] == [
        "DataPilot AnalystAgent", "Agent Decision", "Tool: find_top_n", "Agent Decision"
    ]
    all_trace_payloads = repr(factory.calls) + repr([run.outputs for run in factory.runs])
    assert "secret-a" not in all_trace_payloads and "secret-b" not in all_trace_payloads
    assert "private_customer" not in all_trace_payloads
    assert "Find top customer" not in all_trace_payloads
    assert factory.calls[0]["inputs"]["query_chars"] == len("Find top customer")
    assert factory.calls[0]["inputs"]["query_sha256"]
    assert factory.calls[2]["inputs"]["arguments"]["column"] == {"type": "string", "length": 16}


def test_demo_mode_emits_no_agent_or_llm_trace():
    factory = FakeTraceFactory()
    tracing = TracingService(enabled_settings(), client=object(), trace_factory=factory,
                             activation_factory=lambda **_: nullcontext())
    frame = pd.DataFrame({"x": [1]})
    dataset = DatasetContext("id", "demo.csv", frame, profile_dataframe(frame, "demo.csv"))
    llm = LLMService(LLMSettings.from_env({}), tracing_service=tracing)
    result = AnalystAgent(llm).run(AgentRequest(user_query="Analyze", dataset_context=dataset))
    assert not result.success and result.errors[0].code == "demo_mode"
    assert factory.calls == []


def test_trace_submission_failure_does_not_change_business_result(caplog):
    class BrokenContext(FakeTraceContext):
        def __exit__(self, *_):
            raise RuntimeError("do not log secret details")

    run = FakeRun()
    tracing = TracingService(
        enabled_settings(), client=object(),
        trace_factory=lambda *args, **kwargs: BrokenContext(run),
        activation_factory=lambda **_: nullcontext(),
    )
    with tracing.span("safe") as span:
        span.end({"ok": True})
    assert run.outputs == {"ok": True}
    assert "RuntimeError" in caplog.text
    assert "do not log secret details" not in caplog.text


def test_business_error_is_preserved_but_trace_records_only_error_type():
    factory = FakeTraceFactory()
    tracing = TracingService(enabled_settings(), client=object(), trace_factory=factory,
                             activation_factory=lambda **_: nullcontext())
    with pytest.raises(ValueError, match="private detail"):
        with tracing.span("error"):
            raise ValueError("private detail")
    assert factory.runs[0].error == "ValueError"
