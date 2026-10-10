from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app" / "ui" / "streamlit_app.py"
SAMPLE_ONE = (
    "date,region,sales,quantity\n"
    "2025-01-01,East,10,1\n2025-02-01,West,20,2\n2025-03-01,East,30,3\n"
).encode()
SAMPLE_TWO = b"category,amount\nA,5\nB,7\n"


@pytest.fixture(autouse=True)
def _hermetic_environment(monkeypatch):
    """Keep these tests independent of the developer's real .env.

    The app reads .env at import time, so without this a locally configured LLM or
    Embedding provider changes what the UI renders: the knowledge-base uploader only
    appears once embeddings are configured, which silently shifted positional widget
    lookups (file_uploader[0]) and made the suite pass in CI but fail locally. These
    tests must observe one known state: Demo Mode, no external services.
    """
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
                 "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL",
                 "MYSQL_USER", "MYSQL_PASSWORD",
                 "LANGSMITH_API_KEY", "LANGSMITH_TRACING",
                 "LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "disabled")
    yield


def _submit_form(app: AppTest, form_name: str) -> None:
    button = next(button for button in app.button if button.key.startswith(f"FormSubmitter:{form_name}-"))
    button.click().run()
    assert not app.exception, [error.message for error in app.exception]


def test_the_hermetic_fixture_hides_the_developers_real_configuration():
    """Prove the isolation guarantee, not just that the tests happen to pass.

    Without this, configuring a real provider in .env could change what the UI renders
    (and therefore which widgets these tests find) while CI, which has no .env, stays
    green.
    """
    from app.config.settings import EmbeddingSettings, LLMSettings

    assert not LLMSettings.from_env().is_configured
    assert not LLMSettings.from_env().has_api_key
    assert not EmbeddingSettings.from_env().is_configured


def test_demo_analysis_tools_chart_types_and_language_switch():
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    app.radio(key="dataset_source_choice").set_value("demo").run()
    assert not app.exception
    assert app.session_state["dataset_context"].profile.row_count == 1_208
    assert app.session_state["dataset_context"].profile.quality_score == 92

    for form_name in ("form_top_n", "form_group_stats", "form_time_series"):
        _submit_form(app, form_name)
    assert {"top_n", "group_stats", "time_series"}.issubset(app.session_state["analysis_results"])

    for key in ("run_missing", "run_duplicates", "run_outliers", "run_correlation"):
        app.button(key=key).click().run()
        assert not app.exception, [error.message for error in app.exception]
    assert {"missing", "duplicates", "outliers", "correlation"}.issubset(app.session_state["analysis_results"])

    # Set the form value and submit before rerunning, as a browser does for Streamlit forms.
    app.selectbox(key="chart_type").set_value("bar")
    _submit_form(app, "form_chart")
    assert app.session_state["analysis_results"]["chart"].result.chart_type == "bar"
    app.selectbox(key="chart_type").set_value("line")
    _submit_form(app, "form_chart")
    assert app.session_state["analysis_results"]["chart"].result.chart_type == "line"

    active_digest = app.session_state["dataset_digest"]
    app.selectbox(key="locale").set_value("en-US").run()
    assert not app.exception
    assert app.session_state["dataset_digest"] == active_digest
    app.selectbox(key="locale").set_value("zh-CN").run()
    assert not app.exception
    assert app.session_state["dataset_context"].profile.row_count == 1_208


def test_dataset_switches_clear_analysis_and_stale_column_selections():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.radio(key="dataset_source_choice").set_value("demo").run()
    app.button(key="run_missing").click().run()
    assert "missing" in app.session_state["analysis_results"]

    app.radio(key="dataset_source_choice").set_value("upload").run()
    assert app.session_state.get("dataset_context") is None
    # Address the dataset uploader by key: the knowledge-base uploader also exists and
    # positional indexing would silently target the wrong widget.
    app.file_uploader(key="dataset_uploader").set_value(("first.csv", SAMPLE_ONE, "text/csv")).run()
    assert app.session_state["dataset_context"].filename == "first.csv"
    assert app.session_state["analysis_results"] == {}
    _submit_form(app, "form_top_n")
    assert "top_n" in app.session_state["analysis_results"]

    app.file_uploader(key="dataset_uploader").set_value(("second.csv", SAMPLE_TWO, "text/csv")).run()
    assert not app.exception, [error.message for error in app.exception]
    context = app.session_state["dataset_context"]
    assert context.filename == "second.csv"
    assert context.profile.row_count == 2
    assert app.session_state["analysis_results"] == {}
    assert app.selectbox(key="top_column").value == "amount"

    app.radio(key="dataset_source_choice").set_value("demo").run()
    assert not app.exception, [error.message for error in app.exception]
    assert app.session_state["dataset_context"].filename == "sample_sales.csv"
    assert app.session_state["dataset_context"].profile.row_count == 1_208
    assert app.session_state["analysis_results"] == {}


def test_llm_playground_is_explicit_demo_without_api_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("LLM_MODEL", "")
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    app.text_area(key="llm_user_prompt").set_value("Analyze the active dataset")
    button = next(button for button in app.button if button.key.startswith("FormSubmitter:form_llm_prompt-"))
    button.click().run()
    assert not app.exception
    response = app.session_state["llm_response"]
    assert response.mode == "demo"
    assert response.model == "demo"
    assert "Demo Mode" in response.content
    assert "未调用模型" in response.content


def test_agent_is_unavailable_in_demo_mode(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("LLM_MODEL", "")
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert any("Agent 功能不可用" in item.value for item in app.info)
    assert not any(button.key.startswith("FormSubmitter:form_agent-") for button in app.button)
    assert any("已关闭" in item.value for item in app.info)


def test_langsmith_configuration_status_never_renders_api_key(monkeypatch):
    secret = "langsmith-ui-secret-never-render-this"
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", secret)
    monkeypatch.setenv("LANGCHAIN_PROJECT", "datapilot-ai")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("LLM_MODEL", "")
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    visible = "\n".join(
        str(element.value) for collection in (app.caption, app.info, app.success, app.warning, app.error, app.markdown)
        for element in collection
    )
    assert "已启用" in visible
    assert secret not in visible


def test_langsmith_enabled_without_key_shows_incomplete_status(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("LLM_MODEL", "")
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert any("追踪已开启" in item.value for item in app.warning)


def test_database_unavailable_does_not_block_ui_and_never_renders_password(monkeypatch):
    secret = "mysql-ui-password-never-render-this"
    monkeypatch.setenv("MYSQL_HOST", "127.0.0.1")
    monkeypatch.setenv("MYSQL_PORT", "1")
    monkeypatch.setenv("MYSQL_DATABASE", "datapilot")
    monkeypatch.setenv("MYSQL_USER", "test-user")
    monkeypatch.setenv("MYSQL_PASSWORD", secret)
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    app.radio(key="dataset_source_choice").set_value("demo").run()
    assert not app.exception
    assert app.session_state["dataset_context"].profile.row_count == 1_208
    assert any("MySQL 不可用" in item.value for item in app.warning)
    visible = "\n".join(
        str(element.value) for collection in (app.caption, app.info, app.success, app.warning, app.error, app.markdown)
        for element in collection
    )
    assert secret not in visible


def test_configured_llm_status_never_renders_api_key(monkeypatch):
    secret = "ui-test-secret-must-never-be-rendered"
    monkeypatch.setenv("LLM_API_KEY", secret)
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "ui-test-model")
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    visible_elements = [
        *app.caption, *app.info, *app.success, *app.warning, *app.error, *app.markdown,
    ]
    visible_text = "\n".join(str(element.value) for element in visible_elements)
    assert secret not in visible_text
    assert "ui-test-model" in visible_text
    assert any(button.key.startswith("FormSubmitter:form_agent-") for button in app.button)


def test_evaluation_ui_reports_unavailable_services_without_fake_passes(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "disabled")
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    app.button(key="run_evaluation").click().run()
    assert not app.exception, [error.message for error in app.exception]
    report = app.session_state["evaluation_report"]
    assert report.summary.total_cases == 15
    assert report.summary.passed_cases == 0
    assert report.summary.pass_rate is None
    assert report.summary.unavailable_cases == 9
    assert report.summary.skipped_cases == 6
    assert all(item.status in {"unavailable", "skipped"} for item in report.results)
