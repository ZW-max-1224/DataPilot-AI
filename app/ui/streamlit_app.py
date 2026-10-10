import hashlib
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.analysis_result import (
    CorrelationResult, DatasetOverview, DuplicateResult, MissingValuesResult,
    OutlierResult, TableResult, TimeSeriesResult,
)
from app.models.dataset_context import DatasetContext
from app.agent import AnalystAgent, AgentRequest, AgentResponse
from app.models.dataset_profile import DatasetProfile
from app.config.settings import DatabaseSettings, EmbeddingSettings, LLMSettings, QdrantSettings
from app.models.llm_models import LLMRequest, LLMResponse
from app.services.dataset_service import DatasetLoadError, load_dataset
from app.services.profiler import profile_dataframe
from app.services.llm_service import (
    LLMConfigurationError, LLMService, LLMServiceError, LLMTimeoutError,
)
from app.services.persistence_service import PersistenceService
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError
from app.services.knowledge_service import KnowledgeService
from app.services.tracing_service import TracingService
from app.knowledge.loader import KnowledgeLoadError, load_knowledge_document
from app.vectorstore.qdrant_store import QdrantVectorStore, VectorStoreError
from app.evaluation import EvaluationRunner, load_evaluation_cases
from app.evaluation.judge import OptionalLLMJudge
from app.tools.analysis_tools import (
    AnalysisToolError, analyze_duplicates, analyze_missing_values, calculate_correlation,
    calculate_group_statistics, calculate_time_series, detect_outliers, find_top_n,
    get_dataset_overview,
)
from app.tools.visualization_tools import ChartArtifact, create_chart

logger = logging.getLogger(__name__)
DEFAULT_KNOWLEDGE_BASE_ID = "datapilot-default"
# Stable widget value for "not scoped to a dataset". The translated label is only a
# display concern: using the translated text as the value would leave the stored
# selection invalid the moment the user switches language.
GENERAL_SCOPE_SENTINEL = "__general_knowledge__"


def _qdrant_store_for_session(path: str, collection_name: str, dimensions: int) -> QdrantVectorStore:
    """Return one local Qdrant client per session, closing it when config changes.

    Local Qdrant storage is single-client: it holds a file lock on the directory.
    ``@st.cache_resource`` keyed on the dimensions would build a *second* client for
    the same path while the first is still open, which fails as "Qdrant local storage
    is unavailable". Reusing and explicitly closing the previous client is what makes
    changing EMBEDDING_DIMENSIONS survivable without restarting the app.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    key = (str(resolved), collection_name, dimensions)
    if st.session_state.get("_qdrant_key") != key:
        previous = st.session_state.get("_qdrant_store")
        if isinstance(previous, QdrantVectorStore):
            try:
                previous.close()
            except Exception:
                logger.warning("Could not close the previous Qdrant client (%s)", type(Exception).__name__)
        st.session_state["_qdrant_store"] = QdrantVectorStore(
            path=resolved, collection_name=collection_name, dimensions=dimensions
        )
        st.session_state["_qdrant_key"] = key
    return st.session_state["_qdrant_store"]


def _safe_source_name(filename: str) -> str:
    return Path(filename.replace("\\", "/")).name[:255]

ANALYSIS_WIDGET_KEYS = (
    "top_column", "top_group", "top_n_value", "top_sort_order", "group_column",
    "group_aggregation", "group_metric", "group_sort_order", "group_limit",
    "time_date", "time_metric", "time_frequency", "time_aggregation",
    "outlier_column", "correlation_a", "correlation_b", "chart_type", "chart_x",
    "chart_y", "chart_group", "chart_title_input", "chart_bins",
)


def _clear_analysis_state() -> None:
    st.session_state["analysis_results"] = {}
    st.session_state["analysis_errors"] = {}
    st.session_state.pop("agent_response", None)
    for key in ANALYSIS_WIDGET_KEYS:
        st.session_state.pop(key, None)


def _content_digest(filename: str, content: bytes) -> str:
    return hashlib.sha256(filename.encode("utf-8") + b"\0" + content).hexdigest()


def load_translations(locale: str) -> dict[str, str]:
    path = PROJECT_ROOT / "i18n" / f"{locale}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    if value < 1024**2:
        return f"{value / 1024:.1f} KB"
    return f"{value / 1024**2:.2f} MB"


def set_dataset(
    df: pd.DataFrame, name: str, source: str, digest: str,
    persistence: PersistenceService | None = None,
) -> None:
    profile = profile_dataframe(df, name)
    _clear_analysis_state()
    st.session_state["dataset_df"] = df
    st.session_state["dataset_profile"] = profile
    st.session_state["dataset_context"] = DatasetContext(
        dataset_id=digest, filename=name, dataframe=df, profile=profile
    )
    st.session_state["dataset_source_active"] = source
    st.session_state["dataset_digest"] = digest
    if persistence is not None:
        persistence.persist_dataset(st.session_state["dataset_context"], source)


def _quality_issue_description(issue_code: str, profile: DatasetProfile, text: dict[str, str]) -> str:
    if issue_code in {"high_missing_rate", "moderate_missing_rate", "some_missing_values"}:
        rate = profile.missing.missing_rate
        suffix = {"high_missing_rate": "high_missing", "moderate_missing_rate": "moderate_missing",
                  "some_missing_values": "some_missing"}[issue_code]
    elif issue_code in {"high_duplicate_rate", "moderate_duplicate_rate", "some_duplicate_rows"}:
        rate = profile.duplicates.duplicate_rate
        suffix = {"high_duplicate_rate": "high_duplicates", "moderate_duplicate_rate": "moderate_duplicates",
                  "some_duplicate_rows": "some_duplicates"}[issue_code]
    else:
        unknown = sum(column.semantic_type == "unknown" for column in profile.columns)
        rate = unknown * 100 / profile.column_count
        suffix = {"many_unknown_types": "many_unknown", "some_unknown_types": "some_unknown",
                  "unknown_type_columns": "unknown"}.get(issue_code, "fallback")
    return text.get(f"quality_desc_{suffix}", text["quality_issue_fallback"]).format(rate=rate)


def _render_structured_answer(structured, translations: dict[str, str]) -> None:
    """Render the structured report, keeping facts and hypotheses visibly distinct."""
    badges = {"fact": "✅", "interpretation": "🔎", "hypothesis": "⚠️"}
    labels = {"fact": translations["claim_fact"],
              "interpretation": translations["claim_interpretation"],
              "hypothesis": translations["claim_hypothesis"]}
    st.markdown(f"##### {translations['agent_structured_title']}")
    st.markdown(f"**{translations['result_executive_summary']}**")
    st.write(structured.executive_summary)
    if structured.key_findings:
        st.markdown(f"**{translations['result_key_findings']}**")
        for index, finding in enumerate(structured.key_findings, start=1):
            badge = badges.get(finding.claim_type, "")
            label = labels.get(finding.claim_type, finding.claim_type)
            # The evidence tool sits on the finding itself instead of on a separate
            # caption line: the badge and the tool it was verified with belong together,
            # and a repeated caption added five extra lines for five findings.
            evidence = f" · `{finding.evidence_tool}`" if finding.evidence_tool else ""
            st.markdown(f"{index}. {finding.statement} — {badge} {label}{evidence}")
    if structured.recommendations:
        st.markdown(f"**{translations['result_recommendations']}**")
        for item in structured.recommendations:
            st.markdown(f"- {item}")


def _show_result(key: str, title: str, translations: dict[str, str]) -> None:
    results: dict[str, Any] = st.session_state.get("analysis_results", {})
    errors: dict[str, str] = st.session_state.get("analysis_errors", {})
    if key in errors:
        st.error(f"{translations['tool_error']} {errors[key]}")
        return
    if key not in results:
        return
    result = results[key]
    st.markdown(f"**{title}**")
    if isinstance(result, ChartArtifact):
        st.pyplot(result.figure, clear_figure=False)
        for warning in result.result.warnings:
            st.warning(warning)
        st.caption(f"{translations[result.result.chart_type]} · {result.result.plotted_rows:,} {translations['plotted_rows']}")
        # Matplotlib keeps every figure alive until it is closed; a long Streamlit
        # session would otherwise accumulate one figure per rendered chart.
        result.release()
    elif isinstance(result, TimeSeriesResult):
        value_column = f"{result.metric}_{result.aggregation}"
        rows = [{translations["period"]: row["period"], translations["value"]: row[value_column]}
                for row in result.rows]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        cols = st.columns(3)
        cols[0].metric(translations["trend"], translations.get(result.trend, result.trend))
        cols[1].metric(translations["change"], "—" if result.change is None else f"{result.change:,.2f}")
        cols[2].metric(translations["change_pct"], "—" if result.change_percentage is None else f"{result.change_percentage:.2f}%")
        if result.rows_excluded:
            st.caption(f"{translations['rows_excluded']}: {result.rows_excluded}")
    elif isinstance(result, TableResult):
        st.dataframe(pd.DataFrame(result.rows), width="stretch", hide_index=True)
    elif isinstance(result, MissingValuesResult):
        rows = [{translations["column"]: item.column, translations["missing_count"]: item.missing_count,
                 translations["missing_pct"]: item.missing_percentage} for item in result.rows]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(f"{translations['missing']}: {result.total_missing_values} · {result.missing_rate:.2f}%")
    elif isinstance(result, DuplicateResult):
        st.metric(translations["duplicates"], result.duplicate_rows, f"{result.duplicate_percentage:.2f}%")
        if result.duplicate_examples:
            st.dataframe(pd.DataFrame(result.duplicate_examples), width="stretch", hide_index=True)
    elif isinstance(result, OutlierResult):
        st.dataframe(pd.DataFrame([
            {translations["outlier_field"]: translations.get(name, name),
             translations["value"]: json.dumps(value, ensure_ascii=False)}
            for name, value in result.model_dump(mode="json").items()
        ]), width="stretch", hide_index=True)
    elif isinstance(result, CorrelationResult):
        st.metric(translations["correlation"], result.correlation, translations.get(result.interpretation, result.interpretation))
        st.caption(translations["correlation_note"])
    else:
        st.json(result.model_dump(mode="json"))


st.set_page_config(page_title="DataPilot AI", page_icon="📊", layout="wide")
if "locale" not in st.session_state:
    st.session_state["locale"] = "zh-CN"
initial_text = load_translations(st.session_state["locale"])
llm_settings = LLMSettings.from_env()
tracing_service = TracingService()
langsmith_status = tracing_service.status
database_settings = DatabaseSettings.from_env()
if st.session_state.get("_database_settings") != database_settings:
    old_persistence = st.session_state.get("_persistence_service")
    if isinstance(old_persistence, PersistenceService):
        old_persistence.close()
    persistence_service = PersistenceService(database_settings)
    st.session_state["_database_settings"] = database_settings
    st.session_state["_persistence_service"] = persistence_service
else:
    persistence_service = st.session_state["_persistence_service"]

embedding_settings = EmbeddingSettings.from_env()
qdrant_settings = QdrantSettings.from_env()
knowledge_service: KnowledgeService | None = None
knowledge_runtime_error: str | None = None
qdrant_store: QdrantVectorStore | None = None
try:
    qdrant_store = _qdrant_store_for_session(
        str(qdrant_settings.path), qdrant_settings.collection, embedding_settings.dimensions,
    )
    available, availability_code = qdrant_store.availability()
    if not available:
        # Keep the distinct reason: a dimension mismatch is actionable (rebuild the
        # index) while a backend outage is not, and they must not look identical.
        knowledge_runtime_error = availability_code or "knowledge_base_unavailable"
    else:
        knowledge_service = KnowledgeService(
            EmbeddingService(embedding_settings), qdrant_store,
            repository=persistence_service.knowledge_repository,
            tracing=tracing_service,
        )
except (EmbeddingServiceError, VectorStoreError, ValueError) as exc:
    knowledge_runtime_error = getattr(exc, "code", "knowledge_base_unavailable")

with st.sidebar:
    locale = st.selectbox(
        initial_text["language"], ["zh-CN", "en-US"],
        format_func=lambda value: "中文" if value == "zh-CN" else "English", key="locale",
    )
    t = load_translations(locale)
    source = st.radio(
        t["dataset_source"], ["upload", "demo"],
        format_func=lambda value: t[value], key="dataset_source_choice",
    )
    uploaded_file = None
    if source == "upload":
        st.subheader(t["upload_heading"])
        st.caption(t["upload_help"])
        uploaded_file = st.file_uploader(
            t["upload_heading"], type=["csv", "xlsx", "xls"], label_visibility="collapsed",
            key="dataset_uploader",
        )
    else:
        st.caption(t["demo_note"])

    st.divider()
    st.subheader(t["llm_status"])
    if not llm_settings.has_api_key:
        st.info(t["demo_mode"])
        st.caption(f"{t['api_key_status']}: {t['not_configured']}")
    elif llm_settings.is_configured:
        st.success(t["llm_mode"])
        st.caption(f"{t['api_key_status']}: {t['configured']}")
        st.caption(f"{t['provider']}: OpenAI-compatible API")
        st.caption(f"{t['base_url']}: {llm_settings.base_url}")
        st.caption(f"{t['model']}: {llm_settings.model}")
    else:
        st.warning(t["llm_config_incomplete"])
        st.caption(f"{t['api_key_status']}: {t['configured']}")

    st.divider()
    st.subheader(t["database_status"])
    if persistence_service.status == "connected":
        st.success(t["database_connected"])
    elif persistence_service.status == "unavailable":
        st.warning(t["database_unavailable"])
    else:
        st.info(t["database_disabled"])

    st.divider()
    st.subheader(t["kb_title"])
    if knowledge_runtime_error:
        st.warning(t["kb_unavailable"])
    elif not embedding_settings.is_configured:
        st.info(t["kb_embedding_unavailable"])
    else:
        st.success(t["kb_ready"])
    if persistence_service.knowledge_repository is None:
        st.caption(t["kb_no_metadata_store"])

    st.divider()
    st.subheader(t["langsmith_status"])
    if langsmith_status == "enabled":
        st.success(t["langsmith_enabled"])
        st.caption(t["langsmith_privacy_note"])
    elif langsmith_status == "configuration_incomplete":
        st.warning(t["langsmith_incomplete"])
    elif langsmith_status == "unavailable":
        st.warning(t["langsmith_unavailable"])
    else:
        st.info(t["langsmith_disabled"])

source_changed = st.session_state.get("dataset_source_active") != source
if source_changed:
    for key in ("dataset_df", "dataset_profile", "dataset_context", "dataset_digest", "analysis_results", "analysis_errors"):
        st.session_state.pop(key, None)
    for key in ANALYSIS_WIDGET_KEYS:
        st.session_state.pop(key, None)
    st.session_state["dataset_source_active"] = source

if source == "demo":
    demo_path = PROJECT_ROOT / "data" / "sample_sales.csv"
    try:
        content = demo_path.read_bytes()
        digest = _content_digest(demo_path.name, content)
        if st.session_state.get("dataset_digest") != digest:
            demo_df = load_dataset(demo_path.name, content)
            set_dataset(demo_df, demo_path.name, "demo", digest, persistence_service)
    except (OSError, DatasetLoadError, ValueError):
        st.error(t["profile_error"])
    except Exception:
        logger.exception("Unexpected error while loading demo dataset")
        st.error(t["profile_error"])
else:
    if uploaded_file is not None:
        content = uploaded_file.getvalue()
        digest = _content_digest(uploaded_file.name, content)
        if st.session_state.get("dataset_digest") != digest:
            try:
                uploaded_df = load_dataset(uploaded_file.name, content)
                set_dataset(uploaded_df, uploaded_file.name, "upload", digest, persistence_service)
            except DatasetLoadError as exc:
                for key in ("dataset_df", "dataset_profile", "dataset_context", "dataset_digest"):
                    st.session_state.pop(key, None)
                _clear_analysis_state()
                # DatasetLoadError messages are author-controlled and actionable (empty
                # file, size limit, undecodable CSV, duplicate headers). Replacing them
                # with a generic notice left the user with no next step.
                st.error(f"{t['error']} {exc}")
            except ValueError:
                for key in ("dataset_df", "dataset_profile", "dataset_context", "dataset_digest"):
                    st.session_state.pop(key, None)
                _clear_analysis_state()
                st.error(t["profile_error"])
            except Exception:
                logger.exception("Unexpected error while loading or profiling uploaded dataset")
                for key in ("dataset_df", "dataset_profile", "dataset_context", "dataset_digest"):
                    st.session_state.pop(key, None)
                _clear_analysis_state()
                st.error(t["profile_error"])

with st.container(border=True):
    st.title(t["title"])
    hero_left, hero_right = st.columns([4, 1])
    with hero_left:
        st.subheader(t["subtitle"])
        st.caption(t["tagline"])
    with hero_right:
        st.caption(t["current_phase"])
        st.info(t["no_agent_notice"], icon="ℹ️")
context: DatasetContext | None = st.session_state.get("dataset_context")
if context is None:
    st.info(t["waiting"])
else:
    if persistence_service.status == "unavailable":
        st.warning(t["database_fallback_note"])
    df, profile = context.dataframe, context.profile
    overview: DatasetOverview = get_dataset_overview(df, context.filename, profile=profile)
    st.text(f"{t['file_name']}: {profile.dataset_name}")
    st.caption(f"{t['memory']}: {format_bytes(profile.memory_usage_bytes)}")
    with st.sidebar:
        st.divider()
        st.subheader(t["active_dataset"])
        st.write(f"**{t['file_name']}:** {context.filename}")
        st.write(f"**{t['source_label']}:** {t[st.session_state['dataset_source_active']]}")
        st.write(f"**{t['rows']}:** {overview.rows:,} · **{t['columns']}:** {overview.columns:,}")
        st.metric(t["quality"], f"{profile.quality_score}/100")
    st.header(t["dataset_overview"])
    metrics = st.columns(5)
    metrics[0].metric(t["rows"], f"{overview.rows:,}")
    metrics[1].metric(t["columns"], f"{overview.columns:,}")
    metrics[2].metric(t["missing"], f"{overview.missing_values:,}", f"{profile.missing.missing_rate:.2f}%")
    metrics[3].metric(t["duplicates"], f"{overview.duplicate_rows:,}", f"{profile.duplicates.duplicate_rate:.2f}%")
    metrics[4].metric(t["quality"], f"{profile.quality_score}/100")
    st.caption(t["quality_note"])

    if profile.quality_issues:
        issue_rows = [{t["issue"]: t.get(issue.code, issue.code), t["penalty"]: issue.penalty,
                       t["description"]: _quality_issue_description(issue.code, profile, t)}
                      for issue in profile.quality_issues]
        st.dataframe(pd.DataFrame(issue_rows), width="stretch", hide_index=True)
    else:
        st.success(t["quality_good"])

    profile_tabs = st.tabs([t["column_profile"], t["numeric_summary"], t["categorical_summary"], t["preview"]])
    column_rows = [
        {t["column_profile"]: item.column_name, t["dtype"]: item.pandas_dtype,
         t["semantic"]: item.semantic_type, t["non_null"]: item.non_null_count,
         t["missing_count"]: item.missing_count, t["missing_pct"]: item.missing_percentage,
         t["unique"]: item.unique_count}
        for item in profile.columns
    ]
    with profile_tabs[0]:
        st.dataframe(pd.DataFrame(column_rows), width="stretch", hide_index=True)
    with profile_tabs[1]:
        if profile.numeric_summary:
            labels = {"column_name": t["column"], "count": t["count"], "mean": t["mean"], "median": t["median"],
                      "std": t["std"], "min": t["min"], "max": t["max"]}
            st.dataframe(pd.DataFrame([{labels[key]: value for key, value in item.model_dump().items()}
                                       for item in profile.numeric_summary]), width="stretch", hide_index=True)
        else:
            st.caption(t["no_numeric_columns"])
    with profile_tabs[2]:
        if profile.categorical_summary:
            for item in profile.categorical_summary:
                st.markdown(f"**{item.column_name}** · {item.unique_count} {t['unique'].lower()}")
                st.dataframe(pd.DataFrame([{t["value"]: value.value, t["count"]: value.count,
                                            t["percentage"]: value.percentage}
                                           for value in item.top_values]), width="stretch", hide_index=True)
        else:
            st.caption(t["no_categorical_columns"])
    with profile_tabs[3]:
        st.dataframe(df.head(20), width="stretch", hide_index=True)

    st.divider()
    st.header(t["analysis_playground"])
    numeric_columns = overview.numeric_columns
    categorical_columns = overview.categorical_columns
    all_columns = list(df.columns)
    result_tabs = st.tabs([t["top_n"], t["group_stats"], t["time_series"], t["quality_tools"], t["charts"]])

    with result_tabs[0]:
        with st.form("form_top_n"):
            top_column = st.selectbox(
                t["metric"], all_columns,
                index=all_columns.index(numeric_columns[0]) if numeric_columns else 0,
                key="top_column",
            )
            top_group = st.selectbox(t["group_by_optional"], [None] + categorical_columns, format_func=lambda v: t["none"] if v is None else v, key="top_group")
            top_sort_order = st.selectbox(t["sort_order"], ["desc", "asc"], format_func=lambda v: t[v], key="top_sort_order")
            top_n = st.number_input(t["n"], min_value=1, max_value=100, value=10, step=1, key="top_n_value")
            run_top = st.form_submit_button(t["run_analysis"], type="primary")
        if run_top:
            try:
                st.session_state["analysis_results"]["top_n"] = find_top_n(
                    df, top_column, int(top_n), ascending=top_sort_order == "asc", group_by=top_group
                )
                st.session_state["analysis_errors"].pop("top_n", None)
            except AnalysisToolError as exc:
                st.session_state["analysis_errors"]["top_n"] = str(exc)
            except Exception:
                logger.exception("Unexpected error in Top N analysis")
                st.session_state["analysis_errors"]["top_n"] = t["unexpected_tool_error"]
        _show_result("top_n", t["results"], t)

    with result_tabs[1]:
        if not categorical_columns or not numeric_columns:
            st.info(t["requires_categorical_numeric"])
        else:
            with st.form("form_group_stats"):
                group_column = st.selectbox(t["group_by"], categorical_columns, key="group_column")
                aggregation = st.selectbox(t["aggregation"], ["sum", "mean", "median", "min", "max", "count"], format_func=lambda v: t[v], key="group_aggregation")
                group_metric_options = all_columns if aggregation == "count" else numeric_columns
                group_metric = st.selectbox(t["metric"], group_metric_options, key="group_metric")
                sort_order = st.selectbox(t["sort_order"], ["desc", "asc"], format_func=lambda v: t[v], key="group_sort_order")
                group_limit = st.number_input(t["limit"], min_value=1, max_value=100, value=10, key="group_limit")
                run_group = st.form_submit_button(t["run_analysis"], type="primary")
            if run_group:
                try:
                    st.session_state["analysis_results"]["group_stats"] = calculate_group_statistics(
                        df, group_column, group_metric, aggregation, sort_order=sort_order, limit=int(group_limit)
                    )
                    st.session_state["analysis_errors"].pop("group_stats", None)
                except AnalysisToolError as exc:
                    st.session_state["analysis_errors"]["group_stats"] = str(exc)
                except Exception:
                    logger.exception("Unexpected error in group statistics")
                    st.session_state["analysis_errors"]["group_stats"] = t["unexpected_tool_error"]
            _show_result("group_stats", t["results"], t)

    with result_tabs[2]:
        if not overview.datetime_columns or not numeric_columns:
            st.info(t["requires_date_numeric"])
        else:
            with st.form("form_time_series"):
                date_column = st.selectbox(t["date_column"], overview.datetime_columns, key="time_date")
                time_metric = st.selectbox(t["metric"], numeric_columns, key="time_metric")
                frequency = st.selectbox(t["frequency"], ["day", "week", "month", "quarter", "year"], format_func=lambda v: t[v], index=2, key="time_frequency")
                time_aggregation = st.selectbox(t["aggregation"], ["sum", "mean", "median", "min", "max"], format_func=lambda v: t[v], key="time_aggregation")
                run_time = st.form_submit_button(t["run_analysis"], type="primary")
            if run_time:
                try:
                    st.session_state["analysis_results"]["time_series"] = calculate_time_series(
                        df, date_column, time_metric, time_aggregation, frequency
                    )
                    st.session_state["analysis_errors"].pop("time_series", None)
                except AnalysisToolError as exc:
                    st.session_state["analysis_errors"]["time_series"] = str(exc)
                except Exception:
                    logger.exception("Unexpected error in time-series analysis")
                    st.session_state["analysis_errors"]["time_series"] = t["unexpected_tool_error"]
            _show_result("time_series", t["results"], t)

    with result_tabs[3]:
        quality_cols = st.columns(2)
        with quality_cols[0]:
            if st.button(t["analyze_missing"], key="run_missing"):
                try:
                    st.session_state["analysis_results"]["missing"] = analyze_missing_values(df)
                    st.session_state["analysis_errors"].pop("missing", None)
                except AnalysisToolError as exc:
                    st.session_state["analysis_errors"]["missing"] = str(exc)
                except Exception:
                    logger.exception("Unexpected error in missing-value analysis")
                    st.session_state["analysis_errors"]["missing"] = t["unexpected_tool_error"]
            _show_result("missing", t["results"], t)
            if st.button(t["analyze_duplicates"], key="run_duplicates"):
                try:
                    st.session_state["analysis_results"]["duplicates"] = analyze_duplicates(df)
                    st.session_state["analysis_errors"].pop("duplicates", None)
                except AnalysisToolError as exc:
                    st.session_state["analysis_errors"]["duplicates"] = str(exc)
                except Exception:
                    logger.exception("Unexpected error in duplicate analysis")
                    st.session_state["analysis_errors"]["duplicates"] = t["unexpected_tool_error"]
            _show_result("duplicates", t["results"], t)
        with quality_cols[1]:
            if numeric_columns:
                outlier_column = st.selectbox(t["numeric_column"], numeric_columns, key="outlier_column")
                if st.button(t["detect_outliers"], key="run_outliers"):
                    try:
                        st.session_state["analysis_results"]["outliers"] = detect_outliers(df, outlier_column)
                        st.session_state["analysis_errors"].pop("outliers", None)
                    except AnalysisToolError as exc:
                        st.session_state["analysis_errors"]["outliers"] = str(exc)
                    except Exception:
                        logger.exception("Unexpected error in outlier analysis")
                        st.session_state["analysis_errors"]["outliers"] = t["unexpected_tool_error"]
                _show_result("outliers", t["results"], t)
                if len(numeric_columns) >= 2:
                    correlation_cols = st.columns(2)
                    col_a = correlation_cols[0].selectbox(t["column_a"], numeric_columns, key="correlation_a")
                    col_b = correlation_cols[1].selectbox(t["column_b"], numeric_columns, index=1, key="correlation_b")
                    if st.button(t["calculate_correlation"], key="run_correlation"):
                        try:
                            st.session_state["analysis_results"]["correlation"] = calculate_correlation(df, col_a, col_b)
                            st.session_state["analysis_errors"].pop("correlation", None)
                        except AnalysisToolError as exc:
                            st.session_state["analysis_errors"]["correlation"] = str(exc)
                        except Exception:
                            logger.exception("Unexpected error in correlation analysis")
                            st.session_state["analysis_errors"]["correlation"] = t["unexpected_tool_error"]
                    _show_result("correlation", t["results"], t)

    with result_tabs[4]:
        with st.form("form_chart"):
            chart_type = st.selectbox(t["chart_type"], ["line", "bar", "scatter", "histogram"], format_func=lambda v: t[v], key="chart_type")
            chart_x_options = numeric_columns if chart_type == "histogram" else all_columns
            chart_x = st.selectbox(t["x_axis"], chart_x_options, key="chart_x")
            chart_y = None if chart_type == "histogram" else st.selectbox(t["y_axis"], numeric_columns, key="chart_y")
            chart_group = None
            if chart_type in {"line", "bar"} and categorical_columns:
                chart_group = st.selectbox(t["group_by_optional"], [None] + categorical_columns, format_func=lambda v: t["none"] if v is None else v, key="chart_group")
            chart_title = st.text_input(t["chart_title"], key="chart_title_input")
            chart_bins = st.number_input(t["bins"], min_value=5, max_value=100, value=20, key="chart_bins") if chart_type == "histogram" else 20
            run_chart = st.form_submit_button(t["create_chart"], type="primary")
        if run_chart:
            try:
                st.session_state["analysis_results"]["chart"] = create_chart(
                    df, chart_type, chart_x, chart_y, chart_group, chart_title or None, int(chart_bins)
                )
                st.session_state["analysis_errors"].pop("chart", None)
            except AnalysisToolError as exc:
                st.session_state["analysis_errors"]["chart"] = str(exc)
            except Exception:
                logger.exception("Unexpected error while creating chart")
                st.session_state["analysis_errors"]["chart"] = t["unexpected_tool_error"]
        _show_result("chart", t["results"], t)

st.divider()
st.header(t["kb_title"])
st.caption(t["kb_scope_caption"])
st.caption(t["kb_privacy_caption"])
st.caption(t["kb_collection_info"].format(
    knowledge_base=DEFAULT_KNOWLEDGE_BASE_ID, collection=qdrant_settings.collection,
))

if knowledge_runtime_error:
    if knowledge_runtime_error == "embedding_dimension_mismatch":
        st.error(t["kb_dimension_mismatch"])
        if st.button(t["kb_rebuild_index"], key="kb_rebuild_index"):
            try:
                assert qdrant_store is not None
                qdrant_store.rebuild_collection()
                st.session_state.pop("_qdrant_key", None)  # force a clean reconnect
                st.success(t["kb_rebuild_done"])
                st.rerun()
            except Exception:
                logger.exception("Knowledge base rebuild failed")
                st.error(t["kb_rebuild_failed"])
    else:
        st.error(t["kb_unavailable"])
elif not embedding_settings.is_configured:
    issue = embedding_settings.configuration_problem
    st.warning(t["kb_embedding_unavailable"] + (f" ({issue})" if issue else ""))
else:
    builtin_button = st.button(t["kb_index_builtin"], key="index_builtin_knowledge")
    if builtin_button:
        builtins = PROJECT_ROOT / "data" / "knowledge"
        for source_path in sorted([*builtins.glob("*.md"), *builtins.glob("*.txt")]):
            try:
                content = source_path.read_bytes()
                document = load_knowledge_document(
                    source_path.name, content,
                    document_id=f"builtin-{source_path.stem}"[:64],
                    knowledge_base_id=DEFAULT_KNOWLEDGE_BASE_ID,
                    source_type="builtin",
                )
                indexed = knowledge_service.index_document(document)
                st.session_state.setdefault("knowledge_documents_local", {})[document.document_id] = {
                    "document_id": document.document_id, "title": document.title,
                    "source": document.source, "status": indexed.status,
                    "chunk_count": indexed.chunk_count, "indexing_error": indexed.error_code,
                }
                if indexed.status == "indexed":
                    st.success(t["kb_indexed_source"].format(
                        source=document.source, chunks=indexed.chunk_count))
                else:
                    st.error(t["kb_index_failed_code"].format(
                        source=document.source, code=indexed.error_code))
            except Exception as exc:
                logger.warning("Built-in knowledge indexing failed (%s)", type(exc).__name__)
                st.error(t["kb_index_failed_name"].format(name=source_path.name))

    active_context: DatasetContext | None = st.session_state.get("dataset_context")
    current_dataset_id = active_context.dataset_id if active_context else None
    general_scope = GENERAL_SCOPE_SENTINEL
    scope_options = [general_scope]
    if current_dataset_id:
        scope_options.append(current_dataset_id)
    with st.form("form_knowledge_upload", clear_on_submit=True):
        knowledge_file = st.file_uploader(
            t["kb_upload_label"], type=["md", "txt"], key="knowledge_uploader",
        )
        selected_scope = st.selectbox(
            t["kb_scope_label"], scope_options,
            format_func=lambda value: t["kb_scope_general"] if value == general_scope else f"{value[:12]}…",
            key=f"knowledge_scope_{current_dataset_id or 'none'}",
        )
        index_upload = st.form_submit_button(
            t["kb_index_document"], type="primary", disabled=knowledge_service is None,
        )
    if index_upload:
        if knowledge_file is None:
            st.error(t["kb_choose_document"])
        elif knowledge_service is None:
            st.error(t["kb_unavailable_short"])
        else:
            document_id = uuid.uuid4().hex
            filename = _safe_source_name(knowledge_file.name)
            try:
                file_bytes = knowledge_file.getvalue()
                document = load_knowledge_document(
                    filename, file_bytes,
                    document_id=document_id,
                    knowledge_base_id=DEFAULT_KNOWLEDGE_BASE_ID,
                    dataset_id=None if selected_scope == general_scope else selected_scope,
                    source_type="upload",
                )
                source_path = PROJECT_ROOT / "data" / "knowledge" / "uploads" / f"{document_id}{Path(filename).suffix.lower()}"
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_bytes(file_bytes)
                indexed = knowledge_service.index_document(document)
                st.session_state.setdefault("knowledge_documents_local", {})[document.document_id] = {
                    "document_id": document.document_id, "title": document.title,
                    "source": document.source, "status": indexed.status,
                    "chunk_count": indexed.chunk_count, "indexing_error": indexed.error_code,
                }
                if indexed.status == "indexed":
                    st.success(t["kb_indexed_title"].format(
                        title=document.title, chunks=indexed.chunk_count))
                else:
                    st.error(t["kb_index_failed_reason"].format(code=indexed.error_code))
            except KnowledgeLoadError as exc:
                st.error(str(exc))
            except Exception as exc:
                logger.warning("Knowledge document indexing failed (%s)", type(exc).__name__)
                st.error(t["kb_document_index_failed"])

try:
    if persistence_service.knowledge_repository is not None:
        knowledge_documents = persistence_service.knowledge_repository.list_documents(DEFAULT_KNOWLEDGE_BASE_ID)
    else:
        knowledge_documents = list(st.session_state.get("knowledge_documents_local", {}).values())
except Exception as exc:
    logger.warning("Knowledge document listing failed (%s)", type(exc).__name__)
    knowledge_documents = list(st.session_state.get("knowledge_documents_local", {}).values())
if knowledge_documents:
    st.dataframe(pd.DataFrame([
        {t["kb_table_document"]: item["title"], t["kb_table_source"]: item["source"],
         t["kb_table_status"]: item["status"], t["kb_table_chunks"]: item["chunk_count"]}
        for item in knowledge_documents
    ]), width="stretch", hide_index=True)
else:
    st.caption(t["kb_no_documents"])

st.divider()
st.header(t["eval_title"])
try:
    evaluation_cases = load_evaluation_cases()
    evaluation_load_error = False
except (OSError, ValueError, ValidationError):
    logger.exception("Evaluation dataset could not be loaded")
    evaluation_cases = []
    evaluation_load_error = True
if evaluation_load_error:
    st.error(t["eval_start_failed"])
else:
    st.caption(t["eval_caption"].format(count=len(evaluation_cases)))
    judge_requested = st.checkbox(t["eval_judge_checkbox"], value=False, key="evaluation_llm_judge")
    if st.button(t["eval_run"], key="run_evaluation", type="primary"):
        try:
            evaluation_runner = EvaluationRunner(
                evaluation_cases,
                lambda _: AnalystAgent(LLMService(llm_settings, tracing_service=tracing_service), knowledge_service),
                dependency_status={
                    "llm": llm_settings.is_configured,
                    "embedding": embedding_settings.is_configured,
                    "qdrant": knowledge_service is not None and knowledge_runtime_error is None,
                    "mysql": persistence_service.status == "connected",
                },
                tracing_service=tracing_service,
                llm_judge=OptionalLLMJudge(LLMService(llm_settings, tracing_service=tracing_service)),
            )
            st.session_state["evaluation_report"] = evaluation_runner.run(
                enable_llm_judge=judge_requested
            )
        except Exception as exc:
            logger.warning("Evaluation UI could not start (%s)", type(exc).__name__)
            st.error(t["eval_start_failed"])

    evaluation_report = st.session_state.get("evaluation_report")
    if evaluation_report is None:
        if not llm_settings.is_configured:
            st.info(t["eval_not_run_missing_llm"])
        else:
            st.info(t["eval_not_run"])
    else:
        summary = evaluation_report.summary
        count_columns = st.columns(5)
        for column, label_key, value in zip(
            count_columns, ("eval_total", "eval_passed", "eval_failed", "eval_skipped", "eval_unavailable"),
            (summary.total_cases, summary.passed_cases, summary.failed_cases,
             summary.skipped_cases, summary.unavailable_cases), strict=True,
        ):
            column.metric(t[label_key], value)
        pass_text = "N/A" if summary.pass_rate is None else f"{summary.pass_rate:.1%}"
        st.metric(t["eval_pass_rate"], pass_text)

        # Iterate the metrics the runner actually produced instead of a fixed list, so a
        # newly added metric is visible rather than silently dropped from the table.
        metric_rows = []
        for key, metric in summary.metrics.items():
            display = "N/A" if metric.value is None else f"{metric.value:.1%}"
            metric_rows.append({
                t["eval_metric_name"]: t.get(f"eval_metric_{key}", key),
                t["eval_metric_value"]: display,
                t["eval_metric_numerator"]: metric.numerator,
                t["eval_metric_denominator"]: metric.denominator,
            })
        st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)
        judge_status_text = {
            "enabled": t["eval_judge_enabled"],
            "unavailable": t["eval_judge_unavailable"],
            "disabled": t["eval_judge_disabled"],
        }
        st.caption(judge_status_text.get(evaluation_report.llm_judge_status, evaluation_report.llm_judge_status))
        case_rows = [{
            t["eval_case_id"]: item.case_id, t["eval_case_category"]: item.category,
            t["eval_case_status"]: item.status, t["eval_case_question"]: item.question,
            t["eval_case_expected_tools"]: ", ".join(item.expected_tools),
            t["eval_case_actual_tools"]: ", ".join(item.actual_tools),
            t["eval_case_sources"]: ", ".join(item.actual_sources),
            t["eval_case_retrieved"]: ", ".join(item.retrieved_document_ids),
            t["eval_case_iterations"]: item.iterations,
            t["eval_case_failure"]: item.failure_reason or "",
        } for item in evaluation_report.results]
        st.dataframe(pd.DataFrame(case_rows), width="stretch", hide_index=True)
        st.download_button(
            t["eval_download"],
            data=evaluation_report.model_dump_json(indent=2),
            file_name="datapilot-evaluation-report.json", mime="application/json",
            key="download_evaluation_report",
        )

st.divider()
st.header(t["agent_title"])
st.caption(t["agent_description"])
if not llm_settings.is_configured:
    if not llm_settings.has_api_key:
        st.info(t["agent_demo_unavailable"])
    else:
        st.warning(t["agent_config_unavailable"])
else:
    if st.session_state.get("dataset_context") is None:
        st.info(t["agent_dataset_required"])
    with st.form("form_agent"):
        agent_query = st.text_area(t["agent_query"], max_chars=4_000, key="agent_query_input")
        run_agent = st.form_submit_button(t["agent_run"], type="primary")
    if run_agent:
        st.session_state.pop("agent_response", None)
        try:
            agent_request = AgentRequest(
                user_query=agent_query,
                dataset_context=st.session_state.get("dataset_context"),
                knowledge_base_id=DEFAULT_KNOWLEDGE_BASE_ID,
            )
            started_at = time.perf_counter()
            run_result = AnalystAgent(
                LLMService(llm_settings, tracing_service=tracing_service), knowledge_service,
            ).run(agent_request)
            duration_ms = round((time.perf_counter() - started_at) * 1_000)
            st.session_state["agent_response"] = run_result
            if agent_request.dataset_context is not None:
                persistence_service.record_analysis_run(
                    dataset_id=agent_request.dataset_context.dataset_id,
                    question=agent_request.user_query,
                    mode="agent",
                    status="success" if run_result.success else "failed",
                    iterations=run_result.iterations,
                    tools_used=run_result.tools_used,
                    duration_ms=duration_ms,
                )
            if persistence_service.status == "unavailable":
                st.warning(t["database_fallback_note"])
        except ValidationError:
            st.error(t["agent_query_required"])
        except Exception:
            # The agent loop guards itself, but the UI must never show a traceback.
            logger.exception("Agent run failed unexpectedly in the UI")
            st.error(t["agent_unexpected_error"])
    agent_response: AgentResponse | None = st.session_state.get("agent_response")
    if agent_response is not None:
        if agent_response.structured is not None:
            _render_structured_answer(agent_response.structured, t)
        else:
            st.caption(t["structured_missing_note"])
        if agent_response.downgraded_findings:
            st.warning(t["claim_downgraded_note"].format(count=agent_response.downgraded_findings))
        if not agent_response.success:
            st.warning(t["agent_partial_title"])
            st.caption(t["agent_partial_note"])
        (st.markdown if agent_response.success else st.warning)(agent_response.final_answer)
        # Never let an answer written from the model's own knowledge look like
        # knowledge-base evidence.
        if agent_response.retrieval_status == "unavailable":
            st.warning(t["retrieval_unavailable_note"])
        elif agent_response.retrieval_status == "no_results":
            st.warning(t["retrieval_no_results_note"])
        if agent_response.charts:
            st.markdown(f"**{t['result_visualization']}**")
            for chart in agent_response.charts:
                st.pyplot(chart.figure, clear_figure=False)
                for warning in chart.result.warnings:
                    st.warning(warning)
                st.caption(f"{t[chart.result.chart_type]} · {chart.result.plotted_rows:,} {t['plotted_rows']}")
                chart.release()
        if agent_response.tools_used:
            st.caption(f"{t['agent_tools_used']}: {', '.join(agent_response.tools_used)}")
        if agent_response.steps:
            st.markdown(f"**{t['agent_steps']}**")
            for index, step in enumerate(agent_response.steps, start=1):
                st.write(f"{index}. {step.summary}")
                if step.detail:
                    # What the tool was actually asked to compute. Two calls to the same
                    # tool with different arguments looked like an accidental repeat
                    # while only the summary was shown.
                    st.caption(step.detail)
                if step.tool_name == "search_knowledge" and step.tool_output:
                    response_data = (step.tool_output.get("data") or {})
                    for result in response_data.get("results") or []:
                        st.caption(
                            f"{t['agent_source_label']}: {result.get('title')} · {result.get('source')} · "
                            f"score={result.get('score', 0):.3f}"
                        )
        for error in agent_response.errors:
            st.caption(f"{error.code}: {error.message}")

st.header(t["llm_playground"])
st.caption(t["llm_scope"])
if not llm_settings.has_api_key:
    st.info(t["demo_mode_notice"])
elif not llm_settings.is_configured:
    st.warning(t["llm_config_incomplete"])

with st.form("form_llm_prompt"):
    system_prompt = st.text_area(
        t["system_prompt"], value="You are a helpful assistant.", max_chars=12_000, key="llm_system_prompt"
    )
    user_prompt = st.text_area(t["user_prompt"], max_chars=20_000, key="llm_user_prompt")
    temperature = st.number_input(
        t["temperature"], min_value=0.0, max_value=2.0, value=0.2, step=0.1, key="llm_temperature"
    )
    run_llm = st.form_submit_button(t["send_prompt"], type="primary")

if run_llm:
    st.session_state.pop("llm_response", None)
    try:
        request = LLMRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        response = LLMService(llm_settings).generate(request)
        st.session_state["llm_response"] = response
    except ValidationError:
        st.error(t["prompt_required"])
    except LLMConfigurationError:
        logger.warning("LLM configuration is incomplete")
        st.error(t["llm_config_incomplete"])
    except LLMTimeoutError:
        logger.warning("LLM request timed out")
        st.error(t["llm_timeout"])
    except LLMServiceError as exc:
        logger.warning("LLM provider request failed (%s)", type(exc).__name__)
        st.error(t["llm_request_failed"])

llm_response: LLMResponse | None = st.session_state.get("llm_response")
if llm_response is not None:
    if llm_response.mode == "demo":
        st.info(t["demo_response_label"])
    else:
        st.success(t["llm_response_label"])
    st.markdown(llm_response.content)
    st.caption(f"{t['model']}: {llm_response.model} · {t['finish_reason']}: {llm_response.finish_reason or '—'}")
    if llm_response.usage:
        st.caption(
            f"{t['prompt_tokens']}: {llm_response.usage.prompt_tokens or 0} · "
            f"{t['completion_tokens']}: {llm_response.usage.completion_tokens or 0} · "
            f"{t['total_tokens']}: {llm_response.usage.total_tokens or 0}"
        )
