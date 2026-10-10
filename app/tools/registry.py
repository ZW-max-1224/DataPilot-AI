import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.models.analysis_result import (
    ChartRequest, CorrelationRequest, DuplicateRequest, FindTopNRequest,
    GroupStatisticsRequest, OutlierRequest, TimeSeriesRequest, ToolError,
    ToolExecutionResult,
)
from app.tools.analysis_tools import (
    AnalysisToolError, analyze_duplicates, analyze_missing_values, calculate_correlation,
    calculate_group_statistics, calculate_time_series, detect_outliers, find_top_n,
    get_dataset_overview,
)
from app.tools.visualization_tools import ChartArtifact, create_chart

logger = logging.getLogger(__name__)


class EmptyToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchKnowledgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=5)


@dataclass(frozen=True)
class AnalysisToolSpec:
    """A registered tool and the schema that validates its public arguments."""

    name: str
    description: str
    input_model: type[BaseModel]
    function: Callable[..., Any]
    requires_dataset: bool = True


_TOOL_SPECS = {
    "get_dataset_overview": AnalysisToolSpec(
        "get_dataset_overview", "Summarize the active dataset size, semantic columns, missing values, and duplicates.",
        EmptyToolInput, get_dataset_overview,
    ),
    "find_top_n": AnalysisToolSpec(
        "find_top_n", "Return the top or bottom N rows by a column, optionally summing a numeric metric by group.",
        FindTopNRequest, find_top_n,
    ),
    "calculate_group_statistics": AnalysisToolSpec(
        "calculate_group_statistics", "Aggregate a numeric measure by a categorical field using sum, mean, median, min, max, or count.",
        GroupStatisticsRequest, calculate_group_statistics,
    ),
    "calculate_time_series": AnalysisToolSpec(
        "calculate_time_series", "Aggregate a numeric measure by day, week, month, quarter, or year.",
        TimeSeriesRequest, calculate_time_series,
    ),
    "analyze_missing_values": AnalysisToolSpec(
        "analyze_missing_values", "Report missing-value counts and rates for dataset columns.",
        EmptyToolInput, analyze_missing_values,
    ),
    "analyze_duplicates": AnalysisToolSpec(
        "analyze_duplicates", "Count exact duplicate rows and return a small bounded set of examples.",
        DuplicateRequest, analyze_duplicates,
    ),
    "detect_outliers": AnalysisToolSpec(
        "detect_outliers", "Detect numeric outliers using IQR fences.", OutlierRequest, detect_outliers,
    ),
    "calculate_correlation": AnalysisToolSpec(
        "calculate_correlation", "Calculate Pearson correlation for two numeric columns; correlation is not causation.",
        CorrelationRequest, calculate_correlation,
    ),
    "create_chart": AnalysisToolSpec(
        "create_chart", "Create a supported line, bar, scatter, or histogram chart from dataset columns.",
        ChartRequest, create_chart,
    ),
    "search_knowledge": AnalysisToolSpec(
        "search_knowledge", "Search approved analysis methods and business definitions for supporting evidence.",
        SearchKnowledgeInput, lambda **_: None, requires_dataset=False,
    ),
}
ANALYSIS_TOOL_SPECS: Mapping[str, AnalysisToolSpec] = MappingProxyType(_TOOL_SPECS)
ALLOWED_ANALYSIS_TOOLS: Mapping[str, Callable[..., Any]] = MappingProxyType(
    {name: spec.function for name, spec in ANALYSIS_TOOL_SPECS.items()}
)


def get_analysis_tool_spec(name: str) -> AnalysisToolSpec:
    """Return metadata for an explicitly registered tool by exact name."""
    try:
        return ANALYSIS_TOOL_SPECS[name]
    except KeyError as exc:
        raise AnalysisToolError(f"Tool '{name}' is not in the analysis allowlist.") from exc


def describe_analysis_tools() -> list[dict[str, Any]]:
    """Return only the public descriptions and JSON schemas for registered tools."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_model.model_json_schema(),
        }
        for spec in ANALYSIS_TOOL_SPECS.values()
    ]


def get_analysis_tool(name: str) -> Callable[..., Any]:
    """Return a registered function by exact name, rejecting everything else."""
    return get_analysis_tool_spec(name).function


def _json_result(value: Any, artifacts: list[Any] | None = None) -> dict[str, Any]:
    if isinstance(value, ChartArtifact):
        if artifacts is not None:
            # The caller will render the figure and release it afterwards.
            artifacts.append(value)
        else:
            # Nobody can render this figure through a JSON-only boundary, so release
            # it now instead of leaking it into pyplot's global registry.
            value.release()
        return value.result.model_dump(mode="json")
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError("Registered tools must return a Pydantic result model.")


def invoke_analysis_tool(
    dataset,
    name: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    artifacts: list[Any] | None = None,
    knowledge_service=None,
    trusted_knowledge_base_id: str = "datapilot-default",
    **legacy_parameters: Any,
) -> ToolExecutionResult:
    """Validate and invoke one allowlisted tool, returning a safe JSON envelope.

    Untrusted (model-supplied) arguments MUST be passed as the ``parameters``
    mapping. Never splat them as keyword arguments: a key such as ``name``,
    ``dataset``, ``knowledge_service`` or ``trusted_knowledge_base_id`` would
    collide with this function's own parameters and raise ``TypeError`` while
    Python binds the call, i.e. before the validation below can ever run.

    ``legacy_parameters`` remains only for trusted local callers (tests, scripts)
    that pass literal keyword arguments.

    Charts are binary artifacts, not JSON: when a caller passes a mutable
    ``artifacts`` list, the ``ChartArtifact`` is appended to it so the caller can
    render (and release) it, while the model context receives metadata only.
    """
    argument_source: dict[str, Any] = dict(parameters or {})
    if legacy_parameters:
        argument_source.update(legacy_parameters)
    try:
        spec = get_analysis_tool_spec(name)
        validated = spec.input_model.model_validate(argument_source)
        if spec.requires_dataset and dataset is None:
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                error=ToolError(code="dataset_missing", message="This analysis tool requires an active dataset."),
            )
        if name == "search_knowledge":
            if knowledge_service is None:
                return ToolExecutionResult(
                    tool_name=name,
                    success=True,
                    data={"status": "knowledge_base_unavailable", "results": [], "result_count": 0,
                          "error_code": "knowledge_base_unavailable"},
                )
            response = knowledge_service.search(
                validated.query,
                knowledge_base_id=trusted_knowledge_base_id,
                dataset_id=getattr(dataset, "dataset_id", None),
                top_k=validated.top_k,
            )
            # A retrieval status is an observation, not a tool execution crash. The
            # Agent may continue with Pandas or clearly explain the unavailable source.
            return ToolExecutionResult(tool_name=name, success=True, data=response.model_dump(mode="json"))
        frame = getattr(dataset, "dataframe", dataset)
        if name == "get_dataset_overview" and hasattr(dataset, "profile"):
            result = spec.function(frame, dataset.filename, profile=dataset.profile,
                                   **validated.model_dump(exclude_none=True))
        else:
            result = spec.function(frame, **validated.model_dump(exclude_none=True))
        return ToolExecutionResult(tool_name=name, success=True, data=_json_result(result, artifacts))
    except ValidationError:
        return ToolExecutionResult(
            tool_name=name, success=False,
            error=ToolError(code="invalid_tool_request", message="Tool arguments do not match the registered schema."),
        )
    except AnalysisToolError as exc:
        return ToolExecutionResult(
            tool_name=name, success=False, error=ToolError(code="invalid_tool_request", message=str(exc))
        )
    except TypeError:
        return ToolExecutionResult(
            tool_name=name, success=False,
            error=ToolError(code="invalid_tool_request", message="Tool parameters do not match the registered schema."),
        )
    except Exception as exc:
        logger.warning("Registered analysis tool failed (%s): %s", name, type(exc).__name__)
        return ToolExecutionResult(
            tool_name=name, success=False,
            error=ToolError(code="tool_execution_error", message="The registered tool could not complete the request."),
        )
