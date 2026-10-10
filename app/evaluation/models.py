from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CaseCategory = Literal[
    "tool_selection", "pandas", "rag_retrieval", "rag_grounding",
    "citation", "agent_execution", "combined",
]
CaseStatus = Literal["passed", "failed", "skipped", "unavailable"]
# Why a case did not pass, without changing the `status` strings the UI renders:
# "expectation" = the agent ran but missed a deterministic expectation,
# "dependency"  = a required optional service was missing (excluded from pass rate),
# "unexpected"  = the code raised an unhandled exception (counts as a real failure).
FailureKind = Literal["expectation", "dependency", "unexpected"]
Comparator = Literal["exact_match", "numeric_tolerance", "set_match"]


class GroundTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1, max_length=80)
    parameters: dict[str, Any] = Field(default_factory=dict)
    result_path: str = Field(min_length=1, max_length=300)
    expected: Any
    comparator: Comparator = "exact_match"
    tolerance: float = Field(default=1e-6, ge=0)


class EvaluationExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_tools: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    required_sources: list[str] = Field(default_factory=list)
    expected_retrieval: Literal["hit", "miss"] | None = None
    ground_truth: GroundTruth | None = None
    expect_success: bool | None = None


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=80)
    category: CaseCategory
    question: str = Field(min_length=1, max_length=4_000)
    dataset: str | None = Field(default="sample_sales", max_length=100)
    expectation: EvaluationExpectation = Field(default_factory=EvaluationExpectation)
    requires: list[Literal["llm", "embedding", "qdrant", "mysql"]] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class MetricValue(BaseModel):
    numerator: float = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float | None = Field(default=None, ge=0, le=1)


class EvaluationResult(BaseModel):
    case_id: str
    category: CaseCategory
    status: CaseStatus
    # Optional and backward compatible: old reports simply deserialize to None.
    failure_kind: FailureKind | None = None
    question: str
    expected_tools: list[str] = Field(default_factory=list)
    actual_tools: list[str] = Field(default_factory=list)
    actual_tool_inputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    expected_sources: list[str] = Field(default_factory=list)
    actual_sources: list[str] = Field(default_factory=list)
    retrieved_document_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieval_results: list[dict[str, Any]] = Field(default_factory=list)
    iterations: int = Field(default=0, ge=0)
    agent_success: bool | None = None
    duration_ms: int = Field(default=0, ge=0)
    invalid_tool: bool = False
    duplicate_tool_call: bool = False
    judge_status: Literal["disabled", "completed", "unavailable"] = "disabled"
    judge_output: dict[str, Any] | None = None
    metric_results: dict[str, bool | float | int | None] = Field(default_factory=dict)
    failure_reason: str | None = None
    # Copied from the case so reports surface provenance without the UI reading the dataset.
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    total_cases: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    skipped_cases: int = Field(ge=0)
    unavailable_cases: int = Field(ge=0)
    pass_rate: float | None = Field(default=None, ge=0, le=1)
    metrics: dict[str, MetricValue] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    dataset_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: EvaluationSummary
    results: list[EvaluationResult] = Field(default_factory=list)
    llm_judge_enabled: bool = False
    llm_judge_status: Literal["disabled", "enabled", "unavailable"] = "disabled"
