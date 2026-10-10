from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from app.models.dataset_context import DatasetContext

# Single source of truth for the agent-loop budget. AgentResponse validates
# against this same constant so the two can never drift apart.
MAX_ITERATIONS = 8

# A finding is either computed from data, an explanation of that data, or an
# unverified guess. The distinction is enforced in code, not only asked for in a
# prompt: a "fact" without a real tool call is downgraded to "hypothesis".
ClaimType = Literal["fact", "interpretation", "hypothesis"]


class Finding(BaseModel):
    """One conclusion, explicitly labelled by how well it is supported."""

    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1, max_length=500)
    claim_type: ClaimType = "interpretation"
    evidence_tool: str | None = Field(default=None, max_length=80)


class FinalAnswer(BaseModel):
    """The structured analysis report shown to the user."""

    model_config = ConfigDict(extra="forbid")

    executive_summary: str = Field(min_length=1, max_length=1_500)
    key_findings: list[Finding] = Field(default_factory=list, max_length=10)
    recommendations: list[str] = Field(default_factory=list, max_length=10)


# Models use different words for the same idea. Normalizing instead of rejecting is
# what keeps a strict fact/hypothesis guarantee from also making every answer fail.
CLAIM_TYPE_ALIASES = {
    "fact": "fact", "facts": "fact", "evidence": "fact", "observed": "fact",
    "data": "fact", "finding": "fact", "事实": "fact", "数据": "fact",
    "interpretation": "interpretation", "inference": "interpretation", "explanation": "interpretation",
    "analysis": "interpretation", "insight": "interpretation", "解释": "interpretation", "解读": "interpretation",
    "hypothesis": "hypothesis", "assumption": "hypothesis", "guess": "hypothesis",
    "speculation": "hypothesis", "possible_cause": "hypothesis", "assume": "hypothesis",
    "假设": "hypothesis", "推测": "hypothesis",
}

# Providers name the same field differently (a real DeepSeek reply used "claim" and
# returned recommendations as {"text": ...} objects). Dropping those silently would
# delete the user's conclusions, so known aliases are resolved before giving up.
STATEMENT_KEYS = ("statement", "claim", "text", "finding", "description", "conclusion",
                  "insight", "summary", "content", "结论")
CLAIM_TYPE_KEYS = ("claim_type", "type", "kind", "category", "claimType")
EVIDENCE_TOOL_KEYS = ("evidence_tool", "tool", "evidence", "source_tool", "evidence_source")
RECOMMENDATION_KEYS = ("text", "recommendation", "action", "suggestion", "advice", "title", "建议")


def _first_text(mapping: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class RawFinding(BaseModel):
    """Lenient inbound shape for one finding; unknown keys are ignored."""

    model_config = ConfigDict(extra="ignore")

    statement: str = Field(default="", max_length=800)
    claim_type: str | None = Field(default=None, max_length=40)
    evidence_tool: str | None = Field(default=None, max_length=80)


class RawFinalAnswer(BaseModel):
    """Lenient inbound shape for the structured block.

    The security-critical part of the contract (action, tool_name, tool_input) stays
    strict. This block is presentation, so a slightly-off answer is normalized rather
    than treated as an invalid model output that discards the whole analysis. Item
    lists are typed ``Any`` on purpose: a provider that emits a number, a nested
    object, or a bare string there must not fail the entire decision.
    """

    model_config = ConfigDict(extra="ignore")

    executive_summary: str = Field(default="", max_length=2_000)
    key_findings: list[Any] = Field(default_factory=list, max_length=20)
    recommendations: list[Any] = Field(default_factory=list, max_length=20)

    @staticmethod
    def _coerce_finding(item: Any) -> tuple[str, str, str | None] | None:
        """Return (statement, claim_type, evidence_tool) or None when unusable."""
        if isinstance(item, str):
            statement = item.strip()
            return (statement, "interpretation", None) if statement else None
        if isinstance(item, RawFinding):
            item = item.model_dump()
        if not isinstance(item, dict):
            return None
        statement = _first_text(item, STATEMENT_KEYS)
        if not statement:
            return None
        raw_type = _first_text(item, CLAIM_TYPE_KEYS).casefold()
        claim_type = CLAIM_TYPE_ALIASES.get(raw_type, "interpretation")
        evidence_tool = _first_text(item, EVIDENCE_TOOL_KEYS) or None
        return statement, claim_type, evidence_tool

    @staticmethod
    def _coerce_recommendation(item: Any) -> str:
        """Recommendations arrive as strings or as objects; accept both."""
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            return _first_text(item, RECOMMENDATION_KEYS)
        return ""

    def normalized(self, fallback_summary: str) -> FinalAnswer:
        """Coerce inbound variants into the strict output contract."""
        findings: list[Finding] = []
        for item in self.key_findings:
            coerced = self._coerce_finding(item)
            if coerced is None:
                continue
            statement, claim_type, evidence_tool = coerced
            findings.append(Finding(
                statement=statement[:500], claim_type=claim_type, evidence_tool=evidence_tool,
            ))
        if self.key_findings and not findings:
            # Never drop conclusions silently: make the loss visible in the logs.
            logging.getLogger(__name__).warning(
                "Structured answer carried %d finding(s) but none matched a known field "
                "name; consider adding the provider's key to STATEMENT_KEYS.", len(self.key_findings),
            )
        summary = self.executive_summary.strip() or fallback_summary.strip() or "（模型未提供总体结论）"
        recommendations = [text for text in
                           (self._coerce_recommendation(item) for item in self.recommendations) if text]
        return FinalAnswer(
            executive_summary=summary[:1_500],
            key_findings=findings[:10],
            recommendations=recommendations[:10],
        )


class AgentAction(BaseModel):
    """Strict JSON contract returned by the model for one agent decision."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["tool_call", "final"]
    tool_name: str | None = Field(default=None, min_length=1, max_length=80)
    tool_input: dict[str, Any] | None = None
    final_answer: str | None = Field(default=None, max_length=8_000)
    analysis: RawFinalAnswer | None = None

    @model_validator(mode="after")
    def validate_action_fields(self):
        if self.action == "tool_call":
            if not self.tool_name or self.tool_input is None or self.final_answer is not None:
                raise ValueError("tool_call requires tool_name and tool_input only")
            if self.analysis is not None:
                raise ValueError("analysis is only allowed for action=final")
        elif self.final_answer is None or not self.final_answer.strip() or self.tool_name is not None or self.tool_input is not None:
            raise ValueError("final requires a non-empty final_answer only")
        return self


class AgentRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    user_query: str = Field(min_length=1, max_length=4_000)
    dataset_context: DatasetContext | None = None
    knowledge_base_id: str = Field(default="datapilot-default", min_length=1, max_length=64)


class AgentStep(BaseModel):
    action_type: Literal["tool_call", "final", "error"]
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: dict[str, Any] | None = None
    summary: str = Field(max_length=300)


class AgentError(BaseModel):
    code: str
    message: str


class AgentResponse(BaseModel):
    final_answer: str
    steps: list[AgentStep] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    success: bool
    errors: list[AgentError] = Field(default_factory=list)
    # Structured report when the model supplied one; None means the narrative could
    # not be broken into verified sections. The UI must say so instead of implying
    # the answer was validated.
    structured: FinalAnswer | None = None
    # How many findings claimed to be facts were downgraded to hypotheses because no
    # matching tool call had run. Surfaced so the UI can state it explicitly.
    downgraded_findings: int = Field(default=0, ge=0)

    # Matplotlib artifacts are binary resources, not data: they are carried outside
    # the serialization boundary so the model context stays JSON-only while the UI can
    # still render the chart the agent asked for.
    _charts: list[Any] = PrivateAttr(default_factory=list)

    @property
    def charts(self) -> list[Any]:
        """Chart artifacts produced by tool calls; never part of model_dump()."""
        return self._charts

    def attach_charts(self, charts: list[Any]) -> None:
        self._charts = list(charts)
    # Number of model decisions, including rejected or malformed decisions.
    # This is structured execution telemetry used by Evaluation and the UI.
    iterations: int = Field(default=0, ge=0, le=MAX_ITERATIONS)
