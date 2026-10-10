from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from app.agent import AgentRequest, AnalystAgent
from app.config.settings import EmbeddingSettings, LLMSettings, QdrantSettings
from app.evaluation.metrics import aggregate_metrics, evaluate_case_metrics
from app.evaluation.judge import OptionalLLMJudge
from app.evaluation.models import EvaluationCase, EvaluationReport, EvaluationResult, EvaluationSummary
from app.models.dataset_context import DatasetContext
from app.services.profiler import profile_dataframe
from app.services.tracing_service import TracingService

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_AGENT_ITERATIONS = 8


class EvaluationRunner:
    """Execute fixed evaluation cases through AnalystAgent's public `run` method."""

    def __init__(
        self,
        cases: list[EvaluationCase],
        agent_factory: Callable[[EvaluationCase], AnalystAgent],
        *,
        dependency_status: Mapping[str, bool] | None = None,
        dataset_resolver: Callable[[str | None], DatasetContext | None] | None = None,
        tracing_service: TracingService | None = None,
        llm_judge: OptionalLLMJudge | None = None,
    ) -> None:
        self.cases = cases
        self.agent_factory = agent_factory
        self.dependencies = dict(dependency_status or self._configured_dependencies())
        self.dataset_resolver = dataset_resolver or _resolve_bundled_dataset
        self.tracing = tracing_service or TracingService()
        self.llm_judge = llm_judge

    @staticmethod
    def _configured_dependencies() -> dict[str, bool]:
        return {
            "llm": LLMSettings.from_env().is_configured,
            "embedding": EmbeddingSettings.from_env().is_configured,
            # Local persistent Qdrant is part of the app; it is verified when
            # the UI creates KnowledgeService and can be overridden by callers.
            "qdrant": True,
            "mysql": False,
        }

    def run(self, *, case_ids: set[str] | None = None, enable_llm_judge: bool = False) -> EvaluationReport:
        selected = [case for case in self.cases if case_ids is None or case.case_id in case_ids]
        judge_enabled = bool(enable_llm_judge and self.llm_judge and self.llm_judge.is_available)
        results: list[EvaluationResult] = []
        with self.tracing.span("DataPilot Evaluation", run_type="chain",
                inputs={"dataset": "datapilot_mvp_v1", "case_count": len(selected),
                        "case_ids": [case.case_id for case in selected]}) as run_span:
            for case in selected:
                results.append(self._run_case(case, enable_llm_judge=judge_enabled))
            summary = _build_summary(results)
            run_span.end({"total_cases": summary.total_cases, "passed": summary.passed_cases,
                          "failed": summary.failed_cases, "skipped": summary.skipped_cases,
                          "unavailable": summary.unavailable_cases,
                          "metrics": {name: metric.value for name, metric in summary.metrics.items()}})
        return EvaluationReport(dataset_name="datapilot_mvp_v1", summary=summary, results=results,
                                llm_judge_enabled=judge_enabled,
                                llm_judge_status=("enabled" if judge_enabled else
                                                  "unavailable" if enable_llm_judge else "disabled"))

    def _run_case(self, case: EvaluationCase, *, enable_llm_judge: bool = False) -> EvaluationResult:
        started = time.perf_counter()
        missing = [name for name in case.requires if not self.dependencies.get(name, False)]
        missing_optional_services = [name for name in missing if name in {"embedding", "qdrant"}]
        if missing_optional_services:
            return self._not_run(case, "skipped", f"{', '.join(missing_optional_services)} service unavailable")
        if missing:
            return self._not_run(case, "unavailable", f"{', '.join(missing)} service unavailable")

        query_fingerprint = hashlib.sha256(case.question.encode("utf-8")).hexdigest()
        try:
            with self.tracing.span("Evaluation Case", run_type="chain",
                    inputs={"case_id": case.case_id, "category": case.category,
                            "question_sha256": query_fingerprint, "question_chars": len(case.question)}) as span:
                context = self.dataset_resolver(case.dataset)
                response = self.agent_factory(case).run(AgentRequest(
                    user_query=case.question,
                    dataset_context=context,
                    knowledge_base_id="datapilot-default",
                ))
                result = self._collect_result(case, response, started)
                sources = _retrieved_source_labels(result.retrieval_results)
                tool_data = {
                    step.tool_name: step.tool_output
                    for step in response.steps if step.tool_name and step.tool_output
                }
                metrics = evaluate_case_metrics(case, result, sources, tool_data, response.final_answer)
                result.metric_results.update(metrics)
                if enable_llm_judge and self.llm_judge:
                    evidence = [
                        {key: item.get(key) for key in ("document_id", "title", "source", "content") if item.get(key)}
                        for step in response.steps if step.tool_name == "search_knowledge" and step.tool_output
                        for item in ((step.tool_output.get("data") or {}).get("results") or [])
                        if isinstance(item, dict)
                    ]
                    try:
                        judge_result = self.llm_judge.evaluate(
                            question=case.question,
                            criteria=case.expectation.expected_facts or case.expectation.required_sources,
                            answer=response.final_answer,
                            evidence=evidence,
                        )
                        result.judge_status = "completed"
                        result.judge_output = judge_result.model_dump(mode="json")
                        result.metric_results["llm_judge_grounded"] = judge_result.grounded
                    except Exception as exc:
                        logger.info("Optional evaluation judge unavailable (%s)", type(exc).__name__)
                        result.judge_status = "unavailable"
                result.metric_results["agent_success"] = response.success
                result.metric_results["max_iteration_violation"] = response.iterations > MAX_AGENT_ITERATIONS
                result.metric_results["invalid_tool"] = result.invalid_tool
                result.metric_results["duplicate_tool_call"] = result.duplicate_tool_call
                result.status, result.failure_reason = _case_outcome(case, response, result)
                span.end({"status": result.status, "tools": result.actual_tools,
                          "retrieved_document_ids": result.retrieved_document_ids,
                          "retrieved_chunk_ids": result.retrieved_chunk_ids,
                          "iterations": result.iterations, "metrics": result.metric_results,
                          "judge_status": result.judge_status,
                          "duration_ms": result.duration_ms})
                return result
        except Exception as exc:
            logger.warning("Evaluation case %s unavailable (%s)", case.case_id, type(exc).__name__)
            return self._not_run(case, "unavailable", f"case execution unavailable ({type(exc).__name__})",
                                 duration_ms=_duration(started))

    @staticmethod
    def _collect_result(case: EvaluationCase, response: Any, started: float) -> EvaluationResult:
        retrieval_results: list[dict[str, Any]] = []
        for step in response.steps:
            if step.tool_name != "search_knowledge" or not step.tool_output:
                continue
            data = step.tool_output.get("data") or {}
            for item in data.get("results", []) if isinstance(data, dict) else []:
                if isinstance(item, dict):
                    retrieval_results.append({key: item.get(key) for key in (
                        "document_id", "chunk_id", "title", "source", "score"
                    ) if item.get(key) is not None})

        invalid_tool = any(error.code == "tool_not_allowed" for error in response.errors)
        signatures: list[tuple[str, str]] = []
        duplicate = any(error.code in {"duplicate_knowledge_search", "knowledge_search_limit"}
                        for error in response.errors)
        for step in response.steps:
            if not step.tool_name:
                continue
            signature = (step.tool_name, json.dumps(step.tool_input or {}, sort_keys=True,
                                                    ensure_ascii=False, separators=(",", ":")))
            if signature in signatures:
                duplicate = True
            signatures.append(signature)
        actual_sources = _parse_citations(response.final_answer)
        documents = list(dict.fromkeys(str(item["document_id"]) for item in retrieval_results
                                       if item.get("document_id")))
        chunks = list(dict.fromkeys(str(item["chunk_id"]) for item in retrieval_results
                                    if item.get("chunk_id")))
        return EvaluationResult(
            case_id=case.case_id, category=case.category, status="failed", question=case.question,
            expected_tools=case.expectation.expected_tools,
            actual_tools=list(response.tools_used), expected_sources=case.expectation.required_sources,
            actual_tool_inputs={step.tool_name: dict(step.tool_input or {})
                                for step in response.steps if step.tool_name},
            actual_sources=actual_sources, retrieved_document_ids=documents,
            retrieved_chunk_ids=chunks, retrieval_results=retrieval_results,
            iterations=int(getattr(response, "iterations", len(response.steps))),
            agent_success=bool(response.success), duration_ms=_duration(started),
            invalid_tool=invalid_tool, duplicate_tool_call=duplicate,
        )

    @staticmethod
    def _not_run(case: EvaluationCase, status: str, reason: str, duration_ms: int = 0) -> EvaluationResult:
        return EvaluationResult(
            case_id=case.case_id, category=case.category, status=status, question=case.question,
            expected_tools=case.expectation.expected_tools, expected_sources=case.expectation.required_sources,
            duration_ms=duration_ms, failure_reason=reason,
        )


def _case_outcome(case: EvaluationCase, response: Any, result: EvaluationResult) -> tuple[str, str | None]:
    failures: list[str] = []
    expected_success = case.expectation.expect_success
    if expected_success is not None and bool(response.success) != expected_success:
        failures.append("Agent success did not match the expected outcome")
    score = result.metric_results.get("tool_selection_score")
    if score is not None and score < 1:
        failures.append("Expected tool set was not fully selected")
    if case.expectation.expected_retrieval:
        if case.expectation.expected_retrieval == "hit" and not result.metric_results.get("retrieval_hit_at_5"):
            failures.append("Expected relevant document was not retrieved in top 5")
        if case.expectation.expected_retrieval == "miss" and result.retrieved_document_ids:
            failures.append("Expected no relevant retrieval, but the retriever returned documents")
    if case.expectation.required_sources:
        if result.metric_results.get("citations_valid") is not True:
            failures.append("Required source citation is missing or was not retrieved")
    if case.expectation.ground_truth is not None and result.metric_results.get("ground_truth_match") is not True:
        failures.append("Pandas tool result did not match deterministic ground truth")
    if result.iterations > MAX_AGENT_ITERATIONS:
        failures.append("Agent exceeded the configured iteration limit")
    if result.invalid_tool:
        failures.append("Agent attempted an unregistered tool")
    if result.duplicate_tool_call:
        failures.append("Agent repeated a tool call or knowledge search")
    if result.metric_results.get("grounded_answer") is False:
        failures.append("Answer did not satisfy deterministic evidence checks")
    return ("failed", "; ".join(failures)) if failures else ("passed", None)


def _build_summary(results: list[EvaluationResult]) -> EvaluationSummary:
    passed = sum(result.status == "passed" for result in results)
    failed = sum(result.status == "failed" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    unavailable = sum(result.status == "unavailable" for result in results)
    attempted = passed + failed
    return EvaluationSummary(
        total_cases=len(results), passed_cases=passed, failed_cases=failed,
        skipped_cases=skipped, unavailable_cases=unavailable,
        pass_rate=passed / attempted if attempted else None,
        metrics=aggregate_metrics(results),
    )


def _parse_citations(answer: str) -> list[str]:
    lines = answer.splitlines()
    active = False
    citations: list[str] = []
    for line in lines:
        if line.strip().casefold().rstrip(":：") in {"sources", "来源"}:
            active = True
            continue
        if active:
            stripped = line.strip()
            if stripped.startswith(("-", "*")):
                value = stripped[1:].strip()
                if value:
                    citations.append(value)
            elif stripped:
                break
    return citations


def _retrieved_source_labels(results: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for item in results:
        for key in ("title", "source", "document_id", "chunk_id"):
            value = item.get(key)
            if value and value not in labels:
                labels.append(str(value))
        title, source = item.get("title"), item.get("source")
        if title and source:
            labels.append(f"{title} ({source})")
    return labels


def _resolve_bundled_dataset(name: str | None) -> DatasetContext | None:
    """Only expose the versioned, synthetic sample dataset to evaluation cases."""
    if name is None:
        return None
    if name != "sample_sales":
        raise ValueError("Evaluation dataset alias is not allowlisted.")
    path = PROJECT_ROOT / "data" / "sample_sales.csv"
    frame = pd.read_csv(path)
    return DatasetContext(
        dataset_id="evaluation-sample-sales", filename=path.name, dataframe=frame,
        profile=profile_dataframe(frame, path.name),
    )


def _duration(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1_000))
