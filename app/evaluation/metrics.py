from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.evaluation.comparator import compare_values
from app.evaluation.models import EvaluationCase, EvaluationResult, MetricValue


def tool_selection_score(expected: Iterable[str], actual: Iterable[str]) -> float | None:
    """Expected tools are a required set; ordering and extra safe tools are ignored."""
    required = set(expected)
    if not required:
        return None
    return len(required.intersection(actual)) / len(required)


def tool_selection_precision(expected: Iterable[str], actual: Iterable[str]) -> float | None:
    """Precision-side signal: share of the called tools that the case expected.

    Recall (`tool_selection_score`) ignores irrelevant extra tools; this metric does not,
    so calling the expected tool plus twenty unrelated tools no longer looks clean.
    Returns None when the actual tool list is empty, or when nothing was expected.
    """
    required = set(expected)
    actual_tools = list(actual)
    if not required or not actual_tools:
        return None
    return len(required.intersection(actual_tools)) / len(actual_tools)


def unnecessary_tool_calls(expected: Iterable[str], actual: Iterable[str]) -> int | None:
    """Count of called tools that the case did not expect; None when nothing was expected."""
    required = set(expected)
    if not required:
        return None
    return sum(1 for tool in actual if tool not in required)


def retrieval_hit_at_k(expected_documents: Iterable[str], expected_chunks: Iterable[str],
                       actual_documents: list[str], actual_chunks: list[str], k: int) -> bool | None:
    expected = set(expected_documents) | set(expected_chunks)
    if not expected:
        return None
    actual = set(actual_documents[:k]) | set(actual_chunks[:k])
    return bool(expected.intersection(actual))


def citation_scores(required_sources: Iterable[str], actual_sources: list[str],
                    retrieved_sources: Iterable[str]) -> tuple[float | None, float | None, bool | None]:
    required = {_norm(item) for item in required_sources if item}
    actual = [_norm(item) for item in actual_sources if item]
    retrieved = {_norm(item) for item in retrieved_sources if item}
    if not required and not actual:
        return None, None, None
    valid = sum(any(_citation_matches(item, source) for source in retrieved) for item in actual)
    precision = valid / len(actual) if actual else (1.0 if not required else 0.0)
    cited_required = sum(any(_citation_matches(item, source) for item in actual) for source in required)
    recall = cited_required / len(required) if required else None
    return precision, recall, bool(len(actual) == valid and (not required or cited_required == len(required)))


def evaluate_case_metrics(case: EvaluationCase, result: EvaluationResult,
                          retrieved_sources: Iterable[str], tool_data: dict[str, Any] | None,
                          answer_text: str = "") -> dict[str, bool | float | int | None]:
    expected_tools = case.expectation.expected_tools
    tool_score = tool_selection_score(expected_tools, result.actual_tools)
    metrics: dict[str, bool | float | int | None] = {
        "tool_selection_score": tool_score,
        "tool_selection_pass": None if tool_score is None else tool_score == 1.0,
        "tool_selection_precision": tool_selection_precision(expected_tools, result.actual_tools),
        "unnecessary_tool_calls": unnecessary_tool_calls(expected_tools, result.actual_tools),
        "retrieval_hit_at_1": retrieval_hit_at_k(case.expectation.expected_document_ids,
            case.expectation.expected_chunk_ids, result.retrieved_document_ids, result.retrieved_chunk_ids, 1),
        "retrieval_hit_at_3": retrieval_hit_at_k(case.expectation.expected_document_ids,
            case.expectation.expected_chunk_ids, result.retrieved_document_ids, result.retrieved_chunk_ids, 3),
        "retrieval_hit_at_5": retrieval_hit_at_k(case.expectation.expected_document_ids,
            case.expectation.expected_chunk_ids, result.retrieved_document_ids, result.retrieved_chunk_ids, 5),
        "retrieval_miss": (not result.retrieved_document_ids
                           if case.expectation.expected_retrieval == "miss" else None),
    }
    precision, recall, citations_valid = citation_scores(
        case.expectation.required_sources, result.actual_sources, retrieved_sources
    )
    metrics.update({"citation_precision": precision, "citation_recall": recall,
                    "citations_valid": citations_valid})
    expectation = case.expectation
    if expectation.ground_truth is not None:
        ground_truth = expectation.ground_truth
        candidate = tool_data.get(ground_truth.tool) if tool_data else None
        actual_parameters = result.actual_tool_inputs.get(ground_truth.tool)
        parameters_match = bool(actual_parameters is not None and
                                _matches_expected_parameters(ground_truth.parameters, actual_parameters))
        try:
            actual = _extract_ground_truth_value(candidate, ground_truth.result_path)
            matched = (parameters_match and compare_values(
                ground_truth.expected, actual, ground_truth.comparator, ground_truth.tolerance
            ))
        except (KeyError, IndexError, TypeError, ValueError):
            matched = False
        metrics["ground_truth_match"] = matched
        metrics["ground_truth_parameters_match"] = parameters_match
        metrics["exact_match"] = matched if ground_truth.comparator == "exact_match" else None
        metrics["numeric_accuracy"] = matched if ground_truth.comparator == "numeric_tolerance" else None
    if expectation.expected_facts:
        # Deterministic evidence proxy: facts must be present verbatim, and required
        # citations must point to retrieved sources. It does not claim semantic truth.
        facts_present = all(_norm(fact) in _norm(answer_text) for fact in expectation.expected_facts)
        metrics["grounded_answer"] = bool(facts_present and citations_valid is not False)
    elif expectation.required_sources:
        # `required_sources` alone cannot measure grounding: `all([])` is True, so an
        # unrelated answer ("I do not know.") used to score as grounded. Citation
        # validity is already reported by `citations_valid`; grounding is not measurable.
        metrics["grounded_answer"] = None
    elif expectation.ground_truth is not None:
        metrics["grounded_answer"] = bool(metrics.get("ground_truth_match") and
                                           expectation.ground_truth.tool in result.actual_tools)
    return metrics


def _is_attempted(result: EvaluationResult) -> bool:
    """Skipped/unavailable cases leave the denominator; an unexpected crash never does."""
    if result.failure_kind == "unexpected":
        return True
    return result.status not in {"skipped", "unavailable"}


def aggregate_metrics(results: list[EvaluationResult]) -> dict[str, MetricValue]:
    names = {
        "tool_selection_accuracy": "tool_selection_pass",
        "tool_selection_precision": "tool_selection_precision",
        "retrieval_hit_at_1": "retrieval_hit_at_1",
        "retrieval_hit_at_3": "retrieval_hit_at_3",
        "retrieval_hit_at_5": "retrieval_hit_at_5",
        "retrieval_miss_rate": "retrieval_miss",
        "citation_precision": "citation_precision",
        "citation_recall": "citation_recall",
        "grounded_answer_rate": "grounded_answer",
        "agent_success_rate": "agent_success",
        "max_iteration_violation_rate": "max_iteration_violation",
        "invalid_tool_rate": "invalid_tool",
        "duplicate_tool_rate": "duplicate_tool_call",
        "exact_match_rate": "exact_match",
        "numeric_accuracy": "numeric_accuracy",
    }
    output: dict[str, MetricValue] = {}
    for public_name, name in names.items():
        values: list[float] = []
        for result in results:
            if not _is_attempted(result):
                continue
            value = result.metric_results.get(name)
            if isinstance(value, bool):
                values.append(float(value))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        output[public_name] = MetricValue(
            numerator=sum(values), denominator=len(values),
            value=sum(values) / len(values) if values else None,
        )
    # Unnecessary tool calls are counts, so the aggregate reports the share of observed
    # tool calls that were unnecessary; the numerator keeps the raw count over 0..1.
    unnecessary_calls: list[tuple[int, int]] = []
    for result in results:
        count = result.metric_results.get("unnecessary_tool_calls")
        if _is_attempted(result) and isinstance(count, int) and not isinstance(count, bool):
            unnecessary_calls.append((count, len(result.actual_tools)))
    total_unnecessary = sum(count for count, _ in unnecessary_calls)
    total_calls = sum(calls for _, calls in unnecessary_calls)
    output["unnecessary_tool_calls"] = MetricValue(
        numerator=float(total_unnecessary), denominator=total_calls,
        value=total_unnecessary / total_calls if total_calls else None,
    )
    # Rates count violations as 1, including when none were observed.
    for public_name, name in (("max_iteration_violation_rate", "max_iteration_violation"),
                              ("invalid_tool_rate", "invalid_tool"),
                              ("duplicate_tool_rate", "duplicate_tool_call")):
        values = [float(bool(result.metric_results[name])) for result in results
                  if _is_attempted(result) and name in result.metric_results]
        output[public_name] = MetricValue(numerator=sum(values), denominator=len(values),
                                          value=sum(values) / len(values) if values else None)
    return output


def _extract_ground_truth_value(tool_envelope: dict[str, Any] | None, path: str) -> Any:
    if tool_envelope is None:
        raise KeyError(path)
    # Dataset expectations point into the tool's structured `data` object.
    from app.evaluation.comparator import extract_path
    return extract_path(tool_envelope.get("data", {}), path)


def _norm(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def _citation_matches(left: str, right: str) -> bool:
    """Match a citation to an exact retrieved source identity, never a substring."""
    left_normalized, right_normalized = _norm(left), _norm(right)
    left_source = left_normalized.rsplit("(", 1)[-1].rstrip(") ") if "(" in left_normalized else left_normalized
    right_source = right_normalized.rsplit("(", 1)[-1].rstrip(") ") if "(" in right_normalized else right_normalized
    return left_normalized == right_normalized or left_source == right_source


def _matches_expected_parameters(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Compare only declared expectation keys, allowing harmless Agent defaults."""
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict) or not _matches_expected_parameters(expected_value, actual_value):
                return False
        elif type(expected_value) is not type(actual_value) or expected_value != actual_value:
            return False
    return True
