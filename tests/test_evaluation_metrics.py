from app.evaluation.comparator import compare_values, extract_path
from app.evaluation.metrics import (
    aggregate_metrics,
    citation_scores,
    evaluate_case_metrics,
    retrieval_hit_at_k,
    tool_selection_score,
)
from app.evaluation.models import EvaluationCase, EvaluationExpectation, EvaluationResult


def _case(case_id="metrics", category="tool_selection", **expectation):
    return EvaluationCase(case_id=case_id, category=category, question="q",
                          dataset="sample_sales", expectation=EvaluationExpectation(**expectation))


def _result(case, **overrides):
    payload = {"case_id": case.case_id, "category": case.category, "status": "passed",
               "question": case.question}
    payload.update(overrides)
    return EvaluationResult(**payload)


def test_tool_selection_is_set_based_and_allows_safe_extra_tools():
    assert tool_selection_score(["find_top_n"], ["get_dataset_overview", "find_top_n"]) == 1
    assert tool_selection_score(["find_top_n", "search_knowledge"], ["find_top_n"]) == 0.5
    assert tool_selection_score([], ["find_top_n"]) is None


def test_retrieval_hit_at_k_handles_rank_empty_and_miss():
    docs = ["other", "second", "target"]
    assert retrieval_hit_at_k(["target"], [], docs, [], 1) is False
    assert retrieval_hit_at_k(["target"], [], docs, [], 3) is True
    assert retrieval_hit_at_k(["target"], [], [], [], 5) is False
    assert retrieval_hit_at_k([], [], [], [], 5) is None


def test_citation_precision_recall_and_fake_source_detection():
    p, r, valid = citation_scores(["metrics.md"], ["Metrics (metrics.md)"], ["metrics.md", "Metrics"])
    assert (p, r, valid) == (1.0, 1.0, True)
    p, r, valid = citation_scores(["metrics.md"], ["fake_metrics.md"], ["metrics.md"])
    assert p == 0 and r == 0 and valid is False


def test_ground_truth_comparators_are_structured_not_string_similarity():
    assert compare_values(12345.67, 12345.6701, "numeric_tolerance", tolerance=0.001)
    assert not compare_values(12345, 13000, "numeric_tolerance", tolerance=0.001)
    assert not compare_values("12345", 12345, "exact_match")
    assert compare_values(["a", "b"], ["b", "a"], "set_match")
    assert extract_path({"rows": [{"x": 3}]}, "rows.0.x") == 3


def test_grounded_answer_is_unmeasurable_when_only_sources_are_required():
    case = _case(case_id="sources_only", category="citation",
                 expected_tools=["search_knowledge"], required_sources=["metrics_glossary.md"])
    result = _result(case, actual_tools=["search_knowledge"],
                     actual_sources=["Metrics Glossary (metrics_glossary.md)"])
    metrics = evaluate_case_metrics(case, result, ["metrics_glossary.md"], {}, "I do not know.")
    assert metrics["citations_valid"] is True
    # `all([])` is True, so required sources alone must never report grounding.
    assert metrics["grounded_answer"] is None
    result.metric_results.update(metrics)
    aggregated = aggregate_metrics([result])
    assert aggregated["grounded_answer_rate"].denominator == 0
    assert aggregated["grounded_answer_rate"].value is None


def test_grounded_answer_still_requires_expected_facts_verbatim():
    case = _case(case_id="facts", category="rag_grounding", expected_facts=["毛利润"])
    result = _result(case)
    present = evaluate_case_metrics(case, result, [], {}, "毛利润 是收入减去成本。")
    assert present["grounded_answer"] is True
    absent = evaluate_case_metrics(case, result, [], {}, "我不知道。")
    assert absent["grounded_answer"] is False


def test_tool_selection_precision_penalizes_expected_plus_twenty_irrelevant_tools():
    from app.evaluation.metrics import tool_selection_precision

    expected = ["find_top_n"]
    actual = ["find_top_n"] + [f"unrelated_{index}" for index in range(20)]
    assert tool_selection_score(expected, actual) == 1.0
    assert tool_selection_precision(expected, actual) == 1 / 21
    assert tool_selection_precision(expected, []) is None
    assert tool_selection_precision([], ["find_top_n"]) is None


def test_evaluate_case_metrics_reports_precision_and_unnecessary_tool_calls():
    case = _case(expected_tools=["find_top_n"])
    actual = ["find_top_n"] + [f"unrelated_{index}" for index in range(20)]
    result = _result(case, actual_tools=actual)
    metrics = evaluate_case_metrics(case, result, [], {}, "answer")
    assert metrics["tool_selection_score"] == 1.0
    assert metrics["tool_selection_pass"] is True
    assert metrics["tool_selection_precision"] == 1 / 21
    assert metrics["unnecessary_tool_calls"] == 20
    result.metric_results.update(metrics)
    aggregated = aggregate_metrics([result])
    assert aggregated["tool_selection_precision"].value == 1 / 21
    assert aggregated["unnecessary_tool_calls"].numerator == 20
    assert aggregated["unnecessary_tool_calls"].denominator == 21
    assert aggregated["unnecessary_tool_calls"].value == 20 / 21


def test_aggregate_metrics_never_skips_an_unexpected_crash():
    # The runner records a crash as status "failed" plus failure_kind "unexpected";
    # this guards the filter so such a case can never be excluded as a skipped dependency.
    crashed = EvaluationResult(
        case_id="crash", category="pandas", status="unavailable", question="q",
        failure_kind="unexpected", actual_tools=["find_top_n", "find_top_n"],
        metric_results={"agent_success": False, "unnecessary_tool_calls": 1},
    )
    aggregated = aggregate_metrics([crashed])
    assert aggregated["agent_success_rate"].denominator == 1
    assert aggregated["agent_success_rate"].value == 0.0
    assert aggregated["unnecessary_tool_calls"].denominator == 2

