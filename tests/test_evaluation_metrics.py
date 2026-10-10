from app.evaluation.comparator import compare_values, extract_path
from app.evaluation.metrics import citation_scores, retrieval_hit_at_k, tool_selection_score


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
