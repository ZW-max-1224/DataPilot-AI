import json

from app.evaluation.dataset import DATASET_PATH, load_evaluation_cases


def test_bundled_dataset_covers_required_scenarios_and_unique_ids():
    cases = load_evaluation_cases()
    assert 10 <= len(cases) <= 20
    assert len({case.case_id for case in cases}) == len(cases)
    assert sum(case.category == "pandas" for case in cases) >= 6
    assert sum(case.category == "rag_retrieval" for case in cases) >= 3
    assert any(case.category == "combined" for case in cases)
    assert any(case.expectation.ground_truth for case in cases)


def test_dataset_has_no_secrets_or_executable_payload_fields():
    content = DATASET_PATH.read_text(encoding="utf-8").casefold()
    payload = json.loads(content)
    assert "api_key" not in content and "embedding_key" not in content
    assert "dataframe" not in content and "full_prompt" not in content
    assert all(set(case) <= {"case_id", "category", "question", "dataset", "requires", "expectation", "metadata"}
               for case in payload["cases"])


def test_runner_only_accepts_fixed_dataset_aliases():
    from app.evaluation.runner import _resolve_bundled_dataset
    assert _resolve_bundled_dataset(None) is None
    assert _resolve_bundled_dataset("sample_sales").dataframe.shape[0] == 1_208


def test_all_bundled_pandas_ground_truth_values_match_registered_tools():
    from app.evaluation.comparator import compare_values, extract_path
    from app.evaluation.runner import _resolve_bundled_dataset
    from app.tools.registry import invoke_analysis_tool

    context = _resolve_bundled_dataset("sample_sales")
    for case in load_evaluation_cases():
        ground_truth = case.expectation.ground_truth
        if ground_truth is None:
            continue
        result = invoke_analysis_tool(context, ground_truth.tool, **ground_truth.parameters)
        assert result.success, case.case_id
        actual = extract_path(result.data, ground_truth.result_path)
        assert compare_values(ground_truth.expected, actual, ground_truth.comparator,
                              ground_truth.tolerance), case.case_id
