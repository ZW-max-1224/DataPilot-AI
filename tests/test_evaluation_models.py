import pytest
from pydantic import ValidationError

from app.evaluation.dataset import load_evaluation_cases
from app.evaluation.models import EvaluationCase, EvaluationExpectation, EvaluationResult


def test_expectations_are_optional_and_model_validates_case():
    case = EvaluationCase(case_id="minimal", category="agent_execution", question="Hi")
    assert case.expectation.expected_tools == []
    assert case.expectation.ground_truth is None
    assert EvaluationExpectation(expected_tools=["find_top_n"]).expected_tools == ["find_top_n"]


def test_invalid_category_and_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        EvaluationCase(case_id="bad", category="arbitrary", question="Hi")
    with pytest.raises(ValidationError):
        EvaluationCase(case_id="bad", category="pandas", question="Hi", eval="__import__('os')")


def test_result_failure_kind_and_metadata_default_to_backward_compatible_values():
    result = EvaluationResult(case_id="legacy", category="pandas", status="failed", question="Hi")
    assert result.failure_kind is None
    assert result.metadata == {}
    crashed = EvaluationResult(case_id="crash", category="pandas", status="failed", question="Hi",
                               failure_kind="unexpected", metadata={"ground_truth_source": "tool-derived"})
    assert crashed.failure_kind == "unexpected"
    assert crashed.metadata["ground_truth_source"] == "tool-derived"
    with pytest.raises(ValidationError):
        EvaluationResult(case_id="bad", category="pandas", status="failed", question="Hi",
                         failure_kind="made_up")


def test_every_case_declares_ground_truth_provenance_and_never_claims_an_oracle():
    cases = load_evaluation_cases()
    assert cases
    for case in cases:
        source = case.metadata.get("ground_truth_source")
        assert isinstance(source, str) and source.strip(), case.case_id
    grounded = [case for case in cases if case.expectation.ground_truth is not None]
    assert grounded
    for case in grounded:
        declared = str(case.metadata["ground_truth_source"])
        assert "tool-derived" in declared, case.case_id
        assert "not an independent oracle" in declared, case.case_id
