import pytest
from pydantic import ValidationError

from app.evaluation.models import EvaluationCase, EvaluationExpectation


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
