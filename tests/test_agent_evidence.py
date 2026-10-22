"""Regression checks for false facts, scoped references, and deterministic rendering."""
import json

import pandas as pd
import pytest

from app.agent import AnalystAgent, AgentRequest
from app.agent.agent_models import RawFinalAnswer
from app.agent.evidence import ToolEvidence
from app.config.settings import LangSmithSettings
from app.models.dataset_context import DatasetContext
from app.models.llm_models import LLMResponse
from app.services.profiler import profile_dataframe
from app.services.tracing_service import TracingService


class MockLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.tracing = TracingService(LangSmithSettings(tracing_enabled=False))

    def generate(self, request):
        self.requests.append(request)
        return LLMResponse(content=json.dumps(self.responses.pop(0)), model="mock", mode="llm")


def reference(value=30, path="/rows/0/sales", call_id="call_1", **extra):
    return {"call_id": call_id, "result_path": path, "value": value, **extra}


def finding(evidence, statement="A has sales of 999999 dollars.", **extra):
    return {"statement": statement, "claim_type": "fact", "evidence_tool": "find_top_n",
            "evidence": evidence, **extra}


def final(*findings):
    return {"action": "final", "final_answer": "Model narrative.",
            "analysis": {"executive_summary": "Model summary.", "key_findings": list(findings)}}


def tool(column="sales", n=1):
    return {"action": "tool_call", "tool_name": "find_top_n", "tool_input": {"column": column, "n": n}}


@pytest.fixture
def context():
    frame = pd.DataFrame({"product": ["A", "B", "C"], "sales": [10, 30, 20]})
    return DatasetContext("test", "sales.csv", frame, profile_dataframe(frame, "sales.csv"))


def checked(item, data=None, tool_name="find_top_n"):
    calls = {"call_1": ToolEvidence(tool_name, data if data is not None else {
        "rows": [{"product": "B", "sales": 30}], "result_count": 1,
    })}
    answer, downgraded = AnalystAgent._validate_structured(RawFinalAnswer(key_findings=[item]), calls)
    return answer.key_findings[0], downgraded


def test_real_tool_name_alone_no_longer_certifies_an_invented_number(context):
    response = AnalystAgent(MockLLM([tool(), final(finding([]))])).run(
        AgentRequest(user_query="Top product?", dataset_context=context)
    )
    assert response.success
    assert response.tools_used == ["find_top_n"]
    assert response.downgraded_findings == 1
    assert response.structured.key_findings[0].claim_type == "hypothesis"


def test_facts_are_rendered_from_checked_fields_not_misleading_model_prose(context):
    llm = MockLLM([tool(), final(finding([
        reference("B", "/rows/0/product"), reference(),
    ]))])
    response = AnalystAgent(llm).run(AgentRequest(user_query="Top product?", dataset_context=context))
    actual = response.structured.key_findings[0]
    assert response.success and response.downgraded_findings == 0
    assert actual.claim_type == "fact"
    assert actual.statement == 'call_1 /rows/0/product = "B"; call_1 /rows/0/sales = 30'
    assert "999999" not in actual.statement and "dollars" not in actual.statement
    observation = json.loads(llm.requests[1].user_prompt)["tool_history"][0]
    assert observation["call_id"] == response.steps[0].call_id == "call_1"
    assert observation["parameters"] == response.steps[0].tool_input == {
        "column": "sales", "n": 1, "ascending": False,
    }


@pytest.mark.parametrize("bad_reference", [
    reference(999999),
    reference(20),  # A value that exists in another row is not evidence for this row.
    reference(30, call_id="call_2"),
    reference(30, path="/rows/4/sales"),
    reference(30, path="/rows/-1/sales"),
    reference(30, path="/rows/00/sales"),
    reference(30, path="/rows/0/__class__"),
    reference(30, path="/rows/0/sales~2"),
    reference(30, path="/rows/0/sales.__class__"),
    reference("30"),
    reference(True),
    reference(30, unit="percent"),
    reference(30, unit="count"),
    reference(30, unit="CNY"),
    reference(30, decimal_places=-1),
    reference(30, decimal_places=7),
    reference(30, decimal_places=True),
    reference(30, value_type="number"),  # Unknown evidence keys fail closed.
    {"call_id": "call_1", "result_path": "/rows/0/sales"},
    reference(None),
    reference(float("nan")),
    reference(float("inf")),
])
def test_invalid_or_mismatched_references_are_downgraded_without_crashing(bad_reference):
    actual, count = checked(finding([bad_reference]))
    assert count == 1
    assert actual.claim_type == "hypothesis"
    assert actual.evidence == [] and actual.evidence_tool is None


def test_one_bad_reference_invalidates_the_whole_claim():
    actual, count = checked(finding([reference(), reference(999999)]))
    assert count == 1 and actual.claim_type == "hypothesis"


def test_wrong_tool_name_cannot_relabel_an_existing_call():
    actual, count = checked(finding([reference()], evidence_tool="calculate_group_statistics"))
    assert count == 1 and actual.evidence_tool is None


def test_result_paths_support_literal_slashes_tildes_dots_and_numeric_keys():
    actual, count = checked(finding([reference("0012", path="/rows/0/a~1b~0c.d/0")]),
                            {"rows": [{"a/b~c.d": {"0": "0012"}}]})
    assert count == 0 and actual.claim_type == "fact"
    assert '"0012"' in actual.statement


@pytest.mark.parametrize("actual_value,expected,places,accepted", [
    (1.005, 1.01, 2, True),
    (-1.005, -1.01, 2, True),
    (1.005, 1.00, 2, False),
    (1.005, 1.01, None, False),
    (1234.56789, 1234.56789, None, True),
    (1234.56789, 1234.567, 3, False),
    (1234.56789, 1234.568, 3, True),
    (1e20, 1e20 + 1e6, None, False),
    (30, 30.0, None, True),
    (float("nan"), 0, None, False),
    (float("inf"), 0, None, False),
    (True, 1, None, False),
    (None, 0, None, False),
    ("0012", 12, None, False),
])
def test_numeric_comparison_and_explicit_rounding(actual_value, expected, places, accepted):
    actual, count = checked(finding([reference(expected, decimal_places=places)]),
                            {"rows": [{"sales": actual_value}]})
    assert (actual.claim_type == "fact") is accepted
    assert count == (0 if accepted else 1)


@pytest.mark.parametrize("tool_name,data,path,value,unit,accepted", [
    ("detect_outliers", {"outlier_percentage": 5}, "/outlier_percentage", 5, "percent", True),
    ("detect_outliers", {"outlier_percentage": 5}, "/outlier_percentage", .05, "percent", False),
    ("detect_outliers", {"outlier_percentage": 5}, "/outlier_percentage", 5, "source", False),
    ("detect_outliers", {"outlier_count": 5}, "/outlier_count", 5, "count", True),
    ("find_top_n", {"rows": [{"outlier_count": 5}]}, "/rows/0/outlier_count", 5, "source", True),
    ("find_top_n", {"rows": [{"outlier_count": 5}]}, "/rows/0/outlier_count", 5, "count", False),
    ("analyze_missing_values", {"rows": [{"missing_percentage": 5}]},
     "/rows/0/missing_percentage", 5, "percent", True),
    ("calculate_group_statistics", {"aggregation": "count", "output_metric": "sales_count",
                                    "rows": [{"sales_count": 5}]}, "/rows/0/sales_count", 5, "count", True),
    ("calculate_group_statistics", {"aggregation": "sum", "output_metric": "sales_sum",
                                    "rows": [{"sales_sum": 5}]}, "/rows/0/sales_sum", 5, "count", False),
    ("calculate_time_series", {"aggregation": "count", "metric": "sales", "rows": [{"sales_count": 5}]},
     "/rows/0/sales_count", 5, "count", True),
])
def test_units_come_from_the_specific_tool_contract(tool_name, data, path, value, unit, accepted):
    actual, count = checked(finding([reference(value, path, unit=unit)], evidence_tool=tool_name), data, tool_name)
    assert (actual.claim_type == "fact") is accepted
    assert count == (0 if accepted else 1)


def test_two_invocations_of_the_same_tool_have_distinct_scopes(context):
    response = AnalystAgent(MockLLM([
        tool(), {**tool(), "tool_input": {"column": "sales", "n": 1, "ascending": True}},
        final(finding([reference(30, call_id="call_2")]), finding([reference(10, call_id="call_2")])),
    ])).run(AgentRequest(user_query="Compare top and bottom.", dataset_context=context))
    assert response.success
    assert [step.call_id for step in response.steps[:2]] == ["call_1", "call_2"]
    assert [item.claim_type for item in response.structured.key_findings] == ["hypothesis", "fact"]
    assert response.downgraded_findings == 1


def test_cached_results_keep_the_original_call_id(context):
    llm = MockLLM([tool(), tool(), final(finding([reference()]))])
    response = AnalystAgent(llm).run(AgentRequest(user_query="Top product?", dataset_context=context))
    assert response.success and response.downgraded_findings == 0
    assert [step.call_id for step in response.steps[:2]] == ["call_1", "call_1"]
    assert json.loads(llm.requests[2].user_prompt)["tool_history"][-1]["call_id"] == "call_1"


def test_failed_call_cannot_supply_evidence_after_recovery(context):
    response = AnalystAgent(MockLLM([
        tool("missing"), tool(),
        final(finding([reference()]), finding([reference(call_id="call_2")])),
    ])).run(AgentRequest(user_query="Top product?", dataset_context=context))
    assert response.success
    assert response.steps[0].action_type == "error"
    assert [item.claim_type for item in response.structured.key_findings] == ["hypothesis", "fact"]


def test_unavailable_retrieval_is_not_a_successful_evidence_source():
    response = AnalystAgent(MockLLM([
        {"action": "tool_call", "tool_name": "search_knowledge", "tool_input": {"query": "Definition?"}},
        final(finding([reference(0, "/result_count", unit="count")], evidence_tool="search_knowledge")),
    ])).run(AgentRequest(user_query="Definition?"))
    assert response.success and response.retrieval_status == "unavailable"
    assert response.downgraded_findings == 1


def test_previous_runs_do_not_leak_evidence_into_the_next_run(context):
    agent = AnalystAgent(MockLLM([tool(), final(finding([reference()])), final(finding([reference()]))]))
    request = AgentRequest(user_query="Top product?", dataset_context=context)
    assert agent.run(request).downgraded_findings == 0
    assert agent.run(request).downgraded_findings == 1


def test_full_results_remain_verifiable_when_prompt_preview_is_truncated():
    frame = pd.DataFrame({"product": ["x" * 4000], "sales": [30]})
    context = DatasetContext("test", "sales.csv", frame, profile_dataframe(frame, "sales.csv"))
    response = AnalystAgent(MockLLM([tool(), final(finding([reference()]))])).run(
        AgentRequest(user_query="Top sales?", dataset_context=context)
    )
    assert response.steps[0].tool_output["truncated"] is True
    assert response.downgraded_findings == 0
    assert response.structured.key_findings[0].claim_type == "fact"


def test_interpretations_do_not_carry_certified_looking_references():
    actual, count = checked(finding([reference()], claim_type="interpretation"))
    assert count == 0 and actual.claim_type == "interpretation"
    assert actual.evidence == [] and actual.evidence_tool is None
