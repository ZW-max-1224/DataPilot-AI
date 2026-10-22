import json

import pandas as pd
import pytest

from app.agent import AnalystAgent, AgentAction, AgentRequest
from app.models.dataset_context import DatasetContext
from app.models.llm_models import LLMResponse
from app.models.knowledge import KnowledgeSearchResponse, KnowledgeSearchResult
from app.services.llm_service import LLMResponseError, LLMServiceError
from app.services.profiler import profile_dataframe
from app.tools.registry import ALLOWED_ANALYSIS_TOOLS


class MockLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return LLMResponse(content=value, model="mock", mode="llm")


class MockKnowledgeService:
    def __init__(self, response=None):
        self.response = response or KnowledgeSearchResponse(status="ok", result_count=1, results=[
            KnowledgeSearchResult(document_id="doc-1", chunk_id="chunk-1", title="分析方法",
                                  source="analysis_methods.md", content="相关性不代表因果关系。", score=0.9)
        ])
        self.calls = []

    def search(self, query, *, knowledge_base_id, dataset_id=None, top_k=5):
        self.calls.append((query, knowledge_base_id, dataset_id, top_k))
        return self.response


def action_tool(name, arguments):
    return json.dumps({"action": "tool_call", "tool_name": name, "tool_input": arguments})


def action_final(answer):
    return json.dumps({"action": "final", "final_answer": answer})


def action_final_structured(answer, analysis):
    return json.dumps({"action": "final", "final_answer": answer, "analysis": analysis})


@pytest.fixture
def context():
    frame = pd.DataFrame({"product": ["A", "B", "C"], "sales": [10, 30, 20]})
    return DatasetContext("test", "sales.csv", frame, profile_dataframe(frame, "sales.csv"))


def test_simple_final_answer(context):
    llm = MockLLM([action_final("请先运行统计工具。")])
    result = AnalystAgent(llm).run(AgentRequest(user_query="怎么分析？", dataset_context=context))
    assert result.success and result.final_answer == "请先运行统计工具。"
    assert len(llm.requests) == 1


def test_tool_then_final_is_grounded_in_tool_result(context):
    llm = MockLLM([action_tool("find_top_n", {"column": "sales", "n": 1}), action_final("B 的销售额最高，为 30。")])
    result = AnalystAgent(llm).run(AgentRequest(user_query="销售额最高的是谁？", dataset_context=context))
    assert result.success
    assert result.tools_used == ["find_top_n"]
    assert '"sales": 30' in llm.requests[1].user_prompt
    assert "B" in llm.requests[1].user_prompt


def test_two_tool_calls(context):
    llm = MockLLM([
        action_tool("get_dataset_overview", {}),
        action_tool("analyze_missing_values", {}),
        action_final("已检查数据规模和缺失值。"),
    ])
    result = AnalystAgent(llm).run(AgentRequest(user_query="检查数据", dataset_context=context))
    assert result.tools_used == ["get_dataset_overview", "analyze_missing_values"]
    assert result.success


@pytest.mark.parametrize("name", ["run_python", "execute_sql", "shell", "read_file", "unknown"])
def test_rejects_non_registry_tools(context, name):
    llm = MockLLM([action_tool(name, {})])
    result = AnalystAgent(llm).run(AgentRequest(user_query="分析", dataset_context=context))
    assert not result.success and result.errors[0].code == "tool_not_allowed"
    assert result.tools_used == []
    assert set(result.tools_used).issubset(ALLOWED_ANALYSIS_TOOLS)


def test_tool_arguments_are_validated_and_observed(context):
    """Invalid arguments are fed back so the model can correct them, not fatal."""
    llm = MockLLM([
        action_tool("find_top_n", {"column": "sales", "n": 1000}),   # rejected: n > 100
        action_tool("find_top_n", {"column": "sales", "n": 1}),      # corrected
        action_final("B 的销售额最高，为 30。"),
    ])
    result = AnalystAgent(llm).run(AgentRequest(user_query="找最高", dataset_context=context))
    assert result.success
    assert result.tools_used == ["find_top_n"]
    # The validation error is visible to the model on the next decision.
    assert "invalid_tool_request" in llm.requests[1].user_prompt
    assert "B" in result.final_answer


def test_repeated_invalid_tool_arguments_eventually_stop_the_run(context):
    llm = MockLLM([action_tool("find_top_n", {"column": "sales", "n": 1000})] * 3)
    result = AnalystAgent(llm).run(AgentRequest(user_query="找最高", dataset_context=context))
    assert not result.success and result.errors[0].code == "invalid_tool_request"
    assert result.tools_used == ["find_top_n"]


def test_model_supplied_argument_collision_cannot_crash_the_agent(context):
    """A tool_input key named like a function parameter must be a safe error.

    Previously `**tool_input` let the model collide with invoke_analysis_tool's own
    parameters, raising TypeError while Python bound the call -- before validation --
    and escaping AnalystAgent.run entirely.
    """
    llm = MockLLM([
        action_tool("get_dataset_overview", {"name": "collision"}),
        action_final("已忽略非法参数并完成分析。"),
    ])
    result = AnalystAgent(llm).run(AgentRequest(user_query="分析", dataset_context=context))
    assert result.success  # no exception escaped
    assert "invalid_tool_request" in llm.requests[1].user_prompt


def test_invalid_json_and_schema_are_rejected_after_repair_budget(context):
    for output in ('not json', '{"action":"tool_call","tool_name":"find_top_n"}'):
        llm = MockLLM([output] * (1 + 2))  # initial attempt + two repair attempts
        result = AnalystAgent(llm).run(AgentRequest(user_query="分析", dataset_context=context))
        assert not result.success and result.errors[0].code == "invalid_model_output"
        assert len(llm.requests) == 3


def test_markdown_fenced_json_is_accepted_without_a_repair(context):
    """Models intermittently wrap their decision in a fence; that is not a failure."""
    fenced = "```json\n" + action_tool("find_top_n", {"column": "sales", "n": 1}) + "\n```"
    llm = MockLLM([fenced, action_final("B 最高。")])
    result = AnalystAgent(llm).run(AgentRequest(user_query="找最高", dataset_context=context))
    assert result.success and result.tools_used == ["find_top_n"]
    assert len(llm.requests) == 2  # no repair round trip needed


def test_malformed_json_is_repaired_instead_of_aborting(context):
    llm = MockLLM(["I will analyze the data now.", action_final("完成。")])
    result = AnalystAgent(llm).run(AgentRequest(user_query="分析", dataset_context=context))
    assert result.success
    assert len(llm.requests) == 2
    assert "not a valid AgentAction JSON object" in llm.requests[1].user_prompt


def test_empty_llm_response_is_retried_for_the_same_decision(context):
    llm = MockLLM([LLMResponseError("empty"), action_final("完成。")])
    result = AnalystAgent(llm).run(AgentRequest(user_query="分析", dataset_context=context))
    assert result.success and result.iterations == 1
    assert len(llm.requests) == 2


def test_unexpected_exception_never_escapes_the_agent(context, monkeypatch):
    from app.agent import analyst_agent

    def explode(*args, **kwargs):
        raise RuntimeError("simulated internal bug")

    monkeypatch.setattr(analyst_agent.AnalystAgent, "_build_prompt", staticmethod(explode))
    result = AnalystAgent(MockLLM([action_final("never reached")])).run(
        AgentRequest(user_query="分析", dataset_context=context)
    )
    assert not result.success and result.errors[0].code == "agent_internal_error"


def test_identical_tool_call_is_reused_instead_of_reexecuted(context):
    """A repeated call must not burn the decision budget re-computing the same thing."""
    llm = MockLLM([
        action_tool("find_top_n", {"column": "sales", "n": 1}),
        action_tool("find_top_n", {"column": "sales", "n": 1}),   # identical repeat
        action_final("B 最高。"),
    ])
    result = AnalystAgent(llm).run(AgentRequest(user_query="找最高", dataset_context=context))
    assert result.success
    assert len(llm.requests) == 3
    assert "already called this tool" in llm.requests[2].user_prompt
    # The duplicate step reuses the earlier bounded result rather than recomputing it.
    assert result.steps[1].tool_output == result.steps[0].tool_output
    assert result.tools_used == ["find_top_n"]


def test_prompt_reports_the_remaining_decision_budget(context):
    llm = MockLLM([action_tool("get_dataset_overview", {}), action_final("完成。")])
    AnalystAgent(llm).run(AgentRequest(user_query="分析", dataset_context=context))
    assert '"decisions_remaining": 8' in llm.requests[0].user_prompt
    assert '"decisions_remaining": 7' in llm.requests[1].user_prompt
    assert "reuse the earlier result" in llm.requests[0].user_prompt


def test_the_final_decision_is_forced_to_conclude(context):
    """Without this the model can loop through tools and never answer."""
    llm = MockLLM([action_tool("get_dataset_overview", {})] * 8)
    AnalystAgent(llm).run(AgentRequest(user_query="分析", dataset_context=context))
    assert len(llm.requests) == 8
    assert "THIS IS YOUR FINAL DECISION" not in llm.requests[6].user_prompt
    assert "THIS IS YOUR FINAL DECISION" in llm.requests[7].user_prompt


def test_structured_final_answer_is_returned(context):
    llm = MockLLM([action_final_structured("B 最高。", {
        "executive_summary": "B 的销售额最高。",
        "key_findings": [{"statement": "B 的销售额为 30。", "claim_type": "fact",
                          "evidence_tool": "find_top_n"}],
        "recommendations": ["关注 B 的补货节奏。"],
    })])
    result = AnalystAgent(llm).run(AgentRequest(user_query="谁最高？", dataset_context=context))
    assert result.success
    assert result.structured is not None
    assert result.structured.executive_summary == "B 的销售额最高。"
    assert result.structured.recommendations == ["关注 B 的补货节奏。"]
    # No tool ran, so this "fact" cannot stand: it must be downgraded, not shown as fact.
    assert result.structured.key_findings[0].claim_type == "hypothesis"
    assert result.structured.key_findings[0].evidence_tool is None
    assert result.downgraded_findings == 1


def test_fact_backed_by_a_real_tool_call_is_kept(context):
    llm = MockLLM([
        action_tool("find_top_n", {"column": "sales", "n": 1}),
        action_final_structured("B 最高。", {
            "executive_summary": "B 的销售额最高。",
            "key_findings": [
                {"statement": "B 的销售额为 30。", "claim_type": "fact", "evidence_tool": "find_top_n",
                 "evidence": [
                     {"call_id": "call_1", "result_path": "/rows/0/product", "value": "B"},
                     {"call_id": "call_1", "result_path": "/rows/0/sales", "value": 30},
                 ]},
                {"statement": "可能是因为 B 更受欢迎。", "claim_type": "hypothesis"},
            ],
        }),
    ])
    result = AnalystAgent(llm).run(AgentRequest(user_query="谁最高？", dataset_context=context))
    assert result.success
    assert result.downgraded_findings == 0
    kinds = [finding.claim_type for finding in result.structured.key_findings]
    assert kinds == ["fact", "hypothesis"]


def test_final_answer_without_analysis_is_marked_unstructured_not_faked(context):
    result = AnalystAgent(MockLLM([action_final("B 最高。")])).run(
        AgentRequest(user_query="谁最高？", dataset_context=context)
    )
    assert result.success
    # Honest degradation: never invent a structured report the model did not provide.
    assert result.structured is None
    assert result.downgraded_findings == 0


def test_analysis_is_only_allowed_for_final_decisions():
    with pytest.raises(Exception):
        AgentAction.model_validate({
            "action": "tool_call", "tool_name": "find_top_n", "tool_input": {"column": "a", "n": 1},
            "analysis": {"executive_summary": "x"},
        })


def test_structured_block_tolerates_model_wording_variants(context):
    """Real providers say "conclusion"/"inference"; that must not kill the run."""
    llm = MockLLM([action_final_structured("完成。", {
        "executive_summary": "总体情况。",
        "key_findings": [
            {"statement": "纯字符串结论会被当作解释", "claim_type": "conclusion"},
            {"statement": "缺失字段的结论也要保留"},
            {"statement": "", "claim_type": "fact"},
            "这是字符串形式的结论",
        ],
        "recommendations": ["建议一", "", 42],
        "unknown_extra_key": "ignored",
    })])
    result = AnalystAgent(llm).run(AgentRequest(user_query="分析", dataset_context=context))
    assert result.success
    kinds = [finding.claim_type for finding in result.structured.key_findings]
    assert kinds == ["interpretation", "interpretation", "interpretation"]
    # Empty statements are dropped; non-string recommendations are ignored.
    assert result.structured.recommendations == ["建议一"]


def test_unknown_claim_type_is_never_treated_as_a_fact(context):
    llm = MockLLM([action_final_structured("完成。", {
        "executive_summary": "总体情况。",
        "key_findings": [{"statement": "缺少证据的断言", "claim_type": "totally_made_up"}],
    })])
    result = AnalystAgent(llm).run(AgentRequest(user_query="分析", dataset_context=context))
    assert result.success
    assert result.structured.key_findings[0].claim_type == "interpretation"
    assert result.downgraded_findings == 0


def test_structured_summary_falls_back_to_the_narrative(context):
    llm = MockLLM([action_final_structured("这是叙述本身的结论。", {
        "executive_summary": "",
        "key_findings": [],
    })])
    result = AnalystAgent(llm).run(AgentRequest(user_query="分析", dataset_context=context))
    assert result.success
    assert result.structured.executive_summary == "这是叙述本身的结论。"


def test_provider_field_aliases_are_not_silently_dropped(context):
    """A real provider nested findings under "claim" and recommendations as objects.

    Treating those as unknown silently deleted every conclusion the user should see.
    """
    llm = MockLLM([
        action_tool("find_top_n", {"column": "sales", "n": 1}),
        action_final_structured("完成。", {
            "executive_summary": "总体情况。",
            "key_findings": [
                {"claim": "结论一", "claim_type": "fact", "tool": "find_top_n"},
                {"text": "结论二", "type": "hypothesis"},
            ],
            "recommendations": [{"text": "建议一", "priority": "high"}, "建议二"],
        }),
    ])
    result = AnalystAgent(llm).run(AgentRequest(user_query="分析", dataset_context=context))
    assert result.success
    assert [finding.statement for finding in result.structured.key_findings] == ["结论一", "结论二"]
    # Aliases are preserved, but a tool name alone no longer certifies a fact.
    assert result.structured.key_findings[0].claim_type == "hypothesis"
    assert result.structured.key_findings[0].evidence_tool is None
    assert result.structured.key_findings[1].claim_type == "hypothesis"
    assert result.structured.recommendations == ["建议一", "建议二"]
    assert result.downgraded_findings == 1


def test_decision_token_budget_is_large_enough_for_a_structured_answer(context):
    """A truncated reply is unparseable JSON, so the budget must fit the contract."""
    from app.agent import analyst_agent

    llm = MockLLM([action_final("完成。")])
    AnalystAgent(llm).run(AgentRequest(user_query="分析", dataset_context=context))
    assert llm.requests[0].max_tokens == analyst_agent.DECISION_MAX_TOKENS
    assert analyst_agent.DECISION_MAX_TOKENS >= 2_000


def test_agent_surfaces_the_chart_it_asked_for(context):
    """A chart the agent creates must reach the caller, not be silently discarded."""
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    llm = MockLLM([
        action_tool("create_chart", {"chart_type": "bar", "x": "product", "y": "sales"}),
        action_final("已生成柱状图。"),
    ])
    result = AnalystAgent(llm).run(AgentRequest(user_query="画个图", dataset_context=context))
    assert result.success
    assert len(result.charts) == 1
    # The model context keeps JSON metadata only -- no Figure object leaks into it.
    step_output = json.dumps(result.steps[0].tool_output, ensure_ascii=False)
    assert '"chart_type": "bar"' in step_output
    assert "Figure" not in step_output
    for chart in result.charts:
        chart.release()
    assert plt.get_fignums() == before


def test_agent_response_serialization_excludes_chart_objects(context):
    llm = MockLLM([
        action_tool("create_chart", {"chart_type": "line", "x": "product", "y": "sales"}),
        action_final("完成。"),
    ])
    result = AnalystAgent(llm).run(AgentRequest(user_query="画个图", dataset_context=context))
    dumped = result.model_dump_json()   # must not try to serialize a Matplotlib Figure
    assert "charts" not in dumped
    for chart in result.charts:
        chart.release()


def test_retrieval_status_defaults_to_not_attempted(context):
    result = AnalystAgent(MockLLM([action_final("完成。")])).run(
        AgentRequest(user_query="分析", dataset_context=context)
    )
    assert result.retrieval_status == "not_attempted"


def test_knowledge_hit_reports_ok_and_cites_real_sources():
    result = AnalystAgent(
        MockLLM([action_tool("search_knowledge", {"query": "相关性"}), action_final("相关性不代表因果。")]),
        MockKnowledgeService(),
    ).run(AgentRequest(user_query="相关性是什么？"))
    assert result.success and result.retrieval_status == "ok"
    assert "analysis_methods.md" in result.final_answer
    summaries = " ".join(step.summary for step in result.steps)
    assert "命中" in summaries


def test_unavailable_knowledge_is_reported_and_not_dressed_up_as_evidence():
    """Without a knowledge base the model may still answer -- but not as evidence.

    Verified against the real LLM: with knowledge_service=None the tool reported
    knowledge_base_unavailable, yet the run was marked successful and the step said
    the tool had returned "structured analysis results", so a definition answered from
    the model's own knowledge looked knowledge-base grounded.
    """
    result = AnalystAgent(
        MockLLM([action_tool("search_knowledge", {"query": "什么是毛利率"}), action_final("毛利率是……")]),
        None,
    ).run(AgentRequest(user_query="什么是毛利率？"))
    assert result.success
    assert result.retrieval_status == "unavailable"
    summary = next(step.summary for step in result.steps if step.tool_name == "search_knowledge")
    assert "不可用" in summary
    assert "已附在最终答复中" not in summary


def test_knowledge_no_results_is_reported_as_no_results():
    empty = KnowledgeSearchResponse(status="no_results")
    result = AnalystAgent(
        MockLLM([action_tool("search_knowledge", {"query": "退货政策"}), action_final("知识库没有相关内容。")]),
        MockKnowledgeService(empty),
    ).run(AgentRequest(user_query="退货政策是什么？"))
    assert result.success and result.retrieval_status == "no_results"
    summary = next(step.summary for step in result.steps if step.tool_name == "search_knowledge")
    assert "未找到" in summary


def test_embedding_failure_maps_to_unavailable_retrieval_status():
    failure = KnowledgeSearchResponse(status="embedding_failed", error_code="embedding_failed")
    result = AnalystAgent(
        MockLLM([action_tool("search_knowledge", {"query": "方法"}), action_final("说明。")]),
        MockKnowledgeService(failure),
    ).run(AgentRequest(user_query="分析方法有哪些？"))
    assert result.retrieval_status == "unavailable"


def test_step_detail_shows_what_each_call_was_asked_to_compute(context):
    """Two calls to the same tool with different arguments must be distinguishable.

    The live demo showed two identical "调用白名单工具 calculate_time_series" lines, which
    reads like an accidental repeat even though the arguments differed.
    """
    llm = MockLLM([
        action_tool("calculate_group_statistics",
                    {"group_by": "product", "metric": "sales", "aggregation": "sum"}),
        action_tool("calculate_group_statistics",
                    {"group_by": "product", "metric": "sales", "aggregation": "mean"}),
        action_final("完成。"),
    ])
    result = AnalystAgent(llm).run(AgentRequest(user_query="按产品看销售额", dataset_context=context))
    assert result.success
    first, second = result.steps[0].detail, result.steps[1].detail
    assert first == "group_by=product, metric=sales, aggregation=sum"
    assert second == "group_by=product, metric=sales, aggregation=mean"
    assert first != second
    # The privacy-safe summary stays generic and does not carry the column names.
    assert "sales" not in result.steps[0].summary


def test_step_detail_is_absent_for_the_final_step(context):
    result = AnalystAgent(MockLLM([action_final("完成。")])).run(
        AgentRequest(user_query="分析", dataset_context=context)
    )
    assert result.steps[-1].detail is None


def test_step_detail_is_bounded_and_ignores_unsupported_values(context):
    from app.agent import analyst_agent

    digest = analyst_agent.AnalystAgent._argument_digest({
        "column": "x" * 200, "n": 3, "ascending": None, "nested": {"a": 1},
    })
    assert digest is not None and len(digest) <= 300
    assert "n=3" in digest and "nested" not in digest and "ascending" not in digest
    assert analyst_agent.AnalystAgent._argument_digest({}) is None
    assert analyst_agent.AnalystAgent._argument_digest(None) is None


def test_step_details_stay_inside_the_ui_and_never_reach_the_trace(context):
    """The LangSmith boundary must keep recording structure, not column names.

    ``AgentStep.detail`` exists precisely because the trace may not carry arguments, so
    this locks the boundary: if a future change adds details to the traced step payload,
    the promise "工具参数只记录结构类型和字符串长度" would be silently broken.
    """

    class RecordingSpan:
        def __init__(self, log):
            self._log = log

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def end(self, payload=None):
            self._log.append(payload)

    class RecordingTracer:
        def __init__(self):
            self.payloads = []

        def span(self, name, run_type=None, inputs=None, metadata=None):
            return RecordingSpan(self.payloads)

    tracer = RecordingTracer()

    class StubLLM:
        is_configured = True

        def __init__(self, responses):
            self._responses = list(responses)
            self.tracing = tracer

        def generate(self, request):
            return LLMResponse(content=self._responses.pop(0), model="stub", mode="llm")

    agent = AnalystAgent(StubLLM([
        action_tool("find_top_n", {"column": "sales", "n": 1}),
        action_final("完成。"),
    ]))
    result = agent.run(AgentRequest(user_query="找最高", dataset_context=context))
    assert result.success
    # The UI-facing step carries the column name...
    assert "sales" in (result.steps[0].detail or "")

    root_payload = next(payload for payload in tracer.payloads if payload and "steps" in payload)
    serialized = json.dumps(root_payload, ensure_ascii=False, default=str)
    assert "detail" not in serialized
    assert "sales" not in serialized
    for step_payload in root_payload["steps"]:
        assert set(step_payload) == {"action_type", "tool_name", "summary"}


def test_prompt_budget_is_a_real_hard_limit_on_the_worst_case():
    """A wide schema plus a long question used to exceed the budget with no history.

    The old code only trimmed tool_history, so the "limit" could be violated while
    evidence had already been deleted.
    """
    from app.agent import analyst_agent

    frame = pd.DataFrame({f"col_{index}": [1, 2] for index in range(80)})
    frame["date"] = pd.to_datetime(["2024-01-01", "2024-02-01"])
    wide = DatasetContext("wide", "wide.csv", frame, profile_dataframe(frame, "wide.csv"))
    heavy_history = [
        {"tool": "find_top_n", "result": {"tool_name": "find_top_n", "success": True,
                                          "data": {"rows": [{"value": "x" * 3_000}]}}}
        for _ in range(6)
    ]
    prompt = analyst_agent.AnalystAgent._build_prompt("问" * 4_000, wide, heavy_history)
    assert len(prompt) <= analyst_agent.MAX_PROMPT_CHARS
    # The prompt stays valid JSON and still carries the decision contract.
    payload = json.loads(prompt)
    assert "decision_schema" in payload and "available_tools" in payload
    # A short, ordinary request keeps full history and the full schema.
    small = analyst_agent.AnalystAgent._build_prompt("简短问题", wide, heavy_history[:1])
    assert len(small) <= analyst_agent.MAX_PROMPT_CHARS
    assert len(json.loads(small)["dataset"]["schema"]) == analyst_agent.MAX_SCHEMA_COLUMNS


def test_agent_stops_after_maximum_decisions(context):
    llm = MockLLM([action_tool("get_dataset_overview", {})] * 8)
    result = AnalystAgent(llm).run(AgentRequest(user_query="重复分析", dataset_context=context))
    assert not result.success and result.errors[0].code == "max_iterations_reached"
    assert len(llm.requests) == 8


def test_tool_result_is_capped_before_next_prompt(context, monkeypatch):
    from app.agent import analyst_agent
    from app.models.analysis_result import ToolExecutionResult

    monkeypatch.setattr(analyst_agent, "invoke_analysis_tool", lambda *args, **kwargs: ToolExecutionResult(
        tool_name="get_dataset_overview", success=True, data={"payload": "x" * 10_000},
    ))
    llm = MockLLM([action_tool("get_dataset_overview", {}), action_final("完成。")])
    result = AnalystAgent(llm).run(AgentRequest(user_query="分析", dataset_context=context))
    assert result.success
    assert len(json.dumps(result.steps[0].tool_output, ensure_ascii=False)) <= 3_000
    assert result.steps[0].tool_output["truncated"] is True
    assert "truncated" in llm.requests[1].user_prompt


def test_missing_dataset_reaches_agent_decision_and_demo_mode_is_explicit():
    missing = AnalystAgent(MockLLM([])).run(AgentRequest(user_query="分析"))
    # A missing dataset is no longer rejected before the Agent can select RAG.
    assert not missing.success and missing.errors[0].code == "llm_request_failed"
    frame = pd.DataFrame({"x": [1]})
    dataset = DatasetContext("id", "x.csv", frame, profile_dataframe(frame, "x.csv"))
    demo = LLMResponse(content="demo", model="demo", mode="demo")
    class DemoLLM:
        def generate(self, request):
            return demo
    result = AnalystAgent(DemoLLM()).run(AgentRequest(user_query="分析", dataset_context=dataset))
    assert not result.success and result.errors[0].code == "demo_mode" and not result.tools_used


def test_tool_execution_errors_are_observed_and_recovered(context, monkeypatch):
    """A tool failure is an observation; the model gets a chance to answer anyway."""
    from app.agent import analyst_agent
    from app.models.analysis_result import ToolError, ToolExecutionResult

    monkeypatch.setattr(analyst_agent, "invoke_analysis_tool", lambda *args, **kwargs: ToolExecutionResult(
        tool_name="get_dataset_overview", success=False,
        error=ToolError(code="tool_execution_error", message="safe failure"),
    ))
    llm = MockLLM([action_tool("get_dataset_overview", {}), action_final("工具失败，但我可以说明原因。")])
    result = AnalystAgent(llm).run(AgentRequest(user_query="分析", dataset_context=context))
    assert result.success
    assert "safe failure" in llm.requests[1].user_prompt
    assert any(step.action_type == "error" for step in result.steps)


def test_repeated_tool_execution_errors_stop_the_run(context, monkeypatch):
    from app.agent import analyst_agent
    from app.models.analysis_result import ToolError, ToolExecutionResult

    monkeypatch.setattr(analyst_agent, "invoke_analysis_tool", lambda *args, **kwargs: ToolExecutionResult(
        tool_name="get_dataset_overview", success=False,
        error=ToolError(code="tool_execution_error", message="safe failure"),
    ))
    llm = MockLLM([action_tool("get_dataset_overview", {})] * 3)
    result = AnalystAgent(llm).run(AgentRequest(user_query="分析", dataset_context=context))
    assert not result.success and result.errors[0].code == "tool_execution_error"


def test_failed_run_still_returns_the_evidence_already_computed(context):
    """Project rule: a loop that cannot finish must still return what it computed."""
    llm = MockLLM([
        action_tool("find_top_n", {"column": "sales", "n": 1}),
        LLMServiceError("provider down"),
    ])
    result = AnalystAgent(llm).run(AgentRequest(user_query="找最高", dataset_context=context))
    assert not result.success and result.errors[0].code == "llm_request_failed"
    # A bare error message would discard a real, already-computed result.
    assert "find_top_n" in result.final_answer
    assert "30" in result.final_answer
    assert result.tools_used == ["find_top_n"]


def test_max_iterations_returns_partial_evidence_too(context):
    llm = MockLLM([action_tool("find_top_n", {"column": "sales", "n": 1})] + [
        action_tool("get_dataset_overview", {}) for _ in range(7)
    ])
    result = AnalystAgent(llm).run(AgentRequest(user_query="重复分析", dataset_context=context))
    assert not result.success and result.errors[0].code == "max_iterations_reached"
    assert "find_top_n" in result.final_answer


def test_agent_action_rejects_extra_fields():
    with pytest.raises(Exception):
        AgentAction.model_validate({"action": "tool_call", "tool_name": "shell", "tool_input": {}, "code": "bad"})


def test_agent_can_answer_knowledge_only_without_dataset():
    knowledge = MockKnowledgeService()
    llm = MockLLM([
        action_tool("search_knowledge", {"query": "相关性和因果关系有什么区别？"}),
        action_final("相关性不代表因果关系。\n\nSources:\n- fabricated.md"),
    ])
    result = AnalystAgent(llm, knowledge).run(AgentRequest(user_query="相关性是什么？"))
    assert result.success
    assert result.tools_used == ["search_knowledge"]
    assert knowledge.calls[0][2] is None
    assert "Sources:" in result.final_answer
    assert "analysis_methods.md" in result.final_answer
    assert "fabricated.md" not in result.final_answer
    assert "analysis_methods.md" in llm.requests[1].user_prompt


def test_agent_can_use_pandas_then_knowledge(context):
    knowledge = MockKnowledgeService()
    llm = MockLLM([
        action_tool("calculate_group_statistics", {
            "group_by": "product", "metric": "sales", "aggregation": "sum",
        }),
        action_tool("search_knowledge", {"query": "如何解释分组销售额变化？"}),
        action_final("数据证据显示 B 销售额最高；解释参考分析方法。"),
    ])
    result = AnalystAgent(llm, knowledge).run(AgentRequest(user_query="分析销售变化", dataset_context=context))
    assert result.success
    assert result.tools_used == ["calculate_group_statistics", "search_knowledge"]
    assert knowledge.calls[0][2] == context.dataset_id


def test_agent_can_use_knowledge_then_pandas(context):
    knowledge = MockKnowledgeService()
    llm = MockLLM([
        action_tool("search_knowledge", {"query": "异常值分析应注意什么？"}),
        action_tool("detect_outliers", {"column": "sales"}),
        action_final("按 IQR 检查后得出数据结果。"),
    ])
    result = AnalystAgent(llm, knowledge).run(AgentRequest(user_query="检查销售额异常值", dataset_context=context))
    assert result.success
    assert result.tools_used == ["search_knowledge", "detect_outliers"]


def test_agent_rejects_dataset_tool_without_dataset():
    llm = MockLLM([action_tool("find_top_n", {"column": "sales", "n": 1})])
    result = AnalystAgent(llm).run(AgentRequest(user_query="找出销售额最高的产品"))
    assert not result.success
    assert result.errors[0].code == "dataset_missing"


def test_agent_limits_duplicate_and_repeated_knowledge_searches():
    duplicate = AnalystAgent(MockLLM([
        action_tool("search_knowledge", {"query": "sales margin"}),
        action_tool("search_knowledge", {"query": "sales margin"}),
    ]), MockKnowledgeService()).run(AgentRequest(user_query="sales margin definition"))
    assert not duplicate.success and duplicate.errors[0].code == "duplicate_knowledge_search"
    assert duplicate.tools_used == ["search_knowledge"]

    limited = AnalystAgent(MockLLM([
        action_tool("search_knowledge", {"query": "sales margin"}),
        action_tool("search_knowledge", {"query": "profit definition"}),
        action_tool("search_knowledge", {"query": "chart method"}),
    ]), MockKnowledgeService()).run(AgentRequest(user_query="three knowledge lookups"))
    assert not limited.success and limited.errors[0].code == "knowledge_search_limit"
    assert len(limited.tools_used) == 1


def test_agent_receives_rag_failures_and_no_results_as_observations():
    for response, expected in [
        (KnowledgeSearchResponse(status="embedding_failed", error_code="embedding_failed"), "Embedding"),
        (KnowledgeSearchResponse(status="no_results"), "资料不足"),
    ]:
        llm = MockLLM([
            action_tool("search_knowledge", {"query": "未配置知识检索的状态"}),
            action_final(expected),
        ])
        result = AnalystAgent(llm, MockKnowledgeService(response)).run(AgentRequest(user_query="解释当前知识"))
        assert result.success
        assert result.tools_used == ["search_knowledge"]
        assert response.status in llm.requests[1].user_prompt


def test_rag_tool_result_is_truncated_without_losing_structure(monkeypatch):
    from app.agent import analyst_agent
    from app.models.analysis_result import ToolExecutionResult

    results = [{"document_id": f"doc-{index}", "chunk_id": f"chunk-{index}",
                "title": "Analysis", "source": "methods.md", "content": "x" * 1_000,
                "score": 0.8, "metadata": {}} for index in range(5)]
    monkeypatch.setattr(analyst_agent, "invoke_analysis_tool", lambda *args, **kwargs: ToolExecutionResult(
        tool_name="search_knowledge", success=True,
        data={"status": "ok", "results": results, "result_count": 5, "truncated": False},
    ))
    llm = MockLLM([
        action_tool("search_knowledge", {"query": "analysis method"}),
        action_final("Done."),
    ])
    response = AnalystAgent(llm, MockKnowledgeService()).run(AgentRequest(user_query="Explain analysis"))
    assert response.success
    output = response.steps[0].tool_output
    assert len(json.dumps(output, ensure_ascii=False)) <= 3_000
    assert output["data"]["truncated"] is True
    assert isinstance(output["data"]["results"], list)
    assert "preview" not in output
    assert '"results"' in llm.requests[1].user_prompt
