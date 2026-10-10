import json

from app.agent import AnalystAgent
from app.evaluation.dataset import load_evaluation_cases
from app.evaluation.models import EvaluationCase
from app.evaluation.runner import EvaluationRunner
from app.models.llm_models import LLMResponse


class _MockLLM:
    def __init__(self, actions):
        self.actions = list(actions)

    def generate(self, request):
        return LLMResponse(content=self.actions.pop(0), model="test", mode="llm")


def _tool(name, **args):
    return json.dumps({"action": "tool_call", "tool_name": name, "tool_input": args})


def _final(text):
    return json.dumps({"action": "final", "final_answer": text})


def _runner(case, actions, dependencies=None):
    return EvaluationRunner(
        [case], lambda _: AnalystAgent(_MockLLM(actions)),
        dependency_status=dependencies or {"llm": True, "embedding": True, "qdrant": True},
    )


def test_runner_executes_public_agent_api_and_checks_structured_ground_truth():
    case = next(item for item in load_evaluation_cases() if item.case_id == "pandas_top_product")
    report = _runner(case, [
        _tool("get_dataset_overview"),
        _tool("find_top_n", column="sales", n=1, group_by="product"),
        _final("Laptop Pro 的销售额最高。"),
    ]).run()
    result = report.results[0]
    assert result.status == "passed"
    assert result.actual_tools == ["get_dataset_overview", "find_top_n"]
    assert result.iterations == 3
    assert result.metric_results["ground_truth_match"] is True
    assert report.summary.pass_rate == 1
    assert report.summary.metrics["numeric_accuracy"].denominator == 0


def test_runner_reports_failure_for_wrong_numeric_ground_truth():
    case = next(item for item in load_evaluation_cases() if item.case_id == "pandas_top_product")
    case = case.model_copy(update={"expectation": case.expectation.model_copy(update={
        "ground_truth": case.expectation.ground_truth.model_copy(update={"expected": "Wrong Product"})
    })})
    report = _runner(case, [
        _tool("find_top_n", column="sales", n=1, group_by="product"),
        _final("Laptop Pro"),
    ]).run()
    assert report.results[0].status == "failed"
    assert "ground truth" in report.results[0].failure_reason.casefold()


def test_runner_distinguishes_skipped_and_unavailable_dependencies():
    rag_case = next(item for item in load_evaluation_cases() if item.case_id == "rag_metric_definition")
    skipped = _runner(rag_case, [], {"llm": True, "embedding": False, "qdrant": True}).run()
    assert skipped.results[0].status == "skipped"
    assert "embedding" in skipped.results[0].failure_reason
    unavailable = _runner(rag_case, [], {"llm": False, "embedding": True, "qdrant": True}).run()
    assert unavailable.results[0].status == "unavailable"
    assert unavailable.summary.pass_rate is None


def test_runner_detects_invalid_tools_repeated_search_and_iteration_count():
    case = EvaluationCase(case_id="invalid", category="agent_execution", question="q", dataset="sample_sales")
    result = _runner(case, [_tool("shell"), _final("no")]).run().results[0]
    assert result.invalid_tool is True
    assert result.status == "failed"

    repeated = EvaluationCase(case_id="repeat", category="agent_execution", question="q", dataset=None)
    repeated_result = _runner(repeated, [
        _tool("search_knowledge", query="same"), _tool("search_knowledge", query="same"),
    ]).run().results[0]
    assert repeated_result.duplicate_tool_call is True
    assert repeated_result.iterations == 2


def test_agent_decision_count_is_bounded_at_eight():
    case = EvaluationCase(case_id="loop", category="agent_execution", question="q", dataset="sample_sales")
    actions = [_tool("get_dataset_overview") for _ in range(8)]
    result = _runner(case, actions).run().results[0]
    assert result.iterations == 8
    assert result.metric_results["max_iteration_violation"] is False
    assert result.agent_success is False


def test_runner_checks_retrieval_miss_without_inventing_results():
    case = next(item for item in load_evaluation_cases() if item.case_id == "rag_no_relevant_knowledge")
    class _EmptyKnowledge:
        def search(self, *args, **kwargs):
            from app.models.knowledge import KnowledgeSearchResponse
            return KnowledgeSearchResponse(status="no_results")
    result = _runner(case, [_tool("search_knowledge", query="quantum"), _final("没有找到相关资料。")])
    result.agent_factory = lambda _: AnalystAgent(_MockLLM([
        _tool("search_knowledge", query="quantum"), _final("没有找到相关资料。")
    ]), _EmptyKnowledge())
    report = result.run()
    assert report.results[0].retrieved_document_ids == []
    assert report.results[0].metric_results["retrieval_hit_at_5"] is None
    assert report.results[0].metric_results["retrieval_miss"] is True
    assert report.results[0].status == "passed"

    class _IrrelevantKnowledge:
        def search(self, *args, **kwargs):
            from app.models.knowledge import KnowledgeSearchResponse, KnowledgeSearchResult
            return KnowledgeSearchResponse(status="ok", result_count=1, results=[
                KnowledgeSearchResult(document_id="unrelated", chunk_id="chunk-x", title="Unrelated",
                                      source="other.md", content="Unrelated evidence", score=0.3)
            ])
    false_positive_runner = _runner(case, [
        _tool("search_knowledge", query="quantum"), _final("No relevant knowledge.")
    ])
    false_positive_runner.agent_factory = lambda _: AnalystAgent(_MockLLM([
        _tool("search_knowledge", query="quantum"), _final("No relevant knowledge.")
    ]), _IrrelevantKnowledge())
    false_positive = false_positive_runner.run().results[0]
    assert false_positive.status == "failed"
    assert "returned documents" in false_positive.failure_reason


def test_optional_judge_is_opt_in_and_never_overrides_deterministic_failure():
    from app.evaluation.judge import JudgeOutput

    case = next(item for item in load_evaluation_cases() if item.case_id == "pandas_top_product")

    class _Judge:
        is_available = True
        called = False

        def evaluate(self, **kwargs):
            self.called = True
            return JudgeOutput(grounded=True, score=1, reason="model opinion")

    judge = _Judge()
    actions = [_tool("find_top_n", column="sales", n=1, group_by="product"), _final("Laptop Pro")]
    runner = EvaluationRunner([case], lambda _: AnalystAgent(_MockLLM(actions.copy())),
                              dependency_status={"llm": True}, llm_judge=judge)
    default_report = runner.run()
    assert not judge.called and not default_report.llm_judge_enabled

    wrong = case.model_copy(update={"expectation": case.expectation.model_copy(update={
        "ground_truth": case.expectation.ground_truth.model_copy(update={"expected": "Incorrect"})
    })})
    runner = EvaluationRunner([wrong], lambda _: AnalystAgent(_MockLLM(actions.copy())),
                              dependency_status={"llm": True}, llm_judge=judge)
    judged_report = runner.run(enable_llm_judge=True)
    assert judged_report.llm_judge_enabled
    assert judged_report.results[0].judge_status == "completed"
    assert judged_report.results[0].status == "failed"


def test_optional_judge_is_opt_in_and_never_overrides_deterministic_failure():
    from app.evaluation.judge import JudgeOutput

    case = next(item for item in load_evaluation_cases() if item.case_id == "pandas_top_product")

    class _Judge:
        is_available = True
        called = False

        def evaluate(self, **kwargs):
            self.called = True
            return JudgeOutput(grounded=True, score=1, reason="model opinion")

    judge = _Judge()
    actions = [_tool("find_top_n", column="sales", n=1, group_by="product"), _final("Laptop Pro")]
    runner = EvaluationRunner([case], lambda _: AnalystAgent(_MockLLM(actions.copy())),
                              dependency_status={"llm": True}, llm_judge=judge)
    default_report = runner.run()
    assert not judge.called and not default_report.llm_judge_enabled

    wrong = case.model_copy(update={"expectation": case.expectation.model_copy(update={
        "ground_truth": case.expectation.ground_truth.model_copy(update={"expected": "Incorrect"})
    })})
    runner = EvaluationRunner([wrong], lambda _: AnalystAgent(_MockLLM(actions.copy())),
                              dependency_status={"llm": True}, llm_judge=judge)
    judged_report = runner.run(enable_llm_judge=True)
    assert judged_report.llm_judge_enabled
    assert judged_report.results[0].judge_status == "completed"
    assert judged_report.results[0].status == "failed"
