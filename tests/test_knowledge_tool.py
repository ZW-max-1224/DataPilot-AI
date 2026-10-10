from app.models.knowledge import KnowledgeSearchResponse
from app.tools.registry import ALLOWED_ANALYSIS_TOOLS, describe_analysis_tools, get_analysis_tool_spec, invoke_analysis_tool


class FakeKnowledge:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or KnowledgeSearchResponse(status="no_results")

    def search(self, query, *, knowledge_base_id, dataset_id=None, top_k=5):
        self.calls.append((query, knowledge_base_id, dataset_id, top_k))
        return self.response


def test_search_knowledge_is_allowlisted_and_does_not_require_dataset():
    service = FakeKnowledge()
    result = invoke_analysis_tool(
        None,
        "search_knowledge",
        knowledge_service=service,
        trusted_knowledge_base_id="trusted-kb",
        query="metric definition",
        top_k=3,
    )
    assert "search_knowledge" in ALLOWED_ANALYSIS_TOOLS
    assert get_analysis_tool_spec("search_knowledge").requires_dataset is False
    assert result.success and result.data["status"] == "no_results"
    assert service.calls == [("metric definition", "trusted-kb", None, 3)]
    assert any(item["name"] == "search_knowledge" for item in describe_analysis_tools())


def test_search_knowledge_uses_trusted_base_and_dataset_scope():
    import pandas as pd
    from app.models.dataset_context import DatasetContext
    from app.services.profiler import profile_dataframe

    frame = pd.DataFrame({"sales": [1]})
    dataset = DatasetContext("dataset-42", "sales.csv", frame, profile_dataframe(frame, "sales.csv"))
    service = FakeKnowledge()
    result = invoke_analysis_tool(
        dataset,
        "search_knowledge",
        knowledge_service=service,
        trusted_knowledge_base_id="server-selected-kb",
        query="sales definition",
    )
    assert result.success
    assert service.calls == [("sales definition", "server-selected-kb", "dataset-42", 5)]


def test_search_knowledge_rejects_extra_scope_parameters_and_invalid_top_k():
    service = FakeKnowledge()
    for parameters in (
        {"query": "q", "knowledge_base_id": "attacker-kb"},
        {"query": "q", "top_k": 10},
    ):
        result = invoke_analysis_tool(None, "search_knowledge", knowledge_service=service,
                                      trusted_knowledge_base_id="trusted-kb", **parameters)
        assert not result.success
        assert result.error.code == "invalid_tool_request"
    assert service.calls == []


def test_dataset_tools_still_require_dataset_context():
    result = invoke_analysis_tool(None, "find_top_n", column="sales", n=1)
    assert not result.success
    assert result.error.code == "dataset_missing"
