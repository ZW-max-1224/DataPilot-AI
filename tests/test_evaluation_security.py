import inspect

from app.evaluation import runner
from app.evaluation.dataset import load_evaluation_cases
from app.evaluation.metrics import citation_scores


def test_runner_never_accepts_or_executes_arbitrary_dataset_paths():
    try:
        runner._resolve_bundled_dataset("../../.env")
    except ValueError:
        pass
    else:
        raise AssertionError("Untrusted dataset alias was accepted")
    source = inspect.getsource(runner)
    assert "eval(" not in source
    assert "exec(" not in source
    assert "subprocess" not in source


def test_citation_check_requires_exact_retrieved_identity():
    _, _, valid = citation_scores([], ["fake_document"], ["document-actual"])
    assert valid is False


def test_evaluation_cases_use_only_allowlisted_demo_dataset():
    assert {case.dataset for case in load_evaluation_cases()} <= {None, "sample_sales"}
