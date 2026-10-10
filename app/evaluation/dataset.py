from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.models import EvaluationCase

DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "datapilot_mvp_v1.json"


def load_evaluation_cases(path: Path | None = None) -> list[EvaluationCase]:
    """Load a bundled, reviewed evaluation dataset; never execute dataset content."""
    source = path or DATASET_PATH
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("dataset_name") != "datapilot_mvp_v1":
        raise ValueError("Unsupported evaluation dataset format or name.")
    cases = [EvaluationCase.model_validate(item) for item in raw.get("cases", [])]
    case_ids = [case.case_id for case in cases]
    if not cases or len(case_ids) != len(set(case_ids)):
        raise ValueError("Evaluation case IDs must be non-empty and unique.")
    return cases
