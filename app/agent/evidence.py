"""Deterministic scalar evidence checks; never interpret or execute model text."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import Any

from app.agent.agent_models import EvidenceReference, Finding


@dataclass(frozen=True)
class ToolEvidence:
    """Successful invocation data kept only in the current agent run's memory."""

    tool_name: str
    data: dict[str, Any]


def _resolve(data: dict[str, Any], pointer: str) -> Any:
    """Resolve RFC 6901 object keys/list indices without attribute access."""
    value: Any = data
    for token in pointer[1:].split("/"):
        if re.search(r"~(?![01])", token):
            raise ValueError("Invalid JSON Pointer escape")
        key = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            value = value[key]
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", key):
            value = value[int(key)]
        else:
            raise ValueError("Invalid result path")
    return value


def _unit(call: ToolEvidence, pointer: str) -> str:
    """Infer only units explicitly defined by registered result contracts.

    User columns named 'outlier_count' or 'change_percentage' do not inherit
    units from unrelated tool metadata. Dataset measurement units stay unknown.
    """
    percentages = {
        "analyze_missing_values": {"/missing_rate"},
        "analyze_duplicates": {"/duplicate_percentage"},
        "detect_outliers": {"/outlier_percentage"},
        "calculate_time_series": {"/change_percentage"},
    }
    counts = {
        "get_dataset_overview": {"/rows", "/columns", "/missing_values", "/duplicate_rows"},
        "find_top_n": {"/result_count", "/rows_excluded"},
        "calculate_group_statistics": {"/result_count", "/rows_excluded"},
        "calculate_time_series": {"/result_count", "/rows_excluded"},
        "analyze_missing_values": {"/total_missing_values"},
        "analyze_duplicates": {"/duplicate_rows"},
        "detect_outliers": {"/valid_count", "/outlier_count"},
        "calculate_correlation": {"/sample_size"},
        "create_chart": {"/plotted_rows"},
        "search_knowledge": {"/result_count"},
    }
    if pointer in percentages.get(call.tool_name, set()):
        return "percent"
    if pointer in counts.get(call.tool_name, set()):
        return "count"
    if call.tool_name == "analyze_missing_values":
        if re.fullmatch(r"/rows/(0|[1-9][0-9]*)/missing_percentage", pointer):
            return "percent"
        if re.fullmatch(r"/rows/(0|[1-9][0-9]*)/missing_count", pointer):
            return "count"
    if call.tool_name in {"calculate_group_statistics", "calculate_time_series"}:
        if call.data.get("aggregation") == "count":
            metric = call.data.get("output_metric") or f"{call.data.get('metric')}_count"
            escaped = str(metric).replace("~", "~0").replace("/", "~1")
            if re.fullmatch(r"/rows/(0|[1-9][0-9]*)/" + re.escape(escaped), pointer):
                return "count"
    return "source"


def _matches(reference: EvidenceReference, actual: Any) -> bool:
    if type(actual) in (int, float) and type(reference.value) in (int, float):
        try:
            with localcontext() as context:
                context.prec = 400
                number = Decimal(str(actual))
                expected = Decimal(str(reference.value))
                if not number.is_finite() or not expected.is_finite():
                    return False
                if reference.decimal_places is not None:
                    number = number.quantize(Decimal(1).scaleb(-reference.decimal_places), rounding=ROUND_HALF_UP)
                return number == expected
        except InvalidOperation:
            return False
    return (reference.decimal_places is None and type(actual) is type(reference.value)
            and actual == reference.value)


def verify_finding(finding: Finding, calls: dict[str, ToolEvidence]) -> Finding:
    """Render facts from checked fields; arbitrary model prose is never certified.

    Rounding is explicit and decimal based. There is no model-chosen tolerance,
    arithmetic expression, unit conversion, or cross-run lookup.
    """
    if finding.claim_type != "fact":
        # References attached to unverified prose must not look certified.
        return finding.model_copy(update={"evidence": [], "evidence_tool": None})
    valid = bool(finding.evidence)
    parts: list[str] = []
    tool_names: set[str] = set()
    for reference in finding.evidence:
        call = calls.get(reference.call_id)
        if call is None or (finding.evidence_tool is not None and finding.evidence_tool != call.tool_name):
            valid = False
            break
        try:
            actual = _resolve(call.data, reference.result_path)
        except (KeyError, IndexError, ValueError):
            valid = False
            break
        if reference.unit != _unit(call, reference.result_path) or not _matches(reference, actual):
            valid = False
            break
        tool_names.add(call.tool_name)
        value = json.dumps(reference.value, ensure_ascii=False, allow_nan=False)
        suffix = {"source": "", "percent": "%", "count": " [count]"}[reference.unit]
        rounding = f" [round={reference.decimal_places}]" if reference.decimal_places is not None else ""
        parts.append(f"{reference.call_id} {reference.result_path} = {value}{suffix}{rounding}")
    statement = "; ".join(parts)
    if not valid or len(statement) > 500:
        return finding.model_copy(update={"claim_type": "hypothesis", "evidence_tool": None, "evidence": []})
    return finding.model_copy(update={
        "statement": statement,
        "evidence_tool": next(iter(tool_names)) if len(tool_names) == 1 else None,
    })
