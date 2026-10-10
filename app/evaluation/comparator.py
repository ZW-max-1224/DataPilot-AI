from __future__ import annotations

import math
from typing import Any


def extract_path(value: Any, path: str) -> Any:
    """Read a dotted path from JSON-like structures; numeric segments index lists."""
    current = value
    for segment in path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                raise KeyError(path)
            current = current[segment]
        elif isinstance(current, list) and segment.isdigit():
            current = current[int(segment)]
        else:
            raise KeyError(path)
    return current


def compare_values(expected: Any, actual: Any, comparator: str, tolerance: float = 1e-6) -> bool:
    """Compare structured results without string similarity or numeric coercion."""
    if comparator == "exact_match":
        return type(expected) is type(actual) and expected == actual
    if comparator == "numeric_tolerance":
        if isinstance(expected, bool) or isinstance(actual, bool):
            return False
        if not isinstance(expected, (int, float)) or not isinstance(actual, (int, float)):
            return False
        return math.isfinite(float(actual)) and math.isfinite(float(expected)) and math.isclose(
            float(expected), float(actual), rel_tol=tolerance, abs_tol=tolerance
        )
    if comparator == "set_match":
        try:
            return { _freeze(item) for item in expected } == { _freeze(item) for item in actual }
        except TypeError:
            return False
    raise ValueError(f"Unsupported comparator: {comparator}")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value
