#真正计算数据画像。

import math
import re
from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from app.models.dataset_profile import (
    CategoricalSummary,
    ColumnProfile,
    DatasetProfile,
    DuplicateSummary,
    MissingSummary,
    NumericSummary,
    QualityIssue,
    TopValue,
)

_DATE_NAME = re.compile(r"(^|_)(date|time|datetime|timestamp|created_at|updated_at|order_date)(_|$)", re.I)
_DATE_VALUE = re.compile(
    r"^\s*(?:\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?(?:[ T].*)?|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?:[ T].*)?|\d{8})\s*$"
)
_TEXT_NAME = re.compile(r"(description|comment|review|message|notes?|address|feedback|detail|text)", re.I)
_CATEGORY_NAME = re.compile(r"(region|category|product|gender|status|type|customer|country|city|code|\bid\b)", re.I)
# Unique-key style names: `order_id`, `user_id`, `product_id`, `zip_code`, `sku`, `uuid`, `key`.
_IDENTIFIER_NAME = re.compile(r"(^|_)(id|code|sku|uuid|key)s?(_|$)", re.I)
_IDENTIFIER_UNIQUE_RATE = 0.98
_IDENTIFIER_MIN_VALUES = 20


def _finite(value: object) -> float | None:
    if pd.isna(value):
        return None
    number = float(value)
    return round(number, 4) if math.isfinite(number) else None


def _is_datetime_column(name: str, series: pd.Series) -> bool:
    if is_datetime64_any_dtype(series.dtype):
        return True
    if not (pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype)):
        return False
    values = series.dropna().astype(str).str.strip()
    if len(values) < 3:
        return False
    sample = values.iloc[: min(len(values), 200)]
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed", utc=True)
    parse_rate = float(parsed.notna().mean())
    date_shape_rate = float(sample.str.match(_DATE_VALUE).mean())
    has_date_name = bool(_DATE_NAME.search(name))
    return parse_rate >= 0.8 and (has_date_name or date_shape_rate >= 0.8)

#判断一列是什么类型。
def _semantic_type(name: str, series: pd.Series) -> str:
    if _is_datetime_column(name, series):
        return "datetime"
    if is_bool_dtype(series.dtype):
        return "categorical"
    # Identifier-like names win over the numeric/category heuristics: `order_id`
    # is a unique key, not a measure, and must not be reported as an unknown type.
    if _IDENTIFIER_NAME.search(name):
        return "identifier"
    if is_numeric_dtype(series.dtype):
        return "numeric"
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return "unknown"
    if _TEXT_NAME.search(name):
        return "text"
    if _CATEGORY_NAME.search(name):
        return "categorical"
    unique_count = int(series.nunique(dropna=True))
    unique_rate = unique_count / max(len(values), 1)
    average_length = float(values.str.len().mean())
    if average_length >= 50 or (average_length >= 24 and unique_rate >= 0.8):
        return "text"
    if (
        len(values) >= _IDENTIFIER_MIN_VALUES
        and unique_count >= _IDENTIFIER_MIN_VALUES
        and unique_rate >= _IDENTIFIER_UNIQUE_RATE
    ):
        return "identifier"
    if unique_rate <= 0.5 or unique_count <= min(20, max(3, int(len(values) * 0.1))):
        return "categorical"
    return "unknown"


def _top_values(series: pd.Series, limit: int = 10) -> list[TopValue]:
    non_null = int(series.notna().sum())
    if not non_null:
        return []
    counts = series.dropna().astype(str).value_counts().head(limit)
    return [
        TopValue(value=str(value), count=int(count), percentage=round(int(count) * 100 / non_null, 2))
        for value, count in counts.items()
    ]


def _quality_assessment(
    missing_rate: float, duplicate_rate: float, unknown_rate: float
) -> tuple[int, list[QualityIssue]]:
    issues: list[QualityIssue] = []
    if missing_rate > 20:
        issues.append(QualityIssue(code="high_missing_rate", message=f"缺失单元格占 {missing_rate:.2f}%，高于 20%。", penalty=30))
    elif missing_rate > 5:
        issues.append(QualityIssue(code="moderate_missing_rate", message=f"缺失单元格占 {missing_rate:.2f}%，高于 5%。", penalty=15))
    elif missing_rate > 0:
        issues.append(QualityIssue(code="some_missing_values", message=f"存在缺失值，占所有单元格的 {missing_rate:.2f}%。", penalty=5))

    if duplicate_rate > 10:
        issues.append(QualityIssue(code="high_duplicate_rate", message=f"重复行占 {duplicate_rate:.2f}%，高于 10%。", penalty=20))
    elif duplicate_rate > 1:
        issues.append(QualityIssue(code="moderate_duplicate_rate", message=f"重复行占 {duplicate_rate:.2f}%，高于 1%。", penalty=10))
    elif duplicate_rate > 0:
        issues.append(QualityIssue(code="some_duplicate_rows", message=f"检测到重复行，占 {duplicate_rate:.2f}%。", penalty=3))

    if unknown_rate > 30:
        issues.append(QualityIssue(code="many_unknown_types", message=f"{unknown_rate:.2f}% 的列无法可靠推断语义类型。", penalty=15))
    elif unknown_rate > 10:
        issues.append(QualityIssue(code="some_unknown_types", message=f"{unknown_rate:.2f}% 的列无法可靠推断语义类型。", penalty=7))
    elif unknown_rate > 0:
        issues.append(QualityIssue(code="unknown_type_columns", message=f"{unknown_rate:.2f}% 的列无法可靠推断语义类型。", penalty=2))

    score = max(0, 100 - sum(issue.penalty for issue in issues))
    return score, issues

#把整个 DataFrame 体检一遍，最终生成 DatasetProfile。
def profile_dataframe(df: pd.DataFrame, dataset_name: str = "uploaded_dataset") -> DatasetProfile:
    """Create a deterministic, JSON-serializable profile without modifying the input frame."""
    if not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
        raise ValueError("A dataset must contain at least one column.")
    normalized_names = [str(name).strip().casefold() for name in df.columns]
    if any(not name for name in normalized_names) or len(set(normalized_names)) != len(normalized_names):
        raise ValueError("Dataset column names must be non-empty and unique.")
    safe_name = Path(dataset_name).name or "uploaded_dataset"
    row_count, column_count = df.shape
    if row_count == 0:
        raise ValueError("A dataset must contain at least one data row.")

    total_cells = row_count * column_count
    missing_total = int(df.isna().sum().sum())
    missing_rate = missing_total * 100 / total_cells if total_cells else 0.0
    missing_by_column = df.isna().sum()
    columns_with_missing = [str(name) for name, count in missing_by_column.items() if count > 0]
    duplicate_rows = int(df.duplicated().sum())
    duplicate_rate = duplicate_rows * 100 / row_count if row_count else 0.0

    columns: list[ColumnProfile] = []
    numeric_summary: list[NumericSummary] = []
    categorical_summary: list[CategoricalSummary] = []
    datetime_columns: list[str] = []
    unknown_count = 0

    for name in df.columns:
        series = df[name]
        semantic = _semantic_type(str(name), series)
        non_null = int(series.notna().sum())
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        top_values = _top_values(series) if semantic == "categorical" else []
        columns.append(
            ColumnProfile(
                column_name=str(name),
                pandas_dtype=str(series.dtype),
                semantic_type=semantic,
                non_null_count=non_null,
                missing_count=missing_count,
                missing_percentage=round(missing_count * 100 / row_count, 2),
                unique_count=unique_count,
                top_values=top_values,
            )
        )
        if semantic == "numeric":
            values = pd.to_numeric(series, errors="coerce")
            numeric_summary.append(
                NumericSummary(
                    column_name=str(name),
                    count=int(values.count()),
                    mean=_finite(values.mean()),
                    median=_finite(values.median()),
                    std=_finite(values.std()),
                    min=_finite(values.min()),
                    max=_finite(values.max()),
                )
            )
        elif semantic == "categorical":
            categorical_summary.append(
                CategoricalSummary(column_name=str(name), unique_count=unique_count, top_values=top_values)
            )
        elif semantic == "datetime":
            datetime_columns.append(str(name))
        elif semantic == "unknown":
            unknown_count += 1

    unknown_rate = unknown_count * 100 / column_count
    quality_score, quality_issues = _quality_assessment(missing_rate, duplicate_rate, unknown_rate)
    return DatasetProfile(
        dataset_name=safe_name,
        row_count=row_count,
        column_count=column_count,
        memory_usage_bytes=int(df.memory_usage(index=True, deep=True).sum()),
        columns=columns,
        missing=MissingSummary(
            total_missing_values=missing_total,
            missing_rate=round(missing_rate, 2),
            columns_with_missing=columns_with_missing,
        ),
        duplicates=DuplicateSummary(duplicate_rows=duplicate_rows, duplicate_rate=round(duplicate_rate, 2)),
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        datetime_columns=datetime_columns,
        quality_score=quality_score,
        quality_issues=quality_issues,
        quality_score_note="数据质量评分基于规则估算，仅用于快速评估。",
    )
