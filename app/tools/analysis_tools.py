import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import ValidationError

from app.models.analysis_result import (
    CorrelationRequest, CorrelationResult, DatasetOverview, DuplicateRequest,
    DuplicateResult, FindTopNRequest, GroupStatisticsRequest, GroupStatisticsResult,
    MissingValueRow, MissingValuesResult, OutlierRequest, OutlierResult, TableResult,
    TimeSeriesRequest, TimeSeriesResult,
)
from app.models.dataset_profile import DatasetProfile
from app.services.profiler import profile_dataframe

AGGREGATIONS = {
    "sum": "sum", "mean": "mean", "median": "median",
    "min": "min", "max": "max", "count": "count",
}
FREQUENCIES = {"day": "D", "week": "W", "month": "MS", "quarter": "QS", "year": "YS"}


class AnalysisToolError(ValueError):
    """A safe, actionable input or dataset validation error."""


def _validate(model: type, **values: Any):
    try:
        return model.model_validate(values)
    except ValidationError as exc:
        error = exc.errors()[0]
        location = ".".join(str(item) for item in error["loc"]) or "input"
        raise AnalysisToolError(f"Invalid {location}: {error['msg']}.") from exc


def _require_frame(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise AnalysisToolError("The active dataset is not a pandas DataFrame.")
    if df.empty or df.shape[1] == 0:
        raise AnalysisToolError("The active dataset has no rows or columns to analyze.")
    normalized = [str(name).strip().casefold() for name in df.columns]
    if any(not name for name in normalized) or len(set(normalized)) != len(normalized):
        raise AnalysisToolError("Dataset columns must have non-empty, unique names.")


def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise AnalysisToolError(f"Column '{column}' does not exist in the active dataset.")


def _number(value: Any) -> float | None:
    if pd.isna(value):
        return None
    number = float(value)
    return round(number, 6) if math.isfinite(number) else None


def _json_value(value: Any) -> int | float | str | bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _number(value)
    return str(value)


def _rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{str(key): _json_value(value) for key, value in record.items()} for record in records]


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    _require_column(df, column)
    if not is_numeric_dtype(df[column].dtype) or pd.api.types.is_bool_dtype(df[column].dtype):
        raise AnalysisToolError(f"Column '{column}' must be numeric for this operation.")
    values = pd.to_numeric(df[column], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    if int(values.notna().sum()) < 1:
        raise AnalysisToolError(f"Column '{column}' has no finite numeric values.")
    return values


def get_dataset_overview(
    df: pd.DataFrame, dataset_name: str = "active_dataset", profile: DatasetProfile | None = None
) -> DatasetOverview:
    """Summarize the active frame by reusing the Phase 2 Dataset Profiler."""
    _require_frame(df)
    current_profile = profile or profile_dataframe(df, dataset_name)
    return DatasetOverview(
        dataset_name=Path(current_profile.dataset_name).name,
        rows=current_profile.row_count,
        columns=current_profile.column_count,
        numeric_columns=[c.column_name for c in current_profile.columns if c.semantic_type == "numeric"],
        categorical_columns=[c.column_name for c in current_profile.columns if c.semantic_type == "categorical"],
        datetime_columns=current_profile.datetime_columns,
        missing_values=current_profile.missing.total_missing_values,
        duplicate_rows=current_profile.duplicates.duplicate_rows,
    )


def find_top_n(
    df: pd.DataFrame, column: str, n: int = 10, ascending: bool = False, group_by: str | None = None
) -> TableResult:
    """Return top rows or top groups by a numeric metric; never evaluates expressions."""
    request = _validate(FindTopNRequest, column=column, n=n, ascending=ascending, group_by=group_by)
    _require_frame(df)
    _require_column(df, request.column)
    if request.group_by is not None:
        _require_column(df, request.group_by)
        values = _numeric_column(df, request.column)
        work = pd.DataFrame({request.group_by: df[request.group_by], request.column: values})
        grouped = work.groupby(request.group_by, dropna=False, as_index=False)[request.column].sum(min_count=1)
        selected = grouped.sort_values(request.column, ascending=request.ascending, na_position="last").head(request.n)
        return TableResult(
            tool_name="find_top_n", rows=_rows(selected.to_dict(orient="records")),
            result_count=len(selected), rows_excluded=int(values.isna().sum()),
        )
    selected = df.sort_values(request.column, ascending=request.ascending, na_position="last").head(request.n)
    return TableResult(
        tool_name="find_top_n", rows=_rows(selected.to_dict(orient="records")),
        result_count=len(selected),
    )


def calculate_group_statistics(
    df: pd.DataFrame, group_by: str, metric: str, aggregation: str = "sum",
    sort_by: str = "metric", sort_order: str = "desc", limit: int = 10,
) -> GroupStatisticsResult:
    """Aggregate a numeric measure (or count non-null values) by one dataset column."""
    request = _validate(
        GroupStatisticsRequest, group_by=group_by, metric=metric, aggregation=aggregation,
        sort_by=sort_by, sort_order=sort_order, limit=limit,
    )
    _require_frame(df)
    _require_column(df, request.group_by)
    _require_column(df, request.metric)
    if request.group_by == request.metric:
        raise AnalysisToolError("group_by and metric must be different columns.")
    if request.aggregation == "count":
        values = df[request.metric]
        excluded = int(values.isna().sum())
    else:
        values = _numeric_column(df, request.metric)
        excluded = int(values.isna().sum())
    output_metric = f"{request.metric}_{request.aggregation}"
    work = pd.DataFrame({request.group_by: df[request.group_by], "_metric": values})
    if request.aggregation == "sum":
        # min_count=1 keeps "no data" distinct from "zero". Without it a group whose
        # values are all NaN is reported as 0.0, which reads as a real measurement and
        # disagreed with find_top_n's identical question.
        grouped = work.groupby(request.group_by, dropna=False)["_metric"].sum(min_count=1).reset_index(name=output_metric)
    else:
        grouped = work.groupby(request.group_by, dropna=False)["_metric"].agg(AGGREGATIONS[request.aggregation]).reset_index(name=output_metric)
    sort_column = output_metric if request.sort_by == "metric" else request.group_by
    grouped = grouped.sort_values(sort_column, ascending=request.sort_order == "asc", na_position="last").head(request.limit)
    records = _rows(grouped.to_dict(orient="records"))
    return GroupStatisticsResult(
        tool_name="calculate_group_statistics", rows=records, result_count=len(records),
        rows_excluded=excluded, group_by=request.group_by, metric=request.metric,
        aggregation=request.aggregation, output_metric=output_metric,
    )


def _parse_datetime(series: pd.Series, column: str) -> tuple[pd.Series, int]:
    """Parse a date column, refusing the silent nanosecond-epoch trap.

    ``pd.to_datetime`` interprets an integer column as nanoseconds since the epoch,
    so ``[2021, 2022, 2023]`` silently becomes January 1970 and every period label is
    wrong while the tool still reports success. A numeric date column is therefore
    rejected with an actionable message instead of being guessed at.
    """
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        raise AnalysisToolError(
            f"Column '{column}' is numeric, so it cannot be read as a date safely: "
            "numeric values are interpreted as nanosecond timestamps (e.g. 2021 becomes 1970). "
            "Convert it to a real datetime column or to a date string such as '2021-01-01' and retry."
        )
    non_null = int(series.notna().sum())
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    parsed_count = int(parsed.notna().sum())
    if non_null == 0 or parsed_count / non_null < 0.8:
        raise AnalysisToolError(f"Column '{column}' is not reliably parseable as a date (80% valid values required).")
    return parsed, len(series) - parsed_count


def _format_period(value: pd.Timestamp, frequency: str) -> str:
    if frequency == "quarter":
        return f"{value.year}-Q{value.quarter}"
    if frequency == "year":
        return f"{value.year}"
    if frequency == "month":
        return value.strftime("%Y-%m")
    if frequency == "week":
        return value.strftime("%Y-%m-%d")
    return value.strftime("%Y-%m-%d")


def _trend(values: list[float]) -> str:
    if len(values) < 2:
        return "insufficient_data"
    first, last = values[0], values[-1]
    scale = max(abs(first), abs(last), 1.0)
    relative_change = (last - first) / scale
    differences = pd.Series(values).diff().dropna()
    positive_ratio = float((differences > 0).mean())
    negative_ratio = float((differences < 0).mean())
    if abs(relative_change) <= 0.05:
        return "stable"
    if relative_change > 0.05 and positive_ratio >= 0.6:
        return "increasing"
    if relative_change < -0.05 and negative_ratio >= 0.6:
        return "decreasing"
    return "mixed"


def calculate_time_series(
    df: pd.DataFrame, date_column: str, metric: str, aggregation: str = "sum", frequency: str = "month"
) -> TimeSeriesResult:
    """Aggregate values into fixed calendar periods and apply an explicit trend heuristic."""
    request = _validate(
        TimeSeriesRequest, date_column=date_column, metric=metric, aggregation=aggregation, frequency=frequency
    )
    _require_frame(df)
    _require_column(df, request.date_column)
    _require_column(df, request.metric)
    parsed_dates, rows_without_date = _parse_datetime(df[request.date_column], request.date_column)
    if request.aggregation == "count":
        values = df[request.metric]
    else:
        values = _numeric_column(df, request.metric)
    valid = parsed_dates.notna()
    work = pd.DataFrame({"_date": parsed_dates[valid], "_value": values[valid]})
    grouper = work.groupby(pd.Grouper(key="_date", freq=FREQUENCIES[request.frequency]))["_value"]
    if request.aggregation == "sum":
        grouped = grouper.sum(min_count=1).dropna()
    else:
        grouped = grouper.agg(AGGREGATIONS[request.aggregation]).dropna()
    if grouped.empty:
        raise AnalysisToolError("No usable time-series observations remain after parsing.")
    records = [
        {"period": _format_period(pd.Timestamp(period), request.frequency), f"{request.metric}_{request.aggregation}": _number(value)}
        for period, value in grouped.items()
    ]
    numeric_values = [float(value) for value in grouped.tolist()]
    first, last = numeric_values[0], numeric_values[-1]
    change = last - first
    change_percentage = change / abs(first) * 100 if first != 0 else None
    return TimeSeriesResult(
        tool_name="calculate_time_series", rows=records, result_count=len(records),
        rows_excluded=int((~valid | values.isna()).sum()),
        date_column=request.date_column, metric=request.metric, aggregation=request.aggregation,
        frequency=request.frequency, first_period=records[0]["period"], last_period=records[-1]["period"],
        change=_number(change), change_percentage=_number(change_percentage), trend=_trend(numeric_values),
    )


def analyze_missing_values(df: pd.DataFrame) -> MissingValuesResult:
    """Return per-column missing values, ordered from largest missing rate to smallest."""
    _require_frame(df)
    row_count, column_count = df.shape
    items = [
        MissingValueRow(
            column=str(column), missing_count=int(df[column].isna().sum()),
            missing_percentage=round(float(df[column].isna().mean() * 100), 4),
        )
        for column in df.columns
    ]
    items.sort(key=lambda item: (-item.missing_percentage, item.column))
    total_missing = int(df.isna().sum().sum())
    return MissingValuesResult(
        rows=items, total_missing_values=total_missing,
        missing_rate=round(total_missing * 100 / (row_count * column_count), 4),
    )


def analyze_duplicates(df: pd.DataFrame, example_limit: int = 10) -> DuplicateResult:
    """Summarize exact duplicate rows and return no more than ten examples."""
    request = _validate(DuplicateRequest, example_limit=example_limit)
    _require_frame(df)
    duplicate_mask = df.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    percentage = duplicate_count * 100 / len(df.index)
    examples = _rows(df.loc[duplicate_mask].head(request.example_limit).to_dict(orient="records"))
    return DuplicateResult(
        duplicate_rows=duplicate_count, duplicate_percentage=round(percentage, 4), duplicate_examples=examples
    )


def detect_outliers(df: pd.DataFrame, column: str) -> OutlierResult:
    """Detect IQR outliers in one numeric column; returned examples are capped at ten."""
    request = _validate(OutlierRequest, column=column)
    _require_frame(df)
    values = _numeric_column(df, request.column).dropna()
    q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (values < lower) | (values > upper)
    outliers = values[mask]
    return OutlierResult(
        column=request.column, q1=round(q1, 6), q3=round(q3, 6), iqr=round(iqr, 6),
        lower_bound=round(lower, 6), upper_bound=round(upper, 6), valid_count=len(values),
        outlier_count=int(mask.sum()), outlier_percentage=round(float(mask.mean() * 100), 4),
        examples=[round(float(value), 6) for value in outliers.head(10).tolist()],
    )


def calculate_correlation(df: pd.DataFrame, column_a: str, column_b: str) -> CorrelationResult:
    """Calculate Pearson correlation on pairwise complete numeric observations."""
    request = _validate(CorrelationRequest, column_a=column_a, column_b=column_b)
    _require_frame(df)
    if request.column_a == request.column_b:
        raise AnalysisToolError("Choose two different columns for correlation.")
    left = _numeric_column(df, request.column_a)
    right = _numeric_column(df, request.column_b)
    paired = pd.DataFrame({"a": left, "b": right}).dropna()
    if len(paired) < 2:
        raise AnalysisToolError("Correlation requires at least two complete numeric row pairs.")
    if paired["a"].nunique() < 2 or paired["b"].nunique() < 2:
        raise AnalysisToolError("Correlation is undefined when either column has no variation.")
    coefficient = float(paired["a"].corr(paired["b"]))
    magnitude = abs(coefficient)
    if magnitude < 0.1:
        interpretation = "near_zero"
    elif magnitude < 0.4:
        interpretation = "weak_positive" if coefficient > 0 else "weak_negative"
    elif magnitude < 0.7:
        interpretation = "moderate_positive" if coefficient > 0 else "moderate_negative"
    else:
        interpretation = "strong_positive" if coefficient > 0 else "strong_negative"
    return CorrelationResult(
        column_a=request.column_a, column_b=request.column_b, correlation=round(coefficient, 6),
        sample_size=len(paired), interpretation=interpretation,
    )
