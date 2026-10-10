from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Aggregation = Literal["sum", "mean", "median", "min", "max", "count"]
Frequency = Literal["day", "week", "month", "quarter", "year"]
SortOrder = Literal["asc", "desc"]
ChartType = Literal["line", "bar", "scatter", "histogram"]


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FindTopNRequest(ToolInput):
    column: str = Field(min_length=1, max_length=128)
    n: int = Field(ge=1, le=100)
    ascending: bool = False
    group_by: str | None = Field(default=None, min_length=1, max_length=128)


class GroupStatisticsRequest(ToolInput):
    group_by: str = Field(min_length=1, max_length=128)
    metric: str = Field(min_length=1, max_length=128)
    aggregation: Aggregation
    sort_by: Literal["metric", "group"] = "metric"
    sort_order: SortOrder = "desc"
    limit: int = Field(default=10, ge=1, le=100)


class TimeSeriesRequest(ToolInput):
    date_column: str = Field(min_length=1, max_length=128)
    metric: str = Field(min_length=1, max_length=128)
    aggregation: Aggregation = "sum"
    frequency: Frequency = "month"


class DuplicateRequest(ToolInput):
    example_limit: int = Field(default=10, ge=0, le=10)


class OutlierRequest(ToolInput):
    column: str = Field(min_length=1, max_length=128)


class CorrelationRequest(ToolInput):
    column_a: str = Field(min_length=1, max_length=128)
    column_b: str = Field(min_length=1, max_length=128)


class ChartRequest(ToolInput):
    chart_type: ChartType
    x: str = Field(min_length=1, max_length=128)
    y: str | None = Field(default=None, min_length=1, max_length=128)
    group_by: str | None = Field(default=None, min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=150)
    bins: int = Field(default=20, ge=5, le=100)


class DatasetOverview(BaseModel):
    dataset_name: str
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    missing_values: int = Field(ge=0)
    duplicate_rows: int = Field(ge=0)


class TableResult(BaseModel):
    tool_name: str
    rows: list[dict[str, Any]]
    result_count: int = Field(ge=0)
    rows_excluded: int = Field(default=0, ge=0)


class GroupStatisticsResult(TableResult):
    group_by: str
    metric: str
    aggregation: Aggregation
    output_metric: str


class TimeSeriesResult(TableResult):
    date_column: str
    metric: str
    aggregation: Aggregation
    frequency: Frequency
    first_period: str | None
    last_period: str | None
    change: float | None
    change_percentage: float | None
    trend: Literal["increasing", "decreasing", "stable", "mixed", "insufficient_data"]


class MissingValueRow(BaseModel):
    column: str
    missing_count: int = Field(ge=0)
    missing_percentage: float = Field(ge=0, le=100)


class MissingValuesResult(BaseModel):
    tool_name: str = "analyze_missing_values"
    rows: list[MissingValueRow]
    total_missing_values: int = Field(ge=0)
    missing_rate: float = Field(ge=0, le=100)


class DuplicateResult(BaseModel):
    tool_name: str = "analyze_duplicates"
    duplicate_rows: int = Field(ge=0)
    duplicate_percentage: float = Field(ge=0, le=100)
    duplicate_examples: list[dict[str, Any]]


class OutlierResult(BaseModel):
    tool_name: str = "detect_outliers"
    column: str
    q1: float
    q3: float
    iqr: float
    lower_bound: float
    upper_bound: float
    valid_count: int = Field(ge=0)
    outlier_count: int = Field(ge=0)
    outlier_percentage: float = Field(ge=0, le=100)
    examples: list[float]


class CorrelationResult(BaseModel):
    tool_name: str = "calculate_correlation"
    column_a: str
    column_b: str
    correlation: float
    sample_size: int = Field(ge=2)
    interpretation: Literal[
        "strong_positive", "moderate_positive", "weak_positive", "near_zero",
        "weak_negative", "moderate_negative", "strong_negative"
    ]
    note: str = "Correlation does not imply causation."


class ChartResult(BaseModel):
    chart_type: str
    x: str
    y: str | None
    group_by: str | None
    title: str
    plotted_rows: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class ToolError(BaseModel):
    code: str
    message: str


class ToolExecutionResult(BaseModel):
    tool_name: str
    success: bool
    data: dict[str, Any] | None = None
    error: ToolError | None = None
