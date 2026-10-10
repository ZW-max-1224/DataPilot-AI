#定义数据画像长什么样。

from typing import Literal

from pydantic import BaseModel, Field

SemanticType = Literal["numeric", "categorical", "datetime", "text", "identifier", "unknown"]


class TopValue(BaseModel):
    value: str
    count: int = Field(ge=0)
    percentage: float = Field(ge=0, le=100)


class ColumnProfile(BaseModel):
    column_name: str
    pandas_dtype: str
    semantic_type: SemanticType
    non_null_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    missing_percentage: float = Field(ge=0, le=100)
    unique_count: int = Field(ge=0)
    top_values: list[TopValue] = Field(default_factory=list)


class NumericSummary(BaseModel):
    column_name: str
    count: int = Field(ge=0)
    mean: float | None
    median: float | None
    std: float | None
    min: float | None
    max: float | None


class CategoricalSummary(BaseModel):
    column_name: str
    unique_count: int = Field(ge=0)
    top_values: list[TopValue] = Field(default_factory=list)


class MissingSummary(BaseModel):
    total_missing_values: int = Field(ge=0)
    missing_rate: float = Field(ge=0, le=100)
    columns_with_missing: list[str] = Field(default_factory=list)


class DuplicateSummary(BaseModel):
    duplicate_rows: int = Field(ge=0)
    duplicate_rate: float = Field(ge=0, le=100)


class QualityIssue(BaseModel):
    code: str
    message: str
    penalty: int = Field(ge=0)


class DatasetProfile(BaseModel):
    dataset_name: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    memory_usage_bytes: int = Field(ge=0)
    columns: list[ColumnProfile]
    missing: MissingSummary
    duplicates: DuplicateSummary
    numeric_summary: list[NumericSummary]
    categorical_summary: list[CategoricalSummary]
    datetime_columns: list[str]
    quality_score: int = Field(ge=0, le=100)
    quality_issues: list[QualityIssue]
    quality_score_note: str
