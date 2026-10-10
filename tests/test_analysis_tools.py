import matplotlib.pyplot as plt
import pandas as pd
import pytest

from app.models.analysis_result import DatasetOverview
from app.tools.analysis_tools import (
    AnalysisToolError, analyze_duplicates, analyze_missing_values, calculate_correlation,
    calculate_group_statistics, calculate_time_series, detect_outliers, find_top_n,
    get_dataset_overview,
)
from app.tools.registry import (
    ALLOWED_ANALYSIS_TOOLS, get_analysis_tool, invoke_analysis_tool,
)
from app.tools.visualization_tools import ChartArtifact, create_chart


@pytest.fixture
def sales_df():
    return pd.DataFrame(
        {
            "date": ["2025-01-03", "2025-01-20", "2025-02-04", "2025-02-18", "2025-03-02", "2025-03-22"],
            "product": ["A", "A", "B", "B", "C", "C"],
            "region": ["East", "East", "West", "West", "East", "West"],
            "sales": [10, 20, 30, 40, 50, 60],
            "quantity": [1, 2, 3, 4, 5, 6],
            "profit": [2, 4, 3, 8, 5, 12],
        }
    )


def test_dataset_overview_reuses_profiler(sales_df):
    result = get_dataset_overview(sales_df, "orders.csv")
    assert isinstance(result, DatasetOverview)
    assert result.dataset_name == "orders.csv"
    assert result.rows == 6
    assert result.numeric_columns == ["sales", "quantity", "profit"]
    assert result.datetime_columns == ["date"]


def test_top_n_by_group_matches_expected_pandas_totals(sales_df):
    result = find_top_n(sales_df, column="sales", n=2, group_by="product")
    assert result.rows == [{"product": "C", "sales": 110}, {"product": "B", "sales": 70}]


@pytest.mark.parametrize("n", [0, 101, -1])
def test_top_n_rejects_invalid_n(sales_df, n):
    with pytest.raises(AnalysisToolError, match="Invalid n"):
        find_top_n(sales_df, "sales", n=n)


def test_top_n_rejects_unknown_columns(sales_df):
    with pytest.raises(AnalysisToolError, match="does not exist"):
        find_top_n(sales_df, "revenue", n=5)
    with pytest.raises(AnalysisToolError, match="does not exist"):
        find_top_n(sales_df, "sales", n=5, group_by="missing_group")


def test_group_statistics_sum_mean_count_and_sort(sales_df):
    summed = calculate_group_statistics(sales_df, "region", "sales", "sum")
    assert summed.rows[0] == {"region": "West", "sales_sum": 130}
    mean = calculate_group_statistics(sales_df, "region", "sales", "mean", sort_order="asc")
    assert mean.rows[-1]["sales_mean"] == pytest.approx(130 / 3)
    counted = calculate_group_statistics(sales_df, "region", "product", "count")
    assert {row["product_count"] for row in counted.rows} == {3}


def test_group_statistics_rejects_columns_and_bad_aggregation(sales_df):
    with pytest.raises(AnalysisToolError, match="does not exist"):
        calculate_group_statistics(sales_df, "territory", "sales")
    with pytest.raises(AnalysisToolError, match="Invalid aggregation"):
        calculate_group_statistics(sales_df, "region", "sales", "variance")


def test_time_series_month_year_change_and_trend(sales_df):
    monthly = calculate_time_series(sales_df, "date", "sales", "sum", "month")
    assert monthly.rows == [
        {"period": "2025-01", "sales_sum": 30.0},
        {"period": "2025-02", "sales_sum": 70.0},
        {"period": "2025-03", "sales_sum": 110.0},
    ]
    assert monthly.change == 80.0
    assert monthly.change_percentage == pytest.approx(266.666667)
    assert monthly.trend == "increasing"
    yearly = calculate_time_series(sales_df, "date", "sales", "sum", "year")
    assert yearly.rows == [{"period": "2025", "sales_sum": 210.0}]
    assert yearly.trend == "insufficient_data"


def test_time_series_rejects_invalid_date_column(sales_df):
    with pytest.raises(AnalysisToolError, match="does not exist"):
        calculate_time_series(sales_df, "not_a_date", "sales")
    sales_df["bad_date"] = "not a date"
    with pytest.raises(AnalysisToolError, match="80% valid values required"):
        calculate_time_series(sales_df, "bad_date", "sales")


def test_time_series_counts_excluded_rows_once_and_handles_zero_baseline():
    frame = pd.DataFrame({
        "date": ["2025-01-01", "bad-date", "2025-02-01", "2025-03-01", "2025-04-01"],
        "value": [0, 99, None, 4, 8],
    })
    result = calculate_time_series(frame, "date", "value", frequency="month")
    assert result.rows_excluded == 2
    assert result.trend == "increasing"


def test_missing_value_analysis_sorts_and_handles_no_missing(sales_df):
    sales_df.loc[0, "region"] = None
    sales_df.loc[0, "profit"] = None
    result = analyze_missing_values(sales_df)
    assert result.rows[0].column in {"profit", "region"}
    assert result.total_missing_values == 2
    clean = analyze_missing_values(sales_df.fillna(0))
    assert clean.total_missing_values == 0
    assert clean.missing_rate == 0


def test_duplicate_analysis_caps_examples_at_ten(sales_df):
    duplicated = pd.concat([sales_df, pd.concat([sales_df.iloc[[0]]] * 12)], ignore_index=True)
    result = analyze_duplicates(duplicated, example_limit=10)
    assert result.duplicate_rows == 12
    assert len(result.duplicate_examples) == 10
    clean = analyze_duplicates(sales_df)
    assert clean.duplicate_rows == 0
    assert clean.duplicate_examples == []


def test_outlier_detection_iqr_and_numeric_validation():
    df = pd.DataFrame({"sales": [10, 11, 12, 13, 14, 1000], "region": ["E", "E", "W", "W", "N", "S"]})
    result = detect_outliers(df, "sales")
    assert result.q1 == 11.25
    assert result.q3 == 13.75
    assert result.iqr == 2.5
    assert result.upper_bound == 17.5
    assert result.outlier_count == 1
    with pytest.raises(AnalysisToolError, match="must be numeric"):
        detect_outliers(df, "region")


def test_correlation_interpretation_and_validation(sales_df):
    sales_df["quantity_copy"] = sales_df["quantity"] * 2
    result = calculate_correlation(sales_df, "quantity", "quantity_copy")
    assert result.correlation == 1.0
    assert result.interpretation == "strong_positive"
    assert result.note == "Correlation does not imply causation."
    with pytest.raises(AnalysisToolError, match="must be numeric"):
        calculate_correlation(sales_df, "sales", "region")


@pytest.mark.parametrize(
    "chart_type,kwargs",
    [
        ("line", {"x": "date", "y": "sales"}),
        ("bar", {"x": "product", "y": "sales"}),
        ("scatter", {"x": "sales", "y": "profit"}),
        ("histogram", {"x": "sales"}),
    ],
)
def test_chart_generation_supported_types(sales_df, chart_type, kwargs):
    result = create_chart(sales_df, chart_type=chart_type, **kwargs)
    assert isinstance(result, ChartArtifact)
    assert result.result.chart_type == chart_type
    assert result.result.plotted_rows > 0
    assert result.figure.axes
    plt.close(result.figure)


def test_chart_validation(sales_df):
    with pytest.raises(AnalysisToolError, match="Invalid chart_type"):
        create_chart(sales_df, chart_type="pie", x="product", y="sales")
    with pytest.raises(AnalysisToolError, match="requires numeric x and y"):
        create_chart(sales_df, chart_type="scatter", x="region", y="sales")


def test_tool_registry_only_exposes_allowlisted_tools(sales_df):
    expected = {
        "get_dataset_overview", "find_top_n", "calculate_group_statistics",
        "calculate_time_series", "analyze_missing_values", "analyze_duplicates",
        "detect_outliers", "calculate_correlation", "create_chart", "search_knowledge",
    }
    assert set(ALLOWED_ANALYSIS_TOOLS) == expected
    assert get_analysis_tool("find_top_n") is find_top_n
    with pytest.raises(AnalysisToolError, match="not in the analysis allowlist"):
        get_analysis_tool("run_shell")
    denied = invoke_analysis_tool(sales_df, "run_shell", command="whoami")
    assert not denied.success
    assert denied.error.code == "invalid_tool_request"
    allowed = invoke_analysis_tool(sales_df, "find_top_n", column="sales", n=1)
    assert allowed.success
    assert allowed.data["rows"][0]["sales"] == 60


def test_chart_through_the_json_only_boundary_is_released_not_leaked(sales_df):
    """Without a renderer the figure must be closed, or pyplot's registry grows."""
    before = plt.get_fignums()
    result = invoke_analysis_tool(sales_df, "create_chart",
                                  parameters={"chart_type": "line", "x": "date", "y": "sales"})
    assert result.success
    assert result.data["chart_type"] == "line"          # metadata still reaches the model
    assert plt.get_fignums() == before                   # and the figure was released


def test_chart_artifact_is_handed_to_the_caller_for_rendering(sales_df):
    artifacts: list[ChartArtifact] = []
    before = plt.get_fignums()
    result = invoke_analysis_tool(sales_df, "create_chart",
                                  parameters={"chart_type": "bar", "x": "product", "y": "sales"},
                                  artifacts=artifacts)
    assert result.success
    assert len(artifacts) == 1
    assert plt.get_fignums() != before                   # still renderable
    artifacts[0].release()
    assert plt.get_fignums() == before


def test_numeric_date_column_is_rejected_instead_of_becoming_1970():
    """pd.to_datetime reads integers as nanosecond epochs: 2021 silently became 1970."""
    frame = pd.DataFrame({"year": [2021, 2022, 2023], "sales": [10.0, 20.0, 30.0]})
    with pytest.raises(AnalysisToolError, match="numeric"):
        calculate_time_series(frame, date_column="year", metric="sales", frequency="year")


def test_time_series_periods_keep_the_original_calendar_day():
    frame = pd.DataFrame({"date": ["2025-01-01", "2025-02-01", "2025-03-01"],
                          "sales": [1.0, 2.0, 3.0]})
    result = calculate_time_series(frame, date_column="date", metric="sales", frequency="month")
    assert [row["period"] for row in result.rows] == ["2025-01", "2025-02", "2025-03"]


def test_the_two_group_sum_tools_agree_about_all_missing_groups():
    """"sum by group" must not report 0.0 where find_top_n reports "no data"."""
    frame = pd.DataFrame({"g": ["a", "a", "b"], "v": [None, None, 5.0]})
    statistics = calculate_group_statistics(frame, group_by="g", metric="v", aggregation="sum")
    top = find_top_n(frame, column="v", n=5, group_by="g")
    stats_by_group = {row["g"]: row["v_sum"] for row in statistics.rows}
    top_by_group = {row["g"]: row["v"] for row in top.rows}
    assert stats_by_group["a"] is None and top_by_group["a"] is None
    assert stats_by_group["b"] == top_by_group["b"] == 5.0


def test_chart_configures_a_cjk_capable_font():
    """Chinese chart text must not render as tofu boxes.

    A live demo showed the title "各月销售额趋势（2025）" drawn as empty rectangles,
    because matplotlib's default font has no CJK glyphs.
    """
    import matplotlib
    from matplotlib import font_manager

    from app.tools import visualization_tools

    available = {font.name for font in font_manager.fontManager.ttflist}
    installed = [name for name in visualization_tools.CJK_FONT_CANDIDATES if name in available]
    if installed:
        configured = list(matplotlib.rcParams["font.sans-serif"])
        assert any(name in configured for name in installed), (
            "a CJK-capable font is installed but matplotlib is not configured to use it"
        )
    # Several CJK fonts lack the Unicode minus, which turns negative labels into boxes.
    assert matplotlib.rcParams["axes.unicode_minus"] is False


def test_many_x_categories_are_thinned_but_every_point_is_kept():
    """344 daily points once produced 344 tick labels, i.e. a black band on the axis."""
    days = pd.date_range("2025-01-01", periods=200, freq="D").strftime("%Y-%m-%d")
    frame = pd.DataFrame({"day": list(days), "value": [float(index) for index in range(200)]})
    artifact = create_chart(frame, chart_type="line", x="day", y="value")
    try:
        assert len(artifact.figure.axes[0].get_xticks()) <= 8
        assert artifact.result.plotted_rows == 200          # no data point is dropped
        assert any("tick labels" in warning for warning in artifact.result.warnings)
    finally:
        artifact.release()


def test_few_x_categories_are_not_thinned():
    frame = pd.DataFrame({"month": ["2025-01", "2025-02", "2025-03"], "sales": [1.0, 2.0, 3.0]})
    artifact = create_chart(frame, chart_type="line", x="month", y="sales")
    try:
        assert len(artifact.figure.axes[0].get_xticks()) == 3
        assert artifact.result.warnings == []
    finally:
        artifact.release()
