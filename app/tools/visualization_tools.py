from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from app.models.analysis_result import ChartRequest, ChartResult
from app.tools.analysis_tools import (
    AnalysisToolError, _numeric_column, _require_column, _require_frame, _validate,
)

MAX_PLOT_POINTS = 2_000
MAX_BAR_CATEGORIES = 30
MAX_GROUP_SERIES = 10


@dataclass
class ChartArtifact:
    """A Streamlit-ready Matplotlib figure paired with JSON-safe chart metadata."""

    result: ChartResult
    figure: Figure

    def release(self) -> None:
        """Close the underlying figure so long-running sessions do not leak it.

        Matplotlib keeps every created figure in a global registry until it is
        closed. A Streamlit session lives for hours, so every chart must be released
        explicitly after it has been rendered (or when nobody will render it).
        """
        plt.close(self.figure)



def create_chart(
    df: pd.DataFrame, chart_type: str, x: str, y: str | None = None,
    group_by: str | None = None, title: str | None = None, bins: int = 20,
) -> ChartArtifact:
    """Build a supported chart from columns only; no generated code or file paths are accepted."""
    request = _validate(
        ChartRequest, chart_type=chart_type, x=x, y=y, group_by=group_by, title=title, bins=bins
    )
    _require_frame(df)
    _require_column(df, request.x)
    if request.group_by is not None:
        _require_column(df, request.group_by)
        if request.group_by == request.x:
            raise AnalysisToolError("group_by must differ from the x column.")

    warnings: list[str] = []
    default_title = request.title or f"{request.chart_type.title()}: {request.x}"
    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    plotted_rows = 0

    try:
        if request.chart_type == "histogram":
            if request.y is not None or request.group_by is not None:
                raise AnalysisToolError("Histogram uses x only; leave y and group_by empty.")
            values = _numeric_column(df, request.x).dropna()
            axis.hist(values, bins=request.bins, color="#4f7cff", edgecolor="white")
            axis.set_ylabel("Frequency")
            plotted_rows = len(values)

        elif request.chart_type == "scatter":
            if request.y is None:
                raise AnalysisToolError("Scatter chart requires numeric x and y columns.")
            if not pd.api.types.is_numeric_dtype(df[request.x]) or not pd.api.types.is_numeric_dtype(df[request.y]):
                raise AnalysisToolError("Scatter chart requires numeric x and y columns.")
            x_values = _numeric_column(df, request.x)
            y_values = _numeric_column(df, request.y)
            points = pd.DataFrame({"x": x_values, "y": y_values}).dropna()
            if points.empty:
                raise AnalysisToolError("No complete numeric row pairs are available for this chart.")
            if len(points) > MAX_PLOT_POINTS:
                points = points.sample(MAX_PLOT_POINTS, random_state=42)
                warnings.append(f"Scatter plot sampled {MAX_PLOT_POINTS:,} rows for readability.")
            axis.scatter(points["x"], points["y"], alpha=0.65, color="#4f7cff")
            axis.set_ylabel(request.y)
            plotted_rows = len(points)

        else:
            if request.y is None:
                raise AnalysisToolError(f"{request.chart_type.title()} chart requires a numeric y column.")
            y_values = _numeric_column(df, request.y)
            columns = [request.x, request.y] + ([request.group_by] if request.group_by else [])
            work = df[columns].copy()
            work[request.y] = y_values
            work = work.dropna(subset=[request.x, request.y])
            if work.empty:
                raise AnalysisToolError("No complete x/y rows are available for this chart.")

            group_columns = [request.x] + ([request.group_by] if request.group_by else [])
            grouped = work.groupby(group_columns, dropna=False, as_index=False)[request.y].sum()
            if request.group_by:
                group_totals = grouped.groupby(request.group_by, dropna=False)[request.y].sum()
                keep_groups = group_totals.sort_values(ascending=False).head(MAX_GROUP_SERIES).index
                if len(group_totals) > MAX_GROUP_SERIES:
                    warnings.append(f"Only the top {MAX_GROUP_SERIES} groups are shown.")
                    grouped = grouped[grouped[request.group_by].isin(keep_groups)]

            if request.chart_type == "line":
                if request.group_by:
                    for label, series in grouped.groupby(request.group_by, dropna=False):
                        series = series.sort_values(request.x)
                        axis.plot(series[request.x], series[request.y], marker="o", linewidth=1.8, label=str(label))
                    axis.legend(title=request.group_by)
                else:
                    grouped = grouped.sort_values(request.x)
                    axis.plot(grouped[request.x], grouped[request.y], marker="o", linewidth=1.8, color="#4f7cff")
                plotted_rows = len(grouped)
            else:
                category_totals = grouped.groupby(request.x, dropna=False)[request.y].sum()
                keep_categories = category_totals.sort_values(ascending=False).head(MAX_BAR_CATEGORIES).index
                if len(category_totals) > MAX_BAR_CATEGORIES:
                    warnings.append(f"Only the top {MAX_BAR_CATEGORIES} x categories are shown.")
                    grouped = grouped[grouped[request.x].isin(keep_categories)]
                if request.group_by:
                    chart_data = grouped.pivot(index=request.x, columns=request.group_by, values=request.y).fillna(0)
                    chart_data.plot(kind="bar", ax=axis, width=0.82)
                    axis.legend(title=request.group_by)
                else:
                    grouped = grouped.sort_values(request.y, ascending=False)
                    axis.bar(grouped[request.x].astype(str), grouped[request.y], color="#4f7cff")
                plotted_rows = len(grouped)
                axis.tick_params(axis="x", labelrotation=35)
            axis.set_ylabel(request.y)

        axis.set_title(default_title)
        if request.chart_type != "histogram":
            axis.set_xlabel(request.x)
        result = ChartResult(
            chart_type=request.chart_type, x=request.x, y=request.y, group_by=request.group_by,
            title=default_title, plotted_rows=plotted_rows, warnings=warnings,
        )
        return ChartArtifact(result=result, figure=figure)
    except Exception:
        plt.close(figure)
        raise
