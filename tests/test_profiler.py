import pandas as pd

from app.services.profiler import profile_dataframe


def test_profile_counts_missing_duplicates_and_columns():
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-02", None],
            "sales": [10.0, 20.0, 20.0, None],
            "region": ["East", "West", "West", "East"],
            "description": ["a", "b", "b", "c"],
        }
    )
    profile = profile_dataframe(df, "folder/orders.csv")
    assert profile.dataset_name == "orders.csv"
    assert (profile.row_count, profile.column_count) == (4, 4)
    assert profile.missing.total_missing_values == 2
    assert profile.missing.columns_with_missing == ["date", "sales"]
    assert profile.duplicates.duplicate_rows == 1
    assert profile.duplicates.duplicate_rate == 25.0
    assert profile.datetime_columns == ["date"]
    assert profile.quality_score < 100


def test_numeric_statistics_are_structured_and_finite():
    profile = profile_dataframe(pd.DataFrame({"sales": [1, 2, 3, 4]}))
    item = profile.numeric_summary[0]
    assert (item.count, item.mean, item.median, item.min, item.max) == (4, 2.5, 2.5, 1.0, 4.0)
    assert item.std is not None


def test_categorical_top_values_are_counted():
    profile = profile_dataframe(pd.DataFrame({"region": ["East", "East", "West", None]}))
    item = profile.categorical_summary[0]
    assert item.unique_count == 2
    assert item.top_values[0].value == "East"
    assert item.top_values[0].count == 2
    assert item.top_values[0].percentage == 66.67


def test_object_date_column_is_detected_but_arbitrary_strings_are_not():
    df = pd.DataFrame(
        {
            "created_at": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "comment": ["one", "two", "three", "four"],
        }
    )
    profile = profile_dataframe(df)
    types = {column.column_name: column.semantic_type for column in profile.columns}
    assert types == {"created_at": "datetime", "comment": "text"}


def test_quality_score_is_deterministic_and_explained():
    clean = profile_dataframe(pd.DataFrame({"sales": [10, 20, 30], "region": ["E", "W", "N"]}))
    dirty = profile_dataframe(
        pd.DataFrame(
            {"sales": [10, None, None, None, None, None], "region": ["E", "E", "E", "E", "E", "E"]}
        )
    )
    assert clean.quality_score == 100
    assert dirty.quality_score < clean.quality_score
    assert any(issue.code == "high_missing_rate" for issue in dirty.quality_issues)
    assert dirty.quality_score_note


def test_empty_dataframes_are_rejected():
    import pytest

    with pytest.raises(ValueError, match="at least one data row"):
        profile_dataframe(pd.DataFrame({"sales": []}))


def test_profiler_rejects_duplicate_names():
    import pytest

    df = pd.DataFrame([[1, 2]], columns=["sales", "Sales "])
    with pytest.raises(ValueError, match="non-empty and unique"):
        profile_dataframe(df)


def test_identifier_like_column_names_become_identifier():
    df = pd.DataFrame(
        {
            "order_id": [f"ORD{i:03d}" for i in range(40)],
            "user_id": [f"U{i:03d}" for i in range(40)],
            "product_id": [f"P{i:03d}" for i in range(40)],
        }
    )
    profile = profile_dataframe(df)
    types = {column.column_name: column.semantic_type for column in profile.columns}
    assert types == {"order_id": "identifier", "user_id": "identifier", "product_id": "identifier"}


def test_unique_short_values_without_an_identifier_name_are_identifiers():
    profile = profile_dataframe(pd.DataFrame({"reference": [f"REF{i:04d}" for i in range(40)]}))
    assert profile.columns[0].semantic_type == "identifier"


def test_clean_frame_with_an_identifier_column_scores_100():
    profile = profile_dataframe(
        pd.DataFrame(
            {
                "order_id": [f"ORD{i}" for i in range(50)],
                "sales": [float(i) for i in range(50)],
                "region": ["East", "West"] * 25,
            }
        )
    )
    types = {column.column_name: column.semantic_type for column in profile.columns}
    assert types == {"order_id": "identifier", "sales": "numeric", "region": "categorical"}
    assert profile.quality_issues == []
    assert profile.quality_score == 100


def test_unclassifiable_free_text_column_is_still_unknown():
    values = [f"free text note number {index} about this record" for index in range(24)]
    values += ["free text note number 0 about this record"] * 16
    profile = profile_dataframe(pd.DataFrame({"payload": values}))
    assert profile.columns[0].semantic_type == "unknown"
    assert any(issue.code == "many_unknown_types" for issue in profile.quality_issues)
    assert profile.quality_score < 100
