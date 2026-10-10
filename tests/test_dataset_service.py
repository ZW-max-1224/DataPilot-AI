from io import BytesIO

import pandas as pd
import pytest

from app.services import dataset_service
from app.services.dataset_service import DatasetLoadError, load_csv, load_dataset, load_excel


def csv_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def test_load_normal_csv():
    df = load_csv(csv_bytes("date,sales,region\n2025-01-01,10,East\n2025-01-02,20,West\n"))
    assert df.shape == (2, 3)
    assert list(df.columns) == ["date", "sales", "region"]


@pytest.mark.parametrize("content", [b"", b"date,sales\n"])
def test_empty_file_or_header_only_is_rejected(content):
    with pytest.raises(DatasetLoadError):
        load_csv(content)


def test_bad_csv_rows_return_safe_error():
    with pytest.raises(DatasetLoadError, match="Could not read this CSV"):
        load_csv(csv_bytes("a,b\n1,2\n3,4,5\n"))


def test_duplicate_column_names_are_rejected_case_insensitively():
    with pytest.raises(DatasetLoadError, match="Duplicate column names"):
        load_csv(csv_bytes("Region, region \nEast,West\n"))


def test_oversized_upload_is_rejected(monkeypatch):
    monkeypatch.setattr(dataset_service, "MAX_UPLOAD_BYTES", 4)
    with pytest.raises(DatasetLoadError, match="20 MiB upload limit"):
        load_csv(csv_bytes("a,b\n1,2\n"))


def test_unsupported_extension_is_rejected():
    with pytest.raises(DatasetLoadError, match="Unsupported file type"):
        load_dataset("data.json", b"{}")


def test_xlsx_and_dispatch():
    buffer = BytesIO()
    pd.DataFrame({"sales": [10, 20], "region": ["East", "West"]}).to_excel(buffer, index=False)
    content = buffer.getvalue()
    df = load_excel(content, "sales.xlsx")
    dispatched = load_dataset("sales.xlsx", content)
    assert df.equals(dispatched)
    assert df.shape == (2, 2)


def test_invalid_excel_returns_friendly_error():
    with pytest.raises(DatasetLoadError, match="Could not read this Excel file"):
        load_excel(b"not an excel workbook", "broken.xlsx")


def test_excel_duplicate_headers_are_rejected():
    buffer = BytesIO()
    pd.DataFrame([["East", "West"]]).to_excel(
        buffer, index=False, header=["Region", "region"]
    )
    with pytest.raises(DatasetLoadError, match="Duplicate column names"):
        load_excel(buffer.getvalue(), "duplicate.xlsx")


def test_gb18030_csv_is_supported():
    content = "地区,销售额\n青岛,125\n".encode("gb18030")
    df = load_csv(content, "sales.csv")
    assert df.iloc[0]["地区"] == "青岛"


def test_csv_loader_rejects_excel_extension():
    with pytest.raises(DatasetLoadError, match="Use load_excel"):
        load_csv(b"not a csv", "book.xlsx")


@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_delimiter_is_detected_for_every_supported_separator(delimiter):
    text = "\n".join(delimiter.join(row) for row in (["a", "b", "c"], ["1", "2", "3"], ["4", "5", "6"])) + "\n"
    df = load_csv(csv_bytes(text))
    assert list(df.columns) == ["a", "b", "c"]
    assert df.shape == (2, 3)
    assert df.iloc[1].tolist() == [4, 5, 6]


def test_semicolon_file_with_duplicate_headers_is_rejected():
    with pytest.raises(DatasetLoadError, match="Duplicate column names"):
        load_csv(csv_bytes("A;A\n1;2\n3;4\n"))


def test_single_column_csv_still_uses_the_comma_default():
    df = load_csv(csv_bytes("name\nAlpha\nBeta\n"))
    assert list(df.columns) == ["name"]
    assert df.shape == (2, 1)


def test_leading_zeros_are_preserved_for_identifier_columns():
    df = load_csv(csv_bytes("zip,city\n01234,Beijing\n05678,Shanghai\n"))
    assert df["zip"].tolist() == ["01234", "05678"]
    assert not pd.api.types.is_numeric_dtype(df["zip"])


@pytest.mark.parametrize(
    "column",
    ["zip", "postal", "code", "sku", "order_id", "product_id", "Zip_Code", "postalcode", "phone", "account", "productId"],
)
def test_identifier_like_column_names_are_read_as_text(column):
    df = load_csv(csv_bytes(f"{column},sales\n01234,5.5\n05678,7.25\n"))
    assert df[column].tolist() == ["01234", "05678"]
    assert pd.api.types.is_numeric_dtype(df["sales"])


def test_ordinary_numeric_columns_are_still_inferred_as_numbers():
    df = load_csv(csv_bytes("sales,quantity,profit,width\n1.5,2,3.25,10\n4.5,5,6.75,20\n"))
    for column in ("sales", "quantity", "profit", "width"):
        assert pd.api.types.is_numeric_dtype(df[column]), column


def test_excel_identifier_column_keeps_leading_zeros():
    buffer = BytesIO()
    pd.DataFrame({"zip": ["01234", "05678"], "sales": [5, 7]}).to_excel(buffer, index=False)
    df = load_excel(buffer.getvalue(), "codes.xlsx")
    assert df["zip"].tolist() == ["01234", "05678"]
    assert pd.api.types.is_numeric_dtype(df["sales"])


def test_csv_row_limit_is_enforced_with_a_bounded_read(monkeypatch):
    captured = {}
    real_read_csv = pd.read_csv

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(dataset_service.pd, "read_csv", spy)
    content = csv_bytes("a,b\n" + "1,2\n" * (dataset_service.MAX_ROWS + 5))
    with pytest.raises(DatasetLoadError, match=f"exceeds the {dataset_service.MAX_ROWS:,} row limit"):
        load_csv(content)
    assert captured["nrows"] == dataset_service.MAX_ROWS + 1


def test_excel_row_limit_is_enforced_with_a_bounded_read(monkeypatch):
    captured = {}
    real_read_excel = pd.read_excel

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_read_excel(*args, **kwargs)

    monkeypatch.setattr(dataset_service, "MAX_ROWS", 5)
    monkeypatch.setattr(dataset_service.pd, "read_excel", spy)
    buffer = BytesIO()
    pd.DataFrame({"sales": range(10), "region": ["East"] * 10}).to_excel(buffer, index=False)
    with pytest.raises(DatasetLoadError, match="exceeds the 5 row limit"):
        load_excel(buffer.getvalue(), "big.xlsx")
    assert captured["nrows"] == 6
