import csv
import io
import re
from collections import Counter
from pathlib import Path

import pandas as pd

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 200
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# Candidate separators considered when sniffing the delimiter of an uploaded CSV.
_DELIMITER_CANDIDATES = (",", ";", "\t", "|")
_SNIFF_BYTES = 8 * 1024

# Column names that look like identifiers (order_id, Zip_Code, postalcode, sku, ...).
# Those columns are forced to text so that leading zeros and long digit strings
# ("01234") are not silently turned into numbers by the dtype inference.
_IDENTIFIER_COLUMN_NAME = re.compile(
    r"(?i:(?<![a-z0-9])(?:id|code|sku|uuid|zip|postal|phone|account)s?(?![a-z0-9]))"
    r"|(?i:(?<![a-z0-9])(?:zip|postal)code(?![a-z0-9]))"
    r"|(?<=[a-z])(?:ID|Id|CODE|Code|SKU|Sku|UUID|Uuid|ZIP|Zip|POSTAL|Postal|PHONE|Phone|ACCOUNT|Account)s?(?![A-Za-z0-9])"
)


class DatasetLoadError(ValueError):
    """A safe, user-facing dataset loading error."""


def _validate_file(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DatasetLoadError("Unsupported file type. Upload a CSV or Excel file (.csv, .xlsx, .xls).")
    if not content:
        raise DatasetLoadError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise DatasetLoadError("The file is larger than the 20 MiB upload limit.")
    return suffix


def _width_profile(sample: str, delimiter: str) -> tuple[int, float]:
    """Return the most common row width for a delimiter and how consistent it is."""
    try:
        rows = list(csv.reader(io.StringIO(sample), delimiter=delimiter))
    except csv.Error:
        return 0, 0.0
    widths = [len(row) for row in rows if any(cell.strip() for cell in row)]
    if not widths:
        return 0, 0.0
    mode_width = Counter(widths).most_common(1)[0][0]
    return mode_width, widths.count(mode_width) / len(widths)


def _detect_delimiter(text: str) -> str:
    """Pick the whitelisted separator that consistently yields the widest rows.

    Only `_DELIMITER_CANDIDATES` are considered. If every candidate yields a
    single column the file is genuinely ambiguous, so the comma default is kept.
    """
    sample = text[:_SNIFF_BYTES]
    best_delimiter, best_width, best_consistency = ",", 1, 0.0
    for candidate in _DELIMITER_CANDIDATES:
        width, consistency = _width_profile(sample, candidate)
        if width > best_width or (width == best_width and consistency > best_consistency):
            best_delimiter, best_width, best_consistency = candidate, width, consistency
    return best_delimiter


def _text_columns(names: list[object]) -> dict[str, type]:
    """Map identifier-like column names to `str` for the pandas dtype argument."""
    mapping: dict[str, type] = {}
    for name in names:
        if _IDENTIFIER_COLUMN_NAME.search(str(name).strip()):
            mapping[str(name)] = str
    return mapping


def _validate_header(names: list[object]) -> None:
    if not names or all(pd.isna(value) or not str(value).strip() for value in names):
        raise DatasetLoadError("No column headers were found in the first row.")
    normalized = [str(value).strip().casefold() if not pd.isna(value) else "" for value in names]
    if any(not name for name in normalized):
        raise DatasetLoadError("Column names cannot be blank.")
    if len(set(normalized)) != len(normalized):
        raise DatasetLoadError("Duplicate column names were found. Rename them and upload again.")


def _validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df.index) == 0:
        raise DatasetLoadError("The dataset has headers but no data rows.")
    if len(df.columns) == 0:
        raise DatasetLoadError("The dataset does not contain any columns.")
    if len(df.index) > MAX_ROWS:
        raise DatasetLoadError(f"The dataset exceeds the {MAX_ROWS:,} row limit.")
    if len(df.columns) > MAX_COLUMNS:
        raise DatasetLoadError(f"The dataset exceeds the {MAX_COLUMNS} column limit.")
    names = [str(column).strip() for column in df.columns]
    _validate_header(names)
    df.columns = names
    return df


def load_csv(content: bytes, filename: str = "uploaded.csv") -> pd.DataFrame:
    """Read an uploaded CSV byte stream; never opens an arbitrary filesystem path."""
    suffix = _validate_file(filename, content)
    if suffix != ".csv":
        raise DatasetLoadError("This is an Excel file. Use load_excel to read it.")
    try:
        text = None
        for encoding in ("utf-8-sig", "gb18030"):
            try:
                text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise DatasetLoadError("Could not decode this CSV. Save it as UTF-8 or GB18030 and retry.")
        delimiter = _detect_delimiter(text)
        # The same delimiter must be used for the header guard and for the parse itself.
        header = next(
            (
                row
                for row in csv.reader(io.StringIO(text), delimiter=delimiter)
                if any(cell.strip() for cell in row)
            ),
            [],
        )
        _validate_header(header)
        df = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            dtype=_text_columns(header) or None,
            nrows=MAX_ROWS + 1,
            on_bad_lines="error",
        )
    except DatasetLoadError:
        raise
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError, csv.Error) as exc:
        raise DatasetLoadError("Could not read this CSV. Check its encoding and row formatting.") from exc
    except Exception as exc:
        raise DatasetLoadError("Could not read this CSV file.") from exc
    return _validate_frame(df)


def load_excel(content: bytes, filename: str = "uploaded.xlsx") -> pd.DataFrame:
    """Read the first worksheet from an uploaded XLS/XLSX byte stream."""
    suffix = _validate_file(filename, content)
    if suffix == ".csv":
        raise DatasetLoadError("This is a CSV file. Use load_csv to read it.")
    engine = "xlrd" if suffix == ".xls" else "openpyxl"
    try:
        header_frame = pd.read_excel(io.BytesIO(content), header=None, nrows=1, engine=engine)
        if header_frame.empty:
            raise DatasetLoadError("The Excel file contains no header row.")
        header = header_frame.iloc[0].tolist()
        _validate_header(header)
        # `nrows` keeps a huge workbook from being materialised before the row guard runs.
        df = pd.read_excel(
            io.BytesIO(content),
            engine=engine,
            dtype=_text_columns(header) or None,
            nrows=MAX_ROWS + 1,
        )
    except DatasetLoadError:
        raise
    except ImportError as exc:
        raise DatasetLoadError("Excel support is unavailable. Install the Excel reader dependency.") from exc
    except Exception as exc:
        raise DatasetLoadError("Could not read this Excel file. Check that it is a valid .xls or .xlsx workbook.") from exc
    return _validate_frame(df)


def load_dataset(filename: str, content: bytes) -> pd.DataFrame:
    """Dispatch an uploaded file by its extension."""
    suffix = _validate_file(filename, content)
    if suffix == ".csv":
        return load_csv(content, filename)
    return load_excel(content, filename)
