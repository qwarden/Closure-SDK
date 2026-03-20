"""Read files into list[bytes] for the SDK.

Supports JSONL, CSV, and plain text. Each line/row becomes one record.
The SDK doesn't care about structure — it sees bytes.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    return "text"


def read_file(
    path: Path,
    *,
    fmt: str | None = None,
    columns: list[int] | None = None,
    header: bool = False,
    delimiter: str = ",",
) -> list[bytes]:
    """Read a file into a list of byte records.

    Each line (JSONL/text) or row (CSV) becomes one bytes entry.
    For CSV, you can select specific columns with `columns`.
    """
    if fmt is None:
        fmt = detect_format(path)

    text = path.read_text(encoding="utf-8")

    if fmt == "csv":
        return _read_csv(text, columns=columns, header=header, delimiter=delimiter)
    else:
        return _read_lines(text, header=header)


def _read_lines(text: str, *, header: bool = False) -> list[bytes]:
    lines = text.splitlines()
    if header and lines:
        lines = lines[1:]
    return [line.encode("utf-8") for line in lines if line.strip()]


def _read_csv(
    text: str,
    *,
    columns: list[int] | None = None,
    header: bool = False,
    delimiter: str = ",",
) -> list[bytes]:
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)
    if header and rows:
        rows = rows[1:]
    records = []
    for row in rows:
        if not any(cell.strip() for cell in row):
            continue
        if columns:
            selected = delimiter.join(row[c] for c in columns if c < len(row))
        else:
            selected = delimiter.join(row)
        records.append(selected.encode("utf-8"))
    return records
