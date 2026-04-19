"""Exporter module for paper_miner.

Provides functions to serialize lists of NumericRecord objects to JSON or CSV
format, writing to a file path or to stdout. Both formats are supported as
first-class citizens, with clean handling of optional fields and Unicode text.

Functions
---------
export_json   : Serialize records to a JSON file or stdout.
export_csv    : Serialize records to a CSV file or stdout.
export_records: Dispatch to the correct format based on a format string.
records_to_json_str : Convert records to a JSON string (without file I/O).
records_to_csv_str  : Convert records to a CSV string (without file I/O).
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
from typing import List, Optional, TextIO

from paper_miner.models import NumericRecord

# ---------------------------------------------------------------------------
# Supported export format identifiers
# ---------------------------------------------------------------------------

FORMAT_JSON = "json"
FORMAT_CSV = "csv"
_SUPPORTED_FORMATS = frozenset([FORMAT_JSON, FORMAT_CSV])


# ---------------------------------------------------------------------------
# Core serialization helpers (no I/O)
# ---------------------------------------------------------------------------


def records_to_json_str(
    records: List[NumericRecord],
    indent: int = 2,
    ensure_ascii: bool = False,
) -> str:
    """Convert a list of NumericRecord objects to a JSON-formatted string.

    Each record is serialised via its :py:meth:`~paper_miner.models.NumericRecord.to_dict`
    method.  The resulting string is a JSON array of objects.

    Parameters
    ----------
    records:
        The records to serialise.  An empty list produces ``"[]"``.  
    indent:
        Number of spaces used for JSON pretty-printing.  Defaults to ``2``.
        Pass ``0`` or ``None`` for compact output.
    ensure_ascii:
        When ``True``, non-ASCII characters are escaped with ``\\uXXXX``
        sequences.  Defaults to ``False`` (UTF-8 characters are preserved).

    Returns
    -------
    str
        A JSON string representation of *records*.
    """
    data = [record.to_dict() for record in records]
    indent_arg: Optional[int] = indent if indent else None
    return json.dumps(data, indent=indent_arg, ensure_ascii=ensure_ascii)


def records_to_csv_str(records: List[NumericRecord]) -> str:
    """Convert a list of NumericRecord objects to a CSV-formatted string.

    The first row is a header row containing the canonical field names from
    :py:meth:`~paper_miner.models.NumericRecord.csv_fieldnames`.  Subsequent
    rows contain the string representation of each record's fields (``None``
    values are written as empty strings).

    Parameters
    ----------
    records:
        The records to serialise.  An empty list produces a header-only CSV.

    Returns
    -------
    str
        A CSV string with ``\r\n`` line endings (Python's :mod:`csv` default).
    """
    output = io.StringIO()
    fieldnames = NumericRecord.csv_fieldnames()
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        lineterminator="\r\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for record in records:
        writer.writerow(record.to_csv_row())
    return output.getvalue()


# ---------------------------------------------------------------------------
# File / stream export functions
# ---------------------------------------------------------------------------


def export_json(
    records: List[NumericRecord],
    output_path: Optional[str] = None,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Write *records* as a JSON array to a file or stdout.

    Parameters
    ----------
    records:
        The list of :class:`~paper_miner.models.NumericRecord` objects to
        export.
    output_path:
        Filesystem path for the output file.  When ``None`` (default) or an
        empty string, output is written to *stdout*.
    indent:
        JSON indentation level.  Defaults to ``2`` spaces.
    ensure_ascii:
        When ``True``, non-ASCII characters are escaped.  Defaults to
        ``False``.

    Raises
    ------
    OSError
        If the output file cannot be opened for writing.
    """
    json_str = records_to_json_str(records, indent=indent, ensure_ascii=ensure_ascii)

    if output_path:
        _ensure_parent_dirs(output_path)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(json_str)
            fh.write("\n")
    else:
        sys.stdout.write(json_str)
        sys.stdout.write("\n")


def export_csv(
    records: List[NumericRecord],
    output_path: Optional[str] = None,
) -> None:
    """Write *records* as CSV to a file or stdout.

    The output always includes a header row.  ``None`` field values are
    written as empty strings.  The file is UTF-8 encoded with
    ``\r\n`` line endings.

    Parameters
    ----------
    records:
        The list of :class:`~paper_miner.models.NumericRecord` objects to
        export.
    output_path:
        Filesystem path for the output file.  When ``None`` (default) or an
        empty string, output is written to *stdout*.

    Raises
    ------
    OSError
        If the output file cannot be opened for writing.
    """
    csv_str = records_to_csv_str(records)

    if output_path:
        _ensure_parent_dirs(output_path)
        with open(output_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(csv_str)
    else:
        sys.stdout.write(csv_str)


def export_records(
    records: List[NumericRecord],
    fmt: str = FORMAT_JSON,
    output_path: Optional[str] = None,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Dispatch export to the correct serialiser based on *fmt*.

    This is the primary entry-point for export operations when the desired
    format is not known at import time (e.g. from a CLI flag).

    Parameters
    ----------
    records:
        The list of :class:`~paper_miner.models.NumericRecord` objects to
        export.
    fmt:
        Output format identifier.  Must be one of ``"json"`` or ``"csv"``
        (case-insensitive).  Defaults to ``"json"``.
    output_path:
        Filesystem path for the output file.  When ``None`` (default),
        output is written to *stdout*.
    indent:
        JSON indentation level (ignored for CSV output).  Defaults to ``2``.
    ensure_ascii:
        JSON ``ensure_ascii`` flag (ignored for CSV output).

    Raises
    ------
    ValueError
        If *fmt* is not one of the supported format identifiers.
    OSError
        If the output file cannot be opened for writing.
    """
    normalised = fmt.strip().lower()
    if normalised not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported export format: {fmt!r}. "
            f"Choose from: {sorted(_SUPPORTED_FORMATS)}."
        )

    if normalised == FORMAT_JSON:
        export_json(
            records,
            output_path=output_path,
            indent=indent,
            ensure_ascii=ensure_ascii,
        )
    else:  # FORMAT_CSV
        export_csv(records, output_path=output_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_parent_dirs(path: str) -> None:
    """Create any missing parent directories for *path*.

    Parameters
    ----------
    path:
        A filesystem path whose parent directories should exist before
        writing.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
