"""Unit tests for paper_miner.exporter.

Tests cover:
- records_to_json_str: correct JSON structure, empty list, indentation,
  Unicode preservation, optional-field None handling.
- records_to_csv_str: header row, field values, None-as-empty-string,
  empty list produces header-only output.
- export_json: writes to file, writes to stdout, creates parent directories.
- export_csv: writes to file, writes to stdout, header present.
- export_records: dispatches correctly to JSON/CSV, raises on unknown format.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_miner.models import NumericRecord
from paper_miner.exporter import (
    records_to_json_str,
    records_to_csv_str,
    export_json,
    export_csv,
    export_records,
    FORMAT_JSON,
    FORMAT_CSV,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    value: str = "32.4",
    unit: str = "mg/dL",
    data_type: str = "measurement",
    context: str = "LDL was reduced by 32.4 mg/dL.",
    relationship: str = "LDL cholesterol reduction",
    raw_text: str = "32.4 mg/dL",
    section: str | None = "Results",
    confidence: float | None = 0.95,
    source: str | None = "paper.pdf",
) -> NumericRecord:
    """Create a fully populated NumericRecord for testing."""
    return NumericRecord(
        value=value,
        unit=unit,
        data_type=data_type,
        context=context,
        relationship=relationship,
        raw_text=raw_text,
        section=section,
        confidence=confidence,
        source=source,
    )


def _make_minimal_record() -> NumericRecord:
    """Create a NumericRecord with all optional fields set to None."""
    return NumericRecord(
        value="42",
        unit="none",
        data_type="measurement",
        context="The value was 42.",
        relationship="",
        raw_text="42",
        section=None,
        confidence=None,
        source=None,
    )


SAMPLE_RECORDS = [
    _make_record(),
    _make_record(
        value="0.003",
        unit="none",
        data_type="p-value",
        context="p = 0.003 was observed.",
        relationship="statistical significance",
        raw_text="p = 0.003",
        section="Results",
        confidence=0.99,
        source="paper.pdf",
    ),
    _make_minimal_record(),
]


# ---------------------------------------------------------------------------
# records_to_json_str
# ---------------------------------------------------------------------------


class TestRecordsToJsonStr:
    """Tests for records_to_json_str."""

    def test_returns_valid_json(self) -> None:
        result = records_to_json_str(SAMPLE_RECORDS)
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_length_matches_records(self) -> None:
        result = records_to_json_str(SAMPLE_RECORDS)
        parsed = json.loads(result)
        assert len(parsed) == len(SAMPLE_RECORDS)

    def test_empty_list_returns_empty_array(self) -> None:
        result = records_to_json_str([])
        assert json.loads(result) == []

    def test_field_values_preserved(self) -> None:
        record = _make_record(value="12.5", unit="mmHg", data_type="measurement")
        result = records_to_json_str([record])
        parsed = json.loads(result)
        assert parsed[0]["value"] == "12.5"
        assert parsed[0]["unit"] == "mmHg"
        assert parsed[0]["data_type"] == "measurement"

    def test_optional_none_fields_are_null(self) -> None:
        record = _make_minimal_record()
        result = records_to_json_str([record])
        parsed = json.loads(result)
        assert parsed[0]["section"] is None
        assert parsed[0]["confidence"] is None
        assert parsed[0]["source"] is None

    def test_confidence_float_preserved(self) -> None:
        record = _make_record(confidence=0.87)
        result = records_to_json_str([record])
        parsed = json.loads(result)
        assert abs(parsed[0]["confidence"] - 0.87) < 1e-9

    def test_unicode_preserved_by_default(self) -> None:
        record = _make_record(context="LDL réduction de 32.4 mg/dL. ± μg")
        result = records_to_json_str([record], ensure_ascii=False)
        assert "réduction" in result
        assert "±" in result

    def test_ensure_ascii_escapes_unicode(self) -> None:
        record = _make_record(context="value ±")
        result = records_to_json_str([record], ensure_ascii=True)
        # The ± character should be escaped.
        assert "\\u" in result or "±" not in result

    def test_indentation_applied(self) -> None:
        record = _make_record()
        result_indented = records_to_json_str([record], indent=4)
        result_compact = records_to_json_str([record], indent=0)
        # Indented output is longer (has newlines and spaces).
        assert len(result_indented) > len(result_compact)

    def test_all_canonical_fields_present(self) -> None:
        record = _make_record()
        result = records_to_json_str([record])
        parsed = json.loads(result)
        item = parsed[0]
        for field in NumericRecord.csv_fieldnames():
            assert field in item, f"Missing field in JSON output: {field}"

    def test_context_field_preserved(self) -> None:
        ctx = "The treatment group showed a reduction of 32.4 mg/dL from baseline."
        record = _make_record(context=ctx)
        result = records_to_json_str([record])
        parsed = json.loads(result)
        assert parsed[0]["context"] == ctx

    def test_relationship_field_preserved(self) -> None:
        rel = "LDL cholesterol reduction after 12-week treatment"
        record = _make_record(relationship=rel)
        result = records_to_json_str([record])
        parsed = json.loads(result)
        assert parsed[0]["relationship"] == rel

    def test_source_field_preserved(self) -> None:
        record = _make_record(source="my_study.pdf")
        result = records_to_json_str([record])
        parsed = json.loads(result)
        assert parsed[0]["source"] == "my_study.pdf"

    def test_multiple_records_ordered(self) -> None:
        records = [
            _make_record(value="1.0"),
            _make_record(value="2.0"),
            _make_record(value="3.0"),
        ]
        result = records_to_json_str(records)
        parsed = json.loads(result)
        assert [p["value"] for p in parsed] == ["1.0", "2.0", "3.0"]


# ---------------------------------------------------------------------------
# records_to_csv_str
# ---------------------------------------------------------------------------


class TestRecordsToCsvStr:
    """Tests for records_to_csv_str."""

    def _parse_csv(self, csv_str: str) -> list[dict]:
        """Parse a CSV string into a list of row dicts."""
        reader = csv.DictReader(io.StringIO(csv_str))
        return list(reader)

    def test_returns_string(self) -> None:
        result = records_to_csv_str(SAMPLE_RECORDS)
        assert isinstance(result, str)

    def test_has_header_row(self) -> None:
        result = records_to_csv_str(SAMPLE_RECORDS)
        first_line = result.splitlines()[0]
        for field in NumericRecord.csv_fieldnames():
            assert field in first_line

    def test_empty_list_produces_header_only(self) -> None:
        result = records_to_csv_str([])
        rows = self._parse_csv(result)
        assert rows == []
        # Header must still be present.
        assert "value" in result

    def test_row_count_matches_records(self) -> None:
        result = records_to_csv_str(SAMPLE_RECORDS)
        rows = self._parse_csv(result)
        assert len(rows) == len(SAMPLE_RECORDS)

    def test_value_field_correct(self) -> None:
        record = _make_record(value="99.9")
        result = records_to_csv_str([record])
        rows = self._parse_csv(result)
        assert rows[0]["value"] == "99.9"

    def test_unit_field_correct(self) -> None:
        record = _make_record(unit="mmHg")
        result = records_to_csv_str([record])
        rows = self._parse_csv(result)
        assert rows[0]["unit"] == "mmHg"

    def test_data_type_field_correct(self) -> None:
        record = _make_record(data_type="p-value")
        result = records_to_csv_str([record])
        rows = self._parse_csv(result)
        assert rows[0]["data_type"] == "p-value"

    def test_none_fields_written_as_empty_string(self) -> None:
        record = _make_minimal_record()  # section, confidence, source are None
        result = records_to_csv_str([record])
        rows = self._parse_csv(result)
        assert rows[0]["section"] == ""
        assert rows[0]["confidence"] == ""
        assert rows[0]["source"] == ""

    def test_confidence_written_as_string(self) -> None:
        record = _make_record(confidence=0.95)
        result = records_to_csv_str([record])
        rows = self._parse_csv(result)
        assert rows[0]["confidence"] == "0.95"

    def test_all_canonical_fields_in_header(self) -> None:
        result = records_to_csv_str([_make_record()])
        reader = csv.DictReader(io.StringIO(result))
        fieldnames = reader.fieldnames or []
        for field in NumericRecord.csv_fieldnames():
            assert field in fieldnames

    def test_context_with_comma_escaped_correctly(self) -> None:
        record = _make_record(context="The value, which was 32.4, is significant.")
        result = records_to_csv_str([record])
        rows = self._parse_csv(result)
        assert "32.4" in rows[0]["context"]

    def test_context_with_quotes_escaped_correctly(self) -> None:
        record = _make_record(context='The "reduction" was 32.4 mg/dL.')
        result = records_to_csv_str([record])
        rows = self._parse_csv(result)
        assert "reduction" in rows[0]["context"]

    def test_multiple_records_order_preserved(self) -> None:
        records = [
            _make_record(value="1.0"),
            _make_record(value="2.0"),
        ]
        result = records_to_csv_str(records)
        rows = self._parse_csv(result)
        assert rows[0]["value"] == "1.0"
        assert rows[1]["value"] == "2.0"

    def test_raw_text_field_correct(self) -> None:
        record = _make_record(raw_text="p < 0.001")
        result = records_to_csv_str([record])
        rows = self._parse_csv(result)
        assert rows[0]["raw_text"] == "p < 0.001"


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------


class TestExportJson:
    """Tests for export_json (file and stdout modes)."""

    def test_writes_valid_json_to_file(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        export_json(SAMPLE_RECORDS, output_path=str(output))
        assert output.is_file()
        with open(output, encoding="utf-8") as fh:
            parsed = json.load(fh)
        assert len(parsed) == len(SAMPLE_RECORDS)

    def test_written_file_contains_correct_values(self, tmp_path: Path) -> None:
        record = _make_record(value="12.5", unit="mmHg")
        output = tmp_path / "out.json"
        export_json([record], output_path=str(output))
        with open(output, encoding="utf-8") as fh:
            parsed = json.load(fh)
        assert parsed[0]["value"] == "12.5"
        assert parsed[0]["unit"] == "mmHg"

    def test_empty_records_writes_empty_array(self, tmp_path: Path) -> None:
        output = tmp_path / "empty.json"
        export_json([], output_path=str(output))
        with open(output, encoding="utf-8") as fh:
            parsed = json.load(fh)
        assert parsed == []

    def test_writes_to_stdout_when_no_path(self, capsys: pytest.CaptureFixture) -> None:
        record = _make_record(value="99")
        export_json([record], output_path=None)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_stdout_output_contains_value(self, capsys: pytest.CaptureFixture) -> None:
        record = _make_record(value="55.5")
        export_json([record])
        captured = capsys.readouterr()
        assert "55.5" in captured.out

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        output = tmp_path / "nested" / "dir" / "out.json"
        export_json([_make_record()], output_path=str(output))
        assert output.is_file()

    def test_empty_string_path_writes_to_stdout(self, capsys: pytest.CaptureFixture) -> None:
        export_json([_make_record()], output_path="")
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)

    def test_file_is_utf8_encoded(self, tmp_path: Path) -> None:
        record = _make_record(context="réduction de cholestérol ±")
        output = tmp_path / "utf8.json"
        export_json([record], output_path=str(output), ensure_ascii=False)
        raw = output.read_bytes()
        text = raw.decode("utf-8")
        assert "réduction" in text

    def test_indentation_in_file(self, tmp_path: Path) -> None:
        output = tmp_path / "indented.json"
        export_json([_make_record()], output_path=str(output), indent=4)
        content = output.read_text(encoding="utf-8")
        # Indented JSON should have newlines.
        assert "\n" in content


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------


class TestExportCsv:
    """Tests for export_csv (file and stdout modes)."""

    def _parse_csv_file(self, path: str) -> list[dict]:
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            return list(reader)

    def test_writes_csv_to_file(self, tmp_path: Path) -> None:
        output = tmp_path / "out.csv"
        export_csv(SAMPLE_RECORDS, output_path=str(output))
        assert output.is_file()

    def test_csv_file_has_correct_row_count(self, tmp_path: Path) -> None:
        output = tmp_path / "out.csv"
        export_csv(SAMPLE_RECORDS, output_path=str(output))
        rows = self._parse_csv_file(str(output))
        assert len(rows) == len(SAMPLE_RECORDS)

    def test_csv_file_has_header(self, tmp_path: Path) -> None:
        output = tmp_path / "out.csv"
        export_csv([_make_record()], output_path=str(output))
        content = output.read_text(encoding="utf-8")
        assert "value" in content.splitlines()[0]

    def test_csv_file_values_correct(self, tmp_path: Path) -> None:
        record = _make_record(value="77", unit="bpm")
        output = tmp_path / "out.csv"
        export_csv([record], output_path=str(output))
        rows = self._parse_csv_file(str(output))
        assert rows[0]["value"] == "77"
        assert rows[0]["unit"] == "bpm"

    def test_none_written_as_empty_in_file(self, tmp_path: Path) -> None:
        record = _make_minimal_record()
        output = tmp_path / "out.csv"
        export_csv([record], output_path=str(output))
        rows = self._parse_csv_file(str(output))
        assert rows[0]["section"] == ""
        assert rows[0]["confidence"] == ""

    def test_empty_records_writes_header_only(self, tmp_path: Path) -> None:
        output = tmp_path / "empty.csv"
        export_csv([], output_path=str(output))
        rows = self._parse_csv_file(str(output))
        assert rows == []
        content = output.read_text(encoding="utf-8")
        assert "value" in content

    def test_writes_to_stdout_when_no_path(self, capsys: pytest.CaptureFixture) -> None:
        record = _make_record(value="33.3")
        export_csv([record], output_path=None)
        captured = capsys.readouterr()
        assert "33.3" in captured.out
        assert "value" in captured.out  # header present

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        output = tmp_path / "sub" / "dir" / "out.csv"
        export_csv([_make_record()], output_path=str(output))
        assert output.is_file()

    def test_stdout_has_header(self, capsys: pytest.CaptureFixture) -> None:
        export_csv([_make_record()])
        captured = capsys.readouterr()
        first_line = captured.out.splitlines()[0]
        assert "value" in first_line

    def test_empty_string_path_writes_to_stdout(self, capsys: pytest.CaptureFixture) -> None:
        export_csv([_make_record()], output_path="")
        captured = capsys.readouterr()
        assert "value" in captured.out


# ---------------------------------------------------------------------------
# export_records (dispatcher)
# ---------------------------------------------------------------------------


class TestExportRecords:
    """Tests for the export_records dispatcher function."""

    def test_json_format_writes_json_file(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        export_records(SAMPLE_RECORDS, fmt="json", output_path=str(output))
        with open(output, encoding="utf-8") as fh:
            parsed = json.load(fh)
        assert len(parsed) == len(SAMPLE_RECORDS)

    def test_csv_format_writes_csv_file(self, tmp_path: Path) -> None:
        output = tmp_path / "out.csv"
        export_records(SAMPLE_RECORDS, fmt="csv", output_path=str(output))
        with open(output, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == len(SAMPLE_RECORDS)

    def test_case_insensitive_format_json(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        export_records([_make_record()], fmt="JSON", output_path=str(output))
        assert output.is_file()
        with open(output, encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, list)

    def test_case_insensitive_format_csv(self, tmp_path: Path) -> None:
        output = tmp_path / "out.csv"
        export_records([_make_record()], fmt="CSV", output_path=str(output))
        content = output.read_text(encoding="utf-8")
        assert "value" in content

    def test_unknown_format_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported export format"):
            export_records([_make_record()], fmt="xml")

    def test_empty_format_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported export format"):
            export_records([_make_record()], fmt="")

    def test_defaults_to_json_format(self, tmp_path: Path) -> None:
        output = tmp_path / "default.json"
        export_records([_make_record()], output_path=str(output))
        with open(output, encoding="utf-8") as fh:
            parsed = json.load(fh)
        assert isinstance(parsed, list)

    def test_json_stdout(self, capsys: pytest.CaptureFixture) -> None:
        export_records([_make_record(value="11")], fmt="json")
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed[0]["value"] == "11"

    def test_csv_stdout(self, capsys: pytest.CaptureFixture) -> None:
        export_records([_make_record(value="22")], fmt="csv")
        captured = capsys.readouterr()
        assert "22" in captured.out

    def test_format_constants_work(self, tmp_path: Path) -> None:
        out_json = tmp_path / "a.json"
        out_csv = tmp_path / "b.csv"
        export_records([_make_record()], fmt=FORMAT_JSON, output_path=str(out_json))
        export_records([_make_record()], fmt=FORMAT_CSV, output_path=str(out_csv))
        assert out_json.is_file()
        assert out_csv.is_file()

    def test_json_indent_forwarded(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        export_records([_make_record()], fmt="json", output_path=str(output), indent=4)
        content = output.read_text(encoding="utf-8")
        # 4-space indented JSON has lines starting with 4 spaces.
        assert "    " in content

    def test_mixed_case_format_whitespace_stripped(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        export_records([_make_record()], fmt="  Json  ", output_path=str(output))
        assert output.is_file()


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """End-to-end round-trip serialisation tests."""

    def test_json_round_trip(self) -> None:
        """to_dict -> JSON -> parse back and verify field values."""
        record = _make_record(
            value="12.5",
            unit="mmHg",
            data_type="measurement",
            context="Blood pressure reduced by 12.5 mmHg.",
            relationship="blood pressure reduction",
            raw_text="12.5 mmHg",
            section="Results",
            confidence=0.93,
            source="study.pdf",
        )
        json_str = records_to_json_str([record])
        parsed = json.loads(json_str)
        item = parsed[0]
        assert item["value"] == "12.5"
        assert item["unit"] == "mmHg"
        assert item["data_type"] == "measurement"
        assert item["section"] == "Results"
        assert abs(item["confidence"] - 0.93) < 1e-9
        assert item["source"] == "study.pdf"

    def test_csv_round_trip(self) -> None:
        """to_csv_row -> CSV -> parse back and verify field values."""
        record = _make_record(
            value="0.001",
            unit="none",
            data_type="p-value",
            confidence=0.99,
            source="trial.pdf",
        )
        csv_str = records_to_csv_str([record])
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1
        row = rows[0]
        assert row["value"] == "0.001"
        assert row["unit"] == "none"
        assert row["data_type"] == "p-value"
        assert row["confidence"] == "0.99"
        assert row["source"] == "trial.pdf"

    def test_multiple_records_json_round_trip(self) -> None:
        records = SAMPLE_RECORDS
        json_str = records_to_json_str(records)
        parsed = json.loads(json_str)
        assert len(parsed) == len(records)
        for original, serialised in zip(records, parsed):
            assert serialised["value"] == original.value
            assert serialised["unit"] == original.unit

    def test_multiple_records_csv_round_trip(self) -> None:
        records = SAMPLE_RECORDS
        csv_str = records_to_csv_str(records)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == len(records)
        for original, row in zip(records, rows):
            assert row["value"] == original.value
