"""Unit tests for paper_miner.llm_parser.

All OpenAI API calls are mocked so that these tests run without any live
API key. Tests cover:

- build_prompt: correct JSON structure for batches of candidates.
- parse_llm_response: strict JSON, markdown-fenced JSON, malformed JSON,
  missing IDs, wrong type.
- _merge_record: field merging logic, confidence clamping, fallback to
  original values when LLM fields are empty.
- _fallback_dict: structure and defaults.
- _model_supports_json_mode: allowlist logic.
- _make_client: error when no key available, correct kwargs forwarded.
- _call_llm: retry behaviour on transient errors, success on first try.
- parse_candidates: happy path, empty input, batch splitting, API failure
  fallback, partial batch failure.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_miner.models import NumericRecord
from paper_miner.llm_parser import (
    build_prompt,
    parse_llm_response,
    parse_candidates,
    _merge_record,
    _fallback_dict,
    _model_supports_json_mode,
    _call_llm,
    _SYSTEM_PROMPT,
    DEFAULT_BATCH_SIZE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    value: str = "32.4",
    unit: str = "mg/dL",
    data_type: str = "measurement",
    context: str = "LDL was reduced by 32.4 mg/dL.",
    relationship: str = "",
    raw_text: str = "32.4 mg/dL",
    section: str | None = "Results",
    confidence: float | None = None,
    source: str | None = "paper.pdf",
) -> NumericRecord:
    """Create a NumericRecord for use in tests."""
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


def _make_llm_response(records: List[NumericRecord]) -> str:
    """Build a synthetic LLM JSON response for the given records."""
    payload = [
        {
            "id": idx,
            "value": r.value,
            "unit": r.unit,
            "data_type": r.data_type,
            "relationship": f"relationship for {r.value}",
            "confidence": 0.95,
        }
        for idx, r in enumerate(records)
    ]
    return json.dumps(payload)


def _make_openai_response(content: str) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response object."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    """Tests for the build_prompt function."""

    def test_returns_valid_json_string(self) -> None:
        records = [_make_record()]
        prompt = build_prompt(records)
        parsed = json.loads(prompt)
        assert isinstance(parsed, list)

    def test_length_matches_candidates(self) -> None:
        records = [
            _make_record(),
            _make_record(value="0.003", unit="none", data_type="p-value"),
        ]
        prompt = build_prompt(records)
        parsed = json.loads(prompt)
        assert len(parsed) == 2

    def test_ids_are_sequential(self) -> None:
        records = [_make_record() for _ in range(5)]
        prompt = build_prompt(records)
        parsed = json.loads(prompt)
        ids = [item["id"] for item in parsed]
        assert ids == list(range(5))

    def test_required_fields_present(self) -> None:
        records = [_make_record()]
        prompt = build_prompt(records)
        parsed = json.loads(prompt)
        item = parsed[0]
        for field in ("id", "raw_text", "context", "data_type_hint", "unit_hint", "value"):
            assert field in item, f"Missing field: {field}"

    def test_raw_text_preserved(self) -> None:
        record = _make_record(raw_text="32.4 mg/dL")
        prompt = build_prompt([record])
        parsed = json.loads(prompt)
        assert parsed[0]["raw_text"] == "32.4 mg/dL"

    def test_context_preserved(self) -> None:
        record = _make_record(context="The LDL was 32.4 mg/dL.")
        prompt = build_prompt([record])
        parsed = json.loads(prompt)
        assert parsed[0]["context"] == "The LDL was 32.4 mg/dL."

    def test_empty_candidates_returns_empty_array(self) -> None:
        prompt = build_prompt([])
        parsed = json.loads(prompt)
        assert parsed == []

    def test_value_in_prompt(self) -> None:
        record = _make_record(value="0.001")
        prompt = build_prompt([record])
        parsed = json.loads(prompt)
        assert parsed[0]["value"] == "0.001"

    def test_data_type_hint_used(self) -> None:
        record = _make_record(data_type="p-value")
        prompt = build_prompt([record])
        parsed = json.loads(prompt)
        assert parsed[0]["data_type_hint"] == "p-value"

    def test_unit_hint_used(self) -> None:
        record = _make_record(unit="mmHg")
        prompt = build_prompt([record])
        parsed = json.loads(prompt)
        assert parsed[0]["unit_hint"] == "mmHg"

    def test_single_record_id_is_zero(self) -> None:
        record = _make_record()
        prompt = build_prompt([record])
        parsed = json.loads(prompt)
        assert parsed[0]["id"] == 0

    def test_large_batch_sequential_ids(self) -> None:
        records = [_make_record(value=str(i)) for i in range(20)]
        prompt = build_prompt(records)
        parsed = json.loads(prompt)
        assert len(parsed) == 20
        for i, item in enumerate(parsed):
            assert item["id"] == i

    def test_data_type_hint_measurement(self) -> None:
        record = _make_record(data_type="measurement")
        prompt = build_prompt([record])
        parsed = json.loads(prompt)
        assert parsed[0]["data_type_hint"] == "measurement"

    def test_unit_hint_none(self) -> None:
        record = _make_record(unit="none")
        prompt = build_prompt([record])
        parsed = json.loads(prompt)
        assert parsed[0]["unit_hint"] == "none"

    def test_prompt_is_string(self) -> None:
        prompt = build_prompt([_make_record()])
        assert isinstance(prompt, str)

    def test_two_different_records_different_hints(self) -> None:
        r1 = _make_record(value="32.4", unit="mg/dL", data_type="measurement")
        r2 = _make_record(value="0.001", unit="none", data_type="p-value")
        prompt = build_prompt([r1, r2])
        parsed = json.loads(prompt)
        assert parsed[0]["data_type_hint"] == "measurement"
        assert parsed[1]["data_type_hint"] == "p-value"

    def test_unicode_context_preserved(self) -> None:
        record = _make_record(context="LDL r\u00e9duction de 32.4 mg/dL.")
        prompt = build_prompt([record])
        parsed = json.loads(prompt)
        assert "r\u00e9duction" in parsed[0]["context"]


# ---------------------------------------------------------------------------
# parse_llm_response
# ---------------------------------------------------------------------------


class TestParseLlmResponse:
    """Tests for the parse_llm_response function."""

    def test_valid_json_array(self) -> None:
        data = [
            {
                "id": 0,
                "value": "32.4",
                "unit": "mg/dL",
                "data_type": "measurement",
                "relationship": "LDL reduction",
                "confidence": 0.95,
            }
        ]
        result = parse_llm_response(json.dumps(data), num_candidates=1)
        assert len(result) == 1
        assert result[0]["value"] == "32.4"

    def test_markdown_fenced_json(self) -> None:
        data = [
            {
                "id": 0,
                "value": "0.001",
                "unit": "none",
                "data_type": "p-value",
                "relationship": "significance",
                "confidence": 0.99,
            }
        ]
        fenced = f"```json\n{json.dumps(data)}\n```"
        result = parse_llm_response(fenced, num_candidates=1)
        assert len(result) == 1
        assert result[0]["data_type"] == "p-value"

    def test_malformed_json_returns_fallbacks(self) -> None:
        result = parse_llm_response("not json at all!!!", num_candidates=3)
        assert len(result) == 3
        for r in result:
            assert r["confidence"] == 0.0
            assert r["data_type"] == "other"

    def test_wrong_type_json_returns_fallbacks(self) -> None:
        # The model returns a dict instead of a list.
        result = parse_llm_response(json.dumps({"id": 0}), num_candidates=2)
        assert len(result) == 2

    def test_missing_id_uses_fallback(self) -> None:
        # Response has only id=0; id=1 is missing.
        data = [
            {
                "id": 0,
                "value": "32.4",
                "unit": "mg/dL",
                "data_type": "measurement",
                "relationship": "",
                "confidence": 0.9,
            }
        ]
        result = parse_llm_response(json.dumps(data), num_candidates=2)
        assert len(result) == 2
        # Index 1 should be fallback.
        assert result[1]["confidence"] == 0.0

    def test_correct_ordering_by_id(self) -> None:
        # Model returns ids out of order.
        data = [
            {
                "id": 1,
                "value": "B",
                "unit": "none",
                "data_type": "other",
                "relationship": "",
                "confidence": 0.5,
            },
            {
                "id": 0,
                "value": "A",
                "unit": "none",
                "data_type": "other",
                "relationship": "",
                "confidence": 0.8,
            },
        ]
        result = parse_llm_response(json.dumps(data), num_candidates=2)
        assert result[0]["value"] == "A"
        assert result[1]["value"] == "B"

    def test_extra_ids_ignored(self) -> None:
        # Model returns 3 items but only 2 were requested.
        data = [
            {"id": 0, "value": "A", "unit": "none", "data_type": "other",
             "relationship": "", "confidence": 0.8},
            {"id": 1, "value": "B", "unit": "none", "data_type": "other",
             "relationship": "", "confidence": 0.7},
            {"id": 2, "value": "C", "unit": "none", "data_type": "other",
             "relationship": "", "confidence": 0.6},
        ]
        result = parse_llm_response(json.dumps(data), num_candidates=2)
        assert len(result) == 2

    def test_json_embedded_in_prose(self) -> None:
        """Response has explanatory prose before the JSON array."""
        data = [
            {
                "id": 0,
                "value": "42",
                "unit": "mg",
                "data_type": "measurement",
                "relationship": "dose",
                "confidence": 0.88,
            }
        ]
        prose = f"Here is my analysis: {json.dumps(data)} Hope this helps!"
        result = parse_llm_response(prose, num_candidates=1)
        assert len(result) == 1
        assert result[0]["value"] == "42"

    def test_empty_response_returns_fallbacks(self) -> None:
        result = parse_llm_response("", num_candidates=2)
        assert len(result) == 2
        assert all(r["data_type"] == "other" for r in result)

    def test_returns_list(self) -> None:
        data = [{"id": 0, "value": "1", "unit": "none",
                 "data_type": "other", "relationship": "", "confidence": 0.5}]
        result = parse_llm_response(json.dumps(data), num_candidates=1)
        assert isinstance(result, list)

    def test_each_entry_is_dict(self) -> None:
        data = [{"id": 0, "value": "1", "unit": "none",
                 "data_type": "other", "relationship": "", "confidence": 0.5}]
        result = parse_llm_response(json.dumps(data), num_candidates=1)
        for entry in result:
            assert isinstance(entry, dict)

    def test_num_candidates_zero_returns_empty(self) -> None:
        result = parse_llm_response("[]", num_candidates=0)
        assert result == []

    def test_all_fallbacks_have_correct_ids(self) -> None:
        result = parse_llm_response("INVALID", num_candidates=4)
        for i, entry in enumerate(result):
            assert entry["id"] == i

    def test_partial_missing_ids_filled_with_fallbacks(self) -> None:
        # Response has id=0 and id=2 but not id=1.
        data = [
            {"id": 0, "value": "A", "unit": "none",
             "data_type": "measurement", "relationship": "", "confidence": 0.9},
            {"id": 2, "value": "C", "unit": "none",
             "data_type": "measurement", "relationship": "", "confidence": 0.8},
        ]
        result = parse_llm_response(json.dumps(data), num_candidates=3)
        assert len(result) == 3
        assert result[0]["value"] == "A"
        assert result[1]["confidence"] == 0.0  # fallback for id=1
        assert result[2]["value"] == "C"

    def test_relationship_field_returned(self) -> None:
        data = [{
            "id": 0, "value": "32.4", "unit": "mg/dL",
            "data_type": "measurement",
            "relationship": "LDL reduction from baseline",
            "confidence": 0.95,
        }]
        result = parse_llm_response(json.dumps(data), num_candidates=1)
        assert result[0]["relationship"] == "LDL reduction from baseline"

    def test_confidence_field_returned(self) -> None:
        data = [{
            "id": 0, "value": "0.003", "unit": "none",
            "data_type": "p-value", "relationship": "sig", "confidence": 0.99,
        }]
        result = parse_llm_response(json.dumps(data), num_candidates=1)
        assert result[0]["confidence"] == pytest.approx(0.99)

    def test_markdown_without_json_label(self) -> None:
        data = [{"id": 0, "value": "5", "unit": "mg",
                 "data_type": "measurement", "relationship": "", "confidence": 0.7}]
        fenced = f"```\n{json.dumps(data)}\n```"
        result = parse_llm_response(fenced, num_candidates=1)
        assert len(result) == 1

    def test_whitespace_only_response_returns_fallbacks(self) -> None:
        result = parse_llm_response("   \n\t  ", num_candidates=2)
        assert len(result) == 2
        assert all(r["data_type"] == "other" for r in result)


# ---------------------------------------------------------------------------
# _fallback_dict
# ---------------------------------------------------------------------------


class TestFallbackDict:
    """Tests for the _fallback_dict helper."""

    def test_id_preserved(self) -> None:
        fb = _fallback_dict(3)
        assert fb["id"] == 3

    def test_default_values(self) -> None:
        fb = _fallback_dict(0)
        assert fb["unit"] == "none"
        assert fb["data_type"] == "other"
        assert fb["relationship"] == ""
        assert fb["confidence"] == 0.0
        assert fb["value"] == ""

    def test_returns_dict(self) -> None:
        assert isinstance(_fallback_dict(0), dict)

    def test_id_zero(self) -> None:
        fb = _fallback_dict(0)
        assert fb["id"] == 0

    def test_id_large(self) -> None:
        fb = _fallback_dict(999)
        assert fb["id"] == 999

    def test_has_all_expected_keys(self) -> None:
        fb = _fallback_dict(0)
        for key in ("id", "value", "unit", "data_type", "relationship", "confidence"):
            assert key in fb

    def test_confidence_is_float(self) -> None:
        fb = _fallback_dict(0)
        assert isinstance(fb["confidence"], float)

    def test_unit_is_string(self) -> None:
        fb = _fallback_dict(0)
        assert isinstance(fb["unit"], str)


# ---------------------------------------------------------------------------
# _merge_record
# ---------------------------------------------------------------------------


class TestMergeRecord:
    """Tests for the _merge_record helper."""

    def _llm_fields(self, **overrides: Any) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "id": 0,
            "value": "32.4",
            "unit": "mg/dL",
            "data_type": "measurement",
            "relationship": "LDL reduction from baseline",
            "confidence": 0.95,
        }
        base.update(overrides)
        return base

    def test_value_overwritten_by_llm(self) -> None:
        original = _make_record(value="32")
        merged = _merge_record(original, self._llm_fields(value="32.4"))
        assert merged.value == "32.4"

    def test_unit_overwritten_by_llm(self) -> None:
        original = _make_record(unit="none")
        merged = _merge_record(original, self._llm_fields(unit="mg/dL"))
        assert merged.unit == "mg/dL"

    def test_data_type_overwritten_by_llm(self) -> None:
        original = _make_record(data_type="other")
        merged = _merge_record(original, self._llm_fields(data_type="measurement"))
        assert merged.data_type == "measurement"

    def test_relationship_set_by_llm(self) -> None:
        original = _make_record(relationship="")
        merged = _merge_record(original, self._llm_fields(relationship="LDL reduction"))
        assert merged.relationship == "LDL reduction"

    def test_confidence_set_by_llm(self) -> None:
        original = _make_record(confidence=None)
        merged = _merge_record(original, self._llm_fields(confidence=0.87))
        assert merged.confidence == pytest.approx(0.87)

    def test_confidence_clamped_above_one(self) -> None:
        original = _make_record(confidence=None)
        merged = _merge_record(original, self._llm_fields(confidence=1.5))
        assert merged.confidence == 1.0

    def test_confidence_clamped_below_zero(self) -> None:
        original = _make_record(confidence=None)
        merged = _merge_record(original, self._llm_fields(confidence=-0.3))
        assert merged.confidence == 0.0

    def test_confidence_none_on_invalid_value(self) -> None:
        original = _make_record(confidence=None)
        merged = _merge_record(original, self._llm_fields(confidence="high"))
        assert merged.confidence is None

    def test_context_preserved_from_original(self) -> None:
        original = _make_record(context="Original sentence with 32.4 mg/dL.")
        merged = _merge_record(original, self._llm_fields())
        assert merged.context == "Original sentence with 32.4 mg/dL."

    def test_raw_text_preserved_from_original(self) -> None:
        original = _make_record(raw_text="32.4 mg/dL")
        merged = _merge_record(original, self._llm_fields())
        assert merged.raw_text == "32.4 mg/dL"

    def test_section_preserved_from_original(self) -> None:
        original = _make_record(section="Results")
        merged = _merge_record(original, self._llm_fields())
        assert merged.section == "Results"

    def test_source_preserved_from_original(self) -> None:
        original = _make_record(source="paper.pdf")
        merged = _merge_record(original, self._llm_fields())
        assert merged.source == "paper.pdf"

    def test_empty_llm_value_falls_back_to_original(self) -> None:
        original = _make_record(value="32.4")
        merged = _merge_record(original, self._llm_fields(value=""))
        assert merged.value == "32.4"

    def test_empty_llm_unit_falls_back_to_original(self) -> None:
        original = _make_record(unit="mg/dL")
        merged = _merge_record(original, self._llm_fields(unit=""))
        assert merged.unit == "mg/dL"

    def test_returns_numeric_record_instance(self) -> None:
        original = _make_record()
        merged = _merge_record(original, self._llm_fields())
        assert isinstance(merged, NumericRecord)

    def test_none_confidence_in_llm_fields(self) -> None:
        original = _make_record()
        merged = _merge_record(original, self._llm_fields(confidence=None))
        assert merged.confidence is None

    def test_confidence_zero_preserved(self) -> None:
        original = _make_record(confidence=None)
        merged = _merge_record(original, self._llm_fields(confidence=0.0))
        assert merged.confidence == 0.0

    def test_confidence_one_preserved(self) -> None:
        original = _make_record(confidence=None)
        merged = _merge_record(original, self._llm_fields(confidence=1.0))
        assert merged.confidence == 1.0

    def test_whitespace_only_llm_value_falls_back(self) -> None:
        original = _make_record(value="32.4")
        merged = _merge_record(original, self._llm_fields(value="   "))
        assert merged.value == "32.4"

    def test_whitespace_only_llm_unit_falls_back(self) -> None:
        original = _make_record(unit="mg/dL")
        merged = _merge_record(original, self._llm_fields(unit="   "))
        assert merged.unit == "mg/dL"

    def test_section_none_from_original_preserved(self) -> None:
        original = _make_record(section=None)
        merged = _merge_record(original, self._llm_fields())
        assert merged.section is None

    def test_source_none_from_original_preserved(self) -> None:
        original = _make_record(source=None)
        merged = _merge_record(original, self._llm_fields())
        assert merged.source is None

    def test_data_type_p_value_from_llm(self) -> None:
        original = _make_record(data_type="other")
        merged = _merge_record(original, self._llm_fields(data_type="p-value"))
        assert merged.data_type == "p-value"

    def test_empty_data_type_from_llm_falls_back_to_original(self) -> None:
        original = _make_record(data_type="measurement")
        merged = _merge_record(original, self._llm_fields(data_type=""))
        assert merged.data_type == "measurement"

    def test_empty_unit_from_llm_defaults_to_none_string(self) -> None:
        # When both llm unit and original unit are empty, should default to "none".
        original = _make_record(unit="")
        merged = _merge_record(original, self._llm_fields(unit=""))
        assert merged.unit == "none"

    def test_confidence_string_float_parsed(self) -> None:
        original = _make_record(confidence=None)
        merged = _merge_record(original, self._llm_fields(confidence="0.75"))
        assert merged.confidence == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# _model_supports_json_mode
# ---------------------------------------------------------------------------


class TestModelSupportsJsonMode:
    """Tests for the _model_supports_json_mode helper."""

    def test_gpt4o_mini_supported(self) -> None:
        assert _model_supports_json_mode("gpt-4o-mini") is True

    def test_gpt4o_supported(self) -> None:
        assert _model_supports_json_mode("gpt-4o") is True

    def test_gpt4_turbo_supported(self) -> None:
        assert _model_supports_json_mode("gpt-4-turbo") is True

    def test_gpt35_turbo_1106_supported(self) -> None:
        assert _model_supports_json_mode("gpt-3.5-turbo-1106") is True

    def test_gpt35_turbo_0125_supported(self) -> None:
        assert _model_supports_json_mode("gpt-3.5-turbo-0125") is True

    def test_gpt4_1106_supported(self) -> None:
        assert _model_supports_json_mode("gpt-4-1106") is True

    def test_gpt4_0125_supported(self) -> None:
        assert _model_supports_json_mode("gpt-4-0125") is True

    def test_unknown_model_not_supported(self) -> None:
        assert _model_supports_json_mode("llama-3-70b") is False

    def test_empty_model_not_supported(self) -> None:
        assert _model_supports_json_mode("") is False

    def test_case_insensitive(self) -> None:
        assert _model_supports_json_mode("GPT-4O-MINI") is True

    def test_ollama_not_supported(self) -> None:
        assert _model_supports_json_mode("ollama/llama3") is False

    def test_claude_not_supported(self) -> None:
        assert _model_supports_json_mode("claude-3-opus") is False

    def test_returns_bool(self) -> None:
        result = _model_supports_json_mode("gpt-4o")
        assert isinstance(result, bool)

    def test_gpt4o_with_version_suffix(self) -> None:
        assert _model_supports_json_mode("gpt-4o-2024-05-13") is True

    def test_gpt4_turbo_with_suffix(self) -> None:
        assert _model_supports_json_mode("gpt-4-turbo-2024-04-09") is True


# ---------------------------------------------------------------------------
# _make_client
# ---------------------------------------------------------------------------


class TestMakeClient:
    """Tests for the _make_client helper."""

    def test_raises_value_error_without_key(self) -> None:
        """No key argument and no env var should raise ValueError."""
        import paper_miner.llm_parser as llm_mod
        env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict("os.environ", env_without_key, clear=True):
            with pytest.raises(ValueError, match="No API key"):
                llm_mod._make_client(api_key=None, base_url=None)

    def test_reads_key_from_env(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-env"}):
            with patch("openai.OpenAI") as mock_openai:
                import paper_miner.llm_parser as llm_mod
                llm_mod._make_client(api_key=None, base_url=None)
                mock_openai.assert_called_once()
                kwargs = mock_openai.call_args[1]
                assert kwargs["api_key"] == "test-key-env"

    def test_explicit_key_used_directly(self) -> None:
        with patch("openai.OpenAI") as mock_openai:
            import paper_miner.llm_parser as llm_mod
            llm_mod._make_client(api_key="explicit-key", base_url=None)
            kwargs = mock_openai.call_args[1]
            assert kwargs["api_key"] == "explicit-key"

    def test_explicit_key_takes_precedence_over_env(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            with patch("openai.OpenAI") as mock_openai:
                import paper_miner.llm_parser as llm_mod
                llm_mod._make_client(api_key="explicit-key", base_url=None)
                kwargs = mock_openai.call_args[1]
                assert kwargs["api_key"] == "explicit-key"

    def test_base_url_forwarded(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_openai:
                import paper_miner.llm_parser as llm_mod
                llm_mod._make_client(api_key="key", base_url="http://localhost:8080")
                kwargs = mock_openai.call_args[1]
                assert kwargs["base_url"] == "http://localhost:8080"

    def test_no_base_url_not_forwarded(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_openai:
                import paper_miner.llm_parser as llm_mod
                llm_mod._make_client(api_key="key", base_url=None)
                kwargs = mock_openai.call_args[1]
                assert "base_url" not in kwargs

    def test_returns_openai_instance(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_openai:
                mock_instance = MagicMock()
                mock_openai.return_value = mock_instance
                import paper_miner.llm_parser as llm_mod
                result = llm_mod._make_client(api_key="key", base_url=None)
                assert result is mock_instance

    def test_empty_string_key_treated_as_no_key(self) -> None:
        """An empty string api_key with no env var should raise ValueError."""
        import paper_miner.llm_parser as llm_mod
        env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict("os.environ", env_without_key, clear=True):
            with pytest.raises(ValueError, match="No API key"):
                llm_mod._make_client(api_key="", base_url=None)


# ---------------------------------------------------------------------------
# _call_llm
# ---------------------------------------------------------------------------


class TestCallLlm:
    """Tests for the _call_llm helper."""

    def _make_mock_client(self, content: str) -> MagicMock:
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(content)
        return client

    def test_returns_content_on_success(self) -> None:
        client = self._make_mock_client('[{"id": 0}]')
        result = _call_llm(client, model="gpt-4o-mini", user_message="test")
        assert result == '[{"id": 0}]'

    def test_retries_on_exception(self) -> None:
        client = MagicMock()
        # Fail twice, succeed on third attempt.
        client.chat.completions.create.side_effect = [
            RuntimeError("transient error"),
            RuntimeError("transient error"),
            _make_openai_response("[]"),
        ]
        with patch("time.sleep"):  # suppress actual sleep
            result = _call_llm(
                client,
                model="gpt-4o-mini",
                user_message="test",
                max_retries=3,
                retry_delay=0.01,
            )
        assert result == "[]"
        assert client.chat.completions.create.call_count == 3

    def test_raises_runtime_error_after_max_retries(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("always fails")
        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="failed after"):
                _call_llm(
                    client,
                    model="gpt-4o-mini",
                    user_message="test",
                    max_retries=2,
                    retry_delay=0.01,
                )
        assert client.chat.completions.create.call_count == 2

    def test_system_prompt_included_in_messages(self) -> None:
        client = self._make_mock_client("[]")
        _call_llm(client, model="gpt-4o-mini", user_message="user content")
        call_kwargs = client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == _SYSTEM_PROMPT

    def test_user_message_included_in_messages(self) -> None:
        client = self._make_mock_client("[]")
        _call_llm(client, model="gpt-4o-mini", user_message="my user message")
        call_kwargs = client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "my user message"

    def test_model_forwarded_to_api(self) -> None:
        client = self._make_mock_client("[]")
        _call_llm(client, model="gpt-4-turbo", user_message="test")
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4-turbo"

    def test_temperature_is_zero(self) -> None:
        client = self._make_mock_client("[]")
        _call_llm(client, model="gpt-4o-mini", user_message="test")
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    def test_first_attempt_success_no_sleep(self) -> None:
        client = self._make_mock_client("[{\"id\": 0}]")
        with patch("time.sleep") as mock_sleep:
            _call_llm(client, model="gpt-4o-mini", user_message="test", max_retries=3)
        mock_sleep.assert_not_called()

    def test_retry_sleeps_between_attempts(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            RuntimeError("fail"),
            _make_openai_response("[]"),
        ]
        with patch("time.sleep") as mock_sleep:
            _call_llm(
                client,
                model="gpt-4o-mini",
                user_message="test",
                max_retries=2,
                retry_delay=1.0,
            )
        mock_sleep.assert_called_once()

    def test_returns_string(self) -> None:
        client = self._make_mock_client("[{\"id\": 0}]")
        result = _call_llm(client, model="gpt-4o-mini", user_message="test")
        assert isinstance(result, str)

    def test_empty_content_returned(self) -> None:
        client = self._make_mock_client("")
        result = _call_llm(client, model="gpt-4o-mini", user_message="test")
        assert result == ""

    def test_max_retries_one_fails_immediately(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("fail")
        with patch("time.sleep"):
            with pytest.raises(RuntimeError):
                _call_llm(
                    client,
                    model="gpt-4o-mini",
                    user_message="test",
                    max_retries=1,
                    retry_delay=0.01,
                )
        assert client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# parse_candidates – integration tests with mocked API
# ---------------------------------------------------------------------------


class TestParseCandidates:
    """Integration tests for parse_candidates using mocked OpenAI calls."""

    def _patch_client(self, content: str) -> MagicMock:
        """Return a mock client that always returns *content*."""
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(content)
        return client

    def _llm_response_for(self, records: List[NumericRecord]) -> str:
        """Build a synthetic LLM JSON response for *records*."""
        return _make_llm_response(records)

    def test_empty_candidates_returns_empty_list(self) -> None:
        result = parse_candidates([], api_key="dummy")
        assert result == []

    def test_returns_numeric_record_instances(self) -> None:
        records = [_make_record()]
        llm_resp = self._llm_response_for(records)
        with patch("paper_miner.llm_parser._make_client") as mock_make_client:
            mock_make_client.return_value = self._patch_client(llm_resp)
            result = parse_candidates(records, api_key="sk-test")
        assert len(result) == 1
        assert isinstance(result[0], NumericRecord)

    def test_enriched_relationship_set(self) -> None:
        records = [_make_record()]
        llm_resp = json.dumps(
            [
                {
                    "id": 0,
                    "value": "32.4",
                    "unit": "mg/dL",
                    "data_type": "measurement",
                    "relationship": "LDL cholesterol reduction",
                    "confidence": 0.95,
                }
            ]
        )
        with patch("paper_miner.llm_parser._make_client") as mock_make_client:
            mock_make_client.return_value = self._patch_client(llm_resp)
            result = parse_candidates(records, api_key="sk-test")
        assert result[0].relationship == "LDL cholesterol reduction"

    def test_enriched_confidence_set(self) -> None:
        records = [_make_record()]
        llm_resp = json.dumps(
            [
                {
                    "id": 0,
                    "value": "32.4",
                    "unit": "mg/dL",
                    "data_type": "measurement",
                    "relationship": "LDL reduction",
                    "confidence": 0.92,
                }
            ]
        )
        with patch("paper_miner.llm_parser._make_client") as mock_make_client:
            mock_make_client.return_value = self._patch_client(llm_resp)
            result = parse_candidates(records, api_key="sk-test")
        assert result[0].confidence == pytest.approx(0.92)

    def test_source_preserved_through_enrichment(self) -> None:
        records = [_make_record(source="paper.pdf")]
        llm_resp = self._llm_response_for(records)
        with patch("paper_miner.llm_parser._make_client") as mock_make_client:
            mock_make_client.return_value = self._patch_client(llm_resp)
            result = parse_candidates(records, api_key="sk-test")
        assert result[0].source == "paper.pdf"

    def test_section_preserved_through_enrichment(self) -> None:
        records = [_make_record(section="Methods")]
        llm_resp = self._llm_response_for(records)
        with patch("paper_miner.llm_parser._make_client") as mock_make_client:
            mock_make_client.return_value = self._patch_client(llm_resp)
            result = parse_candidates(records, api_key="sk-test")
        assert result[0].section == "Methods"

    def test_multiple_candidates_all_returned(self) -> None:
        records = [_make_record(value=str(i)) for i in range(5)]
        llm_resp = json.dumps(
            [
                {
                    "id": i,
                    "value": str(i),
                    "unit": "none",
                    "data_type": "measurement",
                    "relationship": "",
                    "confidence": 0.8,
                }
                for i in range(5)
            ]
        )
        with patch("paper_miner.llm_parser._make_client") as mock_make_client:
            mock_make_client.return_value = self._patch_client(llm_resp)
            result = parse_candidates(records, api_key="sk-test")
        assert len(result) == 5

    def test_batch_splitting(self) -> None:
        """With batch_size=3 and 7 candidates, 3 API calls should be made."""
        records = [_make_record(value=str(i)) for i in range(7)]

        call_count = 0

        def side_effect(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            batch_input = json.loads(kwargs["messages"][1]["content"])
            resp_data = [
                {
                    "id": item["id"],
                    "value": item["value"],
                    "unit": "none",
                    "data_type": "measurement",
                    "relationship": "",
                    "confidence": 0.8,
                }
                for item in batch_input
            ]
            call_count += 1
            return _make_openai_response(json.dumps(resp_data))

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = side_effect

        with patch("paper_miner.llm_parser._make_client", return_value=mock_client):
            result = parse_candidates(records, api_key="sk-test", batch_size=3)

        # ceil(7/3) = 3 batches
        assert call_count == 3
        assert len(result) == 7

    def test_api_failure_falls_back_to_original_records(self) -> None:
        """When all retries fail, original records are returned unchanged."""
        records = [_make_record(value="42", unit="mg", data_type="measurement")]
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("network error")

        with patch("paper_miner.llm_parser._make_client", return_value=mock_client):
            with patch("time.sleep"):
                result = parse_candidates(
                    records,
                    api_key="sk-test",
                    max_retries=2,
                    retry_delay=0.01,
                )

        assert len(result) == 1
        # Original record returned as-is.
        assert result[0].value == "42"
        assert result[0].unit == "mg"
        assert result[0].data_type == "measurement"

    def test_partial_batch_failure_other_batches_succeed(self) -> None:
        """First batch fails; second batch should still succeed."""
        records_batch1 = [_make_record(value="1")]
        records_batch2 = [_make_record(value="2")]
        all_records = records_batch1 + records_batch2

        call_results = [
            RuntimeError("fail"),  # first batch attempt 1
            RuntimeError("fail"),  # first batch attempt 2 (max_retries=2)
            _make_openai_response(
                json.dumps(
                    [
                        {
                            "id": 0,
                            "value": "2",
                            "unit": "none",
                            "data_type": "measurement",
                            "relationship": "dose",
                            "confidence": 0.9,
                        }
                    ]
                )
            ),  # second batch succeeds
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = call_results

        with patch("paper_miner.llm_parser._make_client", return_value=mock_client):
            with patch("time.sleep"):
                result = parse_candidates(
                    all_records,
                    api_key="sk-test",
                    batch_size=1,
                    max_retries=2,
                    retry_delay=0.01,
                )

        assert len(result) == 2
        # First record: original fallback.
        assert result[0].value == "1"
        # Second record: LLM-enriched.
        assert result[1].relationship == "dose"
        assert result[1].confidence == pytest.approx(0.9)

    def test_malformed_response_returns_fallback_fields(self) -> None:
        """When the LLM returns unparseable JSON, records get fallback fields."""
        records = [_make_record(value="5", unit="mmHg")]
        with patch("paper_miner.llm_parser._make_client") as mock_make_client:
            mock_make_client.return_value = self._patch_client("this is not json")
            result = parse_candidates(records, api_key="sk-test")
        # The merge uses the fallback dict: value="" falls back to original.
        assert len(result) == 1
        assert result[0].value == "5"  # original preserved
        assert result[0].confidence == 0.0

    def test_api_key_forwarded_to_make_client(self) -> None:
        records = [_make_record()]
        llm_resp = self._llm_response_for(records)
        with patch("paper_miner.llm_parser._make_client") as mock_make_client:
            mock_make_client.return_value = self._patch_client(llm_resp)
            parse_candidates(records, api_key="sk-mykey", base_url="http://local")
            mock_make_client.assert_called_once_with(
                api_key="sk-mykey", base_url="http://local"
            )

    def test_model_forwarded_to_call_llm(self) -> None:
        records = [_make_record()]
        llm_resp = self._llm_response_for(records)
        with patch("paper_miner.llm_parser._make_client") as mock_make_client:
            mock_client = self._patch_client(llm_resp)
            mock_make_client.return_value = mock_client
            parse_candidates(records, api_key="sk-test", model="gpt-4-turbo")
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "gpt-4-turbo"

    def test_context_preserved_from_regex_extractor(self) -> None:
        original_context = "LDL was 32.4 mg/dL in the treatment group."
        records = [_make_record(context=original_context)]
        llm_resp = self._llm_response_for(records)
        with patch("paper_miner.llm_parser._make_client") as mock_make_client:
            mock_make_client.return_value = self._patch_client(llm_resp)
            result = parse_candidates(records, api_key="sk-test")
        assert result[0].context == original_context

    def test_raw_text_preserved_from_regex_extractor(self) -> None:
        records = [_make_record(raw_text="32.4 mg/dL")]
        llm_resp = self._llm_response_for(records)
        with patch("paper_miner.llm_parser._make_client") as mock_make_client:
            mock_make_client.return_value = self._patch_client(llm_resp)
            result = parse_candidates(records, api_key="sk-test")
        assert result[0].raw_text == "32.4 mg/dL"

    def test_order_preserved_across_batches(self) -> None:
        """Records should come back in the same order they were submitted."""
        n = 9
        records = [_make_record(value=str(i)) for i in range(n)]

        def side_effect(**kwargs: Any) -> MagicMock:
            batch_input = json.loads(kwargs["messages"][1]["content"])
            resp_data = [
                {
                    "id": item["id"],
                    "value": item["value"],
                    "unit": "none",
                    "data_type": "measurement",
                    "relationship": f"rel_{item['value']}",
                    "confidence": 0.8,
                }
                for item in batch_input
            ]
            return _make_openai_response(json.dumps(resp_data))

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = side_effect

        with patch("paper_miner.llm_parser._make_client", return_value=mock_client):
            result = parse_candidates(records, api_key="sk-test", batch_size=4)

        assert len(result) == n
        for i, record in enumerate(result):
            assert record.value == str(i)

    def test_base_url_forwarded_to_make_client(self) -> None:
        records = [_make_record()]
        llm_resp = self._llm_response_for(records)
        with patch("paper_miner.llm_parser._make_client") as mock_make_client:
            mock_make_client.return_value = self._patch_client(llm_resp)
            parse_candidates(
                records,
                api_key="sk-test",
                base_url="http://localhost:11434/v1",
            )
            _, kwargs = mock_make_client.call_args
            assert kwargs["base_url"] == "http://localhost:11434/v1"

    def test_default_batch_size_constant(self) -> None:
        assert isinstance(DEFAULT_BATCH_SIZE, int)
        assert DEFAULT_BATCH_SIZE > 0

    def test_single_candidate_single_api_call(self) -> None:
        records = [_make_record()]
        llm_resp = self._llm_response_for(records)
        mock_client = self._patch_client(llm_resp)
        with patch("paper_miner.llm_parser._make_client", return_value=mock_client):
            parse_candidates(records, api_key="sk-test", batch_size=10)
        assert mock_client.chat.completions.create.call_count == 1

    def test_exactly_batch_size_candidates_single_call(self) -> None:
        batch_size = 5
        records = [_make_record(value=str(i)) for i in range(batch_size)]
        llm_resp = json.dumps(
            [
                {
                    "id": i,
                    "value": str(i),
                    "unit": "none",
                    "data_type": "measurement",
                    "relationship": "",
                    "confidence": 0.8,
                }
                for i in range(batch_size)
            ]
        )
        mock_client = self._patch_client(llm_resp)
        with patch("paper_miner.llm_parser._make_client", return_value=mock_client):
            result = parse_candidates(
                records, api_key="sk-test", batch_size=batch_size
            )
        assert mock_client.chat.completions.create.call_count == 1
        assert len(result) == batch_size

    def test_llm_corrects_data_type(self) -> None:
        """LLM can override the heuristic data_type from the regex extractor."""
        records = [_make_record(data_type="other")]
        llm_resp = json.dumps(
            [
                {
                    "id": 0,
                    "value": "32.4",
                    "unit": "mg/dL",
                    "data_type": "measurement",
                    "relationship": "LDL reduction",
                    "confidence": 0.95,
                }
            ]
        )
        with patch("paper_miner.llm_parser._make_client") as mock_make_client:
            mock_make_client.return_value = self._patch_client(llm_resp)
            result = parse_candidates(records, api_key="sk-test")
        assert result[0].data_type == "measurement"

    def test_llm_corrects_unit(self) -> None:
        """LLM can override the heuristic unit."""
        records = [_make_record(unit="none")]
        llm_resp = json.dumps(
            [
                {
                    "id": 0,
                    "value": "32.4",
                    "unit": "mg/dL",
                    "data_type": "measurement",
                    "relationship": "LDL",
                    "confidence": 0.9,
                }
            ]
        )
        with patch("paper_miner.llm_parser._make_client") as mock_make_client:
            mock_make_client.return_value = self._patch_client(llm_resp)
            result = parse_candidates(records, api_key="sk-test")
        assert result[0].unit == "mg/dL"

    def test_all_results_are_numeric_records(self) -> None:
        records = [_make_record(value=str(i)) for i in range(3)]
        llm_resp = json.dumps(
            [
                {
                    "id": i,
                    "value": str(i),
                    "unit": "none",
                    "data_type": "measurement",
                    "relationship": "",
                    "confidence": 0.8,
                }
                for i in range(3)
            ]
        )
        with patch("paper_miner.llm_parser._make_client") as mock_make_client:
            mock_make_client.return_value = self._patch_client(llm_resp)
            result = parse_candidates(records, api_key="sk-test")
        assert all(isinstance(r, NumericRecord) for r in result)
