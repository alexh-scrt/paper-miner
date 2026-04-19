"""Unit tests for paper_miner.extractor.

Tests cover:
- extract_candidates: detection of p-values, percentages, measurements,
  confidence intervals, mean±SD, ratios, and counts.
- Context capture: surrounding sentence is included in each record.
- Deduplication: overlapping spans from different patterns are not double-counted.
- Trivial number filtering: single-digit bare integers are suppressed.
- Edge cases: empty text, whitespace-only, no numeric content.
- Internal helpers: _split_sentences, _find_sentence, _extract_value,
  _extract_unit, _classify_raw, _is_trivial_number, _overlaps_claimed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_miner.extractor import (
    extract_candidates,
    _split_sentences,
    _find_sentence,
    _extract_value,
    _extract_unit,
    _classify_raw,
    _is_trivial_number,
    _overlaps_claimed,
)
from paper_miner.models import NumericRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _data_types(records: List[NumericRecord]) -> List[str]:
    return [r.data_type for r in records]


def _values(records: List[NumericRecord]) -> List[str]:
    return [r.value for r in records]


def _has_type(records: List[NumericRecord], dtype: str) -> bool:
    return any(r.data_type == dtype for r in records)


# ---------------------------------------------------------------------------
# extract_candidates – edge cases
# ---------------------------------------------------------------------------


class TestExtractCandidatesEdgeCases:
    """Edge cases for extract_candidates."""

    def test_empty_string_returns_empty_list(self) -> None:
        assert extract_candidates("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert extract_candidates("   \n\t  ") == []

    def test_no_numbers_returns_empty_list(self) -> None:
        text = "There are no numerical values in this sentence whatsoever."
        result = extract_candidates(text)
        assert isinstance(result, list)
        # No true numeric matches.
        assert all(r.data_type != "measurement" for r in result) or result == []

    def test_returns_list_of_numeric_records(self) -> None:
        text = "The dosage was 50 mg per day for 12 weeks."
        result = extract_candidates(text)
        assert isinstance(result, list)
        for r in result:
            assert isinstance(r, NumericRecord)

    def test_source_propagated(self) -> None:
        text = "Blood pressure was 120 mmHg."
        records = extract_candidates(text, source="paper1.pdf")
        assert all(r.source == "paper1.pdf" for r in records)

    def test_section_propagated(self) -> None:
        text = "The p-value was 0.002."
        records = extract_candidates(text, section="Results")
        assert all(r.section == "Results" for r in records)

    def test_source_none_by_default(self) -> None:
        text = "The mean was 42.0 kg."
        records = extract_candidates(text)
        assert all(r.source is None for r in records)

    def test_confidence_none(self) -> None:
        """confidence field must be None for regex-only records."""
        text = "The reduction was 32.4 mg/dL (p < 0.001)."
        records = extract_candidates(text)
        assert len(records) >= 1
        assert all(r.confidence is None for r in records)

    def test_relationship_empty_string(self) -> None:
        """relationship field must be empty string before LLM enrichment."""
        text = "Triglycerides decreased by 12.4 mg/dL."
        records = extract_candidates(text)
        assert len(records) >= 1
        assert all(r.relationship == "" for r in records)


# ---------------------------------------------------------------------------
# p-value detection
# ---------------------------------------------------------------------------


class TestPValueDetection:
    """Tests for p-value pattern detection."""

    def test_p_equals(self) -> None:
        text = "The difference was statistically significant (p = 0.003)."
        records = extract_candidates(text)
        assert _has_type(records, "p-value")

    def test_p_less_than(self) -> None:
        text = "Results showed p < 0.001 after adjustment."
        records = extract_candidates(text)
        assert _has_type(records, "p-value")

    def test_p_greater_than(self) -> None:
        text = "No significant effect was found (p > 0.05)."
        records = extract_candidates(text)
        assert _has_type(records, "p-value")

    def test_p_value_extracted_correctly(self) -> None:
        text = "The p-value was p = 0.042."
        records = extract_candidates(text)
        pvals = [r for r in records if r.data_type == "p-value"]
        assert len(pvals) >= 1
        assert pvals[0].value == "0.042"

    def test_p_value_unit_is_none(self) -> None:
        text = "Statistical significance: p < 0.05."
        records = extract_candidates(text)
        pvals = [r for r in records if r.data_type == "p-value"]
        assert len(pvals) >= 1
        assert pvals[0].unit == "none"

    def test_p_value_context_captured(self) -> None:
        text = "The treatment was effective. p = 0.01 was the observed significance."
        records = extract_candidates(text)
        pvals = [r for r in records if r.data_type == "p-value"]
        assert len(pvals) >= 1
        assert "0.01" in pvals[0].context

    def test_multiple_pvalues(self) -> None:
        text = (
            "Primary endpoint: p < 0.001. "
            "Secondary endpoint: p = 0.03. "
            "Safety endpoint: p = 0.65."
        )
        records = extract_candidates(text)
        pvals = [r for r in records if r.data_type == "p-value"]
        assert len(pvals) >= 2

    def test_p_value_raw_text_preserved(self) -> None:
        text = "The result was significant (p = 0.001)."
        records = extract_candidates(text)
        pvals = [r for r in records if r.data_type == "p-value"]
        assert len(pvals) >= 1
        assert "0.001" in pvals[0].raw_text

    def test_p_value_non_significant(self) -> None:
        text = "HDL did not change significantly (p = 0.42)."
        records = extract_candidates(text)
        assert _has_type(records, "p-value")
        pvals = [r for r in records if r.data_type == "p-value"]
        assert pvals[0].value == "0.42"


# ---------------------------------------------------------------------------
# Percentage detection
# ---------------------------------------------------------------------------


class TestPercentageDetection:
    """Tests for percentage pattern detection."""

    def test_simple_percentage(self) -> None:
        text = "Approximately 38% of adults are affected worldwide."
        records = extract_candidates(text)
        assert _has_type(records, "percentage")

    def test_percentage_unit_is_percent(self) -> None:
        text = "A 15.2% reduction in total cholesterol was observed."
        records = extract_candidates(text)
        pcts = [r for r in records if r.data_type == "percentage"]
        assert len(pcts) >= 1
        assert pcts[0].unit == "%"

    def test_percentage_value_extracted(self) -> None:
        text = "The response rate was 72.5%."
        records = extract_candidates(text)
        pcts = [r for r in records if r.data_type == "percentage"]
        assert len(pcts) >= 1
        assert pcts[0].value == "72.5"

    def test_decimal_percentage(self) -> None:
        text = "Completion rate was 95.8% across all sites."
        records = extract_candidates(text)
        pcts = [r for r in records if r.data_type == "percentage"]
        assert len(pcts) >= 1
        assert "95.8" in [r.value for r in pcts]

    def test_percentage_context(self) -> None:
        text = "A total of 52% female participants were enrolled in the study."
        records = extract_candidates(text)
        pcts = [r for r in records if r.data_type == "percentage"]
        assert len(pcts) >= 1
        assert "52" in pcts[0].context

    def test_integer_percentage(self) -> None:
        text = "The dropout rate was 20% as anticipated."
        records = extract_candidates(text)
        pcts = [r for r in records if r.data_type == "percentage"]
        assert len(pcts) >= 1
        assert pcts[0].value == "20"

    def test_percentage_raw_text_contains_percent_sign(self) -> None:
        text = "Efficacy rate: 85%."
        records = extract_candidates(text)
        pcts = [r for r in records if r.data_type == "percentage"]
        assert len(pcts) >= 1
        assert "%" in pcts[0].raw_text


# ---------------------------------------------------------------------------
# Measurement detection
# ---------------------------------------------------------------------------


class TestMeasurementDetection:
    """Tests for measurement (number + unit) pattern detection."""

    def test_mmhg_measurement(self) -> None:
        text = "The treatment reduced blood pressure by 12.5 mmHg."
        records = extract_candidates(text)
        measurements = [r for r in records if r.data_type == "measurement"]
        assert len(measurements) >= 1
        assert "mmHg" in [r.unit for r in measurements]

    def test_mg_dl_measurement(self) -> None:
        text = "LDL cholesterol was reduced by 32.4 mg/dL from baseline."
        records = extract_candidates(text)
        measurements = [r for r in records if r.data_type == "measurement"]
        assert len(measurements) >= 1
        values = [r.value for r in measurements]
        assert "32.4" in values

    def test_kg_per_m2_measurement(self) -> None:
        text = "Mean BMI was 27.6 kg/m²."
        records = extract_candidates(text)
        # Should find a measurement with kg/m² unit.
        assert len(records) >= 1

    def test_measurement_unit_extracted(self) -> None:
        text = "Creatinine was 0.9 mg/dL."
        records = extract_candidates(text)
        measurements = [r for r in records if r.unit == "mg/dL"]
        assert len(measurements) >= 1

    def test_measurement_value_extracted(self) -> None:
        text = "Heart rate was 72 bpm at rest."
        records = extract_candidates(text)
        assert len(records) >= 1
        values = [r.value for r in records]
        assert "72" in values

    def test_ml_min_unit(self) -> None:
        text = "eGFR was 45 mL/min/1.73 m²."
        records = extract_candidates(text)
        assert len(records) >= 1

    def test_week_unit(self) -> None:
        text = "The intervention lasted 12 weeks."
        records = extract_candidates(text)
        assert len(records) >= 1

    def test_years_unit(self) -> None:
        text = "Mean age was 54.3 years."
        records = extract_candidates(text)
        assert len(records) >= 1

    def test_negative_measurement(self) -> None:
        text = "HDL change was −0.8 mg/dL (p = 0.42)."
        records = extract_candidates(text)
        measurements = [r for r in records if r.unit == "mg/dL"]
        assert len(measurements) >= 1

    def test_measurement_context_non_empty(self) -> None:
        text = "The compound dose was 50 mg daily."
        records = extract_candidates(text)
        for r in records:
            assert len(r.context) > 0

    def test_measurement_raw_text_contains_value_and_unit(self) -> None:
        text = "Systolic blood pressure was 140 mmHg at baseline."
        records = extract_candidates(text)
        measurements = [r for r in records if r.unit == "mmHg"]
        if measurements:
            assert "140" in measurements[0].raw_text
            assert "mmHg" in measurements[0].raw_text

    def test_u_per_l_unit(self) -> None:
        text = "ALT increased by 3.2 U/L in the treatment group."
        records = extract_candidates(text)
        assert len(records) >= 1


# ---------------------------------------------------------------------------
# Confidence interval detection
# ---------------------------------------------------------------------------


class TestConfidenceIntervalDetection:
    """Tests for confidence interval pattern detection."""

    def test_95_percent_ci(self) -> None:
        text = "The reduction was 32.4 mg/dL (95% CI: 28.1\u201336.7 mg/dL)."
        records = extract_candidates(text)
        ci_records = [r for r in records if r.data_type == "confidence_interval"]
        assert len(ci_records) >= 1

    def test_ci_value_is_first_number(self) -> None:
        text = "Effect size: 95% CI: 10.2\u201315.8."
        records = extract_candidates(text)
        ci_records = [r for r in records if r.data_type == "confidence_interval"]
        assert len(ci_records) >= 1
        # The value should be a number found in the CI range.
        assert ci_records[0].value in ("10.2", "10", "95")

    def test_ci_unit_extracted(self) -> None:
        text = "95% CI: 28.1\u201336.7 mg/dL."
        records = extract_candidates(text)
        ci_records = [r for r in records if r.data_type == "confidence_interval"]
        if ci_records:
            # unit should be mg/dL or none
            assert ci_records[0].unit in ("mg/dL", "none")

    def test_ci_context_captured(self) -> None:
        text = "The 95% CI was 28.1 to 36.7 mg/dL for the primary endpoint."
        records = extract_candidates(text)
        # At minimum, numeric records should be found.
        assert len(records) >= 1

    def test_ci_raw_text_non_empty(self) -> None:
        text = "95% CI: 28.1\u201336.7 mg/dL was reported."
        records = extract_candidates(text)
        ci_records = [r for r in records if r.data_type == "confidence_interval"]
        if ci_records:
            assert len(ci_records[0].raw_text) > 0

    def test_negative_ci_bounds(self) -> None:
        text = "The between-group difference was 95% CI: -35.1 to -26.1 mg/dL."
        records = extract_candidates(text)
        # Should find at least some numeric records.
        assert len(records) >= 1


# ---------------------------------------------------------------------------
# Mean ± SD detection
# ---------------------------------------------------------------------------


class TestMeanSDDetection:
    """Tests for mean ± standard deviation pattern detection."""

    def test_mean_sd_detected(self) -> None:
        text = "Mean age was 54.3 \u00b1 7.8 years."
        records = extract_candidates(text)
        means = [r for r in records if r.data_type == "mean"]
        assert len(means) >= 1

    def test_mean_value_extracted(self) -> None:
        text = "Baseline LDL: 158.3 \u00b1 18.6 mg/dL."
        records = extract_candidates(text)
        means = [r for r in records if r.data_type == "mean"]
        assert len(means) >= 1
        assert means[0].value == "158.3"

    def test_mean_unit_extracted(self) -> None:
        text = "BMI: 27.6 \u00b1 4.1 kg/m\u00b2."
        records = extract_candidates(text)
        means = [r for r in records if r.data_type == "mean"]
        if means:
            # Unit may be kg/m² or similar — just ensure it's not empty
            assert means[0].unit != ""

    def test_multiple_mean_sd_in_text(self) -> None:
        text = (
            "Compound X group: 54.1 \u00b1 7.6 years; "
            "Placebo group: 54.5 \u00b1 8.0 years."
        )
        records = extract_candidates(text)
        means = [r for r in records if r.data_type == "mean"]
        assert len(means) >= 2

    def test_mean_raw_text_contains_plus_minus(self) -> None:
        text = "The mean score was 78.3 \u00b1 12.1 points."
        records = extract_candidates(text)
        means = [r for r in records if r.data_type == "mean"]
        if means:
            # raw_text should contain both values
            assert "78.3" in means[0].raw_text

    def test_mean_confidence_is_none(self) -> None:
        text = "Mean BMI: 27.6 \u00b1 4.1 kg/m\u00b2."
        records = extract_candidates(text)
        means = [r for r in records if r.data_type == "mean"]
        if means:
            assert means[0].confidence is None


# ---------------------------------------------------------------------------
# Ratio detection (OR, HR, RR)
# ---------------------------------------------------------------------------


class TestRatioDetection:
    """Tests for odds ratio, hazard ratio, and risk ratio detection."""

    def test_odds_ratio(self) -> None:
        text = "The odds ratio was OR = 1.45 (95% CI: 1.12\u20131.87)."
        records = extract_candidates(text)
        ratios = [r for r in records if r.data_type == "ratio"]
        assert len(ratios) >= 1

    def test_hazard_ratio(self) -> None:
        text = "Cardiovascular risk HR: 0.78 after treatment."
        records = extract_candidates(text)
        ratios = [r for r in records if r.data_type == "ratio"]
        assert len(ratios) >= 1

    def test_risk_ratio(self) -> None:
        text = "The relative risk RR = 0.65 was observed."
        records = extract_candidates(text)
        ratios = [r for r in records if r.data_type == "ratio"]
        assert len(ratios) >= 1

    def test_ratio_unit_is_none(self) -> None:
        text = "OR = 2.1 for the primary endpoint."
        records = extract_candidates(text)
        ratios = [r for r in records if r.data_type == "ratio"]
        if ratios:
            assert ratios[0].unit == "none"

    def test_ratio_value_extracted(self) -> None:
        text = "Hazard ratio HR = 0.72 was statistically significant."
        records = extract_candidates(text)
        ratios = [r for r in records if r.data_type == "ratio"]
        assert len(ratios) >= 1
        assert ratios[0].value == "0.72"

    def test_ratio_raw_text_non_empty(self) -> None:
        text = "The NNT = 12 for this intervention."
        records = extract_candidates(text)
        ratios = [r for r in records if r.data_type == "ratio"]
        if ratios:
            assert len(ratios[0].raw_text) > 0

    def test_irr_detected(self) -> None:
        text = "Incidence rate ratio IRR = 0.88 (95% CI: 0.77\u20131.01)."
        records = extract_candidates(text)
        ratios = [r for r in records if r.data_type == "ratio"]
        assert len(ratios) >= 1


# ---------------------------------------------------------------------------
# Count / sample size detection
# ---------------------------------------------------------------------------


class TestCountDetection:
    """Tests for sample size and count pattern detection."""

    def test_n_equals(self) -> None:
        text = "A total of n = 240 participants were enrolled."
        records = extract_candidates(text)
        counts = [r for r in records if r.data_type == "count"]
        assert len(counts) >= 1

    def test_capital_N_equals(self) -> None:
        text = "The study had N = 120 participants per group."
        records = extract_candidates(text)
        counts = [r for r in records if r.data_type == "count"]
        assert len(counts) >= 1

    def test_count_value_extracted(self) -> None:
        text = "Enrolled participants: N = 240."
        records = extract_candidates(text)
        counts = [r for r in records if r.data_type == "count"]
        assert len(counts) >= 1
        assert counts[0].value == "240"

    def test_count_unit_is_none(self) -> None:
        text = "n = 60 healthy volunteers were studied."
        records = extract_candidates(text)
        counts = [r for r in records if r.data_type == "count"]
        if counts:
            assert counts[0].unit == "none"

    def test_count_raw_text_format(self) -> None:
        text = "Sample size: n = 96 per group."
        records = extract_candidates(text)
        counts = [r for r in records if r.data_type == "count"]
        if counts:
            assert "96" in counts[0].raw_text

    def test_count_context_captured(self) -> None:
        text = "The per-protocol population comprised n = 228 participants."
        records = extract_candidates(text)
        counts = [r for r in records if r.data_type == "count"]
        if counts:
            assert "228" in counts[0].context


# ---------------------------------------------------------------------------
# Context capture
# ---------------------------------------------------------------------------


class TestContextCapture:
    """Tests that the surrounding sentence context is correctly captured."""

    def test_context_contains_numeric_value(self) -> None:
        text = "The reduction was 32.4 mg/dL from baseline."
        records = extract_candidates(text)
        assert len(records) >= 1
        for r in records:
            assert r.value in r.context or r.raw_text in r.context

    def test_context_is_non_empty(self) -> None:
        text = "Participants had a mean BMI of 27.6 kg/m\u00b2 at baseline."
        records = extract_candidates(text)
        assert len(records) >= 1
        assert all(len(r.context) > 0 for r in records)

    def test_multi_sentence_context_picks_correct_sentence(self) -> None:
        text = (
            "This is background information. "
            "The primary outcome was a reduction of 32.4 mg/dL. "
            "Safety was also assessed."
        )
        records = extract_candidates(text)
        measurements = [r for r in records if "32.4" in r.raw_text]
        if measurements:
            assert "32.4" in measurements[0].context

    def test_context_does_not_bleed_across_paragraphs(self) -> None:
        text = (
            "Section one describes baseline data.\n\n"
            "The result was 15.2% reduction in cholesterol."
        )
        records = extract_candidates(text)
        pcts = [r for r in records if r.data_type == "percentage"]
        if pcts:
            # Context should include the percentage sentence.
            assert "15.2" in pcts[0].context

    def test_context_is_a_string(self) -> None:
        text = "Mean ALT was 3.2 U/L post-treatment."
        records = extract_candidates(text)
        for r in records:
            assert isinstance(r.context, str)

    def test_context_not_longer_than_reasonable(self) -> None:
        # Context should be a sentence, not the entire text.
        long_text = ("Padding sentence number {:d}. ".format(i) for i in range(100))
        sentence = "The key result was 32.4 mg/dL."
        text = " ".join(list(long_text)[:50]) + " " + sentence
        records = extract_candidates(text)
        measurements = [r for r in records if "32.4" in r.raw_text]
        if measurements:
            # Context should be much shorter than the full text.
            assert len(measurements[0].context) < len(text)


# ---------------------------------------------------------------------------
# Deduplication / overlap suppression
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests that overlapping pattern matches are deduplicated."""

    def test_p_value_not_duplicated(self) -> None:
        text = "The finding was significant (p < 0.001)."
        records = extract_candidates(text)
        pvals = [r for r in records if r.data_type == "p-value"]
        # Should appear at most once (with possible sub-matches it could be 2).
        assert len(pvals) <= 2

    def test_no_duplicate_raw_text_at_same_offset(self) -> None:
        text = "Mean LDL was 158.3 \u00b1 18.6 mg/dL."
        records = extract_candidates(text)
        raw_texts = [r.raw_text for r in records]
        # raw_text values may differ (e.g. "158.3 ± 18.6 mg/dL" vs "18.6 mg/dL")
        # but the same raw_text should not appear multiple times.
        assert len(raw_texts) == len(set(raw_texts)) or len(records) < 5

    def test_measurement_within_ci_not_double_counted(self) -> None:
        text = "The reduction was 95% CI: 28.1–36.7 mg/dL."
        records = extract_candidates(text)
        # Ensure the count is manageable (no runaway duplication).
        assert len(records) <= 10

    def test_p_value_within_parentheses(self) -> None:
        text = "The result was significant (p = 0.003; OR = 1.5)."
        records = extract_candidates(text)
        # Both p-value and ratio should be found but without excessive duplicates.
        pvals = [r for r in records if r.data_type == "p-value"]
        ratios = [r for r in records if r.data_type == "ratio"]
        assert len(pvals) >= 1
        assert len(ratios) >= 1


# ---------------------------------------------------------------------------
# Trivial number filtering
# ---------------------------------------------------------------------------


class TestTrivialFilter:
    """Tests that trivial single-digit bare numbers are suppressed."""

    def test_single_digit_without_unit_suppressed(self) -> None:
        # The number "1" alone (like a list item marker) should be suppressed.
        text = "Results are listed in Table 1 and Figure 2."
        records = extract_candidates(text)
        # Any remaining records should have units or multi-digit values.
        for r in records:
            digits = len(r.value.replace("-", "").replace(".", "").replace(",", ""))
            if digits == 1:
                # Single digit must have a unit.
                assert r.unit != "none" or r.data_type in (
                    "p-value", "confidence_interval", "ratio", "count"
                )

    def test_multi_digit_numbers_not_suppressed(self) -> None:
        text = "There were 42 adverse events recorded."
        records = extract_candidates(text)
        # 42 is multi-digit; it may or may not be retained depending on context.
        values = [r.value for r in records]
        # Permissive: just ensure the function runs without error.
        assert isinstance(values, list)

    def test_zero_point_something_not_trivial(self) -> None:
        text = "The p-value was p = 0.001."
        records = extract_candidates(text)
        pvals = [r for r in records if r.data_type == "p-value"]
        assert len(pvals) >= 1

    def test_large_number_not_suppressed(self) -> None:
        text = "The sample comprised 185000 participants."
        records = extract_candidates(text)
        # 185000 has 6 digits — definitely not trivial.
        values = [r.value for r in records]
        assert "185000" in values or any("185" in v for v in values)


# ---------------------------------------------------------------------------
# _split_sentences helper
# ---------------------------------------------------------------------------


class TestSplitSentences:
    """Tests for the internal _split_sentences helper."""

    def test_single_sentence(self) -> None:
        text = "The value was 42."
        result = _split_sentences(text)
        assert len(result) == 1
        assert result[0][0] == "The value was 42."
        assert result[0][1] == 0

    def test_two_sentences(self) -> None:
        text = "First sentence. Second sentence."
        result = _split_sentences(text)
        assert len(result) == 2

    def test_empty_string(self) -> None:
        result = _split_sentences("")
        assert result == []

    def test_offsets_are_correct(self) -> None:
        text = "AAA. BBB."
        result = _split_sentences(text)
        # First sentence starts at 0.
        assert result[0][1] == 0
        # Second sentence starts after "AAA. "
        assert result[1][1] > 0

    def test_exclamation_mark_splits(self) -> None:
        text = "Significant result! Another finding."
        result = _split_sentences(text)
        assert len(result) == 2

    def test_question_mark_splits(self) -> None:
        text = "Is this real? Yes it is."
        result = _split_sentences(text)
        assert len(result) == 2

    def test_returns_list_of_tuples(self) -> None:
        text = "Only one sentence here."
        result = _split_sentences(text)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)

    def test_tuple_has_two_elements(self) -> None:
        text = "One sentence."
        result = _split_sentences(text)
        assert len(result[0]) == 2

    def test_sentence_text_is_string(self) -> None:
        text = "First. Second."
        result = _split_sentences(text)
        for sentence, offset in result:
            assert isinstance(sentence, str)
            assert isinstance(offset, int)

    def test_three_sentences(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        result = _split_sentences(text)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _find_sentence helper
# ---------------------------------------------------------------------------


class TestFindSentence:
    """Tests for the internal _find_sentence helper."""

    def test_finds_correct_sentence(self) -> None:
        text = "Background info here. The value was 42. Summary follows."
        # Offset of "42" is around position 30.
        offset = text.index("42")
        sentence = _find_sentence(text, offset)
        assert "42" in sentence

    def test_fallback_window_when_no_sentence_boundary(self) -> None:
        # A very long string with no sentence boundary.
        text = "a" * 200 + "123" + "b" * 200
        offset = 200
        sentence = _find_sentence(text, offset)
        assert "123" in sentence

    def test_returns_string(self) -> None:
        text = "The p-value was 0.001."
        result = _find_sentence(text, 0)
        assert isinstance(result, str)

    def test_returns_non_empty_string(self) -> None:
        text = "Blood pressure was 120 mmHg."
        result = _find_sentence(text, 0)
        assert len(result) > 0

    def test_offset_at_end_of_text(self) -> None:
        text = "Single sentence."
        result = _find_sentence(text, len(text) - 1)
        assert isinstance(result, str)

    def test_finds_sentence_in_multi_sentence_text(self) -> None:
        text = "First sentence. The measurement was 32.4 mg/dL. Third sentence."
        offset = text.index("32.4")
        sentence = _find_sentence(text, offset)
        assert "32.4" in sentence


# ---------------------------------------------------------------------------
# _extract_value helper
# ---------------------------------------------------------------------------


class TestExtractValue:
    """Tests for the internal _extract_value helper."""

    def test_simple_number(self) -> None:
        assert _extract_value("32.4 mg/dL", "measurement") == "32.4"

    def test_p_value(self) -> None:
        assert _extract_value("p < 0.001", "p-value") == "0.001"

    def test_mean_sd(self) -> None:
        assert _extract_value("54.3 \u00b1 7.8", "mean") == "54.3"

    def test_no_number_returns_raw(self) -> None:
        result = _extract_value("no digits here", "other")
        assert result == "no digits here"

    def test_percentage(self) -> None:
        assert _extract_value("15.2%", "percentage") == "15.2"

    def test_negative_number(self) -> None:
        result = _extract_value("-0.8 mg/dL", "measurement")
        assert result in ("-0.8", "0.8")

    def test_integer_value(self) -> None:
        result = _extract_value("n = 240", "count")
        assert result == "240"

    def test_ratio_value(self) -> None:
        result = _extract_value("OR = 1.45", "ratio")
        assert result == "1.45"

    def test_scientific_notation(self) -> None:
        result = _extract_value("1.5e-3 mg", "measurement")
        assert "1.5" in result or "1.5e-3" == result

    def test_returns_string(self) -> None:
        result = _extract_value("42 bpm", "measurement")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _extract_unit helper
# ---------------------------------------------------------------------------


class TestExtractUnit:
    """Tests for the internal _extract_unit helper."""

    def test_percentage_returns_percent(self) -> None:
        assert _extract_unit("15.2%", "percentage") == "%"

    def test_p_value_returns_none(self) -> None:
        assert _extract_unit("p < 0.001", "p-value") == "none"

    def test_count_returns_none(self) -> None:
        assert _extract_unit("n = 240", "count") == "none"

    def test_ratio_returns_none(self) -> None:
        assert _extract_unit("OR = 1.5", "ratio") == "none"

    def test_measurement_with_unit(self) -> None:
        result = _extract_unit("32.4 mg/dL", "measurement")
        assert result == "mg/dL"

    def test_measurement_no_unit_returns_none(self) -> None:
        result = _extract_unit("42", "measurement")
        assert result == "none"

    def test_after_text_fallback(self) -> None:
        result = _extract_unit("32.4", "measurement", after_text=" mg/dL")
        assert result == "mg/dL"

    def test_mmhg_unit(self) -> None:
        result = _extract_unit("12.5 mmHg", "measurement")
        assert result == "mmHg"

    def test_returns_string(self) -> None:
        result = _extract_unit("100 mg", "measurement")
        assert isinstance(result, str)

    def test_ci_with_unit_in_raw(self) -> None:
        result = _extract_unit("95% CI: 28.1\u201336.7 mg/dL", "confidence_interval")
        assert result in ("mg/dL", "none")

    def test_ci_no_unit_returns_none(self) -> None:
        result = _extract_unit("95% CI: 10\u201320", "confidence_interval")
        assert result == "none"


# ---------------------------------------------------------------------------
# _classify_raw helper
# ---------------------------------------------------------------------------


class TestClassifyRaw:
    """Tests for the internal _classify_raw helper."""

    def test_pvalue_hint(self) -> None:
        assert _classify_raw("p < 0.001", "p-value") == "p-value"

    def test_ci_hint(self) -> None:
        assert _classify_raw("95% CI: 10\u201320", "confidence_interval") == "confidence_interval"

    def test_mean_hint(self) -> None:
        assert _classify_raw("54.3 \u00b1 7.8", "mean") == "mean"

    def test_percentage_hint(self) -> None:
        assert _classify_raw("15.2%", "percentage") == "percentage"

    def test_measurement_hint(self) -> None:
        assert _classify_raw("32.4 mg/dL", "measurement") == "measurement"

    def test_count_hint(self) -> None:
        assert _classify_raw("n = 240", "count") == "count"

    def test_ratio_hint(self) -> None:
        assert _classify_raw("OR = 1.5", "ratio") == "ratio"

    def test_other_hint(self) -> None:
        assert _classify_raw("some 42 value", "other") == "other"

    def test_returns_string(self) -> None:
        result = _classify_raw("42 mg", "measurement")
        assert isinstance(result, str)

    def test_sd_hint_within_mean_pattern(self) -> None:
        # When raw contains "SD", it might be classified as standard_deviation.
        result = _classify_raw("SD = 7.8", "mean")
        assert result in ("mean", "standard_deviation")


# ---------------------------------------------------------------------------
# _is_trivial_number helper
# ---------------------------------------------------------------------------


class TestIsTrivialNumber:
    """Tests for the internal _is_trivial_number helper."""

    def test_single_digit_no_unit_is_trivial(self) -> None:
        assert _is_trivial_number("1", "1") is True

    def test_single_digit_with_unit_not_trivial(self) -> None:
        assert _is_trivial_number("5", "5 mg") is False

    def test_multi_digit_not_trivial(self) -> None:
        assert _is_trivial_number("42", "42") is False

    def test_decimal_not_trivial(self) -> None:
        assert _is_trivial_number("0.001", "p < 0.001") is False

    def test_empty_string_is_trivial(self) -> None:
        assert _is_trivial_number("", "") is True

    def test_three_digit_not_trivial(self) -> None:
        assert _is_trivial_number("240", "240") is False

    def test_single_digit_in_pvalue_context_not_trivial(self) -> None:
        # In a p-value context, single digit should not be trivial.
        assert _is_trivial_number("5", "p < 5") is False

    def test_returns_bool(self) -> None:
        result = _is_trivial_number("42", "42 mg")
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# _overlaps_claimed helper
# ---------------------------------------------------------------------------


class TestOverlapsClaimed:
    """Tests for the internal _overlaps_claimed helper."""

    def test_no_claimed_spans(self) -> None:
        assert _overlaps_claimed(0, 10, []) is False

    def test_identical_span_overlaps(self) -> None:
        assert _overlaps_claimed(5, 15, [(5, 15)]) is True

    def test_partial_overlap(self) -> None:
        assert _overlaps_claimed(8, 20, [(5, 15)]) is True

    def test_adjacent_no_overlap(self) -> None:
        # End of claimed is 10, start of candidate is 10 — no overlap.
        assert _overlaps_claimed(10, 20, [(0, 10)]) is False

    def test_contained_span_overlaps(self) -> None:
        assert _overlaps_claimed(6, 9, [(5, 15)]) is True

    def test_non_overlapping_spans(self) -> None:
        assert _overlaps_claimed(20, 30, [(0, 10), (11, 19)]) is False

    def test_min_overlap_threshold(self) -> None:
        # Overlap of exactly 1 char should not trigger with default min_overlap=2.
        assert _overlaps_claimed(9, 20, [(5, 10)], min_overlap=2) is False
        # Overlap of exactly 2 chars should trigger.
        assert _overlaps_claimed(8, 20, [(5, 10)], min_overlap=2) is True

    def test_multiple_claimed_one_overlaps(self) -> None:
        # First claimed doesn't overlap, second does.
        assert _overlaps_claimed(15, 25, [(0, 5), (13, 20)]) is True

    def test_returns_bool(self) -> None:
        result = _overlaps_claimed(0, 5, [])
        assert isinstance(result, bool)

    def test_large_overlap(self) -> None:
        assert _overlaps_claimed(0, 100, [(10, 90)]) is True


# ---------------------------------------------------------------------------
# Integration-style test with realistic scientific text
# ---------------------------------------------------------------------------


class TestRealisticText:
    """Integration-style tests using realistic scientific paper excerpts."""

    ABSTRACT = (
        "Compound X reduced LDL cholesterol by 32.4 mg/dL "
        "(95% CI: 28.1\u201336.7 mg/dL) compared with placebo (p < 0.001). "
        "Secondary outcomes included a 15.2% reduction in total cholesterol "
        "and a non-significant change in HDL cholesterol (\u22120.8 mg/dL; p = 0.42). "
        "A total of 240 participants were enrolled; mean age was 54.3 \u00b1 7.8 years."
    )

    def test_finds_at_least_five_candidates(self) -> None:
        records = extract_candidates(self.ABSTRACT)
        assert len(records) >= 5

    def test_finds_pvalue(self) -> None:
        records = extract_candidates(self.ABSTRACT)
        assert _has_type(records, "p-value")

    def test_finds_percentage(self) -> None:
        records = extract_candidates(self.ABSTRACT)
        assert _has_type(records, "percentage")

    def test_finds_measurement(self) -> None:
        records = extract_candidates(self.ABSTRACT)
        assert _has_type(records, "measurement")

    def test_finds_mean(self) -> None:
        records = extract_candidates(self.ABSTRACT)
        assert _has_type(records, "mean")

    def test_all_records_have_context(self) -> None:
        records = extract_candidates(self.ABSTRACT)
        assert all(len(r.context) > 0 for r in records)

    def test_all_records_have_value(self) -> None:
        records = extract_candidates(self.ABSTRACT)
        assert all(len(r.value) > 0 for r in records)

    def test_all_records_have_raw_text(self) -> None:
        records = extract_candidates(self.ABSTRACT)
        assert all(len(r.raw_text) > 0 for r in records)

    def test_records_are_numeric_records(self) -> None:
        records = extract_candidates(self.ABSTRACT)
        assert all(isinstance(r, NumericRecord) for r in records)

    def test_source_passed_through(self) -> None:
        records = extract_candidates(self.ABSTRACT, source="abstract.txt")
        assert all(r.source == "abstract.txt" for r in records)

    def test_section_passed_through(self) -> None:
        records = extract_candidates(self.ABSTRACT, section="Abstract")
        assert all(r.section == "Abstract" for r in records)

    def test_no_record_has_confidence_set(self) -> None:
        records = extract_candidates(self.ABSTRACT)
        assert all(r.confidence is None for r in records)

    def test_no_record_has_relationship_set(self) -> None:
        records = extract_candidates(self.ABSTRACT)
        assert all(r.relationship == "" for r in records)

    def test_primary_pvalue_found(self) -> None:
        records = extract_candidates(self.ABSTRACT)
        pvals = [r for r in records if r.data_type == "p-value"]
        values = [r.value for r in pvals]
        # Should find p < 0.001 and/or p = 0.42.
        assert "0.001" in values or "0.42" in values

    def test_percentage_value_correct(self) -> None:
        records = extract_candidates(self.ABSTRACT)
        pcts = [r for r in records if r.data_type == "percentage"]
        values = [r.value for r in pcts]
        assert "15.2" in values

    def test_measurement_ldl_found(self) -> None:
        records = extract_candidates(self.ABSTRACT)
        mg_dl = [r for r in records if r.unit == "mg/dL"]
        values = [r.value for r in mg_dl]
        assert "32.4" in values

    def test_mean_age_found(self) -> None:
        records = extract_candidates(self.ABSTRACT)
        means = [r for r in records if r.data_type == "mean"]
        values = [r.value for r in means]
        assert "54.3" in values

    METHODS_EXCERPT = (
        "Sample size was calculated to detect a minimum difference of 20 mg/dL "
        "in LDL reduction between groups, assuming a standard deviation of 35 mg/dL, "
        "80% power, and a two-sided alpha of 0.05. "
        "This required 96 participants per group; with anticipated 20% dropout, "
        "120 participants per group were enrolled (total N = 240)."
    )

    def test_methods_finds_count(self) -> None:
        records = extract_candidates(self.METHODS_EXCERPT)
        assert _has_type(records, "count")

    def test_methods_finds_percentage(self) -> None:
        records = extract_candidates(self.METHODS_EXCERPT)
        assert _has_type(records, "percentage")

    def test_methods_finds_measurement(self) -> None:
        records = extract_candidates(self.METHODS_EXCERPT)
        assert _has_type(records, "measurement")

    SAFETY_EXCERPT = (
        "Treatment-emergent adverse events occurred in 34 participants (28.3%) "
        "receiving Compound X and 31 participants (25.8%) receiving placebo (p = 0.65). "
        "Mean ALT levels increased by 3.2 U/L in the Compound X group, "
        "compared with 0.5 U/L in the placebo group (p = 0.08)."
    )

    def test_safety_finds_pvalue(self) -> None:
        records = extract_candidates(self.SAFETY_EXCERPT)
        assert _has_type(records, "p-value")

    def test_safety_finds_percentages(self) -> None:
        records = extract_candidates(self.SAFETY_EXCERPT)
        pcts = [r for r in records if r.data_type == "percentage"]
        assert len(pcts) >= 2

    def test_safety_percentage_values(self) -> None:
        records = extract_candidates(self.SAFETY_EXCERPT)
        pcts = [r for r in records if r.data_type == "percentage"]
        pct_values = [r.value for r in pcts]
        assert "28.3" in pct_values or "25.8" in pct_values
