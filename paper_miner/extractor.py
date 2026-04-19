"""Regex-based numeric candidate extractor for paper_miner.

This module provides the core pre-filter that scans plain text chunks for
numeric candidates—measurements, percentages, p-values, confidence intervals,
statistics, and more—and captures the surrounding sentence context for each
candidate. The extracted candidates are represented as lightweight
``NumericRecord`` objects with partial data (no LLM enrichment), ready to be
passed to the LLM parser or returned directly when ``use_llm=False``.

Functions
---------
extract_candidates : Scan a text chunk and return a list of NumericRecord objects.
_split_sentences   : Split text into individual sentences.
_find_sentence     : Find the sentence that contains a given character offset.
_classify_raw      : Heuristically classify a raw match string into a data type.
_extract_unit      : Attempt to extract a unit from the text immediately following
                     a numeric match.
_extract_value     : Extract the primary numeric value from a raw match string.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from paper_miner.models import NumericRecord


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Number component: integers, decimals, negative numbers, scientific notation.
_NUM = r"[-−]?\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?"

# SI and common scientific units (case-sensitive grouping handled at match time).
_UNIT_PATTERN = (
    r"(?:"
    r"kg/m[²2]|mg/dL|mg/L|mmol/L|μmol/L|nmol/L|pmol/L"
    r"|mL/min/1\.73\s*m[²2]|mL/min|L/min"
    r"|mmHg|cmHg|kPa|Pa"
    r"|mg|μg|ng|pg|g|kg"
    r"|mL|μL|L|dL|cL"
    r"|mm|cm|m|km|μm|nm"
    r"|s|ms|min|h|hr|hrs|hours|days|weeks|months|years"
    r"|°C|°F|K"
    r"|U/L|IU/L|IU/mL|nmol/mL|μg/mL|ng/mL|pg/mL"
    r"|cells/μL|copies/mL"
    r"|bpm|beats/min"
    r"|kcal|kJ|cal"
    r"|mol|mmol|μmol|nmol"
    r"|M|mM|μM|nM|pM"
    r"|rpm"
    r"|W|kW|mW"
    r"|Hz|kHz|MHz|GHz"
    r"|N|kN"
    r"|J|kJ|MJ"
    r"|Ω|kΩ|MΩ"
    r"|V|mV|μV"
    r"|A|mA|μA"
    r"|T|mT|μT"
    r"|ULN|× ULN|xULN"
    r")"
)

# Percentage: number followed by % or 'percent'.
_PATTERN_PERCENTAGE = re.compile(
    r"(?P<raw>(?P<value>" + _NUM + r")\s*%(?:\s*(?:of|per|reduction|increase|change|CI|confidence))?)",
    re.IGNORECASE,
)

# p-value: p=, p<, p>, p≤, p≥ followed by a number.
_PATTERN_PVALUE = re.compile(
    r"(?P<raw>\bp\s*[=<>≤≥]\s*(?P<value>" + _NUM + r"))",
    re.IGNORECASE,
)

# Confidence interval: 95% CI, 90% CI, CI:, followed by range.
_PATTERN_CI = re.compile(
    r"(?P<raw>"
    r"(?:\d{1,3}%\s*)?CI[:\s]+"
    r"(?P<value>" + _NUM + r")"
    r"(?:\s*[–\-—−to]+\s*" + _NUM + r")"
    r"(?:\s*" + _UNIT_PATTERN + r")?"
    r")",
    re.IGNORECASE,
)

# Measurement with explicit unit.
_PATTERN_MEASUREMENT = re.compile(
    r"(?P<raw>(?P<value>" + _NUM + r")\s*(?P<unit>" + _UNIT_PATTERN + r"))",
    re.IGNORECASE,
)

# Mean ± SD / SEM pattern.
_PATTERN_MEAN_SD = re.compile(
    r"(?P<raw>"
    r"(?:mean\s+)?(?P<value>" + _NUM + r")"
    r"\s*[±\+\-]\s*"
    r"(?P<sd>" + _NUM + r")"
    r"(?:\s*(?P<unit>" + _UNIT_PATTERN + r"))?"
    r")",
    re.IGNORECASE,
)

# Ratio / odds ratio / hazard ratio: e.g. OR = 1.23, HR: 0.87, RR 1.5.
_PATTERN_RATIO = re.compile(
    r"(?P<raw>(?:OR|HR|RR|IRR|NNT|NNH)\s*[=:]?\s*(?P<value>" + _NUM + r"))",
    re.IGNORECASE,
)

# Count / sample size: n = 240, N = 120.
_PATTERN_COUNT = re.compile(
    r"(?P<raw>[Nn]\s*=\s*(?P<value>\d+))",
)

# Generic standalone number with optional unit (catch-all for numbers >= 2 digits
# or decimal numbers not captured by the above patterns).
_PATTERN_GENERIC = re.compile(
    r"(?P<raw>(?P<value>" + _NUM + r")\s*(?P<unit>" + _UNIT_PATTERN + r")?)",
    re.IGNORECASE,
)

# Ordered list of (pattern, data_type_hint) tuples; more specific patterns first.
_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (_PATTERN_PVALUE, "p-value"),
    (_PATTERN_CI, "confidence_interval"),
    (_PATTERN_MEAN_SD, "mean"),
    (_PATTERN_RATIO, "ratio"),
    (_PATTERN_COUNT, "count"),
    (_PATTERN_PERCENTAGE, "percentage"),
    (_PATTERN_MEASUREMENT, "measurement"),
]

# Sentence-splitting pattern: split on period/!/? followed by whitespace or end.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Minimum number of digits in a match to be considered a candidate.
# Prevents matching isolated single-digit ordinals like "1st", "2nd", etc.
_MIN_DIGIT_COUNT = 1

# Maximum characters of surrounding context on each side of the numeric match.
_CONTEXT_WINDOW = 300


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> List[Tuple[str, int]]:
    """Split *text* into sentences and return (sentence, start_offset) pairs.

    Parameters
    ----------
    text:
        The full text to split.

    Returns
    -------
    List[Tuple[str, int]]
        Each tuple contains the sentence text and its starting character offset
        within *text*.
    """
    sentences: List[Tuple[str, int]] = []
    start = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        sentence = text[start : match.start() + 1].strip()
        if sentence:
            sentences.append((sentence, start))
        start = match.end()
    # Append any remaining text as the final sentence.
    tail = text[start:].strip()
    if tail:
        sentences.append((tail, start))
    return sentences


def _find_sentence(text: str, offset: int) -> str:
    """Return the sentence in *text* that contains character *offset*.

    Falls back to a windowed context extract if sentence splitting yields no
    useful result.

    Parameters
    ----------
    text:
        The source text.
    offset:
        Character offset of the numeric match within *text*.

    Returns
    -------
    str
        The sentence containing *offset*, or a context window around it.
    """
    sentences = _split_sentences(text)
    best_sentence = ""
    for sentence, start in sentences:
        end = start + len(sentence)
        if start <= offset <= end:
            best_sentence = sentence
            break

    if not best_sentence:
        # Fallback: extract a fixed-size window around the offset.
        ctx_start = max(0, offset - _CONTEXT_WINDOW)
        ctx_end = min(len(text), offset + _CONTEXT_WINDOW)
        best_sentence = text[ctx_start:ctx_end].strip()

    return best_sentence


def _extract_value(raw: str, data_type: str) -> str:
    """Extract the primary numeric value string from *raw*.

    Parameters
    ----------
    raw:
        The full raw match string (e.g. ``"p < 0.001"``, ``"32.4 mg/dL"``).
    data_type:
        The heuristic data type, used to guide extraction.

    Returns
    -------
    str
        The primary numeric value as a string (e.g. ``"0.001"``, ``"32.4"``).
        Returns the raw string if no numeric component is found.
    """
    # Find all number-like substrings.
    numbers = re.findall(r"[-−]?\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?", raw)
    if not numbers:
        return raw
    # For p-values and ratios, the first number (after operator) is the value.
    # For confidence intervals, the first number is the lower bound.
    # For mean±SD, the first number is the mean.
    return numbers[0]


def _extract_unit(raw: str, data_type: str, after_text: str = "") -> str:
    """Attempt to extract a unit from *raw* or *after_text*.

    Parameters
    ----------
    raw:
        The full raw match string.
    data_type:
        The heuristic data type.
    after_text:
        A short string immediately following the match in the original text,
        used as a fallback unit source.

    Returns
    -------
    str
        The detected unit string, or ``"%"`` for percentages, ``"none"`` when
        no unit is detectable, or a descriptive label for statistical types.
    """
    if data_type == "percentage":
        return "%"
    if data_type == "p-value":
        return "none"
    if data_type == "confidence_interval":
        # Try to find a unit in raw after the numbers.
        unit_match = re.search(_UNIT_PATTERN, raw, re.IGNORECASE)
        if unit_match:
            return unit_match.group(0).strip()
        unit_match = re.search(_UNIT_PATTERN, after_text, re.IGNORECASE)
        if unit_match:
            return unit_match.group(0).strip()
        return "none"
    if data_type in ("ratio", "count"):
        return "none"

    # For measurement and mean, search raw for a unit.
    unit_match = re.search(_UNIT_PATTERN, raw, re.IGNORECASE)
    if unit_match:
        return unit_match.group(0).strip()

    # Try immediately following context.
    if after_text:
        unit_match = re.search(r"^\s*" + _UNIT_PATTERN, after_text, re.IGNORECASE)
        if unit_match:
            return unit_match.group(0).strip()

    return "none"


def _classify_raw(raw: str, pattern_hint: str) -> str:
    """Return a data-type classification for the raw match string.

    Uses *pattern_hint* from the matched pattern as the primary classifier,
    with a few additional heuristic overrides.

    Parameters
    ----------
    raw:
        The full raw match string.
    pattern_hint:
        The data_type string associated with the pattern that produced the match.

    Returns
    -------
    str
        One of: ``"p-value"``, ``"confidence_interval"``, ``"mean"``,
        ``"percentage"``, ``"measurement"``, ``"count"``, ``"ratio"``,
        ``"standard_deviation"``, ``"other"``.
    """
    if pattern_hint == "p-value":
        return "p-value"
    if pattern_hint == "confidence_interval":
        return "confidence_interval"
    if pattern_hint == "mean":
        # Distinguish SD-only patterns.
        if re.search(r"\bSD\b|\bSD\s*[=:]|standard\s+deviation", raw, re.IGNORECASE):
            return "standard_deviation"
        return "mean"
    if pattern_hint == "percentage":
        return "percentage"
    if pattern_hint == "measurement":
        return "measurement"
    if pattern_hint == "count":
        return "count"
    if pattern_hint == "ratio":
        return "ratio"
    return "other"


def _is_trivial_number(value_str: str, raw: str) -> bool:
    """Return True if the numeric value is too trivial to report.

    Filters out very short or obviously non-scientific numbers like page
    numbers, reference superscripts (e.g. "1", "2"), single-digit counts
    without any unit, and similar noise.

    Parameters
    ----------
    value_str:
        The primary numeric value string.
    raw:
        The full raw match string.

    Returns
    -------
    bool
        ``True`` when the candidate should be discarded.
    """
    digits_only = re.sub(r"[^0-9]", "", value_str)
    # Require at least 1 digit.
    if len(digits_only) < _MIN_DIGIT_COUNT:
        return True
    # Single digit integers without any unit or special context are likely noise.
    if len(digits_only) == 1 and not re.search(_UNIT_PATTERN, raw, re.IGNORECASE):
        # Allow single-digit values in p-value / CI contexts.
        if not re.search(r"\bp\s*[=<>≤≥]|\bCI\b|\bOR\b|\bHR\b|\bRR\b|n\s*=", raw, re.IGNORECASE):
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_candidates(
    text: str,
    source: Optional[str] = None,
    section: Optional[str] = None,
) -> List[NumericRecord]:
    """Scan *text* for numeric candidates and return partial NumericRecord objects.

    This function applies a series of compiled regex patterns (in priority order)
    to find numeric expressions—measurements, percentages, p-values, confidence
    intervals, means with standard deviations, ratios, and counts—within the
    input text. For each unique match it captures the surrounding sentence as
    context and constructs a ``NumericRecord`` with heuristic field values.
    These records are intended to be enriched by the LLM parser; the ``confidence``
    field is left as ``None`` and ``relationship`` is set to an empty string.

    Duplicate matches at the same character offset across multiple patterns are
    deduplicated (highest-priority / more-specific pattern wins).

    Parameters
    ----------
    text:
        The plain text chunk to scan for numeric candidates.
    source:
        Optional identifier for the originating document (filename, URL, etc.).
        Passed through to each ``NumericRecord``.
    section:
        Optional document section label (e.g. ``"Results"``, ``"Methods"``).
        Passed through to each ``NumericRecord``.

    Returns
    -------
    List[NumericRecord]
        Ordered list of numeric candidate records found in *text*. May be empty
        if no numeric patterns are found or all candidates are filtered as
        trivial.

    Examples
    --------
    >>> from paper_miner.extractor import extract_candidates
    >>> records = extract_candidates(
    ...     "The treatment reduced blood pressure by 12.5 mmHg (p=0.003)."
    ... )
    >>> len(records)
    2
    >>> records[0].data_type
    'p-value'
    >>> records[1].data_type
    'measurement'
    """
    if not text or not text.strip():
        return []

    records: List[NumericRecord] = []
    # Track character spans already claimed by a higher-priority pattern so
    # that lower-priority patterns do not produce overlapping duplicates.
    claimed_spans: List[Tuple[int, int]] = []

    for pattern, hint in _PATTERNS:
        for match in pattern.finditer(text):
            span_start, span_end = match.span()

            # Skip if this span substantially overlaps a previously claimed span.
            if _overlaps_claimed(span_start, span_end, claimed_spans):
                continue

            raw = match.group("raw") if "raw" in pattern.groupindex else match.group(0)
            raw = raw.strip()

            # Extract primary value.
            value = _extract_value(raw, hint)

            if _is_trivial_number(value, raw):
                continue

            # Gather text after the match for unit fallback.
            after_text = text[span_end : span_end + 30]

            unit = _extract_unit(raw, hint, after_text)
            data_type = _classify_raw(raw, hint)
            context = _find_sentence(text, span_start)

            record = NumericRecord(
                value=value,
                unit=unit,
                data_type=data_type,
                context=context,
                relationship="",  # To be filled by LLM parser.
                raw_text=raw,
                section=section,
                confidence=None,  # To be filled by LLM parser.
                source=source,
            )
            records.append(record)
            claimed_spans.append((span_start, span_end))

    # Sort records by their position in the text (approximate, via raw_text search).
    records.sort(key=lambda r: text.find(r.raw_text) if r.raw_text in text else 0)

    return records


def _overlaps_claimed(
    start: int,
    end: int,
    claimed: List[Tuple[int, int]],
    min_overlap: int = 2,
) -> bool:
    """Return True if [start, end) overlaps any span in *claimed* by at least *min_overlap* chars.

    Parameters
    ----------
    start:
        Start character offset of the candidate span.
    end:
        End character offset of the candidate span.
    claimed:
        List of already-claimed (start, end) spans.
    min_overlap:
        Minimum number of overlapping characters to consider a conflict.

    Returns
    -------
    bool
        ``True`` when a conflicting overlap exists.
    """
    for cs, ce in claimed:
        overlap = min(end, ce) - max(start, cs)
        if overlap >= min_overlap:
            return True
    return False
