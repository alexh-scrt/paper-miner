"""LLM-assisted parser for paper_miner.

Sends batches of numeric candidate contexts to an OpenAI-compatible API and
parses the structured JSON responses into enriched NumericRecord objects.
The LLM is asked to verify and enrich fields that the regex pre-filter set
heuristically (unit, data_type, relationship) and to assign a confidence
score.

Functions
---------
parse_candidates : Enrich a list of partial NumericRecord objects via LLM.
build_prompt     : Build the system + user prompt for a batch of candidates.
parse_llm_response : Parse a raw LLM JSON response into record field dicts.
_make_client     : Construct an OpenAI client from key / base_url.
_call_llm        : Send a single prompt to the LLM and return the text response.
_merge_record    : Merge LLM-returned fields into an existing NumericRecord.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from paper_miner.models import NumericRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of candidates sent to the LLM in a single API call.
# Keeping batches small reduces the chance of the model truncating output.
DEFAULT_BATCH_SIZE: int = 10

# Number of seconds to wait between retries on transient API errors.
_RETRY_DELAY: float = 2.0

# Maximum number of retry attempts per batch.
_MAX_RETRIES: int = 3

# System prompt instructing the LLM on its role and output format.
_SYSTEM_PROMPT: str = """\
You are a scientific data extraction assistant specialised in analysing
numerical findings from scientific papers.

You will receive a JSON array of numeric candidate objects. Each object has:
  - "id":       integer index (preserve exactly in your response)
  - "raw_text": the exact substring matched in the source document
  - "context":  the surrounding sentence
  - "data_type_hint": a heuristic classification (may be wrong)
  - "unit_hint":      a heuristic unit (may be wrong or "none")
  - "value":    the primary numeric value string

For EACH candidate you must return a JSON object with these fields:
  - "id":          (integer) same id as the input
  - "value":       (string)  the primary numeric value, exactly as it appears
  - "unit":        (string)  the correct SI or domain unit, or "none" if unitless
  - "data_type":   (string)  one of: measurement, percentage, p-value,
                             confidence_interval, mean, median,
                             standard_deviation, count, ratio, other
  - "relationship":(string)  brief phrase describing what the number measures
                             (e.g. "LDL cholesterol reduction from baseline")
  - "confidence":  (float)   0.0-1.0, how confident you are in this extraction

Output ONLY a valid JSON array of these objects — no markdown, no commentary.
If a candidate is not a meaningful scientific measurement, set confidence to 0.1
and data_type to "other".
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_client(
    api_key: Optional[str],
    base_url: Optional[str],
) -> Any:
    """Construct and return an OpenAI client instance.

    Parameters
    ----------
    api_key:
        API key string. When ``None``, the client reads the
        ``OPENAI_API_KEY`` environment variable.
    base_url:
        Optional custom endpoint URL for OpenAI-compatible services.

    Returns
    -------
    openai.OpenAI
        A configured OpenAI client.

    Raises
    ------
    ImportError
        If the ``openai`` package is not installed.
    ValueError
        If no API key is available (neither argument nor env var).
    """
    try:
        import openai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "openai is required for LLM-assisted parsing. "
            "Install it with: pip install openai"
        ) from exc

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not resolved_key:
        raise ValueError(
            "No API key provided. Pass api_key= or set the OPENAI_API_KEY "
            "environment variable."
        )

    kwargs: Dict[str, Any] = {"api_key": resolved_key}
    if base_url:
        kwargs["base_url"] = base_url

    return openai.OpenAI(**kwargs)


def build_prompt(candidates: Sequence[NumericRecord]) -> str:
    """Build the user-facing prompt payload for a batch of candidates.

    Serialises *candidates* as a JSON array of lightweight dicts that contain
    only the fields the LLM needs to perform enrichment, avoiding unnecessary
    token usage.

    Parameters
    ----------
    candidates:
        A sequence of (potentially partial) ``NumericRecord`` objects produced
        by the regex extractor.

    Returns
    -------
    str
        A JSON-formatted string ready to be sent as the user message.
    """
    payload = [
        {
            "id": idx,
            "raw_text": record.raw_text,
            "context": record.context,
            "data_type_hint": record.data_type,
            "unit_hint": record.unit,
            "value": record.value,
        }
        for idx, record in enumerate(candidates)
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_llm_response(
    response_text: str,
    num_candidates: int,
) -> List[Dict[str, Any]]:
    """Parse the LLM's raw text response into a list of field dictionaries.

    Attempts strict JSON parsing first. If that fails, tries to extract a JSON
    array from the response using a regex (handles cases where the model wraps
    its output in markdown fences or adds commentary).

    Parameters
    ----------
    response_text:
        The raw string returned by the LLM.
    num_candidates:
        The number of candidates that were sent; used to validate the response
        length and generate fallback placeholders for missing entries.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, one per candidate, with enriched field values.
        Missing or unparseable entries are replaced with safe fallback dicts.
    """
    parsed: Optional[List[Dict[str, Any]]] = None

    # Attempt 1: strict JSON parse.
    stripped = response_text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract JSON array from possible markdown fences or prose.
    if parsed is None:
        array_match = re.search(r"(\[.*\])", stripped, re.DOTALL)
        if array_match:
            try:
                parsed = json.loads(array_match.group(1))
            except json.JSONDecodeError:
                pass

    # If still None, log a warning and return fallback dicts.
    if parsed is None:
        logger.warning(
            "Failed to parse LLM response as JSON. "
            "Using fallback for all %d candidates. "
            "Raw response (first 500 chars): %s",
            num_candidates,
            response_text[:500],
        )
        return [_fallback_dict(i) for i in range(num_candidates)]

    if not isinstance(parsed, list):
        logger.warning(
            "LLM response JSON is not a list (got %s). Using fallback.",
            type(parsed).__name__,
        )
        return [_fallback_dict(i) for i in range(num_candidates)]

    # Build an id -> dict lookup for safe indexed retrieval.
    by_id: Dict[int, Dict[str, Any]] = {}
    for item in parsed:
        if isinstance(item, dict) and "id" in item:
            try:
                by_id[int(item["id"])] = item
            except (ValueError, TypeError):
                pass

    result: List[Dict[str, Any]] = []
    for idx in range(num_candidates):
        if idx in by_id:
            result.append(by_id[idx])
        else:
            logger.debug("LLM did not return entry for candidate id=%d; using fallback.", idx)
            result.append(_fallback_dict(idx))

    return result


def _fallback_dict(idx: int) -> Dict[str, Any]:
    """Return a minimal fallback dict for candidate *idx* when the LLM fails.

    Parameters
    ----------
    idx:
        The zero-based index of the candidate.

    Returns
    -------
    Dict[str, Any]
        A dictionary with default / unknown field values.
    """
    return {
        "id": idx,
        "value": "",
        "unit": "none",
        "data_type": "other",
        "relationship": "",
        "confidence": 0.0,
    }


def _merge_record(
    original: NumericRecord,
    llm_fields: Dict[str, Any],
) -> NumericRecord:
    """Merge LLM-returned fields into a copy of *original*.

    Fields produced by the regex extractor that the LLM has improved
    (``value``, ``unit``, ``data_type``, ``relationship``, ``confidence``) are
    overwritten. Fields that only the original record knows about (``context``,
    ``raw_text``, ``section``, ``source``) are preserved.

    Parameters
    ----------
    original:
        The partial ``NumericRecord`` from the regex extractor.
    llm_fields:
        Dictionary of enriched fields from the LLM response.

    Returns
    -------
    NumericRecord
        A new ``NumericRecord`` with merged data.
    """
    # Safely coerce confidence to float in [0, 1].
    raw_confidence = llm_fields.get("confidence")
    confidence: Optional[float] = None
    if raw_confidence is not None:
        try:
            confidence = float(raw_confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = None

    # Use LLM value only if non-empty; otherwise keep original.
    value = str(llm_fields.get("value", "")).strip() or original.value

    # Use LLM unit if provided and non-empty.
    unit = str(llm_fields.get("unit", "")).strip() or original.unit
    if not unit:
        unit = "none"

    # Use LLM data_type if provided.
    data_type = str(llm_fields.get("data_type", "")).strip() or original.data_type

    # Use LLM relationship.
    relationship = str(llm_fields.get("relationship", "")).strip()

    return NumericRecord(
        value=value,
        unit=unit,
        data_type=data_type,
        context=original.context,
        relationship=relationship,
        raw_text=original.raw_text,
        section=original.section,
        confidence=confidence,
        source=original.source,
    )


def _call_llm(
    client: Any,
    model: str,
    user_message: str,
    max_retries: int = _MAX_RETRIES,
    retry_delay: float = _RETRY_DELAY,
) -> str:
    """Send a prompt to the LLM and return the response text.

    Retries on transient errors (rate limits, server errors) up to
    *max_retries* times with exponential-ish back-off.

    Parameters
    ----------
    client:
        An ``openai.OpenAI`` client instance.
    model:
        Model identifier string (e.g. ``"gpt-4o-mini"``).
    user_message:
        The user-role message content to send.
    max_retries:
        Maximum number of retry attempts.
    retry_delay:
        Base delay in seconds between retries (doubled each attempt).

    Returns
    -------
    str
        The text content of the model's first response choice.

    Raises
    ------
    RuntimeError
        If all retry attempts are exhausted.
    """
    last_exc: Optional[Exception] = None
    delay = retry_delay

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,  # deterministic output for structured extraction
                # Request JSON mode where supported to reduce parse failures.
                response_format={"type": "json_object"} if _model_supports_json_mode(model) else None,
            )
            content = response.choices[0].message.content or ""
            return content
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "LLM API call failed (attempt %d/%d): %s",
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2

    raise RuntimeError(
        f"LLM API call failed after {max_retries} attempts. "
        f"Last error: {last_exc}"
    ) from last_exc


def _model_supports_json_mode(model: str) -> bool:
    """Heuristically determine whether *model* supports the JSON response format.

    OpenAI's JSON mode is supported by ``gpt-4o``, ``gpt-4-turbo``,
    ``gpt-3.5-turbo`` (1106+), and many compatible providers. This function
    applies a conservative allowlist; unknown models default to ``False`` to
    avoid API errors.

    Parameters
    ----------
    model:
        Model identifier string.

    Returns
    -------
    bool
        ``True`` when JSON mode is believed to be supported.
    """
    json_mode_prefixes = (
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125",
        "gpt-4-1106",
        "gpt-4-0125",
    )
    lower = model.lower()
    return any(lower.startswith(prefix.lower()) for prefix in json_mode_prefixes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_candidates(
    candidates: List[NumericRecord],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = _MAX_RETRIES,
    retry_delay: float = _RETRY_DELAY,
) -> List[NumericRecord]:
    """Enrich a list of numeric candidate records using an LLM.

    Candidates are sent to the LLM in batches of *batch_size*. For each
    batch the function builds a structured prompt, calls the API, parses
    the JSON response, and merges the enriched fields back into the original
    records. Any batch that fails after all retries is returned with its
    original heuristic values and ``confidence=None``.

    Parameters
    ----------
    candidates:
        List of partial ``NumericRecord`` objects from the regex extractor.
        Must be non-empty.
    api_key:
        OpenAI-compatible API key. Falls back to the ``OPENAI_API_KEY``
        environment variable when ``None``.
    base_url:
        Optional custom endpoint URL (e.g. for a local LLM server or a
        non-OpenAI provider).
    model:
        Model identifier. Defaults to ``"gpt-4o-mini"``.
    batch_size:
        Number of candidates per API call. Defaults to
        ``DEFAULT_BATCH_SIZE`` (10).
    max_retries:
        Maximum number of retry attempts per batch.
    retry_delay:
        Base delay in seconds between retries.

    Returns
    -------
    List[NumericRecord]
        Enriched records in the same order as *candidates*. Records for
        which LLM enrichment failed retain their original heuristic values.

    Raises
    ------
    ValueError
        If no API key is available.
    ImportError
        If the ``openai`` package is not installed.

    Examples
    --------
    >>> from paper_miner.extractor import extract_candidates
    >>> from paper_miner.llm_parser import parse_candidates
    >>> text = "LDL reduced by 32.4 mg/dL (p < 0.001)."
    >>> candidates = extract_candidates(text)
    >>> # In production, api_key is read from OPENAI_API_KEY env var:
    >>> # records = parse_candidates(candidates)
    """
    if not candidates:
        return []

    client = _make_client(api_key=api_key, base_url=base_url)
    enriched: List[NumericRecord] = []

    total = len(candidates)
    for batch_start in range(0, total, batch_size):
        batch = candidates[batch_start : batch_start + batch_size]
        logger.debug(
            "Processing batch %d-%d of %d candidates.",
            batch_start + 1,
            batch_start + len(batch),
            total,
        )

        user_message = build_prompt(batch)

        try:
            response_text = _call_llm(
                client=client,
                model=model,
                user_message=user_message,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
        except RuntimeError as exc:
            logger.error(
                "Batch %d-%d failed after all retries: %s. "
                "Returning candidates with original heuristic values.",
                batch_start + 1,
                batch_start + len(batch),
                exc,
            )
            enriched.extend(batch)
            continue

        llm_fields_list = parse_llm_response(response_text, num_candidates=len(batch))

        for original, llm_fields in zip(batch, llm_fields_list):
            merged = _merge_record(original, llm_fields)
            enriched.append(merged)

    return enriched
