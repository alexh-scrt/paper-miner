"""paper_miner: Extract numerical data from scientific PDF and HTML papers.

This package provides a programmatic API and CLI tool for ingesting scientific
documents and extracting structured numerical findings—measurements, percentages,
p-values, confidence intervals, and more—using regex-based pre-filtering combined
with LLM-assisted parsing.

Public API
----------
extract_from_pdf   : Extract numeric records from a PDF file path.
extract_from_html  : Extract numeric records from an HTML file path or string.
extract_from_text  : Extract numeric records from a plain text string.
export_records     : Serialize a list of NumericRecord objects to JSON or CSV.
NumericRecord      : The core data model for a single extracted numeric finding.

Example
-------
>>> from paper_miner import extract_from_text
>>> records = extract_from_text(
...     "The treatment reduced blood pressure by 12.5 mmHg (p=0.003).",
...     use_llm=False,
... )
>>> for record in records:
...     print(record.value, record.unit, record.data_type)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "paper_miner contributors"
__all__ = [
    "extract_from_pdf",
    "extract_from_html",
    "extract_from_text",
    "export_records",
    "NumericRecord",
]

from paper_miner.models import NumericRecord
from paper_miner.exporter import export_records


def extract_from_pdf(
    path: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "gpt-4o-mini",
    use_llm: bool = True,
    source: str | None = None,
) -> list[NumericRecord]:
    """Extract numeric records from a PDF file.

    Ingests the PDF at *path* page-by-page using *pdfplumber*, applies the
    regex-based pre-filter to identify numeric candidates, and optionally
    enriches them via an OpenAI-compatible LLM.

    Parameters
    ----------
    path:
        Filesystem path to the PDF file to ingest.
    api_key:
        OpenAI-compatible API key.  If ``None``, reads from the
        ``OPENAI_API_KEY`` environment variable.
    base_url:
        Optional custom base URL for an OpenAI-compatible API endpoint
        (e.g. a local LLM server).
    model:
        Name of the LLM model to use for parsing.  Defaults to
        ``"gpt-4o-mini"``.
    use_llm:
        When ``True`` (default), numeric candidates are enriched by the LLM
        parser.  When ``False``, only regex-extracted candidates are returned
        with heuristic field values and ``confidence=None``.
    source:
        Optional label attached to every returned record as
        :attr:`~paper_miner.models.NumericRecord.source`.  Defaults to
        *path* when ``None``.

    Returns
    -------
    list[NumericRecord]
        A list of structured numeric records extracted from the document,
        in document order.

    Raises
    ------
    FileNotFoundError
        If *path* does not point to an existing file.
    ValueError
        If the file cannot be parsed as a PDF.

    Examples
    --------
    >>> from paper_miner import extract_from_pdf
    >>> records = extract_from_pdf("study.pdf", use_llm=False)
    >>> print(len(records))
    """
    # Deferred imports to keep startup fast and avoid circular imports.
    from paper_miner.ingest import ingest_pdf
    from paper_miner.extractor import extract_candidates
    from paper_miner.llm_parser import parse_candidates

    effective_source = source if source is not None else path

    chunks = ingest_pdf(path)
    candidates: list[NumericRecord] = []
    for chunk in chunks:
        chunk_candidates = extract_candidates(chunk, source=effective_source)
        candidates.extend(chunk_candidates)

    if not use_llm or not candidates:
        return candidates

    records = parse_candidates(
        candidates,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    return records


def extract_from_html(
    source: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "gpt-4o-mini",
    use_llm: bool = True,
    is_file: bool = True,
    source_label: str | None = None,
) -> list[NumericRecord]:
    """Extract numeric records from an HTML file or HTML string.

    Strips boilerplate HTML (scripts, navigation, footers, etc.) via
    *BeautifulSoup4*, chunks the remaining visible text, applies the regex
    pre-filter, and optionally enriches candidates with an LLM.

    Parameters
    ----------
    source:
        Either a filesystem path to an HTML file (when ``is_file=True``) or
        a raw HTML string (when ``is_file=False``).
    api_key:
        OpenAI-compatible API key.  If ``None``, reads from the
        ``OPENAI_API_KEY`` environment variable.
    base_url:
        Optional custom base URL for an OpenAI-compatible API endpoint.
    model:
        Name of the LLM model to use for parsing.  Defaults to
        ``"gpt-4o-mini"``.
    use_llm:
        When ``True`` (default), numeric candidates are enriched by the LLM
        parser.  When ``False``, only regex-extracted candidates are returned.
    is_file:
        When ``True`` (default), *source* is treated as a file path.  When
        ``False``, *source* is treated as raw HTML content.
    source_label:
        Optional label attached to every returned record as
        :attr:`~paper_miner.models.NumericRecord.source`.  When ``None``
        and ``is_file=True``, defaults to *source* (the file path).  When
        ``None`` and ``is_file=False``, defaults to ``"<html string>"``.

    Returns
    -------
    list[NumericRecord]
        A list of structured numeric records extracted from the document.

    Raises
    ------
    FileNotFoundError
        If ``is_file=True`` and the given path does not exist.

    Examples
    --------
    >>> from paper_miner import extract_from_html
    >>> records = extract_from_html("paper.html", use_llm=False)
    >>> print(len(records))
    """
    from paper_miner.ingest import ingest_html
    from paper_miner.extractor import extract_candidates
    from paper_miner.llm_parser import parse_candidates

    if source_label is None:
        effective_source = source if is_file else "<html string>"
    else:
        effective_source = source_label

    chunks = ingest_html(source, is_file=is_file)
    candidates: list[NumericRecord] = []
    for chunk in chunks:
        chunk_candidates = extract_candidates(chunk, source=effective_source)
        candidates.extend(chunk_candidates)

    if not use_llm or not candidates:
        return candidates

    records = parse_candidates(
        candidates,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    return records


def extract_from_text(
    text: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "gpt-4o-mini",
    use_llm: bool = True,
    source: str | None = None,
    section: str | None = None,
) -> list[NumericRecord]:
    """Extract numeric records from a plain text string.

    Applies the regex pre-filter directly to *text* (no ingestion step needed)
    and optionally enriches the resulting candidates with an LLM.

    Parameters
    ----------
    text:
        The plain text content to mine for numeric findings.
    api_key:
        OpenAI-compatible API key.  If ``None``, reads from the
        ``OPENAI_API_KEY`` environment variable.
    base_url:
        Optional custom base URL for an OpenAI-compatible API endpoint.
    model:
        Name of the LLM model to use for parsing.  Defaults to
        ``"gpt-4o-mini"``.
    use_llm:
        When ``True`` (default), numeric candidates are enriched by the LLM
        parser.  When ``False``, only regex-extracted candidates are returned
        with heuristic field values and ``confidence=None``.
    source:
        Optional label attached to every returned record as
        :attr:`~paper_miner.models.NumericRecord.source`.  ``None`` by
        default.
    section:
        Optional document section label (e.g. ``"Abstract"``,
        ``"Results"``) propagated to every record.  ``None`` by default.

    Returns
    -------
    list[NumericRecord]
        A list of structured numeric records extracted from *text*.

    Examples
    --------
    >>> from paper_miner import extract_from_text
    >>> records = extract_from_text(
    ...     "The treatment reduced blood pressure by 12.5 mmHg (p=0.003).",
    ...     use_llm=False,
    ... )
    >>> records[0].data_type
    'p-value'
    """
    from paper_miner.extractor import extract_candidates
    from paper_miner.llm_parser import parse_candidates

    candidates = extract_candidates(text, source=source, section=section)

    if not use_llm or not candidates:
        return candidates

    records = parse_candidates(
        candidates,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    return records
