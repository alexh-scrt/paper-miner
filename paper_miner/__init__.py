"""paper_miner: Extract numerical data from scientific PDF and HTML papers.

This package provides a programmatic API and CLI tool for ingesting scientific
documents and extracting structured numerical findings—measurements, percentages,
p-values, confidence intervals, and more—using regex-based pre-filtering combined
with LLM-assisted parsing.

Public API
----------
extract_from_pdf : Extract numeric records from a PDF file path.
extract_from_html : Extract numeric records from an HTML file path or string.
extract_from_text : Extract numeric records from a plain text string.

Example
-------
>>> from paper_miner import extract_from_text
>>> records = extract_from_text("The treatment reduced blood pressure by 12.5 mmHg (p=0.003).")
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
    "NumericRecord",
]

from paper_miner.models import NumericRecord


def extract_from_pdf(
    path: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "gpt-4o-mini",
    use_llm: bool = True,
) -> list[NumericRecord]:
    """Extract numeric records from a PDF file.

    Parameters
    ----------
    path:
        Filesystem path to the PDF file to ingest.
    api_key:
        OpenAI-compatible API key. If None, reads from the OPENAI_API_KEY
        environment variable.
    base_url:
        Optional custom base URL for an OpenAI-compatible API endpoint.
    model:
        Name of the LLM model to use for parsing. Defaults to ``gpt-4o-mini``.
    use_llm:
        When True (default), numeric candidates are enriched by the LLM parser.
        When False, only regex-extracted candidates are returned with partial data.

    Returns
    -------
    list[NumericRecord]
        A list of structured numeric records extracted from the document.

    Raises
    ------
    FileNotFoundError
        If the given path does not point to an existing file.
    ValueError
        If the file cannot be parsed as a PDF.
    """
    # Deferred imports to keep startup fast and avoid circular imports
    from paper_miner.ingest import ingest_pdf
    from paper_miner.extractor import extract_candidates
    from paper_miner.llm_parser import parse_candidates

    chunks = ingest_pdf(path)
    candidates = []
    for chunk in chunks:
        candidates.extend(extract_candidates(chunk))

    if not use_llm or not candidates:
        return candidates

    records = parse_candidates(candidates, api_key=api_key, base_url=base_url, model=model)
    return records


def extract_from_html(
    source: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "gpt-4o-mini",
    use_llm: bool = True,
    is_file: bool = True,
) -> list[NumericRecord]:
    """Extract numeric records from an HTML file or HTML string.

    Parameters
    ----------
    source:
        Either a filesystem path to an HTML file (when ``is_file=True``) or
        a raw HTML string (when ``is_file=False``).
    api_key:
        OpenAI-compatible API key. If None, reads from the OPENAI_API_KEY
        environment variable.
    base_url:
        Optional custom base URL for an OpenAI-compatible API endpoint.
    model:
        Name of the LLM model to use for parsing. Defaults to ``gpt-4o-mini``.
    use_llm:
        When True (default), numeric candidates are enriched by the LLM parser.
        When False, only regex-extracted candidates are returned with partial data.
    is_file:
        When True (default), ``source`` is treated as a file path. When False,
        ``source`` is treated as raw HTML content.

    Returns
    -------
    list[NumericRecord]
        A list of structured numeric records extracted from the document.

    Raises
    ------
    FileNotFoundError
        If ``is_file=True`` and the given path does not exist.
    """
    from paper_miner.ingest import ingest_html
    from paper_miner.extractor import extract_candidates
    from paper_miner.llm_parser import parse_candidates

    chunks = ingest_html(source, is_file=is_file)
    candidates = []
    for chunk in chunks:
        candidates.extend(extract_candidates(chunk))

    if not use_llm or not candidates:
        return candidates

    records = parse_candidates(candidates, api_key=api_key, base_url=base_url, model=model)
    return records


def extract_from_text(
    text: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "gpt-4o-mini",
    use_llm: bool = True,
) -> list[NumericRecord]:
    """Extract numeric records from a plain text string.

    Parameters
    ----------
    text:
        The plain text content to mine for numeric findings.
    api_key:
        OpenAI-compatible API key. If None, reads from the OPENAI_API_KEY
        environment variable.
    base_url:
        Optional custom base URL for an OpenAI-compatible API endpoint.
    model:
        Name of the LLM model to use for parsing. Defaults to ``gpt-4o-mini``.
    use_llm:
        When True (default), numeric candidates are enriched by the LLM parser.
        When False, only regex-extracted candidates are returned with partial data.

    Returns
    -------
    list[NumericRecord]
        A list of structured numeric records extracted from the text.
    """
    from paper_miner.extractor import extract_candidates
    from paper_miner.llm_parser import parse_candidates

    candidates = extract_candidates(text)

    if not use_llm or not candidates:
        return candidates

    records = parse_candidates(candidates, api_key=api_key, base_url=base_url, model=model)
    return records
