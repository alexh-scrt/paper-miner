"""Ingestion layer for paper_miner.

Provides functions to extract and chunk plain text from PDF files (via
pdfplumber) and HTML documents (via BeautifulSoup4). The resulting text
chunks are passed downstream to the regex-based extractor and LLM parser.

Functions
---------
ingest_pdf : Extract text chunks from a PDF file path.
ingest_html : Extract text chunks from an HTML file path or raw HTML string.
ingest_text : Split a plain text string into processable chunks.
"""

from __future__ import annotations

import os
import re
from typing import List


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Target character length for a single text chunk sent downstream.
# Chosen to balance context richness with LLM token limits.
DEFAULT_CHUNK_SIZE: int = 2000

# Overlap between consecutive chunks (in characters) so that sentences
# straddling chunk boundaries are not silently dropped.
DEFAULT_CHUNK_OVERLAP: int = 200

# Tags whose text content we want to discard from HTML documents (navigation,
# scripts, styles, etc.).
_HTML_SKIP_TAGS: frozenset[str] = frozenset(
    [
        "script",
        "style",
        "noscript",
        "head",
        "nav",
        "footer",
        "aside",
        "figure",
        "figcaption",
        "button",
        "input",
        "select",
        "textarea",
        "iframe",
        "svg",
        "math",
    ]
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split *text* into overlapping chunks of at most *chunk_size* characters.

    Splitting is attempted at sentence boundaries (period, exclamation mark, or
    question mark followed by whitespace) so that individual sentences remain
    intact wherever possible. If no sentence boundary is found within the
    current window the chunk is cut at ``chunk_size`` characters.

    Parameters
    ----------
    text:
        The input text to split.
    chunk_size:
        Maximum number of characters per chunk. Defaults to
        ``DEFAULT_CHUNK_SIZE`` (2 000).
    overlap:
        Number of characters carried over from the end of one chunk to the
        start of the next. Defaults to ``DEFAULT_CHUNK_OVERLAP`` (200).

    Returns
    -------
    List[str]
        Ordered list of text chunks. Returns an empty list when *text* is
        blank after stripping whitespace.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Try to find the last sentence boundary within the window.
            window = text[start:end]
            # Find all sentence-ending positions (., !, ?) followed by space/\n
            matches = list(re.finditer(r'[.!?][\s]', window))
            if matches:
                # Cut after the punctuation character (inclusive).
                boundary = matches[-1].start() + 1
                end = start + boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance start, backing up by *overlap* characters.
        next_start = end - overlap
        if next_start <= start:
            # Guard against infinite loop when overlap >= chunk_size.
            next_start = end
        start = next_start

    return chunks


# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------


def ingest_pdf(
    path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Extract and chunk plain text from a PDF file.

    Each page's text is extracted individually via *pdfplumber*, concatenated
    in page order with a newline separator, then split into overlapping chunks
    using :func:`chunk_text`.

    Parameters
    ----------
    path:
        Absolute or relative filesystem path to the PDF file.
    chunk_size:
        Maximum character length of each output chunk.
    overlap:
        Character overlap between consecutive chunks.

    Returns
    -------
    List[str]
        Ordered list of plain-text chunks ready for downstream processing.
        Returns an empty list if the PDF contains no extractable text.

    Raises
    ------
    FileNotFoundError
        If *path* does not point to an existing file.
    ValueError
        If *path* cannot be opened as a PDF.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF file not found: {path!r}")

    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pdfplumber is required for PDF ingestion. "
            "Install it with: pip install pdfplumber"
        ) from exc

    try:
        page_texts: List[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                except Exception:  # noqa: BLE001
                    # Skip pages that raise extraction errors.
                    continue
                if page_text:
                    page_texts.append(page_text)
    except Exception as exc:
        raise ValueError(f"Failed to open or read PDF at {path!r}: {exc}") from exc

    full_text = "\n\n".join(page_texts)
    return chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)


# ---------------------------------------------------------------------------
# HTML ingestion
# ---------------------------------------------------------------------------


def ingest_html(
    source: str,
    is_file: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Extract and chunk plain text from an HTML document.

    Boilerplate tags (scripts, styles, navigation, etc.) are stripped before
    text extraction. The remaining visible text is normalised and split into
    overlapping chunks.

    Parameters
    ----------
    source:
        Either a filesystem path to an HTML file (``is_file=True``) or a raw
        HTML string (``is_file=False``).
    is_file:
        When ``True`` (default), *source* is interpreted as a file path.
        When ``False``, *source* is interpreted as raw HTML content.
    chunk_size:
        Maximum character length of each output chunk.
    overlap:
        Character overlap between consecutive chunks.

    Returns
    -------
    List[str]
        Ordered list of plain-text chunks ready for downstream processing.
        Returns an empty list if the document contains no extractable text.

    Raises
    ------
    FileNotFoundError
        If ``is_file=True`` and *source* does not point to an existing file.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "beautifulsoup4 is required for HTML ingestion. "
            "Install it with: pip install beautifulsoup4"
        ) from exc

    if is_file:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"HTML file not found: {source!r}")
        with open(source, "r", encoding="utf-8", errors="replace") as fh:
            html_content = fh.read()
    else:
        html_content = source

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove unwanted tags in-place.
    for tag_name in _HTML_SKIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Extract visible text, preserving whitespace between block elements.
    raw_text = soup.get_text(separator="\n")

    # Normalise whitespace: collapse runs of blank lines to a single blank
    # line, and strip leading/trailing spaces from each line.
    lines = [line.strip() for line in raw_text.splitlines()]
    # Collapse multiple consecutive empty lines into one.
    normalised_lines: List[str] = []
    prev_blank = False
    for line in lines:
        if line == "":
            if not prev_blank:
                normalised_lines.append("")
            prev_blank = True
        else:
            normalised_lines.append(line)
            prev_blank = False

    full_text = "\n".join(normalised_lines).strip()
    return chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)


# ---------------------------------------------------------------------------
# Plain-text ingestion (convenience wrapper)
# ---------------------------------------------------------------------------


def ingest_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split a plain text string into processable chunks.

    This is a thin wrapper around :func:`chunk_text` for use cases where the
    caller has already obtained raw text from an external source.

    Parameters
    ----------
    text:
        The plain text content to chunk.
    chunk_size:
        Maximum character length of each output chunk.
    overlap:
        Character overlap between consecutive chunks.

    Returns
    -------
    List[str]
        Ordered list of text chunks. Returns an empty list when *text* is
        blank.
    """
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)
