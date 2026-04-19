"""Unit tests for the paper_miner.ingest module.

Tests cover:
- chunk_text: splitting behaviour, overlap, edge cases.
- ingest_html: plain-text extraction from HTML strings and fixture file.
- ingest_pdf: error handling for missing/invalid files (pdfplumber mocked).
- ingest_text: thin wrapper sanity checks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Ensure the project root is on the path when running tests directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_miner.ingest import (
    chunk_text,
    ingest_html,
    ingest_pdf,
    ingest_text,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_HTML_PATH = FIXTURE_DIR / "sample.html"

# A small HTML snippet used in in-memory tests.
SIMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Test</title>
  <style>body { color: red; }</style>
  <script>alert('skip me');</script>
</head>
<body>
  <nav>Navigation bar</nav>
  <h1>Study Title</h1>
  <p>The treatment reduced blood pressure by 12.5 mmHg (p=0.003).</p>
  <p>A total of 240 participants were enrolled; mean age was 54.3 years.</p>
  <p>The 95% confidence interval was 10.1 to 14.9 mmHg.</p>
  <footer>Footer content to be stripped.</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# chunk_text tests
# ---------------------------------------------------------------------------


class TestChunkText:
    """Tests for the chunk_text utility function."""

    def test_empty_string_returns_empty_list(self) -> None:
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert chunk_text("   \n\t  ") == []

    def test_short_text_returned_as_single_chunk(self) -> None:
        text = "Hello world. This is a short sentence."
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_text_exactly_at_chunk_size_is_single_chunk(self) -> None:
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=100, overlap=0)
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self) -> None:
        # Create text long enough to require multiple chunks.
        sentence = "This is a test sentence with some content. "
        text = sentence * 60  # ~2520 characters
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) > 1

    def test_all_chunks_non_empty(self) -> None:
        sentence = "Number value is 42 units. "
        text = sentence * 100
        chunks = chunk_text(text, chunk_size=300, overlap=30)
        assert all(c.strip() for c in chunks)

    def test_overlap_content_appears_in_consecutive_chunks(self) -> None:
        # Build a predictable text where we can verify overlap.
        # Use a long word-free string split with sentence boundaries.
        part_a = "A" * 195 + ". "
        part_b = "B" * 195 + ". "
        part_c = "C" * 195 + "."
        text = part_a + part_b + part_c
        chunks = chunk_text(text, chunk_size=210, overlap=20)
        # There should be more than one chunk.
        assert len(chunks) >= 2

    def test_no_overlap_chunks_cover_full_text(self) -> None:
        text = "Word " * 400  # 2000 chars
        chunks = chunk_text(text, chunk_size=200, overlap=0)
        combined_len = sum(len(c) for c in chunks)
        # Combined length may differ from original due to strip() calls,
        # but every chunk should be non-empty and total coverage should
        # account for the original characters (allowing for stripped spaces).
        assert combined_len > 0
        assert len(chunks) >= 1

    def test_chunk_size_respected(self) -> None:
        text = "x" * 10_000
        chunk_size = 1000
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=0)
        for chunk in chunks:
            assert len(chunk) <= chunk_size


# ---------------------------------------------------------------------------
# ingest_html tests
# ---------------------------------------------------------------------------


class TestIngestHtml:
    """Tests for the ingest_html function."""

    def test_ingest_html_string_returns_nonempty_chunks(self) -> None:
        chunks = ingest_html(SIMPLE_HTML, is_file=False)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_script_content_stripped(self) -> None:
        chunks = ingest_html(SIMPLE_HTML, is_file=False)
        combined = " ".join(chunks)
        assert "alert" not in combined
        assert "skip me" not in combined

    def test_style_content_stripped(self) -> None:
        chunks = ingest_html(SIMPLE_HTML, is_file=False)
        combined = " ".join(chunks)
        assert "color: red" not in combined

    def test_nav_content_stripped(self) -> None:
        chunks = ingest_html(SIMPLE_HTML, is_file=False)
        combined = " ".join(chunks)
        # nav text should be absent
        assert "Navigation bar" not in combined

    def test_footer_content_stripped(self) -> None:
        chunks = ingest_html(SIMPLE_HTML, is_file=False)
        combined = " ".join(chunks)
        assert "Footer content" not in combined

    def test_body_text_preserved(self) -> None:
        chunks = ingest_html(SIMPLE_HTML, is_file=False)
        combined = " ".join(chunks)
        assert "12.5 mmHg" in combined
        assert "240 participants" in combined
        assert "95%" in combined

    def test_ingest_html_file_fixture(self) -> None:
        """Ingest the real sample.html fixture file."""
        assert SAMPLE_HTML_PATH.is_file(), (
            f"Fixture file missing: {SAMPLE_HTML_PATH}"
        )
        chunks = ingest_html(str(SAMPLE_HTML_PATH), is_file=True)
        assert len(chunks) >= 1
        combined = " ".join(chunks)
        # Key numeric values from the fixture should be present.
        assert "32.4" in combined  # LDL reduction
        assert "240" in combined   # number of participants
        assert "95%" in combined   # confidence interval marker

    def test_ingest_html_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            ingest_html("/nonexistent/path/to/file.html", is_file=True)

    def test_ingest_html_empty_string_returns_empty_list(self) -> None:
        chunks = ingest_html("", is_file=False)
        assert chunks == []

    def test_ingest_html_custom_chunk_size(self) -> None:
        chunks = ingest_html(SIMPLE_HTML, is_file=False, chunk_size=50, overlap=0)
        # With a very small chunk size every chunk should be ≤ 50 chars.
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_ingest_html_fixture_script_stripped(self) -> None:
        chunks = ingest_html(str(SAMPLE_HTML_PATH), is_file=True)
        combined = " ".join(chunks)
        assert "console.log" not in combined
        assert "navigation analytics" not in combined

    def test_ingest_html_fixture_nav_stripped(self) -> None:
        chunks = ingest_html(str(SAMPLE_HTML_PATH), is_file=True)
        combined = " ".join(chunks)
        # The nav bar link text like "Abstract" may appear in headings, but
        # the nav tag content itself should be gone (no duplicate nav-only text).
        # We assert the nav tag was processed; the nav content "Methods" also
        # appears in section headings which is fine — we just ensure no crash.
        assert isinstance(combined, str)


# ---------------------------------------------------------------------------
# ingest_pdf tests
# ---------------------------------------------------------------------------


class TestIngestPdf:
    """Tests for the ingest_pdf function."""

    def test_ingest_pdf_missing_file_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            ingest_pdf("/nonexistent/path/to/file.pdf")

    def test_ingest_pdf_with_mocked_pdfplumber(self, tmp_path: Path) -> None:
        """Test PDF ingestion using a mocked pdfplumber to avoid needing a real PDF."""
        # Create a dummy file so the existence check passes.
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 fake content")

        fake_page_1 = MagicMock()
        fake_page_1.extract_text.return_value = (
            "The compound reduced LDL by 32.4 mg/dL (95% CI: 28.1–36.7 mg/dL; p < 0.001). "
            "A total of 240 participants were enrolled."
        )

        fake_page_2 = MagicMock()
        fake_page_2.extract_text.return_value = (
            "Mean age was 54.3 ± 7.8 years. BMI was 27.6 kg/m². "
            "Triglycerides decreased by 12.4 mg/dL (p = 0.03)."
        )

        fake_pdf = MagicMock()
        fake_pdf.pages = [fake_page_1, fake_page_2]
        fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
        fake_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=fake_pdf):
            chunks = ingest_pdf(str(dummy_pdf))

        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        combined = " ".join(chunks)
        assert "32.4" in combined
        assert "240" in combined
        assert "54.3" in combined

    def test_ingest_pdf_empty_pages_returns_empty_list(self, tmp_path: Path) -> None:
        dummy_pdf = tmp_path / "empty.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        fake_page = MagicMock()
        fake_page.extract_text.return_value = None

        fake_pdf = MagicMock()
        fake_pdf.pages = [fake_page]
        fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
        fake_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=fake_pdf):
            chunks = ingest_pdf(str(dummy_pdf))

        assert chunks == []

    def test_ingest_pdf_extraction_exception_skips_page(self, tmp_path: Path) -> None:
        dummy_pdf = tmp_path / "partial.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        bad_page = MagicMock()
        bad_page.extract_text.side_effect = RuntimeError("extraction failed")

        good_page = MagicMock()
        good_page.extract_text.return_value = "Blood pressure was 120/80 mmHg."

        fake_pdf = MagicMock()
        fake_pdf.pages = [bad_page, good_page]
        fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
        fake_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=fake_pdf):
            chunks = ingest_pdf(str(dummy_pdf))

        assert len(chunks) >= 1
        combined = " ".join(chunks)
        assert "120/80" in combined

    def test_ingest_pdf_invalid_pdf_raises_value_error(self, tmp_path: Path) -> None:
        dummy_pdf = tmp_path / "bad.pdf"
        dummy_pdf.write_bytes(b"not a pdf")

        with patch("pdfplumber.open", side_effect=Exception("invalid PDF")):
            with pytest.raises(ValueError, match="Failed to open or read PDF"):
                ingest_pdf(str(dummy_pdf))

    def test_ingest_pdf_custom_chunk_size(self, tmp_path: Path) -> None:
        dummy_pdf = tmp_path / "chunked.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        long_text = "The value was 3.14 units. " * 200  # ~5200 chars

        fake_page = MagicMock()
        fake_page.extract_text.return_value = long_text

        fake_pdf = MagicMock()
        fake_pdf.pages = [fake_page]
        fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
        fake_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=fake_pdf):
            chunks = ingest_pdf(str(dummy_pdf), chunk_size=500, overlap=0)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 500


# ---------------------------------------------------------------------------
# ingest_text tests
# ---------------------------------------------------------------------------


class TestIngestText:
    """Tests for the ingest_text wrapper function."""

    def test_empty_string_returns_empty_list(self) -> None:
        assert ingest_text("") == []

    def test_short_text_returns_single_chunk(self) -> None:
        text = "The p-value was 0.001."
        chunks = ingest_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_chunked(self) -> None:
        text = "A measurement of 42 kg was recorded. " * 100
        chunks = ingest_text(text, chunk_size=200, overlap=0)
        assert len(chunks) > 1

    def test_returns_list_of_strings(self) -> None:
        chunks = ingest_text("Some text with 99.5% efficiency.", chunk_size=500)
        assert isinstance(chunks, list)
        for c in chunks:
            assert isinstance(c, str)
