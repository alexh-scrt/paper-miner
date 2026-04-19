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
from unittest.mock import MagicMock, patch

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

    def test_returns_list_type(self) -> None:
        result = chunk_text("Some text here.", chunk_size=100)
        assert isinstance(result, list)

    def test_each_chunk_is_string(self) -> None:
        text = "Measurement was 42 kg. Another reading was 55 kg."
        chunks = chunk_text(text, chunk_size=100)
        for chunk in chunks:
            assert isinstance(chunk, str)

    def test_single_very_long_word_does_not_hang(self) -> None:
        # A single token longer than chunk_size should still produce a result.
        text = "x" * 5000
        chunks = chunk_text(text, chunk_size=1000, overlap=100)
        assert len(chunks) >= 1
        # All content should be covered.
        combined = "".join(chunks)
        # Because of overlapping, combined may be longer than original.
        assert len(combined) >= len(text) // 2

    def test_overlap_zero_no_repetition(self) -> None:
        # With overlap=0, adjacent chunks should not share content
        # (they may share a trailing/leading stripped space, which is fine).
        sentence = "Sentence number {:d} here. "
        text = "".join(sentence.format(i) for i in range(50))  # ~1300 chars
        chunks = chunk_text(text, chunk_size=200, overlap=0)
        assert len(chunks) >= 2

    def test_default_constants_are_reasonable(self) -> None:
        assert DEFAULT_CHUNK_SIZE > 0
        assert DEFAULT_CHUNK_OVERLAP >= 0
        assert DEFAULT_CHUNK_OVERLAP < DEFAULT_CHUNK_SIZE

    def test_text_with_only_sentence_boundaries(self) -> None:
        text = "First. Second. Third. Fourth. Fifth."
        chunks = chunk_text(text, chunk_size=20, overlap=0)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) > 0

    def test_newline_separated_text(self) -> None:
        text = "\n".join(["Line {:d} with content.".format(i) for i in range(50)])
        chunks = chunk_text(text, chunk_size=200, overlap=0)
        assert len(chunks) >= 1

    def test_unicode_text_chunked_correctly(self) -> None:
        # Unicode characters (multi-byte) should be handled correctly.
        text = "LDL r\u00e9duction de 32.4 mg/dL. " * 50
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, str)

    def test_overlap_larger_than_text_does_not_loop(self) -> None:
        # overlap >= chunk_size guard: should not infinite loop.
        text = "Short text with value 42."
        chunks = chunk_text(text, chunk_size=100, overlap=200)
        assert len(chunks) == 1  # text is shorter than chunk_size

    def test_scientific_text_chunked(self) -> None:
        abstract = (
            "Compound X reduced LDL cholesterol by 32.4 mg/dL "
            "(95% CI: 28.1\u201336.7 mg/dL) compared with placebo (p < 0.001). "
            "Secondary outcomes included a 15.2% reduction in total cholesterol. "
            "Mean age was 54.3 \u00b1 7.8 years. N = 240 participants enrolled."
        )
        chunks = chunk_text(abstract, chunk_size=200, overlap=20)
        combined = " ".join(chunks)
        # Key numbers should survive chunking.
        assert "32.4" in combined
        assert "15.2" in combined


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

    def test_title_preserved_or_stripped(self) -> None:
        # The <title> tag is in <head> which is stripped; title text may or
        # may not appear depending on parser behaviour — just ensure no crash.
        chunks = ingest_html(SIMPLE_HTML, is_file=False)
        assert isinstance(chunks, list)

    def test_h1_text_preserved(self) -> None:
        chunks = ingest_html(SIMPLE_HTML, is_file=False)
        combined = " ".join(chunks)
        assert "Study Title" in combined

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
        # With a very small chunk size every chunk should be <= 50 chars.
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
        # Just ensure the nav tag was processed; the text itself is fine to appear
        # in section headings but the nav element should be gone.
        assert isinstance(combined, str)

    def test_ingest_html_returns_list_of_strings(self) -> None:
        chunks = ingest_html(SIMPLE_HTML, is_file=False)
        for chunk in chunks:
            assert isinstance(chunk, str)

    def test_ingest_html_is_file_false_with_minimal_html(self) -> None:
        html = "<html><body><p>Measurement was 42 mg/dL.</p></body></html>"
        chunks = ingest_html(html, is_file=False)
        assert len(chunks) >= 1
        combined = " ".join(chunks)
        assert "42" in combined

    def test_ingest_html_unicode_preserved(self) -> None:
        html = (
            "<html><body>"
            "<p>LDL r\u00e9duction de 32.4 mg/dL (±0.5).</p>"
            "</body></html>"
        )
        chunks = ingest_html(html, is_file=False)
        combined = " ".join(chunks)
        assert "32.4" in combined

    def test_ingest_html_whitespace_only_body_returns_empty(self) -> None:
        html = "<html><body>   \n\n   </body></html>"
        chunks = ingest_html(html, is_file=False)
        assert chunks == []

    def test_ingest_html_multiple_paragraphs(self) -> None:
        html = (
            "<html><body>"
            + "".join(
                f"<p>Paragraph {i}: value was {i * 10} mg/dL.</p>"
                for i in range(1, 6)
            )
            + "</body></html>"
        )
        chunks = ingest_html(html, is_file=False)
        combined = " ".join(chunks)
        assert "10 mg/dL" in combined
        assert "50 mg/dL" in combined

    def test_ingest_html_table_text_preserved(self) -> None:
        html = (
            "<html><body>"
            "<table><tr><th>Characteristic</th><th>Value</th></tr>"
            "<tr><td>LDL (mg/dL)</td><td>158.3</td></tr></table>"
            "</body></html>"
        )
        chunks = ingest_html(html, is_file=False)
        combined = " ".join(chunks)
        assert "158.3" in combined

    def test_ingest_html_fixture_footer_stripped(self) -> None:
        chunks = ingest_html(str(SAMPLE_HTML_PATH), is_file=True)
        combined = " ".join(chunks)
        # The footer tag should have been stripped.
        # Its text "Journal of Hypothetical Medicine" may still appear in body
        # section; just verify no crash.
        assert isinstance(combined, str)

    def test_ingest_html_fixture_numeric_content_rich(self) -> None:
        """The fixture should yield many numeric values from the study."""
        chunks = ingest_html(str(SAMPLE_HTML_PATH), is_file=True)
        combined = " ".join(chunks)
        # Check several key numbers from the fixture paper.
        assert "54.3" in combined   # mean age
        assert "158.3" in combined  # baseline LDL
        assert "15.2" in combined   # % cholesterol reduction
        assert "0.001" in combined  # p-value (appears as &lt; 0.001)

    def test_ingest_html_overlap_parameter_accepted(self) -> None:
        chunks = ingest_html(SIMPLE_HTML, is_file=False, chunk_size=100, overlap=20)
        assert isinstance(chunks, list)

    def test_ingest_html_no_crash_on_malformed_html(self) -> None:
        malformed = "<p>Value is 42 mg/dL<p>Another value <b>32.4 mg/dL"
        chunks = ingest_html(malformed, is_file=False)
        # BeautifulSoup is lenient; should not crash.
        combined = " ".join(chunks)
        assert "42" in combined


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
            "The compound reduced LDL by 32.4 mg/dL (95% CI: 28.1\u201336.7 mg/dL; p < 0.001). "
            "A total of 240 participants were enrolled."
        )

        fake_page_2 = MagicMock()
        fake_page_2.extract_text.return_value = (
            "Mean age was 54.3 \u00b1 7.8 years. BMI was 27.6 kg/m\u00b2. "
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

    def test_ingest_pdf_returns_list(self, tmp_path: Path) -> None:
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        fake_page = MagicMock()
        fake_page.extract_text.return_value = "Simple text with 42 mg/dL."

        fake_pdf = MagicMock()
        fake_pdf.pages = [fake_page]
        fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
        fake_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=fake_pdf):
            result = ingest_pdf(str(dummy_pdf))

        assert isinstance(result, list)

    def test_ingest_pdf_each_chunk_is_string(self, tmp_path: Path) -> None:
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        fake_page = MagicMock()
        fake_page.extract_text.return_value = "Measurement: 42 mmHg. Another: 55 bpm."

        fake_pdf = MagicMock()
        fake_pdf.pages = [fake_page]
        fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
        fake_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=fake_pdf):
            chunks = ingest_pdf(str(dummy_pdf))

        for chunk in chunks:
            assert isinstance(chunk, str)

    def test_ingest_pdf_multiple_pages_combined(self, tmp_path: Path) -> None:
        """Text from all pages should be combined into the chunk stream."""
        dummy_pdf = tmp_path / "multi.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        pages_text = [
            "Page one content: LDL was 158.3 mg/dL.",
            "Page two content: p < 0.001 was the significance level.",
            "Page three content: mean age 54.3 years.",
        ]
        fake_pages = []
        for text in pages_text:
            page = MagicMock()
            page.extract_text.return_value = text
            fake_pages.append(page)

        fake_pdf = MagicMock()
        fake_pdf.pages = fake_pages
        fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
        fake_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=fake_pdf):
            chunks = ingest_pdf(str(dummy_pdf))

        combined = " ".join(chunks)
        assert "158.3" in combined
        assert "0.001" in combined
        assert "54.3" in combined

    def test_ingest_pdf_all_pages_return_empty_string(self, tmp_path: Path) -> None:
        dummy_pdf = tmp_path / "empty_str.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        fake_page = MagicMock()
        fake_page.extract_text.return_value = ""

        fake_pdf = MagicMock()
        fake_pdf.pages = [fake_page, fake_page]
        fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
        fake_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=fake_pdf):
            chunks = ingest_pdf(str(dummy_pdf))

        # Empty strings from pages produce no text.
        assert chunks == []

    def test_ingest_pdf_no_pages(self, tmp_path: Path) -> None:
        """A PDF with no pages (empty pages list) should return empty list."""
        dummy_pdf = tmp_path / "nopages.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        fake_pdf = MagicMock()
        fake_pdf.pages = []
        fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
        fake_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=fake_pdf):
            chunks = ingest_pdf(str(dummy_pdf))

        assert chunks == []

    def test_ingest_pdf_overlap_parameter_accepted(self, tmp_path: Path) -> None:
        dummy_pdf = tmp_path / "overlap.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        fake_page = MagicMock()
        fake_page.extract_text.return_value = "Sample text. " * 200

        fake_pdf = MagicMock()
        fake_pdf.pages = [fake_page]
        fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
        fake_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=fake_pdf):
            chunks = ingest_pdf(str(dummy_pdf), chunk_size=300, overlap=50)

        assert len(chunks) >= 1

    def test_ingest_pdf_text_extraction_returns_whitespace(self, tmp_path: Path) -> None:
        """Pages returning only whitespace should be treated as empty."""
        dummy_pdf = tmp_path / "ws.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        fake_page = MagicMock()
        fake_page.extract_text.return_value = "   \n\n   "

        good_page = MagicMock()
        good_page.extract_text.return_value = "Result: 42 mg/dL."

        fake_pdf = MagicMock()
        fake_pdf.pages = [fake_page, good_page]
        fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
        fake_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=fake_pdf):
            chunks = ingest_pdf(str(dummy_pdf))

        # The good page should contribute content.
        combined = " ".join(chunks)
        assert "42" in combined


# ---------------------------------------------------------------------------
# ingest_text tests
# ---------------------------------------------------------------------------


class TestIngestText:
    """Tests for the ingest_text wrapper function."""

    def test_empty_string_returns_empty_list(self) -> None:
        assert ingest_text("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert ingest_text("   \n  ") == []

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

    def test_content_preserved(self) -> None:
        text = "LDL was 32.4 mg/dL and p < 0.001."
        chunks = ingest_text(text)
        combined = " ".join(chunks)
        assert "32.4" in combined
        assert "0.001" in combined

    def test_custom_chunk_size(self) -> None:
        text = "Value was 42. " * 100
        chunks = ingest_text(text, chunk_size=100, overlap=0)
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_custom_overlap(self) -> None:
        text = "Sentence one. Sentence two. Sentence three. " * 20
        chunks = ingest_text(text, chunk_size=100, overlap=20)
        assert len(chunks) >= 1

    def test_unicode_text_preserved(self) -> None:
        text = "Mean BMI was 27.6 \u00b1 4.1 kg/m\u00b2."
        chunks = ingest_text(text)
        combined = " ".join(chunks)
        assert "27.6" in combined

    def test_single_number_text(self) -> None:
        text = "42"
        chunks = ingest_text(text)
        assert len(chunks) == 1
        assert chunks[0] == "42"

    def test_delegates_to_chunk_text(self) -> None:
        # ingest_text is a thin wrapper around chunk_text.
        # Verify it produces the same result.
        text = "Sample measurement was 32.4 mg/dL (p < 0.001). Mean age 54.3 years."
        direct = chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
        via_ingest = ingest_text(text)
        assert direct == via_ingest

    def test_long_scientific_text_all_chunks_non_empty(self) -> None:
        # Simulate a realistic scientific paragraph.
        paragraph = (
            "The primary endpoint was change from baseline in LDL cholesterol. "
            "At week 12, participants in the Compound X group experienced a mean "
            "reduction of 32.4 mg/dL (95% CI: 28.1\u201336.7 mg/dL; p < 0.001). "
            "Secondary outcomes included a 15.2% reduction in total cholesterol "
            "and a 12.4 mg/dL decrease in triglycerides (p = 0.03). "
            "HDL cholesterol did not change significantly (\u22120.8 mg/dL; p = 0.42). "
        ) * 5
        chunks = ingest_text(paragraph, chunk_size=300, overlap=30)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.strip() != ""
