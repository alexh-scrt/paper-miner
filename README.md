# paper_miner

> **Mine quantitative findings from scientific literature at scale.**

`paper_miner` is a Python library and CLI tool that ingests PDF or HTML scientific papers and automatically extracts numerical data—measurements, percentages, p-values, confidence intervals, and more—along with their surrounding context. It uses LLM-assisted parsing (via OpenAI-compatible APIs) to understand units, relationships, and the significance of numbers, then outputs clean structured JSON or CSV.

Designed for researchers and data engineers who need to systematically extract quantitative findings from scientific literature without manual effort.

---

## Quick Start

### Install

```bash
pip install paper_miner
```

Or install from source:

```bash
git clone https://github.com/your-org/paper_miner.git
cd paper_miner
pip install -e .
```

### Basic Usage

```bash
# Extract from a PDF (with LLM enrichment)
export OPENAI_API_KEY=sk-...
paper-miner extract paper.pdf --format json --output results.json

# Extract from an HTML file, no LLM (regex only, no API key needed)
paper-miner extract paper.html --format csv --no-llm

# Extract from inline text
paper-miner text "LDL reduced by 32.4 mg/dL (p < 0.001)" --no-llm
```

That's it. After running, `results.json` contains structured records for every numeric finding detected in the paper.

---

## Features

- **Multi-format ingestion** — Parse scientific papers from PDF (via `pdfplumber`) or HTML (via `BeautifulSoup4`) sources, with automatic text chunking for large documents.
- **Regex pre-filter** — Rapidly identifies numeric candidates (measurements, percentages, p-values, confidence intervals, mean±SD) with surrounding sentence context before any LLM call.
- **LLM-assisted enrichment** — Sends candidates to any OpenAI-compatible API to classify unit, data type (e.g., `mean`, `p-value`, `percentage`), relationship to experimental conditions, and a confidence score.
- **Flexible export** — Output to structured JSON or CSV, to a file or stdout, usable as a standalone CLI or imported as a Python library.
- **CI/CD-friendly testing** — All LLM calls are mockable; the full test suite runs without a live API key.

---

## Usage Examples

### CLI

```bash
# Extract from PDF → JSON file
paper-miner extract paper.pdf --format json --output results.json

# Extract from HTML → CSV, printed to stdout
paper-miner extract paper.html --format csv

# Use a specific model and custom API base URL
paper-miner extract paper.pdf \
  --model gpt-4o \
  --api-key sk-... \
  --api-base https://api.openai.com/v1 \
  --format json --output results.json

# Skip LLM entirely (regex-only, no API key required)
paper-miner extract paper.pdf --no-llm --format csv --output results.csv

# Pipe plain text
echo "BMI was 27.6 kg/m² in the control group." | paper-miner text - --no-llm

# Check version
paper-miner version
```

### Python API

```python
from paper_miner import extract_from_pdf, extract_from_html, extract_from_text, export_records

# From a PDF file
records = extract_from_pdf("paper.pdf", use_llm=True, api_key="sk-...")

# From an HTML file or raw HTML string
records = extract_from_html("paper.html", use_llm=False)

# From a plain text string
records = extract_from_text(
    "The treatment reduced systolic blood pressure by 12.5 mmHg (p=0.003).",
    use_llm=False
)

# Inspect results
for record in records:
    print(record.value, record.unit, record.data_type, record.context)

# Export to JSON or CSV
export_records(records, format="json", output_path="results.json")
export_records(records, format="csv", output_path="results.csv")

# Or convert to string directly
from paper_miner.exporter import records_to_json_str, records_to_csv_str

json_str = records_to_json_str(records)
csv_str = records_to_csv_str(records)
```

### Example Output (JSON)

```json
[
  {
    "value": 12.5,
    "unit": "mmHg",
    "data_type": "measurement",
    "context": "The treatment reduced systolic blood pressure by 12.5 mmHg (p=0.003).",
    "relationship": "reduction in systolic blood pressure",
    "confidence": 0.97,
    "source": null
  },
  {
    "value": 0.003,
    "unit": null,
    "data_type": "p-value",
    "context": "The treatment reduced systolic blood pressure by 12.5 mmHg (p=0.003).",
    "relationship": "statistical significance of blood pressure reduction",
    "confidence": 0.99,
    "source": null
  }
]
```

---

## Project Structure

```
paper_miner/
├── pyproject.toml          # Project metadata, dependencies, CLI entry-point
├── README.md               # This file
├── paper_miner/
│   ├── __init__.py         # Public API: extract_from_pdf, extract_from_html, extract_from_text
│   ├── ingest.py           # PDF and HTML ingestion → plain text chunks
│   ├── extractor.py        # Regex-based numeric candidate pre-filter
│   ├── llm_parser.py       # LLM enrichment of candidate records
│   ├── models.py           # NumericRecord dataclass
│   ├── exporter.py         # JSON and CSV serialization
│   └── cli.py              # Typer CLI entry point
└── tests/
    ├── __init__.py
    ├── test_extractor.py   # Regex extractor unit tests
    ├── test_ingest.py      # Ingestion unit tests
    ├── test_exporter.py    # Exporter unit tests
    ├── test_llm_parser.py  # LLM parser tests (mocked API)
    └── fixtures/
        └── sample.html     # Sample HTML paper snippet for testing
```

---

## Configuration

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--format` | `json` | Output format: `json` or `csv` |
| `--output` / `-o` | stdout | Output file path |
| `--no-llm` | `False` | Skip LLM enrichment; use regex only |
| `--model` | `gpt-4o-mini` | OpenAI-compatible model name |
| `--api-key` | `$OPENAI_API_KEY` | API key (env var preferred) |
| `--api-base` | OpenAI default | Custom API base URL |
| `--batch-size` | `10` | Number of candidates per LLM request |

### Environment Variables

```bash
# Set your API key (recommended over passing via --api-key)
export OPENAI_API_KEY=sk-...

# Optional: use a different OpenAI-compatible provider
export OPENAI_API_BASE=https://your-provider.com/v1
```

### Python API Options

```python
records = extract_from_pdf(
    "paper.pdf",
    use_llm=True,           # Set False for regex-only mode
    api_key="sk-...",       # Falls back to OPENAI_API_KEY env var
    api_base=None,          # Custom base URL for OpenAI-compatible APIs
    model="gpt-4o-mini",    # Any OpenAI-compatible model
    batch_size=10,          # Candidates per LLM call
)
```

---

## Running Tests

No API key required — all LLM calls are mocked.

```bash
pip install -e .[dev]
pytest tests/
```

---

## License

MIT © paper_miner contributors

---

*Built with [Jitter](https://github.com/jitter-ai) - an AI agent that ships code daily.*
