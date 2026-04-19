# paper_miner

> **Mine quantitative findings from scientific literature at scale.**

`paper_miner` is a Python library and CLI tool that ingests PDF or HTML scientific papers and automatically extracts numerical data—measurements, percentages, p-values, confidence intervals, and more—along with their surrounding context. It uses LLM-assisted parsing (via OpenAI-compatible APIs) to understand units, relationships, and the significance of numbers, then outputs clean structured JSON or CSV.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [CLI Reference](#cli-reference)
   - [extract](#extract-command)
   - [text](#text-command)
   - [version](#version-command)
5. [Python API](#python-api)
6. [Output Schema](#output-schema)
7. [LLM Configuration](#llm-configuration)
8. [Development & Testing](#development--testing)
9. [License](#license)

---

## Features

- **PDF ingestion** via [`pdfplumber`](https://github.com/jsvine/pdfplumber) — extracts and chunks text from every page.
- **HTML ingestion** via [`BeautifulSoup4`](https://www.crummy.com/software/BeautifulSoup/) — strips scripts, styles, navigation, and footers before extracting visible text.
- **Regex pre-filter** — rapidly identifies numeric candidates (measurements, percentages, p-values, confidence intervals, mean ± SD, odds ratios, sample sizes) with surrounding sentence context *before* calling any LLM.
- **LLM-assisted enrichment** — sends candidate batches to any OpenAI-compatible API to classify units, data types, and relationships with a confidence score.
- **Flexible export** — JSON, CSV, or a Rich-formatted terminal table.
- **Programmatic API and CLI** — use `paper_miner` as a library in your own pipelines or from the command line.
- **Fully testable** — mocked LLM calls enable CI/CD integration without live API keys.

---

## Installation

### From PyPI (once published)

```bash
pip install paper_miner
```

### From source

```bash
git clone https://github.com/example/paper_miner.git
cd paper_miner
pip install -e .
```

### Development install (with test dependencies)

```bash
pip install -e ".[dev]"
```

### Dependencies

| Package | Purpose |
|---|---|
| `pdfplumber` | PDF text extraction |
| `beautifulsoup4` | HTML parsing and cleaning |
| `openai` | LLM API client |
| `typer[all]` | CLI framework |
| `rich` | Terminal formatting, tables, and progress bars |

---

## Quick Start

### CLI — extract from a PDF

```bash
# LLM-enriched extraction, save as JSON
paper-miner extract study.pdf --output results.json

# Regex-only (no API key needed), display as terminal table
paper-miner extract paper.html --no-llm

# Save as CSV
paper-miner extract trial.pdf --no-llm --format csv --output data.csv
```

### CLI — extract from plain text

```bash
# Inline text
paper-miner text "LDL was reduced by 32.4 mg/dL (p < 0.001)." --no-llm

# From stdin
echo "Mean BMI was 27.6 ± 4.1 kg/m²" | paper-miner text - --no-llm --format json
```

### Python API

```python
from paper_miner import extract_from_pdf, extract_from_html, extract_from_text

# From text (no LLM, no API key required)
records = extract_from_text(
    "The treatment reduced blood pressure by 12.5 mmHg (p=0.003).",
    use_llm=False,
)
for record in records:
    print(record.value, record.unit, record.data_type)
# 0.003  none    p-value
# 12.5   mmHg    measurement

# From PDF with LLM enrichment
records = extract_from_pdf(
    "study.pdf",
    api_key="sk-...",          # or set OPENAI_API_KEY env var
    model="gpt-4o-mini",
)

# Export to JSON
from paper_miner import export_records
export_records(records, fmt="json", output_path="results.json")
```

---

## CLI Reference

All commands share a common structure:

```
paper-miner [COMMAND] [ARGUMENTS] [OPTIONS]
```

Run `paper-miner --help` or `paper-miner [COMMAND] --help` for full usage information.

---

### `extract` command

Extract numerical data from a **PDF** or **HTML** file.

```
paper-miner extract FILE [OPTIONS]
```

#### Arguments

| Argument | Description |
|---|---|
| `FILE` | Path to the input `.pdf`, `.html`, or `.htm` file. |

#### Options

| Option | Default | Description |
|---|---|---|
| `--output`, `-o` PATH | stdout | Output file path. Format is auto-detected from the extension. |
| `--format`, `-f` FORMAT | auto | Output format: `json`, `csv`, or `table`. |
| `--llm` / `--no-llm` | `--llm` | Enable or disable LLM enrichment. |
| `--api-key` KEY | `$OPENAI_API_KEY` | OpenAI-compatible API key. |
| `--base-url` URL | OpenAI default | Custom API base URL (e.g. for Ollama or LM Studio). |
| `--model`, `-m` MODEL | `gpt-4o-mini` | LLM model identifier. |
| `--source` LABEL | filename | Custom source label attached to every record. |
| `--quiet`, `-q` | false | Suppress progress indicators and summary. |

#### Examples

```bash
# Extract from PDF, save as JSON (LLM enabled)
paper-miner extract study.pdf -o results.json

# Extract from HTML, display as table (no LLM)
paper-miner extract paper.html --no-llm

# Extract from PDF, save as CSV, use a local LLM server
paper-miner extract trial.pdf \
  --base-url http://localhost:11434/v1 \
  --model llama3 \
  --format csv \
  --output data.csv

# Regex-only, quiet mode, pipe JSON to another tool
paper-miner extract paper.pdf --no-llm --format json --quiet | jq '.[] | .value'
```

---

### `text` command

Extract numerical data from **plain text** passed as an argument or piped via stdin.

```
paper-miner text CONTENT [OPTIONS]
```

#### Arguments

| Argument | Description |
|---|---|
| `CONTENT` | Plain text to process, or `-` to read from stdin. |

#### Options

| Option | Default | Description |
|---|---|---|
| `--output`, `-o` PATH | stdout | Output file path. |
| `--format`, `-f` FORMAT | `table` | Output format: `json`, `csv`, or `table`. |
| `--llm` / `--no-llm` | `--llm` | Enable or disable LLM enrichment. |
| `--api-key` KEY | `$OPENAI_API_KEY` | OpenAI-compatible API key. |
| `--base-url` URL | OpenAI default | Custom API base URL. |
| `--model`, `-m` MODEL | `gpt-4o-mini` | LLM model identifier. |
| `--source` LABEL | `<inline text>` | Custom source label. |
| `--section` LABEL | None | Document section label (e.g. `Results`). |
| `--quiet`, `-q` | false | Suppress progress and summary output. |

#### Examples

```bash
# Inline text, regex-only
paper-miner text "Compound X reduced LDL by 32.4 mg/dL (p < 0.001)." --no-llm

# Read from stdin
cat abstract.txt | paper-miner text - --no-llm --format json --output abstract_nums.json

# With section label
paper-miner text "n = 240 participants were enrolled." \
  --no-llm --section Methods --format json
```

---

### `version` command

Display the installed `paper_miner` version.

```bash
paper-miner version
```

---

## Python API

### `extract_from_pdf(path, *, api_key, base_url, model, use_llm, source)`

Extract numeric records from a PDF file.

```python
from paper_miner import extract_from_pdf

records = extract_from_pdf(
    "study.pdf",
    use_llm=True,
    api_key="sk-...",
    model="gpt-4o-mini",
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | — | Path to the PDF file. |
| `api_key` | `str \| None` | `None` | API key (falls back to `OPENAI_API_KEY`). |
| `base_url` | `str \| None` | `None` | Custom API endpoint URL. |
| `model` | `str` | `"gpt-4o-mini"` | LLM model name. |
| `use_llm` | `bool` | `True` | Whether to call the LLM. |
| `source` | `str \| None` | `None` | Source label for records (defaults to `path`). |

**Returns:** `list[NumericRecord]`

---

### `extract_from_html(source, *, api_key, base_url, model, use_llm, is_file, source_label)`

Extract numeric records from an HTML file or HTML string.

```python
from paper_miner import extract_from_html

# From a file
records = extract_from_html("paper.html", use_llm=False)

# From a raw HTML string
records = extract_from_html(
    "<p>LDL reduced by 32.4 mg/dL (p < 0.001).</p>",
    is_file=False,
    use_llm=False,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | `str` | — | File path (if `is_file=True`) or raw HTML string. |
| `is_file` | `bool` | `True` | When `True`, `source` is a path; when `False`, raw HTML. |
| `source_label` | `str \| None` | `None` | Source label for records. |
| *(common LLM params)* | | | Same as `extract_from_pdf`. |

**Returns:** `list[NumericRecord]`

---

### `extract_from_text(text, *, api_key, base_url, model, use_llm, source, section)`

Extract numeric records from a plain text string.

```python
from paper_miner import extract_from_text

records = extract_from_text(
    "The treatment reduced blood pressure by 12.5 mmHg (p=0.003).",
    use_llm=False,
)
print(records[0].value)      # "0.003"
print(records[0].data_type)  # "p-value"
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | — | Plain text to mine. |
| `source` | `str \| None` | `None` | Source label for records. |
| `section` | `str \| None` | `None` | Section label for records. |
| *(common LLM params)* | | | Same as `extract_from_pdf`. |

**Returns:** `list[NumericRecord]`

---

### `export_records(records, fmt, output_path, indent, ensure_ascii)`

Serialize a list of `NumericRecord` objects to JSON or CSV.

```python
from paper_miner import export_records

export_records(records, fmt="json", output_path="results.json")
export_records(records, fmt="csv",  output_path="results.csv")
export_records(records, fmt="json")  # prints to stdout
```

---

## Output Schema

Every extracted finding is represented as a `NumericRecord` with the following fields:

| Field | Type | Description |
|---|---|---|
| `value` | `str` | The raw numeric value as it appears in the source (e.g. `"32.4"`). |
| `unit` | `str` | Physical or statistical unit (e.g. `"mg/dL"`, `"%"`, `"none"`). |
| `data_type` | `str` | Classification: `measurement`, `percentage`, `p-value`, `confidence_interval`, `mean`, `median`, `standard_deviation`, `count`, `ratio`, `other`. |
| `context` | `str` | The surrounding sentence from the source document. |
| `relationship` | `str` | Brief description of what the number measures (LLM-generated). |
| `raw_text` | `str` | The exact substring that triggered this record (e.g. `"p < 0.001"`). |
| `section` | `str \| null` | Document section (e.g. `"Results"`), if available. |
| `confidence` | `float \| null` | LLM confidence score in `[0.0, 1.0]`, or `null` when no LLM is used. |
| `source` | `str \| null` | Source document identifier (filename, URL, etc.). |

### JSON example

```json
[
  {
    "value": "32.4",
    "unit": "mg/dL",
    "data_type": "measurement",
    "context": "Compound X reduced LDL cholesterol by 32.4 mg/dL (95% CI: 28.1–36.7 mg/dL) compared with placebo (p < 0.001).",
    "relationship": "LDL cholesterol reduction from baseline in Compound X group",
    "raw_text": "32.4 mg/dL",
    "section": "Results",
    "confidence": 0.97,
    "source": "study.pdf"
  },
  {
    "value": "0.001",
    "unit": "none",
    "data_type": "p-value",
    "context": "Compound X reduced LDL cholesterol by 32.4 mg/dL (95% CI: 28.1–36.7 mg/dL) compared with placebo (p < 0.001).",
    "relationship": "statistical significance of LDL reduction",
    "raw_text": "p < 0.001",
    "section": "Results",
    "confidence": 0.99,
    "source": "study.pdf"
  }
]
```

### CSV example

```
value,unit,data_type,context,relationship,raw_text,section,confidence,source
32.4,mg/dL,measurement,"Compound X reduced LDL cholesterol by 32.4 mg/dL...","LDL cholesterol reduction",32.4 mg/dL,Results,0.97,study.pdf
0.001,none,p-value,"... compared with placebo (p < 0.001).","statistical significance",p < 0.001,Results,0.99,study.pdf
```

---

## LLM Configuration

`paper_miner` works with **any OpenAI-compatible API**, including:

- [OpenAI](https://platform.openai.com/) (default)
- [Ollama](https://ollama.ai/) (local models)
- [LM Studio](https://lmstudio.ai/) (local models)
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- [Groq](https://groq.com/)
- [Together AI](https://www.together.ai/)

### Setting the API key

```bash
# Environment variable (recommended)
export OPENAI_API_KEY="sk-..."

# CLI flag (overrides env var)
paper-miner extract paper.pdf --api-key sk-...

# Python API
records = extract_from_pdf("paper.pdf", api_key="sk-...")
```

### Using a local LLM (Ollama example)

```bash
# Start Ollama with a model
ollama pull llama3
ollama serve

# Point paper-miner at the local server
paper-miner extract paper.pdf \
  --base-url http://localhost:11434/v1 \
  --model llama3 \
  --api-key ollama
```

### Disabling LLM enrichment

When `--no-llm` is used, the tool returns regex-extracted candidates only.
No API key is required.  Records will have `confidence = null` and
`relationship = ""` (empty), but all other fields are populated heuristically.

```bash
paper-miner extract paper.pdf --no-llm --format json
```

---

## Development & Testing

### Running the test suite

```bash
pip install -e ".[dev]"
pytest
```

All LLM calls are mocked — no API key is required to run the tests.

### Running with coverage

```bash
pytest --cov=paper_miner --cov-report=term-missing
```

### Project structure

```
paper_miner/
├── __init__.py      # Public API: extract_from_pdf/html/text, export_records
├── cli.py           # Typer CLI: extract, text, version commands
├── ingest.py        # PDF and HTML ingestion and chunking
├── extractor.py     # Regex-based numeric candidate pre-filter
├── llm_parser.py    # LLM-assisted enrichment via OpenAI-compatible API
├── exporter.py      # JSON and CSV serialization
└── models.py        # NumericRecord dataclass
tests/
├── fixtures/
│   └── sample.html  # Synthetic scientific paper HTML fixture
├── test_extractor.py
├── test_ingest.py
├── test_exporter.py
└── test_llm_parser.py
```

### Code style

- Python 3.9+ with type hints throughout.
- PEP 8 compliant.
- Docstrings on all public functions and classes.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome!  Please open an issue or pull request on
[GitHub](https://github.com/example/paper_miner).

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests.
4. Run the test suite: `pytest`
5. Open a pull request.
