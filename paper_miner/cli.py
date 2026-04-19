"""Command-line interface for paper_miner.

Provides a Typer-based CLI with commands for extracting numerical data from
PDF and HTML scientific papers, with support for format selection (JSON/CSV),
LLM configuration flags, and Rich-powered progress indicators and formatted
output tables.

Commands
--------
extract  : Extract numeric records from a PDF or HTML file.
text     : Extract numeric records from piped / inline plain text.
version  : Display the current paper_miner version.

Usage examples
--------------
  paper-miner extract paper.pdf --format json --output results.json
  paper-miner extract paper.html --format csv --no-llm
  paper-miner text "LDL reduced by 32.4 mg/dL (p < 0.001)" --no-llm
  paper-miner extract paper.pdf --model gpt-4o --api-key sk-...
  echo "BMI was 27.6 kg/m²." | paper-miner text - --no-llm
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from paper_miner import __version__
from paper_miner.exporter import export_records
from paper_miner.models import NumericRecord

# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="paper-miner",
    help=(
        "paper_miner — extract structured numerical data from scientific "
        "PDF and HTML papers using regex pre-filtering and optional "
        "LLM-assisted classification."
    ),
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console(stderr=True)   # status/progress output goes to stderr
out_console = Console()           # primary data output goes to stdout


# ---------------------------------------------------------------------------
# Shared enumerations and helpers
# ---------------------------------------------------------------------------


class OutputFormat(str, Enum):
    """Supported export format identifiers."""

    json = "json"
    csv = "csv"
    table = "table"


def _infer_format_from_path(path: Optional[str]) -> Optional[OutputFormat]:
    """Attempt to infer an output format from a file extension.

    Parameters
    ----------
    path:
        The output file path string, or ``None``.

    Returns
    -------
    Optional[OutputFormat]
        The inferred format, or ``None`` when the extension is unrecognised or
        *path* is ``None``.
    """
    if not path:
        return None
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return OutputFormat.json
    if suffix == ".csv":
        return OutputFormat.csv
    return None


def _render_table(records: list[NumericRecord]) -> None:
    """Render *records* as a Rich table printed to stdout.

    Parameters
    ----------
    records:
        The list of :class:`~paper_miner.models.NumericRecord` objects to
        display.
    """
    if not records:
        out_console.print(Panel("[yellow]No numeric records found.[/yellow]", title="Results"))
        return

    table = Table(
        title=f"Extracted Numeric Records ({len(records)} found)",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
        expand=False,
    )
    table.add_column("#", style="dim", justify="right", no_wrap=True)
    table.add_column("Value", style="bold green", no_wrap=True)
    table.add_column("Unit", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta", no_wrap=True)
    table.add_column("Confidence", justify="right", no_wrap=True)
    table.add_column("Relationship", max_width=40)
    table.add_column("Context", max_width=60)
    table.add_column("Section", style="dim", no_wrap=True)

    for idx, record in enumerate(records, start=1):
        confidence_str = (
            f"{record.confidence:.2f}" if record.confidence is not None else "—"
        )
        section_str = record.section or "—"
        relationship_str = record.relationship or "—"
        # Truncate long context for display.
        context_display = record.context
        if len(context_display) > 120:
            context_display = context_display[:117] + "..."

        table.add_row(
            str(idx),
            record.value,
            record.unit,
            record.data_type,
            confidence_str,
            relationship_str,
            context_display,
            section_str,
        )

    out_console.print(table)


def _print_summary(records: list[NumericRecord], source: str) -> None:
    """Print a brief extraction summary panel to stderr.

    Parameters
    ----------
    records:
        The extracted records.
    source:
        Human-readable label for the source document.
    """
    counts: dict[str, int] = {}
    for r in records:
        counts[r.data_type] = counts.get(r.data_type, 0) + 1

    lines = [f"[bold]Source:[/bold] {source}"]
    lines.append(f"[bold]Total records:[/bold] {len(records)}")
    if counts:
        breakdown = ", ".join(
            f"{dtype}: [green]{n}[/green]" for dtype, n in sorted(counts.items())
        )
        lines.append(f"[bold]By type:[/bold] {breakdown}")

    panel_text = "\n".join(lines)
    console.print(
        Panel(panel_text, title="[bold cyan]Extraction Summary[/bold cyan]", expand=False)
    )


def _write_output(
    records: list[NumericRecord],
    fmt: OutputFormat,
    output: Optional[str],
) -> None:
    """Serialize *records* to the chosen format and destination.

    Parameters
    ----------
    records:
        The records to export.
    fmt:
        The desired output format.
    output:
        File path for the output, or ``None`` for stdout.
    """
    if fmt == OutputFormat.table:
        _render_table(records)
        return

    # JSON and CSV are handled by the exporter module.
    try:
        export_records(
            records,
            fmt=fmt.value,
            output_path=output,
        )
    except OSError as exc:
        console.print(f"[bold red]Error writing output:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if output:
        console.print(
            f"[green]✓[/green] Output written to [bold]{output}[/bold] "
            f"([cyan]{fmt.value.upper()}[/cyan])."
        )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("extract")
def extract_command(
    file: Path = typer.Argument(
        ...,
        help=(
            "Path to the input PDF or HTML file to process. "
            "The file type is inferred from the extension (.pdf or .html/.htm)."
        ),
        exists=True,
        readable=True,
        resolve_path=True,
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Output file path.  When omitted, results are printed to stdout. "
            "The format is auto-detected from the extension if --format is not "
            "specified."
        ),
        metavar="PATH",
    ),
    fmt: Optional[OutputFormat] = typer.Option(
        None,
        "--format",
        "-f",
        help=(
            "Output format: json, csv, or table.  Defaults to 'table' when "
            "writing to stdout and no --output path is given, otherwise 'json'. "
            "Auto-detected from the --output extension when possible."
        ),
        case_sensitive=False,
    ),
    use_llm: bool = typer.Option(
        True,
        "--llm/--no-llm",
        help=(
            "Enable or disable LLM-assisted enrichment of numeric candidates. "
            "When disabled, only regex-extracted heuristic values are returned."
        ),
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="OPENAI_API_KEY",
        help=(
            "OpenAI-compatible API key.  Can also be set via the "
            "OPENAI_API_KEY environment variable."
        ),
        show_default=False,
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        help=(
            "Custom base URL for an OpenAI-compatible API endpoint "
            "(e.g. http://localhost:11434/v1 for a local Ollama server)."
        ),
        show_default=False,
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        "-m",
        help="LLM model identifier to use for candidate enrichment.",
    ),
    source_label: Optional[str] = typer.Option(
        None,
        "--source",
        help=(
            "Custom source label attached to every output record. "
            "Defaults to the input file name."
        ),
        show_default=False,
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress progress indicators and summary output.",
    ),
) -> None:
    """Extract numerical data from a [bold]PDF[/bold] or [bold]HTML[/bold] file.

    Supports both LLM-assisted enrichment (default) and regex-only mode
    (\[dim]--no-llm[/dim]).  Results can be exported as JSON, CSV, or displayed
    as a formatted table.

    [bold]Examples:[/bold]

      [green]paper-miner extract study.pdf --format json -o results.json[/green]

      [green]paper-miner extract paper.html --no-llm --format table[/green]

      [green]paper-miner extract paper.pdf --model gpt-4o --base-url http://localhost:8080/v1[/green]
    """
    suffix = file.suffix.lower()
    if suffix not in (".pdf", ".html", ".htm"):
        console.print(
            f"[bold red]Error:[/bold red] Unsupported file type {suffix!r}. "
            "Only .pdf, .html, and .htm files are supported."
        )
        raise typer.Exit(code=1)

    # Resolve output format.
    resolved_fmt: OutputFormat
    if fmt is not None:
        resolved_fmt = fmt
    else:
        inferred = _infer_format_from_path(output)
        if inferred is not None:
            resolved_fmt = inferred
        elif output:
            resolved_fmt = OutputFormat.json
        else:
            resolved_fmt = OutputFormat.table

    # Determine source label.
    effective_source = source_label if source_label else file.name

    if use_llm and not api_key:
        import os as _os
        if not _os.environ.get("OPENAI_API_KEY"):
            console.print(
                "[bold yellow]Warning:[/bold yellow] LLM mode is enabled but no API key "
                "was found.  Set OPENAI_API_KEY or pass --api-key, or use --no-llm."
            )

    records: list[NumericRecord] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        disable=quiet,
    ) as progress:
        # Step 1: Ingest.
        ingest_task = progress.add_task(f"Ingesting {file.name} …", total=None)
        try:
            if suffix == ".pdf":
                from paper_miner.ingest import ingest_pdf

                chunks = ingest_pdf(str(file))
            else:
                from paper_miner.ingest import ingest_html

                chunks = ingest_html(str(file), is_file=True)
        except FileNotFoundError as exc:
            console.print(f"[bold red]File not found:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc
        except ValueError as exc:
            console.print(f"[bold red]Ingestion error:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc
        progress.update(ingest_task, completed=True, total=1)

        if not chunks:
            console.print(
                "[yellow]Warning:[/yellow] No text could be extracted from the document."
            )
            _write_output([], resolved_fmt, output)
            raise typer.Exit(code=0)

        # Step 2: Regex extraction.
        extract_task = progress.add_task(
            "Extracting numeric candidates …", total=len(chunks)
        )
        from paper_miner.extractor import extract_candidates

        candidates: list[NumericRecord] = []
        for chunk in chunks:
            candidates.extend(
                extract_candidates(chunk, source=effective_source)
            )
            progress.advance(extract_task)

        if not candidates:
            if not quiet:
                console.print(
                    "[yellow]No numeric candidates found in the document.[/yellow]"
                )
            _write_output([], resolved_fmt, output)
            raise typer.Exit(code=0)

        # Step 3: Optional LLM enrichment.
        if use_llm:
            llm_task = progress.add_task(
                f"Enriching {len(candidates)} candidates via LLM …", total=None
            )
            try:
                from paper_miner.llm_parser import parse_candidates

                records = parse_candidates(
                    candidates,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                )
            except (ValueError, ImportError) as exc:
                console.print(f"[bold red]LLM error:[/bold red] {exc}")
                raise typer.Exit(code=1) from exc
            except RuntimeError as exc:
                console.print(
                    f"[bold yellow]LLM enrichment failed:[/bold yellow] {exc}\n"
                    "Falling back to regex-only results."
                )
                records = candidates
            progress.update(llm_task, completed=True, total=1)
        else:
            records = candidates

    # Print summary to stderr.
    if not quiet:
        _print_summary(records, source=effective_source)

    # Write output.
    _write_output(records, resolved_fmt, output)


@app.command("text")
def text_command(
    content: str = typer.Argument(
        ...,
        help=(
            "Plain text to mine for numerical data.  "
            "Pass '-' to read from stdin."
        ),
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path.  When omitted, results are printed to stdout.",
        metavar="PATH",
    ),
    fmt: Optional[OutputFormat] = typer.Option(
        None,
        "--format",
        "-f",
        help=(
            "Output format: json, csv, or table.  Defaults to 'table' when no "
            "--output path is given."
        ),
        case_sensitive=False,
    ),
    use_llm: bool = typer.Option(
        True,
        "--llm/--no-llm",
        help="Enable or disable LLM-assisted enrichment.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="OPENAI_API_KEY",
        help="OpenAI-compatible API key.",
        show_default=False,
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        help="Custom base URL for an OpenAI-compatible API endpoint.",
        show_default=False,
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        "-m",
        help="LLM model identifier.",
    ),
    source: Optional[str] = typer.Option(
        None,
        "--source",
        help="Source label to attach to every output record.",
        show_default=False,
    ),
    section: Optional[str] = typer.Option(
        None,
        "--section",
        help="Document section label to attach to every output record.",
        show_default=False,
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress progress and summary output.",
    ),
) -> None:
    """Extract numerical data from [bold]plain text[/bold] passed as an argument or via stdin.

    Pass [green]-[/green] as the CONTENT argument to read from stdin.

    [bold]Examples:[/bold]

      [green]paper-miner text "LDL was reduced by 32.4 mg/dL (p < 0.001)." --no-llm[/green]

      [green]echo "BMI was 27.6 kg/m²." | paper-miner text - --no-llm --format json[/green]
    """
    # Read from stdin if content is "-".
    if content == "-":
        if not quiet:
            console.print("[dim]Reading from stdin …[/dim]")
        content = sys.stdin.read()

    if not content.strip():
        console.print("[yellow]Warning:[/yellow] No text provided.")
        raise typer.Exit(code=0)

    # Resolve output format.
    resolved_fmt: OutputFormat
    if fmt is not None:
        resolved_fmt = fmt
    else:
        inferred = _infer_format_from_path(output)
        if inferred is not None:
            resolved_fmt = inferred
        elif output:
            resolved_fmt = OutputFormat.json
        else:
            resolved_fmt = OutputFormat.table

    records: list[NumericRecord] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        disable=quiet,
    ) as progress:
        extract_task = progress.add_task("Extracting numeric candidates …", total=None)
        from paper_miner.extractor import extract_candidates

        candidates = extract_candidates(content, source=source, section=section)
        progress.update(extract_task, completed=True, total=1)

        if not candidates:
            if not quiet:
                console.print("[yellow]No numeric candidates found in the text.[/yellow]")
            _write_output([], resolved_fmt, output)
            raise typer.Exit(code=0)

        if use_llm:
            llm_task = progress.add_task(
                f"Enriching {len(candidates)} candidates via LLM …", total=None
            )
            try:
                from paper_miner.llm_parser import parse_candidates

                records = parse_candidates(
                    candidates,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                )
            except (ValueError, ImportError) as exc:
                console.print(f"[bold red]LLM error:[/bold red] {exc}")
                raise typer.Exit(code=1) from exc
            except RuntimeError as exc:
                console.print(
                    f"[bold yellow]LLM enrichment failed:[/bold yellow] {exc}\n"
                    "Falling back to regex-only results."
                )
                records = candidates
            progress.update(llm_task, completed=True, total=1)
        else:
            records = candidates

    if not quiet:
        _print_summary(records, source=source or "<inline text>")

    _write_output(records, resolved_fmt, output)


@app.command("version")
def version_command() -> None:
    """Display the current [bold cyan]paper_miner[/bold cyan] version."""
    out_console.print(
        Panel(
            Text(f"paper_miner  v{__version__}", style="bold green"),
            title="[cyan]Version[/cyan]",
            expand=False,
        )
    )


# ---------------------------------------------------------------------------
# Entry-point guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    app()
