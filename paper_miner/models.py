"""Data models for paper_miner.

Defines the NumericRecord dataclass that serves as the core data contract
for all numeric findings extracted from scientific documents. Every extracted
numeric measurement, statistic, or percentage is represented as a NumericRecord
before being exported to JSON or CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class NumericRecord:
    """Represents a single extracted numerical finding from a scientific document.

    This is the core data contract of paper_miner. Instances are created by
    the LLM parser (or the regex extractor in no-LLM mode) and consumed by
    the exporter for JSON/CSV output.

    Attributes
    ----------
    value : str
        The raw numeric value as it appears in the source text (e.g., ``"12.5"``,
        ``"0.003"``, ``"95"``). Stored as a string to preserve formatting such
        as leading zeros or scientific notation.
    unit : str
        The physical or statistical unit associated with the value (e.g.,
        ``"mmHg"``, ``"%"``, ``"mg/dL"``, ``"none"`` when unitless).
    data_type : str
        Classification of the numeric finding. Common values include
        ``"measurement"``, ``"percentage"``, ``"p-value"``, ``"confidence_interval"``,
        ``"mean"``, ``"median"``, ``"standard_deviation"``, ``"count"``,
        ``"ratio"``, and ``"other"``.
    context : str
        The surrounding sentence or passage from the source document that
        contains the numeric value. Provides provenance for the finding.
    relationship : str
        A brief description of what experimental condition, variable, or
        outcome the numeric value is related to (e.g.,
        ``"blood pressure reduction after treatment"``).
    section : Optional[str]
        The document section where the finding was located (e.g.,
        ``"Results"``, ``"Methods"``, ``"Abstract"``). May be ``None`` if
        section information is unavailable.
    confidence : Optional[float]
        A 0.0–1.0 confidence score assigned by the LLM indicating how
        certain it is about the classification. ``None`` when the LLM is
        not used.
    raw_text : str
        The exact substring from the source document that triggered this
        record (e.g., ``"12.5 mmHg"``, ``"p=0.003"``, ``"95% CI"``).
    source : Optional[str]
        Identifier for the source document (e.g., a filename or URL).
        ``None`` when not provided.
    """

    value: str
    unit: str
    data_type: str
    context: str
    relationship: str
    raw_text: str
    section: Optional[str] = field(default=None)
    confidence: Optional[float] = field(default=None)
    source: Optional[str] = field(default=None)

    def to_dict(self) -> dict:
        """Serialize this record to a plain dictionary.

        Returns
        -------
        dict
            A dictionary representation of the record suitable for JSON
            serialization or CSV row construction. All fields are included;
            optional fields that are ``None`` are represented as ``None``.
        """
        return asdict(self)

    def to_csv_row(self) -> dict[str, str]:
        """Serialize this record to a flat string dictionary for CSV export.

        Optional fields that are ``None`` are converted to empty strings so
        that the CSV writer does not emit the literal string ``"None"``.

        Returns
        -------
        dict[str, str]
            A dictionary mapping field names to their string representations.
        """
        raw = asdict(self)
        return {
            key: "" if value is None else str(value)
            for key, value in raw.items()
        }

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        """Return the ordered list of CSV column names.

        Returns
        -------
        list[str]
            Field names in the canonical export order.
        """
        return [
            "value",
            "unit",
            "data_type",
            "context",
            "relationship",
            "raw_text",
            "section",
            "confidence",
            "source",
        ]

    def __repr__(self) -> str:
        """Return a concise developer-facing string representation."""
        return (
            f"NumericRecord(value={self.value!r}, unit={self.unit!r}, "
            f"data_type={self.data_type!r}, source={self.source!r})"
        )
