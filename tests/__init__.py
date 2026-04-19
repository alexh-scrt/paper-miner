"""Test suite for paper_miner.

This package contains unit tests for all core modules of paper_miner:

- test_extractor.py  : Tests for regex-based numeric candidate extraction.
- test_ingest.py     : Tests for PDF and HTML ingestion functions.
- test_exporter.py   : Tests for JSON and CSV serialization of NumericRecord objects.
- test_llm_parser.py : Tests for LLM-assisted parsing using mocked API responses.

Tests are designed to run without live API keys; LLM calls are fully mocked
to ensure CI/CD compatibility.
"""
