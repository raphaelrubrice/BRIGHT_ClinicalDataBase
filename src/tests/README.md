# Test Suite — Skipped Tests

**Current status:** 726 passed, 39 skipped, 0 failures.

Run with: `pytest src/tests/ -v --tb=short`

---

## Skipped: requires `edsnlp` (7 modules — all tests in each)

These test modules import `edsnlp` at module level. `edsnlp` requires PyTorch and the
full ML environment (installed via `scripts/setup.sh` or `scripts/setup.ps1`). They are
skipped automatically via `pytest.importorskip("edsnlp")` when the library is not present.

| Module | Description |
|--------|-------------|
| `test_benchmark.py` | Full evaluation benchmark against gold-standard data |
| `test_date_extractor.py` | Context-aware consultation date extraction via `eds.dates` |
| `test_eds_extractor.py` | EDS-NLP entity extraction |
| `test_patient_timeline.py` | End-to-end patient timeline builder |
| `test_pipeline.py` | End-to-end extraction pipeline |
| `test_row_duplicator.py` | Multiple-event row splitting |
| `test_temporal_aggregation.py` | Forward-fill and conflict resolution across documents |

---

## Skipped: LLM fallback removed from `DocumentClassifier` (19 tests in `test_document_classifier.py`)

The `DocumentClassifier` was simplified to keyword-only classification; the Ollama LLM
fallback and its helper functions (`_parse_llm_response`, `_truncate_to_tokens`) were
removed. Tests for that functionality are kept but marked skip to preserve intent.

- `TestLLMFallback` (6 tests) — `ollama_client` parameter no longer exists
- `TestLLMResponseParsing` (10 tests) — `_parse_llm_response` removed
- `TestTextTruncation` (3 tests) — `_truncate_to_tokens` removed

---

## Skipped: Phase 0 placeholders (13 tests)

These tests were scaffolded as placeholders in an early development phase and have not
yet been implemented. They are marked with `pytest.mark.skip(reason="Phase 0 placeholder")`.

| Module | Tests skipped |
|--------|---------------|
| `test_ops.py` | 4 (init_database, load_save_roundtrip, append_row, extract_consult_date_regex) |
| `test_pseudonymizer.py` | 3 (pseudonymize_text, deterministic_hashing, long_document_chunking) |
| `test_security.py` | 2 (salt_creation, ope_encryption) |
| `test_text_extraction.py` | 2 (extract_text_from_pdf, scanned_pdf_detection) |
| `test_utils.py` | 2 (model_resolution, eds_nlp_registry_setup) |
