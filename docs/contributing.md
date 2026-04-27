# Contributing to BRIGHT

## Development Environment Setup

Complete the installation steps in the project README first (conda environment,
EDS-NLP, eds-pseudo clone, spaCy models, HuggingFace login). Then verify the
setup:

```bash
conda activate bright_db
cd BRIGHT_ClinicalDataBase

# Run the full test suite
pytest src/tests/ -v

# Verify the CLI is working
python main.py --help
python main.py pseudo --help
python main.py extract --help
```

All tests should pass (tests that require external data files, PDFs or a gold
standard CSV, are decorated with `@pytest.mark.skip` and will be skipped
automatically).

---

## Test Suite Overview

The test suite lives in `src/tests/` and comprises 24 pytest modules.

| Test Module | What it Tests |
|-------------|---------------|
| `test_schema.py` | All 55 bio + 56 clinical fields defined in `schema.py`; `ControlledVocab` constants; `ExtractionValue` validation; `FEATURE_ROUTING` completeness; JSON schema generation |
| `test_pipeline.py` | End-to-end pipeline with rules + EDS on sample documents (anapath, molecular_report, consultation, rcp, radiology); feature routing per document type |
| `test_controlled_extractor.py` | `ControlledExtractor` fuzzy matching for chromosomal, IHC, molecular, and binary fields; positive and negative cases per field |
| `test_date_extractor.py` | Date regex extraction; format parsing; consultation-date context filtering |
| `test_rule_extraction.py` | Rule-based extractors for dates, IHC, molecular, chromosomal, binary, numerical, amplification, and fusion fields |
| `test_new_extractors.py` | Phase A/B/C rule extractors: `sexe`, `tumeur_lateralite`, `evol_clinique`, `type_chirurgie`, `classification_oms`, `chimios`, `tumeur_position`, `diag_histologique` |
| `test_eds_extractor.py` | EDS-NLP CRF model extraction: demographics, dates, IHC, molecular fields |
| `test_negation.py` | `AssertionAnnotator` negation/hypothesis/history detection via regex; span-level assertion classification |
| `test_section_detector.py` | Document section segmentation; IHC/molecular/microscopy/conclusion detection; fallback to `full_text`; section-to-feature mapping coverage |
| `test_document_classifier.py` | Keyword-based classification for all 5 document types; ambiguity detection; LLM fallback mock |
| `test_validation.py` | Controlled-vocabulary enforcement; value normalization; `_is_value_valid()` helper |
| `test_metrics.py` | `compute_per_feature_metrics()` and `compute_aggregate_metrics()`; TP/TN/FP/FN/alteration counts; F1 computation |
| `test_benchmark.py` | `run_benchmark()` with a dummy pipeline; metrics aggregation; CSV output |
| `test_gold_standard.py` | Loading and saving gold-standard JSON files; annotation filtering by schema |
| `test_patient_timeline.py` | Patient timeline assembly from multiple documents; row duplication; temporal forward-fill |
| `test_temporal_aggregation.py` | Forward-fill, specimen-reset, conflict resolution by document-type priority; 5-document synthetic patient case |
| `test_row_duplicator.py` | Multi-event row splitting for surgery dates, chemo lines, radiotherapy courses, and progression events |
| `test_quality_regression.py` | Regression tests for 4 specific bug fixes: date context assignment, pseudo-token rejection, consultation bio routing, pseudo-birthdate patterns |
| `test_similarity.py` | Fuzzy string matching utility; similarity threshold tuning |
| `test_text_normalisation.py` | Text normalization helpers: accent removal, case normalization, whitespace cleanup |
| `test_pseudonymizer.py` | Pseudonymization utilities |
| `test_ops.py` | Database CRUD operations |
| `test_security.py` | Security checks (salt generation, sidecar persistence) |
| `test_utils.py` | Utility functions |

Tests that depend on external files (real PDFs, gold-standard CSVs) are marked
`@pytest.mark.skip` and are not run by default.

---

## Code Style

- **Python 3.12+**: type hints are required for all public functions and methods.
- **Pydantic v2**: use Pydantic models for all data structures that cross module
  boundaries. Do not use bare dicts for structured data.
- **No `print()`**: use the `logging` module for all diagnostic output. The
  pipeline has a `transparent=True` flag for verbose per-field logging during
  debugging.
- **Function size**: keep functions under approximately 50 lines. If a function
  is growing longer, split it into focused helpers.
- **Tests**: new extraction logic must ship with tests in `src/tests/`. Cover at
  least one positive case (field extracted correctly) and one negative case
  (field not extracted when absent).

---

## Adding a New Field

See [`docs/adding_fields.md`](adding_fields.md) for a complete step-by-step
guide covering schema, routing, extraction logic, training data, validation,
and tests.

---

## Pushing Trained Models to Hugging Face Hub

After training new model versions in `bright_models/`, push them to the Hub:

```bash
cd bright_models
python push_to_hub.py <your_hf_username> <output_dir>
```

**Arguments:**
- `<your_hf_username>`, your HuggingFace username (positional, required)
- `<output_dir>`, path to the directory containing trained model subdirectories
  (positional, required; typically `./output`)

**Authentication:** the script uses `HfApi()` from the `huggingface_hub` library.
Set `HF_TOKEN` in your environment or run `huggingface-cli login` beforehand.

The script creates (or updates) repositories named
`{username}/bright-{method}-{group}`, where:
- `method` is `gliner` (GLiNER2 checkpoints) or `eds` (edsnlp CRF pipelines)
- `group` is one of the 10 semantic groups: `diagnosis`, `ihc`, `molecular`,
  `chromosomal`, `demographics`, `first_symptoms`, `current_symptoms`,
  `radiology`, `tumour_location`, `treatment`

The script searches for model artifacts in:
- GLiNER: `output/{group}/gliner/best_merged_crt/` or `best_merged/`
- EDS: `output/{group}/eds/model-balanced-best/` or `model-best/`

---

## Common Issues

### "EDS-NLP model not found" or `eds-pseudo` import error

The `eds-pseudo` pipeline must be cloned manually from GitHub, it is not on
PyPI. Re-run the setup script, or re-clone and install it:

```bash
git clone https://github.com/aphp/eds-pseudo.git
pip install -e eds-pseudo/
```

If the model artifacts are in a non-default location, pass the path explicitly:

```bash
python main.py pseudo --db db.csv --pdfs ./pdfs/ --eds_path /path/to/eds-pseudo/
```

The CLI resolves the EDS model via `resolve_eds_model_path()` in
`src/database/utils.py`, which checks `--eds_path`, then a bundled cache in
`src/database/hf_cache/`, then `src/ui/hf_cache/`.

### "Salt file missing" or pseudonym inconsistency across runs

The secret salt is stored in a sidecar file at `<db_path>.csv.pseudonym_salt`
(next to the CSV). It is created automatically by `get_or_create_salt_file()` on
the first run.

If the file was deleted or moved, re-pseudonymizing will produce **different
tokens** for the same entities, breaking cross-run linking. To restore
consistency:

- Restore the sidecar file from backup (preferred).
- Alternatively, set the `PSEUDO_SALT` environment variable to the original
  salt string before running, the code falls back to this env var if the
  sidecar is missing.
- If neither is possible, you must re-pseudonymize all documents from scratch
  and regenerate any gold-standard annotations that referenced the old tokens.

### "CUDA out of memory" during extraction

The extraction pipeline does not require a GPU. The 10 fine-tuned edsnlp models
use CRF inference on CPU. If you are running the GLiNER2 training pipeline in
`bright_models/` and encounter OOM errors:

- Reduce `config.batch_size` in the training config.
- Enable CPU-only mode by setting `CUDA_VISIBLE_DEVICES=""` in your environment.
- For extraction (not training), the pipeline runs on CPU by default and will not
  use a GPU even if one is present.
