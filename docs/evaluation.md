# Evaluation and Benchmarking

## Overview

Evaluating extraction quality requires a **gold standard** — a set of documents
whose fields have been manually annotated by a clinician or trained annotator.
The benchmark compares the pipeline's output against this gold standard and
reports per-field precision, recall, and F1 scores, along with hallucination,
omission, and alteration rates. There is no automatic ground truth: someone must
annotate real documents for evaluation to be meaningful.

---

## Gold Standard Format

The gold standard consists of **JSON files** (one per document), stored in a
single directory. The format is:

```json
{
  "document_id": "doc_123",
  "patient_id": "pat_456",
  "raw_text": "... pseudonymized document text ...",
  "annotations": {
    "diag_histologique": "glioblastome",
    "grade": "4",
    "ihc_idh1": "negatif",
    "mol_tert": "mute",
    "date_chir": "2024-03-15"
  }
}
```

- `document_id` and `patient_id` are arbitrary string identifiers.
- `raw_text` must be the pseudonymized text (the same text the pipeline will
  receive).
- `annotations` is a flat dict mapping field names (as defined in `schema.py`)
  to their correct string values. Fields absent from the dict are treated as
  `None` (not present in this document).
- Annotation values are strings; numeric fields (`grade`, `chm_cycles`, etc.)
  are stored as their string representation.

### Converting from annotation tool exports

If annotations were collected in `REQ_BIO.csv` and `REQ_CLINIQUE.csv` (the
semicolon-delimited transposed CSVs produced by the project's annotation
spreadsheet), use the conversion script:

```bash
python scripts/convert_annotations_to_gold.py --db path/to/clinical_db_pseudo_only.csv
```

The `--db` argument is optional; if omitted the script uses its default paths
(see the script's argparse help). The script writes two output sets:

- `data/gold_standard/lines/` — one JSON per (patient × visit) line
- `data/gold_standard/aggregates/` — one JSON per patient (all visits merged)

Use the `lines/` directory for per-document benchmarking and `aggregates/` for
patient-level evaluation.

---

## Running the Benchmark

### Python API

```python
from src.evaluation.benchmark import run_benchmark
from src.extraction.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline(use_rules=True, use_eds=True)

metrics_df = run_benchmark(
    gold_standard_dir="data/gold_standard/lines/",
    pipeline=pipeline,
    output_dir="results/benchmark_run_1/",
)
```

`run_benchmark` loads all JSON files from `gold_standard_dir`, runs the pipeline
on each document's `raw_text`, computes metrics, and writes four CSV files to
`output_dir`:

| Output file | Contents |
|-------------|----------|
| `benchmark_metrics.csv` | Per-field TP / FP / FN / alteration / P / R / F1 |
| `category_metrics.csv` | Macro-average F1 per field category (diagnosis, IHC, molecular, etc.) |
| `tier_category_metrics.csv` | F1 broken down by extraction tier (date / controlled / rules / EDS / HF) |
| `error_analysis.csv` | Each FP / FN / alteration with root-cause classification |

### Ablation runner

To compare the three pipeline modes in one pass, use the ablation script:

```bash
# Rules-only (no ML models)
python scripts/run_pipeline_ablation.py --mode rule \
  --data-dir data/gold_standard/lines/ \
  --out-dir tmp/ablation/

# ML-only (no regex rules)
python scripts/run_pipeline_ablation.py --mode ml \
  --data-dir data/gold_standard/lines/ \
  --out-dir tmp/ablation/

# Hybrid (rules + ML, default pipeline behaviour)
python scripts/run_pipeline_ablation.py --mode both \
  --data-dir data/gold_standard/lines/ \
  --out-dir tmp/ablation/
```

Additional flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--max-docs N` | all | Cap the number of documents processed |
| `--no-negation` | off | Disable negation detection |
| `--jobs N` | 1 | Parallel workers (rule mode only; -1 = auto) |
| `--groups g1 g2` | all 10 | Subset of HF model groups to load (ml / both modes) |
| `--local-model-dir PATH` | Hub | Use locally cached model directories instead of Hub |

Each run writes `results_{mode}_{timestamp}.json` (per-document extractions with
intermediate branch results) and `run_{mode}_{timestamp}.log` (full debug log)
to `--out-dir`.

---

## Metrics Explained

### What counts as "correct"

**Exact match** (normalized): both the predicted and gold values are lowercased,
accents are removed, integers and floats are normalized, and dates are unified to
`YYYY-MM-DD`. If the normalized strings are identical, the extraction is a TP.

**Fuzzy match** (85% threshold): for the 12 free-text fields listed below, a
predicted value that scores ≥ 85% rapidfuzz similarity against the gold value
also counts as a TP. These fields contain diagnostic phrases, drug names, or
place names where minor spelling and accent variations are expected:
`diag_histologique`, `diag_integre`, `tumeur_position`, `activite_professionnelle`,
`chimios`, `localisation_chir`, `localisation_radiotherapie`, `neuroncologue`,
`neurochirurgien`, `radiotherapeute`, `infos_deces`, `autre_trouble`.

**Date coarsening**: dates are compared at their shared granularity. A predicted
value of `2024` matches a gold value of `2024-03-15` because the prediction is
correct at the year level.

### Metric formulas

For each field across all evaluated documents:

```
TP               — predicted value matches gold (exact or fuzzy)
TN               — both predicted and gold are None
FP_hallucination — predicted has a value, gold is None
FN_omission      — predicted is None, gold has a value
alteration       — both have values but they do not match
                   (counts 0.5 toward FP and 0.5 toward FN)

precision = TP / (TP + FP_hallucination + 0.5 × alteration)
recall    = TP / (TP + FN_omission     + 0.5 × alteration)
F1        = 2 × precision × recall / (precision + recall)

hallucination_rate = FP_hallucination / (FP_hallucination + TN)
omission_rate      = FN_omission      / (TP + FN_omission + alteration)
alteration_rate    = alteration       / (TP + FN_omission + alteration)
```

### Error root-cause classification

Each FP / FN / alteration in `error_analysis.csv` is annotated with one of six
root causes:

| Root cause | Description |
|------------|-------------|
| `pseudo_token` | Extracted a pseudonymization token (e.g., `[NOM_A3F2]`) instead of a real value |
| `date_mismatch` | Correct date type but wrong temporal context (e.g., date of progression vs. date of surgery) |
| `routing_omission` | Field not in `FEATURE_ROUTING` for this document type; extractor never ran |
| `truncation` | Long document was chunked; relevant passage fell at a chunk boundary |
| `hallucination` | Model produced a value with no textual basis in the document |
| `format_mismatch` | Correct semantic content but wrong format (e.g., `03/2024` vs. `2024-03-01`) |

---

## Ablation Study

The three modes isolate the contribution of each pipeline layer:

| Mode flag | Pipeline configuration | Useful for |
|-----------|----------------------|------------|
| `--mode rule` | Rules only (date regex + controlled vocab + EDS rules, no ML) | Measuring the baseline achievable without training data |
| `--mode ml` | ML only (10 HF models, no rules) | Measuring raw model quality |
| `--mode both` | Hybrid (rules + ML; HF overrides rules on ~50 fields) | Production performance |

A healthy result typically shows: `both` > `ml` > `rule` on free-text fields, and
`rule` ≈ `both` > `ml` on date and controlled-vocab fields (where rules are
precise and models add little).

---

## Interpreting Results

**High precision + low recall**: the extractor is conservative — it only fires
when highly confident, missing many instances. Consider lowering similarity
thresholds in the controlled extractor or adding more training examples.

**High recall + low precision**: the extractor fires too broadly, generating
many false positives. Check whether marker terms in `controlled_vocab_data.py`
are too general, or whether the ML model is overfitting to superficial patterns.

**Fields with controlled vocabularies** (IHC status, molecular status,
chromosomal) tend to have higher precision than free-text fields (diagnostic
descriptions, tumour location) because the value space is finite.

**Synthetic vs. real data gap**: models were trained on LLM-generated documents.
On real hospital records, expect F1 scores 10–20 points lower than on held-out
synthetic test sets, primarily due to vocabulary shift and OCR artefacts. The
`error_analysis.csv` root-cause breakdown helps distinguish synthetic-training
errors (hallucination, format_mismatch) from structural pipeline errors
(routing_omission, truncation).
