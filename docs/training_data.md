# Synthetic Training Data

## Overview

No annotated real clinical records were available during development — IRB
approval and data-sharing agreements were out of scope. Synthetic data generated
by a large language model was therefore the only feasible path for producing
labelled training examples at scale. The pipeline prompts an LLM with patient
profiles and few-shot templates to generate French clinical documents
(anatomopathology reports, consultation notes, RCP summaries) together with
character-level entity annotations. The primary trade-off is that LLM-generated
text differs from real records in vocabulary, structure, and error patterns,
which is the main source of performance degradation on live hospital data.

---

## Generation Pipeline Architecture

```
bright_models/generated_data/
│
├── config/
│   ├── fields.py              111 field descriptions (French) + 21 semantic groups
│   └── prompts/
│       ├── system_common.txt  Shared system instructions for all document types
│       ├── anapath.txt        Anatomopathology report persona + template
│       ├── consultation.txt   Consultation note persona + template
│       └── rcp.txt            RCP (multidisciplinary meeting) persona + template
│
├── data/
│   └── few_shot/              Few-shot examples per document type
│       ├── anapath_example_1.json
│       ├── consultation_example_1.json
│       └── rcp_example_1.json
│
└── pipeline/
    ├── llm_client.py          Unified LLM interface (vLLM / transformers / API)
    ├── step2_generate.py      LLM call: patient profile + prompt → synthetic doc + annotations
    ├── step3_resolve.py       Resolve entity labels to character offsets in the text
    └── step4_filter.py        6-level quality cascade
```

---

## How to Run the Generation Pipeline

The three pipeline steps are **library modules**, not standalone CLI scripts.
They are orchestrated via a `PipelineConfig` object and called from a top-level
training script. The key configuration parameters are:

```python
from bright_models.generated_data.pipeline import step2_generate, step3_resolve, step4_filter
from bright_models.generated_data.pipeline.llm_client import LLMClient

config = PipelineConfig(
    llm_provider="local",        # "local" (vLLM/transformers), or an API provider
    batch_size=32,               # documents per LLM call
    few_shot_dir="data/few_shot/",
    prompts_dir="config/prompts/",
    fuzzy_threshold=85,          # minimum rapidfuzz ratio for span resolution (step 3)
    min_resolution_rate=0.7,     # minimum annotation resolution rate (step 3)
    length_limits={              # per-document-type text length bounds (step 4)
        "anapath": {"min": 200, "max": 6000},
        "consultation": {"min": 200, "max": 6000},
        "rcp": {"min": 200, "max": 6000},
    },
    dedup_similarity=0.85,       # MinHash LSH threshold for near-duplicate removal (step 4)
)

llm = LLMClient(provider=config.llm_provider)

# Step 2: Generate raw synthetic documents + structured annotations
raw_documents = step2_generate.generate_documents(
    profiles=patient_profiles,   # list of patient profile dicts
    config=config,
    llm_client=llm,
    checkpoint_mgr=checkpoint_mgr,
)

# Step 3: Resolve entity labels to character offsets
resolved_documents = step3_resolve.resolve_all_spans(
    raw_documents=raw_documents,
    config=config,
)

# Step 4: Filter for quality
filtered_documents = step4_filter.run_filter_cascade(
    documents=resolved_documents,
    config=config,
)
```

The final `filtered_documents` list is the training dataset passed to
`bright_models/gliner_bright/training_gliner.py` or
`bright_models/eds_bright/training_eds.py`.

---

## Quality Filtering Levels

`step4_filter.py` applies six filters in sequence. A document must pass all
six to be included in the training set.

| Level | Filter | What it checks |
|-------|--------|----------------|
| 1 | **Resolution Rate** | The fraction of annotated entity spans that were successfully resolved to character offsets in step 3. Documents below `config.reject_resolution_below` are rejected; those below `config.review_resolution_below` are flagged for review. |
| 2 | **Clinical Coherence** | WHO 2021 biological coherence rules (e.g., IDH-wildtype glioblastoma cannot also be 1p/19q co-deleted). Documents that violate hard constraints are rejected. |
| 3 | **Document Length** | Text length must fall within the per-document-type bounds in `config.length_limits`. Documents outside the bounds are flagged for review. |
| 4 | **Profile-to-Annotation Match** | Annotation values are compared against the original patient profile using 80% fuzzy matching. Documents where the LLM substantially invented values not present in the profile are rejected or flagged. |
| 5 | **Near-Duplicate Detection** | MinHash LSH identifies clusters of near-identical documents (similarity ≥ `config.dedup_similarity`). Only the document with the highest resolution rate from each cluster is kept. |
| 6 | **Language Quality** | Checks for a minimum of 5 French medical vocabulary terms, rejects documents with > 10% English function words, and detects prompt leakage (verbatim prompt fragments in the generated text). |

---

## Adding Few-Shot Examples for New Fields

Few-shot examples live in `bright_models/generated_data/data/few_shot/`. One
JSON file exists per document type:

```
data/few_shot/
├── anapath_example_1.json
├── consultation_example_1.json
└── rcp_example_1.json
```

Each JSON file has the following structure:

```json
{
  "document_type": "anapath",
  "text": "...<full synthetic French document text>...",
  "annotations": {
    "field_name": "extracted_value",
    "another_field": "another_value"
  }
}
```

To add a few-shot example for a new field:

1. Identify the document type(s) from which the field will be extracted
   (check `FEATURE_ROUTING` in `schema.py`).
2. Add a representative annotation for the new field to the relevant JSON file,
   or create a new example file and register it in the `FEW_SHOT_MAP` constant
   in `fields.py`.
3. The prompt templates in `config/prompts/` reference these examples via the
   `few_shot_dir` config path. When the LLM is prompted, the examples are
   included verbatim in the user message to illustrate the expected output format.

For new fields that represent a clinical concept not covered by existing examples,
write at least 2–3 varied examples to reduce overfitting to a single phrasing
pattern.

---

## Important Caveats

**Synthetic ≠ Real**: LLM-generated documents differ from real clinical records
in vocabulary, structure, and error patterns. This is the primary source of
performance degradation on real data.

**Hallucinations**: LLMs may generate plausible-but-incorrect medical statements.
The quality filter catches structural errors but cannot detect clinical inaccuracies.

**Annotation noise**: Span resolution (step 3) has approximately a 5–10% error
rate on complex multi-word entities. Review generated data before using it for
training, especially for fields with long or hyphenated values.

**Language model dependency**: Generation quality depends heavily on the LLM
used. Larger models produce more realistic documents and more accurate
annotations. Switching LLM providers requires re-tuning `config.fuzzy_threshold`
and `config.min_resolution_rate`.
