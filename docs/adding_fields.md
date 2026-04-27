# Adding a New Clinical Field

Adding a field to BRIGHT requires consistent changes in 4–5 places across two
sub-projects. If any step is skipped the field will be silently ignored: the
pipeline will route around it, extract nothing, and write `None` to the output
CSV without raising an error.

---

## Step 1, Define the field in the schema (`src/extraction/schema.py`)

Open `schema.py` and locate either `BiologicalFeatures` or `ClinicalFeatures`
(both are Pydantic v2 models). Add:

```python
my_new_field: Optional[ExtractionValue] = None
```

All field values are wrapped in `ExtractionValue`, which carries the extracted
string value plus provenance metadata (`extraction_tier`, `source_span`,
`confidence`, `vocab_valid`, `flagged`).

If the field has a fixed set of allowed values, add a constant to `ControlledVocab`
(the namespace class defined earlier in `schema.py`):

```python
# In ControlledVocab, add a new class variable:
MY_NEW_STATUS: set[str] = {"value_a", "value_b", "NA"}
```

For simple yes/no fields, reuse `ControlledVocab.BINARY = {"oui", "non", "NA"}`.

**Existing vocabulary constants** (reuse if appropriate):

| Constant | Values |
|----------|--------|
| `BINARY` | `oui`, `non`, `NA` |
| `IHC_STATUS` | `positif`, `negatif`, `maintenu`, `NA` |
| `MOLECULAR_STATUS` | `wt`, `mute`, `autre`, `NA` |
| `CHROMOSOMAL` | `gain`, `perte`, `perte partielle`, `NA` |
| `METHYLATION` | `methyle`, `non methyle`, `NA` |
| `GRADE` | `1`, `2`, `3`, `4`, `autre`, `NA` |
| `SURGERY_TYPE` | `exerese complete`, `exerese partielle`, `biopsie`, `en attente`, `autre`, `NA` |

---

## Step 2, Add to `FEATURE_ROUTING` (`src/extraction/schema.py`)

`FEATURE_ROUTING` is a dict that maps each document type to the set of fields
that should be extracted from it. Locate the dict and add `"my_new_field"` to
the appropriate document type(s):

```python
FEATURE_ROUTING = {
    "anapath": {
        "bio": [..., "my_new_field"],   # biological field from pathology reports
        "clinique": [],
    },
    "consultation": {
        "bio": [...],
        "clinique": [..., "my_new_field"],  # clinical field from consultations
    },
    ...
}
```

Available document types: `anapath`, `molecular_report`, `consultation`, `rcp`,
`radiology`. A field can appear in multiple document types.

If a field is missing from `FEATURE_ROUTING`, the pipeline will never attempt to
extract it, regardless of which extractor it is wired to.

---

## Step 3, Add extraction logic

Choose the appropriate extraction tier based on the field's nature.

### Date fields

Add a regex pattern and context keywords to `src/extraction/date_extractor.py`.
Follow the existing pattern: each date field has a list of context keywords
(searched within ±200–300 characters of any detected date) that determine which
date gets assigned to which field.

### Controlled-vocabulary fields

Add the field's marker terms and candidate values to
`src/extraction/controlled_vocab_data.py`. The structure is:

```python
"my_new_field": {
    "markers": ["marker term a", "marker term b"],
    "candidates": MY_NEW_STATUS,   # the ControlledVocab set from Step 1
}
```

The `ControlledExtractor` scans each section for marker terms, then applies
fuzzy matching against the candidate set.

### Complex free-text fields (ML)

Fields that require understanding context or phrasing belong in the ML tier.
You can add a regex fallback in `src/extraction/rule_extraction.py` to catch
the most common patterns while the ML model is being trained (see Step 4).
Follow existing extractor functions for structure.

### Fields needing EDS-NLP patterns

Add an `edsnlp` rule component to `src/extraction/eds_extractor.py`. EDS-NLP
patterns use spaCy-style matchers and can leverage French clinical terminology
resources.

---

## Step 4, Add to the ML training pipeline

Open `bright_models/generated_data/config/fields.py`. Add an entry describing
the new field **in French**, following the structure of existing entries. Each
entry includes:

- The field name (matching `schema.py`)
- A French description of what the field represents
- The allowed values (for prompting the LLM)
- The semantic group it belongs to

The 21 semantic groups in `fields.py` (`_IHC`, `_MOLECULAR`, `_DEMOGRAPHICS`,
`_SYMPTOMS_INITIAL`, `_TREATMENT`, etc.) map to the **10 fine-tuned edsnlp
model groups** loaded by `HFExtractor` in `src/extraction/hf_extractor.py`.
If you add a field to an existing semantic group, also verify that the group's
mapping in `hf_extractor.py` includes the new field name so the extractor's
output is correctly attributed.

If the field belongs to a new clinical category not covered by any existing
group, you will need to create a new semantic group in `fields.py` **and** train
a new model for it (adding a new entry to the group list in `hf_extractor.py`).
This is rare, prefer assigning to the closest existing group.

---

## Step 5, Generate new training data and retrain

Follow the full pipeline described in [`docs/training_data.md`](training_data.md).
In summary: update the field config, regenerate synthetic documents with the LLM,
resolve span offsets, filter for quality, then retrain the relevant edsnlp model
group.

---

## Step 6, Add validation (`src/extraction/validation.py`)

If the new field has constraints, register them:

- **Controlled vocabulary**: add the field name → `ControlledVocab` constant
  mapping to the validation lookup table in `validation.py`. The validator will
  flag values outside the allowed set.
- **Numeric range**: add a range check (e.g., Ki67 % must be 0–100).
- **Date format**: dates are already normalised to `YYYY-MM-DD` by the
  `DateExtractor`; no additional validator is usually needed.
- **French spelling variants**: add normalization entries to `NORMALIZATION_MAP`
  (e.g., `"négatif"` → `"negatif"`) so accent-variant forms are accepted.

---

## Step 7, Write a test

Add a test in `src/tests/` that exercises the new field end-to-end. The
simplest approach is to add cases to an existing extractor test file, or create
`test_my_new_field.py` following the pattern of `test_rule_extraction.py` or
`test_controlled_extractor.py`:

```python
def test_my_new_field_positive():
    text = "... document text containing the field ..."
    result = extractor.extract(text)
    assert result["my_new_field"].value == "expected_value"

def test_my_new_field_negative():
    text = "... text that should not trigger the field ..."
    result = extractor.extract(text)
    assert result["my_new_field"] is None
```

---

## Checklist

- [ ] Field added to `schema.py` (`BiologicalFeatures` or `ClinicalFeatures`)
- [ ] `ControlledVocab` constant added (if field has a fixed value set)
- [ ] Field added to `FEATURE_ROUTING` for the correct document type(s)
- [ ] Extraction logic added in at least one tier (`date_extractor.py`,
  `controlled_vocab_data.py`, `rule_extraction.py`, or `eds_extractor.py`)
- [ ] Field description added to `bright_models/generated_data/config/fields.py`
  and semantic group mapping updated in `src/extraction/hf_extractor.py`
- [ ] Validator added in `validation.py` (if the field has constraints)
- [ ] Test written in `src/tests/`
- [ ] `pytest src/tests/` passes with no regressions
