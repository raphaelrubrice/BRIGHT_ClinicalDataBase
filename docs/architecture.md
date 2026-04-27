# BRIGHT — Architecture

## Module Map

| Module | Path | Responsibility |
|--------|------|----------------|
| Pseudonymization | `src/database/` | Extract text from PDFs; detect and replace 13+ PII entity types using EDS-NLP; produce deterministic pseudonyms via SHA-256 + secret salt; write `clinical_db_pseudo_only.csv` |
| Feature Extraction | `src/extraction/` | Classify documents by type; detect sections; run a four-tier extraction cascade (date regex → controlled-vocab fuzzy match → EDS-NLP rules → 10 fine-tuned HF models); apply negation detection and controlled-vocab validation; emit `DocumentExtraction` objects |
| Aggregation | `src/aggregation/` | Duplicate rows for multi-event documents (multiple surgeries, chemo lines, etc.); forward-fill fields across the patient timeline with document-type priority; produce `bio.csv` and `clinique.csv` |
| Evaluation | `src/evaluation/` | Load JSON gold-standard annotations; run `ExtractionPipeline` on the same documents; compute per-field precision/recall/F1 with weighted alteration scoring; write benchmark CSV reports |
| GUI | `src/ui/` | PySide6 Qt6 desktop application for the pseudonymization pipeline; provides file-browser dialogs, progress tracking, and multi-threaded processing so the UI stays responsive |
| CLI | `main.py` | Argparse entry point exposing two subcommands: `pseudo` (pseudonymize PDFs → CSV) and `extract` (extract 111 clinical fields from a pseudo CSV) |

---

## Full Data Flow

```
Input: PDF files in a folder
       + clinical_db.csv  (columns: IPP, SOURCE_FILE, DOCUMENT, ORDER)
    │
    ├─ Text extraction (EDS-PDF → PyMuPDF → pypdf fallback chain; surya-ocr for scans)
    ├─ EDS-NLP pseudonymization  (13 entity types: NOM, PRENOM, DATE, DATE_NAISSANCE,
    │   ADRESSE, ZIP, VILLE, TEL, MAIL, HOPITAL, IPP, NDA, SECU)
    │   Deterministic token: SHA256(IPP | label | text | salt)[:10].upper()
    │   Practitioner names matched against BRIGHT_PRACTITIONERS whitelist
    └─ Output: clinical_db_pseudo_only.csv  (adds PSEUDO column, masks DATE_NAISSANCE)
         │
         ├─ Document classification  (keyword scoring → anapath / molecular_report /
         │   consultation / rcp / radiology)
         ├─ Section detection  (IHC / molecular / microscopy / conclusion / full_text)
         ├─ Feature routing  (FEATURE_ROUTING maps doc type → extractable field set)
         │
         ├─ Tier 1 — DateExtractor      (regex + context-keyword date assignment)
         ├─ Tier 2 — ControlledExtractor (marker-term fuzzy matching for vocab fields)
         ├─ Tier 3 — RuleExtractor       (regex heuristics for IHC, molecular, binary, etc.)
         ├─ Tier 4a— EDSExtractor        (edsnlp CRF rules)
         ├─ Tier 4b— HFExtractor         (10 fine-tuned edsnlp models, one per semantic group)
         │
         ├─ Priority merge  (date > controlled > EDS > rules; HF overrides rules on
         │   ~50 fields listed in _HF_PASSING_FIELDS)
         ├─ Negation detection  (EDS-NLP AssertionAnnotator + regex fallback;
         │   inverts binary/status values via SIMILARITY_FLIP map)
         ├─ Controlled-vocab validation  (flags values outside allowed sets)
         └─ Output: list of DocumentExtraction objects (one per document)
              │
              ├─ Row duplication  (split multi-event documents: N surgeries → N rows)
              ├─ Temporal aggregation
              │   ├─ STATIC_FEATURES: set once, persist unchanged
              │   ├─ SPECIMEN_BOUND_FEATURES: forward-fill until next surgery
              │   └─ TIME_VARYING_FEATURES: latest explicit value wins
              ├─ Conflict resolution by document-type priority
              │   (anapath > molecular_report > rcp > consultation > radiology for bio;
              │    consultation > rcp > anapath > ... for clinique)
              └─ Output: bio.csv (55 biological fields)
                         clinique.csv (56 clinical fields)
```

---

## Key Design Decisions

**Why deterministic pseudonyms?**
The SHA-256 formula `SHA256(IPP | label | text | salt)[:10]` maps the same entity to
the same token every time for a given patient, so NER spans can be linked across
documents without storing a lookup table. The secret salt (stored in a sidecar file
alongside the CSV) provides the security guarantee: the mapping cannot be inverted
without it.

**Why synthetic training data?**
No annotated real clinical records were available during development — access would have
required IRB approval and data-sharing agreements that were not in scope. LLM-generated
documents were the only feasible path for creating labelled training examples at scale.
The gap between synthetic and real vocabulary and structure is the primary source of
performance degradation on live data.

**Why three extraction tiers?**
Date regex is cheap and highly accurate for well-formatted date strings. Controlled-vocab
fuzzy matching handles structured fields (IHC status, molecular mutations) without model
overhead. ML handles complex free-text fields (diagnostic phrases, tumour locations) where
hand-written rules would be brittle. Routing fields to the cheapest adequate tier keeps
inference latency low on CPU-only hospital workstations.

**Why 10 semantic model groups?**
EDS-NLP's CRF models operate on documents tokenized with a sliding context window.
Grouping the 111 fields by clinical semantics (IHC, molecular, chromosomal, demographics,
treatment, etc.) keeps intra-group context coherent, makes each model's label space
manageable, and lets the system load only the relevant groups for a given document type.

**Why EDS-NLP / ONNX runtime?**
`edsnlp` is the standard French clinical NLP framework at AP-HP / Inria and ships with
pre-built pseudonymization and rule pipelines that would otherwise require months of
engineering. ONNX export of the fine-tuned CRF models enables CPU inference without a
GPU, making the system deployable on standard hospital workstations without special
hardware.

---

## External Dependencies of Note

| Package | Role |
|---------|------|
| `edsnlp` | French clinical NLP framework (AP-HP / Inria). Required for pseudonymization (`eds-pseudo` pipeline), EDS-based extraction rules, and the CRF models that back all 10 HF model groups. |
| `eds-pseudo` | EDS-NLP pseudonymization pipeline. Must be cloned separately from GitHub during setup (`git clone https://github.com/aphp/eds-pseudo.git`). Not on PyPI. |
| `gliner2` / `gliner2-onnx` | Named-entity recognition framework used in early model experiments. The production models are edsnlp CRF pipelines, but GLiNER2 variants may also be trained in `bright_models/`. |
| `surya-ocr` | OCR engine for scanned PDF pages. Heavier dependency; GPU-optional. Only activated when EDS-PDF and PyMuPDF both fail to extract usable text. |
| `PySide6` | Qt6 Python bindings for the desktop GUI (`src/ui/`). Not required for CLI usage. |
