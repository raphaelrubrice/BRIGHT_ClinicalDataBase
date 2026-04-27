# Feature Extraction Pipeline

## Overview

The feature extraction pipeline reads a pseudonymized clinical database CSV and produces two structured output files: `bio.csv` (biological and pathology features) and `clinique.csv` (clinical and treatment features). It processes each document independently, then aggregates results across time per patient.

The pipeline extracts 111 standardized fields covering IHC markers, molecular mutations, chromosomal alterations, diagnosis, demographics, treatment dates, surgical details, and clinical evolution. Extraction uses a hybrid architecture: regex and rule-based methods handle deterministic fields such as dates and categorical values with fixed vocabularies; 10 fine-tuned EDS-NLP CRF models handle the remaining fields using learned clinical NER.

---

## Architecture

```
Pseudonymized CSV (text columns)
    │
    ▼
Document Classifier (document_classifier.py)
├── anatomopathology report   → bio fields only
├── molecular report          → molecular/chromosomal/amplification/fusion fields
├── consultation / RCP report → key bio fields + all clinical fields
└── radiology report          → radiology-specific clinical fields
    │
    ▼
Section Detector (section_detector.py)
└── Identifies relevant sections (findings, conclusion, molecular results, etc.)
    │
    ▼
Four-Tier Extraction (pipeline.py)
│
├── Tier 1 — Date Extractor (date_extractor.py)
│   └── Regex-based date parsing for all date fields
│
├── Tier 2 — Controlled Vocabulary Extractor (controlled_extractor.py)
│   ├── Find & Check: locate marker terms, then match category values
│   ├── Regex for short terms (≤3 chars), fuzzy matching (rapidfuzz) for longer
│   └── Covers: IHC status, molecular status, grade, surgery type, etc.
│
├── Tier 3 — EDS Extractor (eds_extractor.py)
│   └── Rule-based EDS-NLP patterns for specialized fields
│
└── Tier 4 — ML Extractor (hf_extractor.py)
    ├── 10 fine-tuned EDS-NLP CRF models (raphael-r/bright-eds-{group})
    ├── 10 semantic groups covering all 111 fields
    └── GPU: window=510 tokens, stride=382 / CPU fallback: window=128, stride=96
    │
    ▼
Negation Handler (negation.py)
└── AssertionAnnotator: detects negation / hypothesis / history context
    │
    ▼
Result Merger (pipeline.py)
├── Rules merged (priority: date > controlled > eds > rule)
├── ML fields: HF wins for ~150 fields with F1≥0.6 on synthetic benchmark
└── Rules win on remaining fields; confidence boosted when both agree
    │
    ▼
Validator (validation.py)
└── Schema + controlled vocabulary checks; flags out-of-vocab values
    │
    ▼
Aggregation (patient_timeline.py)
├── Row Duplicator: detects multiple treatment events per document
├── Temporal Aggregation: forward-fill + conflict resolution across timepoints
└── Output: bio.csv + clinique.csv
```

---

## Training Data

The ML models were trained on **synthetic clinical documents** generated using a large language model (LLM) pipeline located in `../bright_models/generated_data/`.

### How the synthetic data was created

1. **Patient profiles** were sampled from templates in `bright_models/generated_data/profiles/`
2. **Few-shot prompts** (in `bright_models/generated_data/config/prompts/`) guided an LLM to generate realistic-looking French anatomopathology and consultation reports
3. **Ground truth spans** were resolved: the LLM was asked to return exact character offsets for each extracted entity (`step3_resolve.py`)
4. **Quality filtering** (`step4_filter.py`) applied a 6-level cascade:
   - Span validity (offsets exist in the text)
   - Entity type consistency (values match the schema)
   - WHO 2021 biological coherence (e.g., IDH1 IHC and molecular results must agree)
   - Duplicate filtering
   - Length checks
   - Format validation

### Why this matters for performance

LLM-generated documents differ from real clinical records in vocabulary, sentence structure, abbreviation patterns, and error frequency. Models trained on synthetic data often fail to generalize to real-world clinical text. See the Performance Warning section below.

---

## The 10 Model Groups

The 111 fields are split across 10 semantic groups, each served by a separate fine-tuned EDS-NLP CRF model loaded from HuggingFace Hub (`raphael-r/bright-eds-{group}`):

| Group | Fields covered |
|---|---|
| `diagnosis` | diag_histologique, diag_integre, classification_oms, grade, num_labo |
| `ihc` | ihc_idh1, ihc_atrx, ihc_p53, ihc_fgfr3, ihc_braf, ihc_gfap, ihc_olig2, ihc_ki67, ihc_hist_h3k27m, ihc_hist_h3k27me3, ihc_egfr_hirsch, ihc_mmr |
| `histology` | histo_necrose, histo_pec, histo_mitoses, aspect_cellulaire |
| `molecular` | mol_idh1, mol_idh2, mol_mgmt, mol_h3f3a, mol_hist1h3b, mol_tert, mol_CDKN2A, mol_atrx, mol_cic, mol_fubp1, mol_fgfr1, mol_egfr_mut, mol_prkca, mol_pten, mol_p53, mol_braf |
| `chromosomal` | ch1p, ch19q, ch1p19q_codel, ch7p, ch7q, ch10p, ch10q, ch9p, ch9q, ampli_egfr, ampli_cdk4, ampli_mdm2, ampli_mdm4, ampli_met, fusion_fgfr, fusion_ntrk, fusion_autre |
| `demographics` | sexe, annee_de_naissance, activite_professionnelle, antecedent_tumoral, ik_clinique, dominance_cerebrale, neuroncologue, neurochirurgien, radiotherapeute, anatomo_pathologiste |
| `tumor_location` | tumeur_lateralite, tumeur_position, localisation_chir |
| `treatment` | chimios, chimio_protocole, chm_cycles, chm_date_debut, chm_date_fin, type_chirurgie, qualite_exerese, chir_date, rx_dose, rx_fractionnement, rx_date_debut, rx_date_fin, localisation_radiotherapie, corticoides, anti_epileptiques, optune, essai_therapeutique |
| `symptoms_evolution` | epilepsie_1er_symptome, ceph_hic_1er_symptome, ceph_hic, deficit_1er_symptome, deficit, cognitif_1er_symptome, cognitif, autre_trouble_1er_symptome, contraste_1er_symptome, prise_de_contraste, oedeme_1er_symptome, calcif_1er_symptome, epilepsie, autre_trouble, evol_clinique, progress_clinique, progress_radiologique, reponse_radiologique |
| `dates_outcomes` | date_chir, date_rcp, dn_date, date_deces, date_1er_symptome, exam_radio_date_decouverte, date_progression, survie_globale, infos_deces |

Models are loaded one at a time to minimize memory usage. On GPU, the CRF encoder uses a sliding window of 510 tokens with a stride of 382 tokens. On CPU, a smaller window (128 tokens, stride 96) is used for a ~4× speed improvement at a minor accuracy cost.

Splitting into groups is necessary because EDS-NLP CRF models have a 512-token context limit, and running all 111 field labels simultaneously would degrade performance severely. Each model is fine-tuned specifically on the fields in its group.

---

## Controlled Vocabulary Fields

The `ControlledExtractor` (`src/extraction/controlled_extractor.py`) handles fields whose values come from a fixed vocabulary (e.g., IHC status, molecular status, chromosomal status). It uses a three-step Find & Check algorithm:

**Step 1 — Find:** scan the document for field-specific identification terms (gene names, marker names). Short terms (≤3 characters) are matched using word-boundary regex; longer terms use sliding-window fuzzy matching via `rapidfuzz.fuzz.partial_ratio`.

**Step 2 — Check:** for each identification hit, extract a context window around it and fuzzy-match each candidate category value (e.g., `positif`, `negatif`, `maintenu`) against that context. A length bonus rewards longer matching terms to reduce false positives from short coincidental matches.

**Step 3 — Assign:** sort candidates by combined score (identification score × 0.3 + category score × 0.7) and assign each field to its best category, greedy (one value per field).

The controlled vocabularies are defined in `src/extraction/schema.py` and the term lists in `src/extraction/controlled_vocab_data.py`. Covered vocabulary types:

| Vocab | Values |
|---|---|
| `IHC_STATUS` | positif, negatif, maintenu, NA |
| `MOLECULAR_STATUS` | wt, mute, autre, NA |
| `CHROMOSOMAL` | gain, perte, perte partielle, NA |
| `METHYLATION` | methyle, non methyle, NA |
| `MGMT_STATUS` | methyle, non methyle, wt, mute, NA |
| `GRADE` | 1, 2, 3, 4, autre, NA |
| `WHO_CLASSIFICATION` | 2007, 2016, 2021, NA |
| `SURGERY_TYPE` | exerese complete, exerese partielle, exerese, biopsie, en attente, autre, NA |
| `BINARY` | oui, non, NA |
| `SEX` | M, F, NA |
| `LATERALITY` | gauche, droite, bilateral, median, NA |
| `HANDEDNESS` | droitier, gaucher, ambidextre, droitier contrarié, gaucher contrarié, NA |

---

## Negation Handling

The `AssertionAnnotator` (`src/extraction/negation.py`) checks whether each extracted entity appears in a negated, hypothetical, or historical context.

**EDS-NLP backend (preferred when available):** wraps the `eds.negation`, `eds.hypothesis`, and `eds.history` EDS-NLP components. These use clinical NLP patterns tuned for French medical text.

**Regex fallback (when EDS-NLP is unavailable):** scans a 60-character window before (and after, for hypothesis/history) each entity span for patterns such as:
- Negation: `pas de`, `absence de`, `sans`, `aucun`, `non`, `ne...pas`, `négatif`
- Hypothesis: `possible`, `probable`, `suspecté`, `à confirmer`, `éventuel`
- History: `antécédent`, `histoire de`, `précédemment`, `en YYYY`

When an entity is detected as negated, its value is inverted using the `SIMILARITY_FLIP` map defined in `src/extraction/pipeline.py`:

| Original value | Negated value |
|---|---|
| positif | negatif |
| maintenu | negatif |
| mute / muté | wt |
| methyle / méthylé | non methyle / non méthylé |

Note: chromosomal gain/perte are **not** inverted under negation because "pas de gain" and "perte" are logically distinct states requiring manual review.

For PRESENCE-type fields (e.g., `epilepsie`, `oedeme`), negation flips the value to `non`. For FREE_TEXT fields, the prefix `non ` is prepended.

---

## ⚠️ Performance Warning

Benchmarks run on the **synthetic test set** show reasonable F1 scores. However, when evaluated on real annotated clinical records, performance drops substantially.

Observed failure modes on real data:
- Abbreviations not seen in training (e.g., regional laboratory abbreviations, institution-specific shorthand)
- Negation constructs not covered by the assertion rules (nested negation, implicit negation via contrast)
- Multi-sentence entities that span chunk boundaries
- Documents with non-standard formatting (tables, OCR artifacts, handwritten annotations)
- Molecular report formats that differ from the templates used for synthetic data generation

**Do not use extraction results in any clinical or scientific analysis without manual spot-checking on a representative sample.**

---

## Output Format

**`bio.csv`** — one row per surgical event per patient. Contains all 55 biological fields (`date_chir`, `num_labo`, `diag_histologique`, all IHC, histology, molecular, chromosomal, amplification, and fusion fields). When a single document references multiple surgical events (e.g., re-operation), `row_duplicator.py` detects this and produces one row per event.

**`clinique.csv`** — one row per consultation event per patient. Contains all 56 clinical fields (demographics, care team, outcome, symptoms, radiology, tumour location, evolution, treatment, clinical state, surgery date, and adjunct therapies). Consultation events are identified by `date_rcp`; each distinct date produces a separate row.

Both files include a patient identifier column (`IPP`) and the source document reference. Fields that could not be extracted are left empty (not filled with a default value) so that missingness is distinguishable from a truly extracted `NA` value.

Temporal aggregation (`src/aggregation/temporal_aggregation.py`) applies forward-fill across timepoints: a value extracted from an earlier consultation is carried forward to later rows unless a newer value is found. Conflicts (two documents with different values for the same field at the same timepoint) are flagged in the audit log.

---

## CLI Reference

```
usage: main.py extract [-h] --db DB --output OUTPUT [--parallel PARALLEL]

Extract clinical/bio features to output folder

options:
  -h, --help           show this help message and exit
  --db DB              Path to the pseudonymized document database CSV
                       (required)
  --output OUTPUT      Path to output directory; created if it does not exist
                       (required)
  --parallel PARALLEL  Number of worker processes for parallel document
                       processing (default: cpu_count - 2)
```

**Examples:**

```bash
# Basic extraction
python main.py extract \
    --db data/clinical_db_pseudo_only.csv \
    --output results/

# Limit to 4 workers (useful on shared servers)
python main.py extract \
    --db data/clinical_db_pseudo_only.csv \
    --output results/ \
    --parallel 4
```

**Output files:**
- `results/bio.csv` — biological features, one row per surgical event
- `results/clinique.csv` — clinical features, one row per consultation
- `results/cli_extraction_log.txt` — full run log with per-document timing and flagged fields
