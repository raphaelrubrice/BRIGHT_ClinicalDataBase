# Feature Extraction Pipeline

## Overview

The extraction pipeline reads a pseudonymized clinical database CSV and outputs `bio.csv` (55 biological fields, one row per surgical event) and `clinique.csv` (55 clinical fields, one row per consultation). It processes each document independently, then aggregates results per patient across time.

Extraction is hybrid: regex and fuzzy-matching rules cover dates and fixed-vocabulary categoricals; 10 fine-tuned EDS-NLP CRF models handle the remaining fields.

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
    ├── 10 semantic groups covering all 110 fields
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

The 110 fields are split across 10 semantic groups, each served by a separate fine-tuned EDS-NLP CRF model loaded from HuggingFace Hub (`raphael-r/bright-eds-{group}`):

| Group | Fields covered |
|---|---|
| `diagnosis` | diag_histologique, diag_integre, classification_oms, grade, num_labo |
| `ihc` | ihc_idh1, ihc_atrx, ihc_p53, ihc_fgfr3, ihc_braf, ihc_gfap, ihc_olig2, ihc_ki67, ihc_hist_h3k27m, ihc_hist_h3k27me3, ihc_egfr_hirsch, ihc_mmr |
| `histology` | histo_necrose, histo_pec, histo_mitoses, aspect_cellulaire |
| `molecular` | mol_idh1, mol_idh2, mol_mgmt, mol_h3f3a, mol_hist1h3b, mol_tert, mol_CDKN2A, mol_atrx, mol_cic, mol_fubp1, mol_fgfr1, mol_egfr_mut, mol_prkca, mol_pten, mol_p53, mol_braf |
| `chromosomal` | ch1p, ch19q, ch1p19q_codel, ch7p, ch7q, ch10p, ch10q, ch9p, ch9q, ampli_egfr, ampli_cdk4, ampli_mdm2, ampli_mdm4, ampli_met, fusion_fgfr, fusion_ntrk, fusion_autre |
| `demographics` | sexe, annee_de_naissance, activite_professionnelle, antecedent_tumoral, ik_clinique, dominance_cerebrale, neuroncologue, neurochirurgien, radiotherapeute, anatomo_pathologiste |
| `tumor_location` | tumeur_lateralite, tumeur_position, localisation_chir |
| `treatment` | chimios, chimio_protocole, chm_cycles, chm_date_debut, chm_date_fin, type_chirurgie, qualite_exerese, rx_dose, rx_fractionnement, rx_date_debut, rx_date_fin, localisation_radiotherapie, corticoides, anti_epileptiques, optune, essai_therapeutique |
| `symptoms_evolution` | epilepsie_1er_symptome, ceph_hic_1er_symptome, ceph_hic, deficit_1er_symptome, deficit, cognitif_1er_symptome, cognitif, autre_trouble_1er_symptome, contraste_1er_symptome, prise_de_contraste, oedeme_1er_symptome, calcif_1er_symptome, epilepsie, autre_trouble, evol_clinique, progress_clinique, progress_radiologique, reponse_radiologique |
| `dates_outcomes` | date_chir, date_rcp, dn_date, date_deces, date_1er_symptome, exam_radio_date_decouverte, date_progression, survie_globale, infos_deces |

> **Note:** `bright_models/utils.py` defines the `treatment` training group with an extra field `chir_date` that no longer exists in the extraction schema (`schema.py`). The model was trained on it, but it is never populated in the pipeline output. If the models are retrained, `chir_date` should either be removed from `GROUPS["treatment"]` or re-added to `CLINIQUE_FIELDS`.

Models are loaded one at a time. GPU: window=510 tokens, stride=382. CPU fallback: window=128, stride=96 (~4× faster, minor accuracy loss).

Splitting into groups is necessary because EDS-NLP CRF models have a 512-token context limit; running all fields simultaneously degrades performance significantly.

---

## Controlled Vocabulary Fields

`ControlledExtractor` (`src/extraction/controlled_extractor.py`) handles fields with a fixed value set using a three-step Find & Check algorithm:

1. **Find** — scan for field-specific identification terms (gene names, marker names). Terms ≤3 chars: word-boundary regex. Longer terms: sliding-window fuzzy match via `rapidfuzz`.
2. **Check** — for each hit, fuzzy-match candidate category values against the surrounding context. A length bonus penalises short coincidental matches.
3. **Assign** — combined score = id_score × 0.3 + category_score × 0.7; greedy assignment, one value per field.

Vocabularies are in `src/extraction/schema.py`; term lists in `src/extraction/controlled_vocab_data.py`. Covered types:

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

`AssertionAnnotator` (`src/extraction/negation.py`) checks whether each entity is negated, hypothetical, or historical.

**EDS-NLP backend (preferred):** uses `eds.negation`, `eds.hypothesis`, `eds.history` — clinical patterns tuned for French.

**Regex fallback:** 60-character context window before each span (and after for hypothesis/history):
- Negation: `pas de`, `absence de`, `sans`, `aucun`, `non`, `ne...pas`, `négatif`
- Hypothesis: `possible`, `probable`, `suspecté`, `à confirmer`, `éventuel`
- History: `antécédent`, `histoire de`, `précédemment`, `en YYYY`

Negated values are inverted via the `SIMILARITY_FLIP` map in `src/extraction/pipeline.py`:

| Original value | Negated value |
|---|---|
| positif | negatif |
| maintenu | negatif |
| mute / muté | wt |
| methyle / méthylé | non methyle / non méthylé |

Chromosomal gain/perte are **not** inverted — "pas de gain" and "perte" are logically distinct. PRESENCE fields flip to `non`; FREE_TEXT fields get `non ` prepended.

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

**`bio.csv`** — one row per surgical event per patient, 55 biological fields. If a document references multiple surgical events, `row_duplicator.py` splits it into one row per event.

**`clinique.csv`** — one row per consultation per patient, 55 clinical fields. Events are identified by `date_rcp`.

Both files include an `IPP` column and a source document reference. Unextracted fields are left empty (distinguishable from an explicitly extracted `NA`). Temporal aggregation (`src/aggregation/temporal_aggregation.py`) forward-fills values across timepoints; conflicts are flagged in the audit log.

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
