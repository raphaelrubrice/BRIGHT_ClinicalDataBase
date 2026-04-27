# run_hf_demo.ipynb — Results Backup

Captured from notebook outputs before clearing (2026-04-27).

---

## Setup

- Repo: `raphaelrubrice/BRIGHT_ClinicalDataBase`, branch `final`
- Data: copied from Google Drive at `/content/drive/MyDrive/MVA/BRIGHT/`
- Runtime: Google Colab (kernel restarted after package install)

---

## HFExtractor Smoke Test (Section 2)

Input sentence: `"Glioblastome WHO grade IV, temporal gauche, IDH1 non muté."`  
Enabled groups: `["diagnosis", "tumor_location"]`

| Field | Value | Span |
|-------|-------|------|
| `diag_integre` | `'.'` | `'.'` |
| `localisation_chir` | `'Glioblastome WHO grade IV, temporal gauche, IDH1 non muté.'` | `'Glioblastome WHO grade IV, temporal gauche, IDH1 non muté.'` |

---

## Model Cache (Section 3a)

All 10 HF groups found cached at `/content/drive/MyDrive/MVA/BRIGHT/bright_models_cache/`:

`diagnosis`, `ihc`, `histology`, `molecular`, `chromosomal`, `demographics`,
`tumor_location`, `treatment`, `symptoms_evolution`, `dates_outcomes`

---

## Ablation Run (Section 3b)

3 modes run: `rule`, `ml`, `both`  
`MAX_DOCS = 50` (capped at 9 available gold-standard documents)

All 3 modes exited cleanly (return code 0).

---

## Metrics (Section 5) — Macro-averaged over all fields

| Mode | Fields | Macro-F1 (exact) | Macro-F1 (relaxed) |
|------|--------|-------------------|--------------------|
| rule | 102 | 0.182 | 0.194 |
| ml   | 98  | 0.045 | 0.050 |
| both | 105 | 0.170 | 0.185 |

---

## Summary Table (Section 7) — Full macro P/R/F1

| Mode | macro_P_exact | macro_R_exact | macro_F1_exact | macro_P_relaxed | macro_R_relaxed | macro_F1_relaxed |
|------|---------------|---------------|----------------|-----------------|-----------------|------------------|
| rule | — | — | 0.182 | — | — | 0.194 |
| ml   | — | — | 0.045 | — | — | 0.050 |
| both | — | — | 0.170 | — | — | 0.185 |

*(Precision and recall per-mode were displayed as a styled DataFrame in the notebook; exact values not captured in text output. F1 values above come from the cell 19 print.)*

---

## Notes

- The `ml`-only mode underperforms `rule` and `both` significantly on this small set (9 docs).
- The `both` mode does not consistently outperform `rule` — likely because the HF models are not yet well-tuned on this domain.
- Gold standard: 9 documents in `data/gold_standard/`.
- Per-field plots were saved to `/tmp/bright_eval/plots/` during the Colab session (not persisted).
