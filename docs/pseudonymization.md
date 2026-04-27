# Pseudonymization Pipeline

## Overview

The pseudonymization module detects and replaces PHI in the free-text columns of the BRIGHT clinical database CSV. It covers text extraction from source PDFs, PHI detection via the `eds-pseudo` EDS-NLP model, and deterministic pseudonym generation using a secret salt. It is designed for internal research use only — not as a GDPR-compliant public release mechanism.

---

## Architecture

```
PDF files
    │
    ▼
Text Extraction
├── EDS-PDF (primary)
├── PyMuPDF (fallback)
└── pypdf (last resort)
    │
    ▼ (if scanned: <30 chars or >50% images)
OCR Pipeline
├── EasyOCR (primary)
└── Surya-OCR (fallback)
    │
    ▼
Raw Text
    │
    ▼
EDS-NLP Pseudonymization Pipeline (eds-pseudo model)
├── Entity Detection
│   ├── NOM, PRENOM (names)
│   ├── DATE (dates in running text)
│   ├── ADRESSE, ZIP, VILLE (addresses)
│   ├── TEL (phone numbers)
│   ├── MAIL (email addresses)
│   ├── HOPITAL (hospital names)
│   ├── IPP, NDA, SECU (patient administrative IDs)
│   └── Practitioner names (title-prefixed regex, separate pass)
├── Salt-Based Pseudonym Generation
│   └── SHA256(IPP + label + entity_text + secret_salt) → 10-char hex pseudonym
└── Right-to-Left Text Replacement (preserves character offsets)
    │
    ▼
Pseudonymized Text → clinical_db_pseudo_only.csv
```

Large texts (>1000 characters) are split into overlapping chunks (350-character overlap) before being passed to the EDS-NLP model to stay within its context limit. Overlapping spans are deduplicated by keeping the longest match.

---

## Supported Entity Types

The following entity types are detected and replaced:

| Label | What it covers |
|---|---|
| `NOM` | Family names |
| `PRENOM` | Given names |
| `DATE` | Dates in running text (birth dates replaced with year only) |
| `ADRESSE` | Street addresses |
| `ZIP` | Postal codes |
| `VILLE` | City names |
| `TEL` | Phone numbers |
| `MAIL` | Email addresses |
| `HOPITAL` | Hospital and clinic names |
| `IPP` | Identifiant Permanent du Patient (internal patient ID) |
| `NDA` | Numéro de Dossier Administratif |
| `SECU` | Numéro de sécurité sociale |
| Practitioner names | Names preceded by Dr, Pr, Professeur, Docteur, or Interne (detected by regex, separate from the EDS-NLP pass) |

---

## Deterministic Pseudonymization

```
pseudonym = SHA256(IPP + "|" + label + "|" + entity_text + "|" + secret_salt)[:10].upper()
```

The 10-character hex token is embedded in a label-specific template (e.g., `[NOM_A3F9C12B70]`). The same entity text always produces the same pseudonym for a given patient, preserving cross-document entity linking within a patient's records.

By default, pseudonyms are scoped per patient (via `ipp`). Setting `consistent_across_ipp=True` produces globally consistent pseudonyms — useful for linking practitioner names across patients.

---

## Salt Management

The salt is stored in a sidecar file next to the database CSV:

```
path/to/clinical_db.csv
path/to/clinical_db.csv.pseudonym_salt   ← auto-created on first run
```

**If the salt is lost:** re-running pseudonymization on the same documents will produce different tokens, breaking all cross-run linking. Back it up in a secure location separate from the clinical data — treat it like a password.

Add this to `.gitignore` — never commit the salt:
```
*.pseudonym_salt
```

---

## Practitioner Whitelist

BRIGHT team members are listed in `BRIGHT_PRACTITIONERS` in `src/database/pseudonymizer.py`. When `keep_practitioner_names=True` (the default), whitelisted names are preserved in the output. Non-whitelisted practitioners are pseudonymized like any other entity.

Detection uses a title-prefix regex (`Dr`, `Pr`, `Professeur`, `Docteur`, `Interne` followed by a capitalized name). To add a team member, append their name to `BRIGHT_PRACTITIONERS`; matching is case-insensitive.

---

## Limitations

- OCR quality degrades for poor-quality scans (low resolution, skewed pages, handwritten annotations). The pipeline does not detect or flag low-confidence OCR output.
- Entity detection may miss rare name forms, hyphenated surnames, or names that appear without a preceding title or context cue.
- Dates embedded in structured tables or grid layouts are often not detected by the EDS-NLP model, which is optimized for running text.
- The pipeline does not pseudonymize images, charts, signatures, or stamps embedded in PDFs — only extracted text is processed.
- This system has not been reviewed for GDPR compliance. Consult a data protection officer before using it in any regulated context or for any purpose beyond internal research.

---

## CLI Reference

```
usage: main.py pseudo [-h] [--gui] [--db DB] [--pdfs PDFS]
                      [--eds_path EDS_PATH] [--no_pseudo_only]

Create/Update Pseudonymized DB from PDFs

options:
  -h, --help            show this help message and exit
  --gui                 Launch the Qt GUI. Other CLI arguments will be ignored.
  --db DB               Path to the document database CSV
  --pdfs PDFS           Folder containing PDFs or a specific PDF file
  --eds_path EDS_PATH   Path to the HuggingFace cache directory containing
                        the eds-pseudo model (uses bundled cache if omitted)
  --no_pseudo_only      Do not create a _pseudo_only.csv copy alongside the
                        main output
```

**Examples:**

```bash
# CLI mode
python main.py pseudo --db data/clinical_db.csv --pdfs data/pdfs/

# CLI with a single PDF
python main.py pseudo --db data/clinical_db.csv --pdfs data/report_2024.pdf

# Desktop GUI
python main.py pseudo --gui

# Use a custom local eds-pseudo model cache
python main.py pseudo --db data/clinical_db.csv --pdfs data/pdfs/ \
    --eds_path /path/to/hf_cache/
```

**Output:** a new CSV file at `<db_path>_pseudo_only.csv` with all text columns pseudonymized. The original CSV is not modified. The log is written to `cli_pseudo_log.txt` next to the database file.
