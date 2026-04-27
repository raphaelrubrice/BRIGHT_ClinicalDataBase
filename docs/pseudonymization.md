# Pseudonymization Pipeline

## Overview

The pseudonymization module detects and replaces protected health information (PHI) in the free-text columns of the BRIGHT clinical database. It is designed for internal research use at Institut de Neurologie, Hôpital Pitié-Salpêtrière: the goal is to reduce re-identification risk when working with extracted clinical features, not to produce a GDPR-compliant public dataset. The pipeline covers text extraction from source PDFs, PHI detection via the `eds-pseudo` EDS-NLP model, and deterministic pseudonym generation using a secret salt.

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

Each detected entity is replaced by a pseudonym generated as follows:

```
pseudonym = SHA256(IPP + "|" + label + "|" + entity_text + "|" + secret_salt)[:10].upper()
```

The resulting 10-character uppercase hex string is then embedded in a label-specific template (e.g., `[NOM_A3F9C12B70]`). Because the hash is deterministic given a fixed salt, the same entity text always produces the same pseudonym within a patient's records. This means that cross-document entity linking is preserved: if the same name appears in three reports for the same patient, all three will be replaced by the same token.

The `ipp` argument scopes pseudonyms to a single patient by default. Setting `consistent_across_ipp=True` produces globally consistent pseudonyms regardless of patient — useful when you need to link entities across patients (e.g., same practitioner appearing in multiple records).

---

## Salt Management

The salt is stored in a sidecar file next to the clinical database CSV:

```
path/to/clinical_db.csv          ← your database
path/to/clinical_db.csv.pseudonym_salt  ← salt file (auto-created on first run)
```

The salt is generated once using `secrets.token_urlsafe(32)` and written atomically (write to a temp file, then rename) to prevent corruption from concurrent runs.

**What happens if the salt is lost:** pseudonyms become inconsistent across pipeline runs. Re-running pseudonymization on the same documents will produce different tokens, breaking any cross-run entity linking. Existing pseudonymized CSVs will no longer be reproducible.

**How to back it up:** copy the `.pseudonym_salt` file to a secure, access-controlled location (separate from the clinical data). Treat it like a password — whoever has the salt and the original documents can reconstruct all pseudonyms.

**Git:** add the salt file pattern to `.gitignore`:
```
*.pseudonym_salt
```
The salt must never be committed to version control.

---

## Practitioner Whitelist

BRIGHT team members at Pitié-Salpêtrière are listed in the `BRIGHT_PRACTITIONERS` constant in `src/database/pseudonymizer.py`. When `keep_practitioner_names=True` (the default), any practitioner name that matches the whitelist is preserved in the output rather than replaced with a pseudonym.

Practitioner names are detected by a title-prefix regex pattern that looks for tokens like `Dr`, `Pr`, `Professeur`, `Docteur`, or `Interne` followed by a capitalized name. Non-whitelisted practitioners whose names appear after a title are still pseudonymized.

To add a new team member to the whitelist, edit the `BRIGHT_PRACTITIONERS` list in `src/database/pseudonymizer.py`. Names are matched case-insensitively.

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
