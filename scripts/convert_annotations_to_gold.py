"""Convert REQ_BIO.csv and REQ_CLINIQUE.csv into gold standard JSON files.

Produces two sets of files:

  data/gold_standard/lines/       — one file per (patient × visit) line.
                                    Annotations = merged BIO + CLINIQUE for that visit.
  data/gold_standard/aggregates/  — one file per patient.
                                    Annotations = union of ALL their REQ rows.

Both use raw_text = all pseudonymised documents for that patient concatenated
(sorted by ORDER), which mirrors what the extraction pipeline will process.

Matching logic
--------------
Every REQ_BIO row and every REQ_CLINIQUE row is guaranteed to appear in exactly
one line entry:

1. BIO rows are collected first, keyed by (nip, date_chir).
2. CLINIQUE rows are matched to an existing BIO key via fuzzy date comparison
   (nip must match; dates compared by dates_match_fuzzy).  Unmatched CLINIQUE
   rows create new line keys using (nip, evol_clinique) as fallback identifier.

Per-line entries:  document_id = "{nip}_{evol}"
Aggregate entries: document_id = "{nip}_aggregate"

Usage
-----
    python scripts/convert_annotations_to_gold.py [--db PATH_TO_CLINICAL_DB.csv]
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ANNOTATIONS_DIR = PROJECT_ROOT.parent / "test_annotated" / "ANNOTATIONS_RE MAJ Infos cliniques Braincap"
CLINICAL_DB_DIR = PROJECT_ROOT.parent / "test_annotated" / "RE MAJ Infos cliniques Braincap"
OUTPUT_DIR = PROJECT_ROOT / "data" / "gold_standard"

REQ_BIO_PATH = ANNOTATIONS_DIR / "REQ_BIO.csv"
REQ_CLINIQUE_PATH = ANNOTATIONS_DIR / "REQ_CLINIQUE.csv"
DEFAULT_DB_PATH = CLINICAL_DB_DIR / "clinical_db_20260317_pseudo_only.csv"

sys.path.append(str(PROJECT_ROOT))
from src.extraction.schema import ALL_FIELDS_BY_NAME  # noqa: E402

SKIPPED_FIELDS: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Parsing helpers (unchanged from previous version)
# ---------------------------------------------------------------------------

def load_clinical_db(db_path: Path) -> dict[str, str]:
    """Load the clinical_db CSV and return a mapping IPP → concatenated raw_text.

    Documents for each patient are sorted by ORDER and joined with a double
    newline separator so the extraction pipeline sees all reports at once.
    """
    import pandas as pd

    df = pd.read_csv(db_path, sep=",", dtype=str).dropna(subset=["PSEUDO"])
    df["ORDER"] = pd.to_numeric(df["ORDER"], errors="coerce")
    df = df.sort_values(["IPP", "ORDER"])

    texts_by_ipp: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        texts_by_ipp[row["IPP"]].append(row["PSEUDO"].strip())

    return {ipp: "\n\n".join(texts) for ipp, texts in texts_by_ipp.items()}


def parse_transposed_csv(filepath: Path) -> list[dict[str, str]]:
    """Parse a transposed semicolon-delimited CSV.

    Returns a list of dicts (one per column / patient-visit),
    where keys are the field names from column 0.
    """
    rows = []
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(filepath, encoding=encoding, newline="") as f:
                reader = csv.reader(f, delimiter=";")
                rows = list(reader)
            break
        except UnicodeDecodeError:
            continue

    if not rows:
        raise RuntimeError(f"Could not read {filepath} with any encoding")

    n_columns = len(rows[0]) - 1
    entries: list[dict[str, str | None]] = [{} for _ in range(n_columns)]

    for row in rows:
        if not row or not row[0].strip():
            continue
        field_name = row[0].strip()
        for col_idx in range(n_columns):
            value = row[col_idx + 1].strip() if col_idx + 1 < len(row) else ""
            entries[col_idx][field_name] = value if value else None

    return entries


def _normalize_date_str(s: str) -> str:
    """Lowercase and replace non-ASCII characters so encoding variants match."""
    return s.lower().encode("ascii", errors="ignore").decode("ascii")


def dates_match_fuzzy(date_a: str | None, date_b: str | None) -> bool:
    """Check if two date strings refer to the same event (fuzzy).

    Handles cases like '01/03/2016' matching '2016' (year-only),
    'déc-10' matching 'dec-10' (encoding variants), and exact equality.
    """
    if date_a is None or date_b is None:
        return False
    if date_a == date_b:
        return True
    a_norm, b_norm = _normalize_date_str(date_a), _normalize_date_str(date_b)
    if a_norm == b_norm:
        return True
    for a, b in [(a_norm, b_norm), (b_norm, a_norm)]:
        if len(a) == 4 and a.isdigit():
            if a in b:
                return True
    return False


def clean_annotations(annotations: dict[str, str | None]) -> dict[str, object]:
    """Convert annotation dict to gold standard format.

    Removes None values, skips identifier fields, validates against schema,
    and converts numeric strings where appropriate.
    """
    cleaned = {}
    for field, value in annotations.items():
        if value is None:
            continue
        if field in ("nip", "date_chir", "chir_date"):
            # Identifier / matching fields — not features to be extracted
            continue
        if field not in ALL_FIELDS_BY_NAME:
            SKIPPED_FIELDS[field] = SKIPPED_FIELDS.get(field, 0) + 1
            continue

        if field in ("grade", "chm_cycles", "histo_mitoses", "ik_clinique"):
            try:
                cleaned[field] = {"value": int(value)}
                continue
            except (ValueError, TypeError):
                pass
        if field in ("rx_dose",):
            try:
                cleaned[field] = {"value": float(value)}
                continue
            except (ValueError, TypeError):
                pass
        cleaned[field] = {"value": value}

    return cleaned


# ---------------------------------------------------------------------------
# Line-building logic
# ---------------------------------------------------------------------------

def _canonical_date(entry: dict, date_key: str) -> str | None:
    """Return the date string for an entry, or None."""
    return entry.get(date_key) or None


def build_lines(
    bio_entries: list[dict],
    clinique_entries: list[dict],
) -> list[dict]:
    """Merge BIO and CLINIQUE entries into per-visit lines.

    Returns a list of line dicts, each with:
        nip, evol, bio (dict), clinique (dict)

    Every input row appears in exactly one line.
    """
    # Each line: {"nip": str, "evol": str|None, "bio": dict, "clinique": dict}
    lines: list[dict] = []

    # --- Collect BIO rows ---
    for bio in bio_entries:
        nip = bio.get("nip")
        if not nip:
            continue
        lines.append({"nip": nip, "evol": None, "bio": bio, "clinique": {}})

    # --- Match CLINIQUE rows to existing BIO lines, or create new lines ---
    used_line_indices: set[int] = set()

    for clinique in clinique_entries:
        nip = clinique.get("nip")
        if not nip:
            continue
        chir_date = clinique.get("chir_date")
        evol = clinique.get("evol_clinique")

        # Try to find a matching BIO line (same nip, fuzzy date)
        matched_idx = None
        for idx, line in enumerate(lines):
            if idx in used_line_indices:
                continue
            if line["nip"] != nip:
                continue
            bio_date = _canonical_date(line["bio"], "date_chir")
            if dates_match_fuzzy(chir_date, bio_date):
                matched_idx = idx
                break

        if matched_idx is not None:
            lines[matched_idx]["clinique"] = clinique
            lines[matched_idx]["evol"] = evol or lines[matched_idx]["evol"]
            used_line_indices.add(matched_idx)
        else:
            # No matching BIO row — create a CLINIQUE-only line
            lines.append({"nip": nip, "evol": evol, "bio": {}, "clinique": clinique})

    return lines


def make_document_id(nip: str, evol: str | None, fallback_idx: int) -> str:
    """Create a human-readable document ID."""
    evol_part = evol.strip() if evol else f"visit{fallback_idx}"
    return f"{nip}_{evol_part}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB_PATH,
        help="Path to clinical_db pseudo-only CSV (provides raw_text).",
    )
    args = parser.parse_args()

    lines_dir = OUTPUT_DIR / "lines"
    aggregates_dir = OUTPUT_DIR / "aggregates"
    lines_dir.mkdir(parents=True, exist_ok=True)
    aggregates_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load raw texts
    # ------------------------------------------------------------------
    db_path: Path = args.db
    if db_path.exists():
        raw_texts_by_ipp = load_clinical_db(db_path)
        print(f"Loaded document texts for {len(raw_texts_by_ipp)} patients from {db_path.name}")
    else:
        raw_texts_by_ipp = {}
        print(f"WARNING: clinical_db not found at {db_path} — gold standard will have no raw_text")

    # ------------------------------------------------------------------
    # Parse REQ CSVs
    # ------------------------------------------------------------------
    bio_entries = parse_transposed_csv(REQ_BIO_PATH)
    clinique_entries = parse_transposed_csv(REQ_CLINIQUE_PATH)
    print(f"Parsed {len(bio_entries)} BIO entries, {len(clinique_entries)} CLINIQUE entries")

    # ------------------------------------------------------------------
    # Build per-visit lines
    # ------------------------------------------------------------------
    lines = build_lines(bio_entries, clinique_entries)
    print(f"Built {len(lines)} lines across {len({l['nip'] for l in lines})} patients")

    line_entries: list[dict] = []
    lines_by_nip: dict[str, list[dict]] = defaultdict(list)

    for idx, line in enumerate(lines):
        nip = line["nip"]
        evol = line["evol"]
        doc_id = make_document_id(nip, evol, idx)

        # Merge annotations: BIO first, CLINIQUE wins on conflict
        bio_annotations = {
            k: v for k, v in line["bio"].items() if k != "nip" and v is not None
        }
        clinique_annotations = {
            k: v for k, v in line["clinique"].items() if k != "nip" and v is not None
        }
        merged_raw = {**bio_annotations, **clinique_annotations}
        annotations = clean_annotations(merged_raw)

        if not annotations:
            print(f"  [skip] {doc_id} — no annotations after cleaning")
            continue

        entry = {
            "document_id": doc_id,
            "patient_id": nip,
            "entry_type": "line",
            "raw_text": raw_texts_by_ipp.get(nip, ""),
            "annotations": annotations,
        }
        line_entries.append(entry)
        lines_by_nip[nip].append(entry)

        out_path = lines_dir / f"{doc_id}.json"
        out_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [line]      {out_path.name}  ({len(annotations)} fields)")

    # ------------------------------------------------------------------
    # Build per-patient aggregates
    # ------------------------------------------------------------------
    aggregate_entries: list[dict] = []

    for nip, patient_lines in sorted(lines_by_nip.items()):
        # Union all annotations; later lines win on conflict
        agg_annotations: dict[str, object] = {}
        for line_entry in patient_lines:
            agg_annotations.update(line_entry["annotations"])

        doc_id = f"{nip}_aggregate"
        entry = {
            "document_id": doc_id,
            "patient_id": nip,
            "entry_type": "patient_aggregate",
            "raw_text": raw_texts_by_ipp.get(nip, ""),
            "annotations": agg_annotations,
        }
        aggregate_entries.append(entry)

        out_path = aggregates_dir / f"{doc_id}.json"
        out_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [aggregate] {out_path.name}  ({len(agg_annotations)} fields, {len(patient_lines)} lines merged)")

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    manifest = {
        "total_lines": len(line_entries),
        "total_aggregates": len(aggregate_entries),
        "source_files": {
            "REQ_BIO": str(REQ_BIO_PATH),
            "REQ_CLINIQUE": str(REQ_CLINIQUE_PATH),
            "clinical_db": str(db_path),
        },
        "lines": [
            {
                "document_id": e["document_id"],
                "patient_id": e["patient_id"],
                "n_annotations": len(e["annotations"]),
                "has_raw_text": bool(e.get("raw_text")),
            }
            for e in line_entries
        ],
        "aggregates": [
            {
                "document_id": e["document_id"],
                "patient_id": e["patient_id"],
                "n_annotations": len(e["annotations"]),
                "has_raw_text": bool(e.get("raw_text")),
            }
            for e in aggregate_entries
        ],
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(line_entries)} line files → {lines_dir}")
    print(f"Wrote {len(aggregate_entries)} aggregate files → {aggregates_dir}")
    print(f"Wrote manifest → {manifest_path}")

    if SKIPPED_FIELDS:
        print(f"\nSkipped {sum(SKIPPED_FIELDS.values())} values across {len(SKIPPED_FIELDS)} non-schema fields:")
        for field, count in sorted(SKIPPED_FIELDS.items()):
            print(f"  - {field}: {count} occurrences")


if __name__ == "__main__":
    main()
