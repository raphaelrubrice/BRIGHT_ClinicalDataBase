"""Convert REQ_BIO.csv and REQ_CLINIQUE.csv into gold standard JSON files.

The annotation CSVs are transposed (rows = fields, columns = patient-visits),
semicolon-delimited. This script parses them and produces one JSON file per
patient-visit, matching BIO and CLINIQUE entries by NIP + surgery date.

Usage:
    python scripts/convert_annotations_to_gold.py
"""

import csv
import json
import os
from pathlib import Path


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ANNOTATIONS_DIR = PROJECT_ROOT.parent / "test_annotated" / "ANNOTATIONS_RE MAJ Infos cliniques Braincap"
OUTPUT_DIR = PROJECT_ROOT / "data" / "gold_standard"

REQ_BIO_PATH = ANNOTATIONS_DIR / "REQ_BIO.csv"
REQ_CLINIQUE_PATH = ANNOTATIONS_DIR / "REQ_CLINIQUE.csv"


def parse_transposed_csv(filepath: Path) -> list[dict[str, str]]:
    """Parse a transposed semicolon-delimited CSV.

    Returns a list of dicts (one per column / patient-visit),
    where keys are the field names from column 0.
    """
    rows = []
    # Try multiple encodings since the files have French accents
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

    # rows[i][0] = field name, rows[i][1..n] = values for each patient-visit
    n_columns = len(rows[0]) - 1  # exclude the field-name column
    entries = [{} for _ in range(n_columns)]

    for row in rows:
        if not row or not row[0].strip():
            continue
        field_name = row[0].strip()
        for col_idx in range(n_columns):
            value = row[col_idx + 1].strip() if col_idx + 1 < len(row) else ""
            entries[col_idx][field_name] = value if value else None

    return entries


def make_document_id(nip: str, evol: str | None, idx: int) -> str:
    """Create a human-readable document ID."""
    evol_part = evol if evol else f"visit{idx}"
    return f"{nip}_{evol_part}"


def dates_match_fuzzy(date_a: str | None, date_b: str | None) -> bool:
    """Check if two date strings refer to the same event (fuzzy).

    Handles cases like '01/03/2016' matching '2016' (year-only),
    and 'déc-10' matching 'déc-10'.
    """
    if date_a is None or date_b is None:
        return False
    if date_a == date_b:
        return True
    # If one is year-only (4 digits), check if the other contains that year
    for a, b in [(date_a, date_b), (date_b, date_a)]:
        if len(a) == 4 and a.isdigit():
            if a in b:
                return True
    return False


def clean_annotations(annotations: dict[str, str | None]) -> dict[str, object]:
    """Convert annotation dict to gold standard format.

    Removes None values and converts numeric strings where appropriate.
    """
    cleaned = {}
    for field, value in annotations.items():
        if value is None:
            continue
        # Skip the identifier fields (handled separately)
        if field in ("nip",):
            continue
        # Try integer conversion for grade, cycles, etc.
        if field in ("grade", "chm_cycles", "histo_mitoses", "ik_clinique"):
            try:
                cleaned[field] = {"value": int(value)}
                continue
            except (ValueError, TypeError):
                pass
        # Try float for rx_dose
        if field in ("rx_dose",):
            try:
                cleaned[field] = {"value": float(value)}
                continue
            except (ValueError, TypeError):
                pass
        # Everything else is a string
        cleaned[field] = {"value": value}

    return cleaned


def match_bio_to_clinique(
    bio_entries: list[dict], clinique_entries: list[dict]
) -> list[tuple[dict, dict]]:
    """Match BIO and CLINIQUE entries by NIP + surgery date (fuzzy).

    Returns list of (bio_entry_or_empty, clinique_entry_or_empty) pairs.
    """
    # Build all CLINIQUE entries indexed for matching
    used_clinique = set()
    pairs: list[tuple[dict, dict]] = []

    # First pass: match each BIO entry to a CLINIQUE entry
    for bio in bio_entries:
        nip_bio = bio.get("nip")
        date_bio = bio.get("date_chir")
        matched = False
        for j, clinique in enumerate(clinique_entries):
            if j in used_clinique:
                continue
            nip_clin = clinique.get("nip")
            date_clin = clinique.get("chir_date")
            if nip_bio == nip_clin and dates_match_fuzzy(date_bio, date_clin):
                pairs.append((bio, clinique))
                used_clinique.add(j)
                matched = True
                break
        if not matched:
            pairs.append((bio, {}))

    # Second pass: add unmatched CLINIQUE entries
    for j, clinique in enumerate(clinique_entries):
        if j not in used_clinique:
            pairs.append(({}, clinique))

    return pairs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parse both CSVs
    bio_entries = parse_transposed_csv(REQ_BIO_PATH)
    clinique_entries = parse_transposed_csv(REQ_CLINIQUE_PATH)

    print(f"Parsed {len(bio_entries)} BIO entries, {len(clinique_entries)} CLINIQUE entries")

    for i, entry in enumerate(bio_entries):
        nip = entry.get("nip")
        date_chir = entry.get("date_chir")
        print(f"  BIO[{i}]: NIP={nip}, date_chir={date_chir}")

    for i, entry in enumerate(clinique_entries):
        nip = entry.get("nip")
        chir_date = entry.get("chir_date")
        evol = entry.get("evol_clinique")
        print(f"  CLINIQUE[{i}]: NIP={nip}, chir_date={chir_date}, evol={evol}")

    # Match BIO and CLINIQUE entries by NIP + fuzzy date
    matched_pairs = match_bio_to_clinique(bio_entries, clinique_entries)
    print(f"\nMatched {len(matched_pairs)} pairs:")

    gold_entries = []

    for pair_idx, (bio, clinique) in enumerate(matched_pairs):

        # Get identifiers from whichever is available
        nip = bio.get("nip") or clinique.get("nip")
        evol = clinique.get("evol_clinique")
        date_chir = bio.get("date_chir") or clinique.get("chir_date")

        if not nip:
            continue

        doc_id = make_document_id(nip, evol, len(gold_entries))

        # Determine document type from available annotations
        # If we have BIO fields filled, it's likely anapath or molecular
        # If we have CLINIQUE fields filled, it's likely consultation
        has_bio = any(
            v is not None
            for k, v in bio.items()
            if k not in ("nip", "date_chir", "num_labo")
        )
        has_clinique = any(
            v is not None
            for k, v in clinique.items()
            if k not in ("nip", "date_de_naissance", "sexe", "date_deces", "infos_deces")
        )

        # Merge annotations
        annotations = {}

        # Add BIO annotations (skip nip — it's an identifier, not a feature)
        bio_annotations = {
            k: v for k, v in bio.items()
            if k != "nip" and v is not None
        }
        annotations.update(clean_annotations(bio_annotations))

        # Add CLINIQUE annotations (skip nip)
        clinique_annotations = {
            k: v for k, v in clinique.items()
            if k != "nip" and v is not None
        }
        annotations.update(clean_annotations(clinique_annotations))

        if not annotations:
            continue

        gold_entry = {
            "document_id": doc_id,
            "patient_id": nip,
            "date_chir": date_chir,
            "evol_clinique": evol,
            "has_bio_annotations": has_bio,
            "has_clinique_annotations": has_clinique,
            "annotations": annotations,
        }
        gold_entries.append(gold_entry)

        # Write individual JSON file
        out_path = OUTPUT_DIR / f"{doc_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(gold_entry, f, ensure_ascii=False, indent=2)
        print(f"  -> Wrote {out_path.name} ({len(annotations)} annotated fields)")

    # Write a manifest of all gold standard entries
    manifest = {
        "total_entries": len(gold_entries),
        "source_files": {
            "REQ_BIO": str(REQ_BIO_PATH),
            "REQ_CLINIQUE": str(REQ_CLINIQUE_PATH),
        },
        "entries": [
            {
                "document_id": e["document_id"],
                "patient_id": e["patient_id"],
                "date_chir": e["date_chir"],
                "evol_clinique": e["evol_clinique"],
                "n_annotations": len(e["annotations"]),
                "has_bio": e["has_bio_annotations"],
                "has_clinique": e["has_clinique_annotations"],
            }
            for e in gold_entries
        ],
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nWrote manifest with {len(gold_entries)} entries to {manifest_path}")


if __name__ == "__main__":
    main()
