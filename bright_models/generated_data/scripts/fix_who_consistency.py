#!/usr/bin/env python3
"""Fix per-patient WHO classification consistency.

For each patient with conflicting classification_oms values across their
documents, harmonize all to the oldest year (min value) to preserve variety.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "data" / "generated_dataset_extended.jsonl"

NOTE_ID_RE = re.compile(r"synth-P(\d+)-(\w+)")


def load_jsonl(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def write_jsonl(path, docs):
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def main():
    print(f"Loading {DATASET} ...")
    docs = load_jsonl(DATASET)
    print(f"  {len(docs)} documents loaded")

    # Group doc indices by patient ID
    patient_docs = defaultdict(list)
    for i, doc in enumerate(docs):
        m = NOTE_ID_RE.match(doc["note_id"])
        if m:
            patient_docs[m.group(1)].append(i)

    # Find patients with inconsistent classification_oms
    patients_fixed = 0
    entities_changed = 0
    span_skipped = 0

    for pid, doc_indices in patient_docs.items():
        # Collect all classification_oms values for this patient
        years = set()
        for i in doc_indices:
            for ent in docs[i].get("entities", []):
                if ent["label"] == "classification_oms":
                    years.add(ent["value"])

        if len(years) <= 1:
            continue

        # Pick the oldest (min) year
        target_year = min(years)
        patients_fixed += 1

        # Update all docs for this patient
        for i in doc_indices:
            doc = docs[i]
            text = doc["note_text"]
            for ent in doc["entities"]:
                if ent["label"] == "classification_oms" and ent["value"] != target_year:
                    old_value = ent["value"]
                    s, e = ent["start"], ent["end"]
                    # Update text span only if it matches the current entity value
                    if text[s:e] == old_value:
                        text = text[:s] + target_year + text[e:]
                    else:
                        span_skipped += 1
                    ent["value"] = target_year
                    entities_changed += 1
            doc["note_text"] = text

    print(f"  {patients_fixed} patients had inconsistent values, fixed")
    print(f"  {entities_changed} entities changed")
    if span_skipped:
        print(f"  {span_skipped} spans skipped (text mismatch from augmentation)")

    # Verify: no more inconsistencies
    patient_years = defaultdict(set)
    for doc in docs:
        m = NOTE_ID_RE.match(doc["note_id"])
        if not m:
            continue
        for ent in doc.get("entities", []):
            if ent["label"] == "classification_oms":
                patient_years[m.group(1)].add(ent["value"])

    inconsistent = sum(1 for y in patient_years.values() if len(y) > 1)
    print(f"\n  Post-fix inconsistencies: {inconsistent}")

    # Final distribution
    counts = defaultdict(int)
    for doc in docs:
        for ent in doc.get("entities", []):
            if ent["label"] == "classification_oms":
                counts[ent["value"]] += 1

    print(f"\nFinal classification_oms distribution:")
    for year, cnt in sorted(counts.items()):
        print(f"  {year}: {cnt}")

    write_jsonl(DATASET, docs)
    print(f"\nWritten {len(docs)} docs to {DATASET}")


if __name__ == "__main__":
    main()
