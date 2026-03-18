#!/usr/bin/env python3
"""Vary WHO classification year in the extended dataset.

Assigns 10% of docs to "2007", 20% to "2016", keeping 70% as "2021".
Updates both entity values and corresponding text spans.
"""

import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "data" / "generated_dataset_extended.jsonl"

SEED = 42
FRACTION_2007 = 0.10
FRACTION_2016 = 0.20


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

    # Find docs with classification_oms = "2021" AND matching text span
    eligible = []
    skipped = 0
    for i, doc in enumerate(docs):
        for ent in doc.get("entities", []):
            if ent["label"] == "classification_oms" and ent["value"] == "2021":
                span_text = doc["note_text"][ent["start"]:ent["end"]]
                if span_text == "2021":
                    eligible.append(i)
                else:
                    skipped += 1
                break  # one per doc is enough

    print(f"  {len(eligible)} documents have classification_oms = '2021' with matching span")
    if skipped:
        print(f"  {skipped} documents skipped (span mismatch from augmentation)")

    # Shuffle and assign
    rng = random.Random(SEED)
    rng.shuffle(eligible)

    n_2007 = int(len(eligible) * FRACTION_2007)
    n_2016 = int(len(eligible) * FRACTION_2016)

    assign_2007 = set(eligible[:n_2007])
    assign_2016 = set(eligible[n_2007 : n_2007 + n_2016])

    print(f"  Assigning: {n_2007} -> '2007', {n_2016} -> '2016', "
          f"{len(eligible) - n_2007 - n_2016} -> keep '2021'")

    # Apply changes
    changed = 0
    for i in assign_2007 | assign_2016:
        doc = docs[i]
        new_year = "2007" if i in assign_2007 else "2016"
        text = doc["note_text"]

        for ent in doc["entities"]:
            if ent["label"] == "classification_oms" and ent["value"] == "2021":
                s, e = ent["start"], ent["end"]
                # Update text
                text = text[:s] + new_year + text[e:]
                # Update entity
                ent["value"] = new_year
                changed += 1

        doc["note_text"] = text

        # Post-check: verify span matches
        for ent in doc["entities"]:
            if ent["label"] == "classification_oms":
                assert doc["note_text"][ent["start"]:ent["end"]] == ent["value"]

    print(f"  {changed} entities updated")

    # Summary counts
    counts = {"2007": 0, "2016": 0, "2021": 0, "other": 0}
    for doc in docs:
        for ent in doc.get("entities", []):
            if ent["label"] == "classification_oms":
                val = ent["value"]
                counts[val] = counts.get(val, 0) + 1

    print(f"\nFinal distribution of classification_oms:")
    for year, cnt in sorted(counts.items()):
        if cnt > 0:
            print(f"  {year}: {cnt}")

    # Write back
    write_jsonl(DATASET, docs)
    print(f"\nWritten {len(docs)} docs to {DATASET}")


if __name__ == "__main__":
    main()
