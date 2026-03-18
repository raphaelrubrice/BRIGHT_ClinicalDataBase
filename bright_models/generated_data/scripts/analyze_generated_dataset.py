#!/usr/bin/env python3
"""Analyze the quality of the generated dataset (final filtered JSONL)."""

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent          # generated_data/
DATASET = BASE / "data" / "generated_dataset.jsonl"
CHECKPOINTS = BASE / "checkpoints" / "step2"
OUTPUT_JSON = BASE / "scripts" / "analysis_summary.json"

# ---------------------------------------------------------------------------
# 111 canonical fields (copied from profiles_validation for standalone use)
# ---------------------------------------------------------------------------
ALL_111_FIELDS = [
    # Identifiers & dates
    "date_chir", "num_labo", "date_rcp", "dn_date", "date_deces",
    # Demographics
    "sexe", "annee_de_naissance", "activite_professionnelle",
    "antecedent_tumoral", "ik_clinique",
    # Diagnosis
    "diag_histologique", "diag_integre", "classification_oms", "grade",
    # Tumor location
    "tumeur_lateralite", "tumeur_position", "dominance_cerebrale",
    # Radiology
    "exam_radio_date_decouverte", "contraste_1er_symptome",
    "prise_de_contraste", "oedeme_1er_symptome", "calcif_1er_symptome",
    # Initial symptoms
    "date_1er_symptome", "epilepsie_1er_symptome", "ceph_hic_1er_symptome",
    "deficit_1er_symptome", "cognitif_1er_symptome", "autre_trouble_1er_symptome",
    # Current symptoms
    "epilepsie", "ceph_hic", "deficit", "cognitif", "autre_trouble",
    # Histology
    "histo_necrose", "histo_pec", "histo_mitoses", "aspect_cellulaire",
    # IHC
    "ihc_idh1", "ihc_atrx", "ihc_p53", "ihc_fgfr3", "ihc_braf",
    "ihc_gfap", "ihc_olig2", "ihc_ki67", "ihc_hist_h3k27m",
    "ihc_hist_h3k27me3", "ihc_egfr_hirsch", "ihc_mmr",
    # Molecular 1
    "mol_idh1", "mol_idh2", "mol_mgmt", "mol_h3f3a", "mol_hist1h3b",
    "mol_tert", "mol_CDKN2A",
    # Chromosomal
    "ch1p", "ch19q", "ch1p19q_codel", "ch7p", "ch7q",
    "ch10p", "ch10q", "ch9p", "ch9q",
    # Molecular 2
    "mol_p53", "mol_atrx", "mol_cic", "mol_fubp1", "mol_fgfr1",
    "mol_egfr_mut", "mol_prkca", "mol_pten", "mol_braf",
    # Amplifications & fusions
    "ampli_egfr", "ampli_mdm2", "ampli_cdk4", "ampli_met", "ampli_mdm4",
    "fusion_fgfr", "fusion_ntrk", "fusion_autre",
    # Surgery
    "type_chirurgie", "localisation_chir", "qualite_exerese", "chir_date",
    # Chemotherapy
    "chimios", "chimio_protocole", "chm_date_debut", "chm_date_fin", "chm_cycles",
    # Radiotherapy
    "rx_date_debut", "rx_date_fin", "rx_dose", "rx_fractionnement",
    "localisation_radiotherapie",
    # Adjuvant
    "anti_epileptiques", "essai_therapeutique", "corticoides", "optune",
    # Evolution
    "evol_clinique", "progress_clinique", "progress_radiologique",
    "reponse_radiologique", "date_progression",
    # Care team
    "neuroncologue", "neurochirurgien", "radiotherapeute",
    "anatomo_pathologiste",
    # Outcome
    "infos_deces", "survie_globale",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NOTE_ID_RE = re.compile(r"synth-P(\d+)-(\w+)")


def parse_note_id(note_id: str):
    m = NOTE_ID_RE.match(note_id)
    if m:
        return m.group(1), m.group(2)  # patient_id, doc_type
    return None, None


def stats_summary(values):
    if not values:
        return {}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
        "stdev": round(statistics.stdev(values), 1) if len(values) > 1 else 0,
    }


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
print(f"Loading dataset from {DATASET}")
docs = []
with open(DATASET, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            docs.append(json.loads(line))

print(f"Loaded {len(docs)} documents\n")

# ---------------------------------------------------------------------------
# 1. Volume & document-type distribution
# ---------------------------------------------------------------------------
doc_type_counter = Counter()
patient_ids = set()
patient_doc_counts = Counter()

for doc in docs:
    pid, dtype = parse_note_id(doc["note_id"])
    if dtype:
        doc_type_counter[dtype] += 1
    if pid:
        patient_ids.add(pid)
        patient_doc_counts[pid] += 1

print("=" * 60)
print("1. VOLUME & DOCUMENT-TYPE DISTRIBUTION")
print("=" * 60)
print(f"  Total documents: {len(docs)}")
print(f"  Unique patients: {len(patient_ids)}")
for dtype, count in doc_type_counter.most_common():
    print(f"  {dtype:15s}: {count:5d}  ({100*count/len(docs):.1f}%)")

# Docs per patient
dpc_values = list(patient_doc_counts.values())
dpc_stats = stats_summary(dpc_values)
print(f"\n  Docs per patient: mean={dpc_stats['mean']}, "
      f"median={dpc_stats['median']}, min={dpc_stats['min']}, max={dpc_stats['max']}")

# ---------------------------------------------------------------------------
# 2. Text statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. TEXT STATISTICS (characters)")
print("=" * 60)

text_lengths_by_type = defaultdict(list)
word_counts_by_type = defaultdict(list)

for doc in docs:
    _, dtype = parse_note_id(doc["note_id"])
    text = doc["note_text"]
    text_lengths_by_type[dtype].append(len(text))
    word_counts_by_type[dtype].append(len(text.split()))

all_lengths = [len(d["note_text"]) for d in docs]
print(f"  Overall: {stats_summary(all_lengths)}")
for dtype in sorted(text_lengths_by_type):
    s = stats_summary(text_lengths_by_type[dtype])
    w = stats_summary(word_counts_by_type[dtype])
    print(f"  {dtype:15s}: chars={s}, words(mean={w['mean']}, median={w['median']})")

# ---------------------------------------------------------------------------
# 3. Annotation statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. ANNOTATION STATISTICS")
print("=" * 60)

total_entities = 0
entities_per_doc = []
field_counter = Counter()  # field -> number of docs containing it
field_total = Counter()    # field -> total occurrences

for doc in docs:
    ents = doc.get("entities", [])
    total_entities += len(ents)
    entities_per_doc.append(len(ents))
    seen_fields = set()
    for e in ents:
        label = e["label"]
        field_total[label] += 1
        seen_fields.add(label)
    for f in seen_fields:
        field_counter[f] += 1

print(f"  Total entities: {total_entities}")
print(f"  Entities per doc: {stats_summary(entities_per_doc)}")

# Field coverage
fields_present = set(field_counter.keys())
fields_expected = set(ALL_111_FIELDS)
missing_fields = fields_expected - fields_present
extra_fields = fields_present - fields_expected

print(f"\n  Fields present: {len(fields_present)} / {len(ALL_111_FIELDS)}")
if missing_fields:
    print(f"  Missing fields ({len(missing_fields)}): {sorted(missing_fields)}")
if extra_fields:
    print(f"  Extra fields ({len(extra_fields)}): {sorted(extra_fields)}")

# Top 30 fields by doc frequency
print("\n  Top 30 fields by document frequency:")
for rank, (field, count) in enumerate(field_counter.most_common(30), 1):
    pct = 100 * count / len(docs)
    tot = field_total[field]
    print(f"    {rank:2d}. {field:35s}: {count:5d} docs ({pct:5.1f}%), {tot:5d} total")

# Bottom 15 fields
print("\n  Bottom 15 fields by document frequency:")
for field, count in field_counter.most_common()[-15:]:
    pct = 100 * count / len(docs)
    print(f"      {field:35s}: {count:5d} docs ({pct:5.1f}%)")

# ---------------------------------------------------------------------------
# 4. Value distributions for key fields
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. VALUE DISTRIBUTIONS (key controlled-vocab fields)")
print("=" * 60)

KEY_FIELDS = [
    "sexe", "grade", "classification_oms", "tumeur_lateralite",
    "type_chirurgie", "mol_idh1", "mol_mgmt", "ch1p19q_codel",
]
FREE_TEXT_TOP = {
    "diag_histologique": 15,
    "chimio_protocole": 10,
    "tumeur_position": 15,
    "diag_integre": 15,
}

value_distributions = defaultdict(Counter)
for doc in docs:
    for e in doc.get("entities", []):
        value_distributions[e["label"]][e["value"]] += 1

for field in KEY_FIELDS:
    dist = value_distributions[field]
    total = sum(dist.values())
    print(f"\n  {field} (n={total}):")
    for val, cnt in dist.most_common():
        print(f"    {val:40s}: {cnt:5d} ({100*cnt/total:5.1f}%)")

for field, top_n in FREE_TEXT_TOP.items():
    dist = value_distributions[field]
    total = sum(dist.values())
    print(f"\n  {field} — top {top_n} (n={total}):")
    for val, cnt in dist.most_common(top_n):
        print(f"    {val:50s}: {cnt:5d} ({100*cnt/total:5.1f}%)")

# ---------------------------------------------------------------------------
# 5. Span validation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. SPAN VALIDATION")
print("=" * 60)

span_ok = 0
span_empty = 0
span_oob = 0
span_start_ge_end = 0
span_examples = []

for doc in docs:
    text = doc["note_text"]
    text_len = len(text)
    for e in doc.get("entities", []):
        s, end = e["start"], e["end"]
        if s >= end:
            span_start_ge_end += 1
            if len(span_examples) < 5:
                span_examples.append((doc["note_id"], e["label"], s, end))
        elif end > text_len or s < 0:
            span_oob += 1
        elif text[s:end].strip() == "":
            span_empty += 1
        else:
            span_ok += 1

total_spans = span_ok + span_empty + span_oob + span_start_ge_end
print(f"  Total spans: {total_spans}")
print(f"  Valid:        {span_ok} ({100*span_ok/total_spans:.2f}%)")
print(f"  Empty text:   {span_empty}")
print(f"  Out of bounds:{span_oob}")
print(f"  start >= end: {span_start_ge_end}")
if span_examples:
    print(f"  Examples of bad spans:")
    for nid, label, s, e in span_examples:
        print(f"    {nid}: {label} start={s} end={e}")

# ---------------------------------------------------------------------------
# 6. WHO 2021 coherence spot-checks
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. WHO 2021 COHERENCE SPOT-CHECKS")
print("=" * 60)


def get_doc_field_values(doc):
    """Return dict of label -> list of values for a document."""
    result = defaultdict(list)
    for e in doc.get("entities", []):
        result[e["label"]].append(e["value"].lower().strip())
    return result


gbm_total = 0
gbm_idh_wt = 0
gbm_idh_missing = 0
gbm_idh_violation = 0

oligo_total = 0
oligo_codel_present = 0
oligo_codel_missing = 0

astro_idh_total = 0
astro_idh_correct = 0

for doc in docs:
    fv = get_doc_field_values(doc)
    diag = " ".join(fv.get("diag_histologique", [])).lower()

    # GBM check
    if "glioblastome" in diag or "gbm" in diag:
        gbm_total += 1
        idh1_vals = fv.get("mol_idh1", [])
        if not idh1_vals:
            gbm_idh_missing += 1
        elif any("non" in v or "wt" in v or "sauvage" in v or "wild" in v for v in idh1_vals):
            gbm_idh_wt += 1
        else:
            gbm_idh_violation += 1

    # Oligodendroglioma check
    if "oligodendrogliome" in diag or "oligo" in diag:
        oligo_total += 1
        codel_vals = fv.get("ch1p19q_codel", [])
        if codel_vals:
            oligo_codel_present += 1
        else:
            oligo_codel_missing += 1

    # Astrocytoma IDH-mutant check
    if "astrocytome" in diag and ("idh" in diag or any("mut" in v for v in fv.get("mol_idh1", []))):
        astro_idh_total += 1
        idh1_vals = fv.get("mol_idh1", [])
        if any("mut" in v and "non" not in v for v in idh1_vals):
            astro_idh_correct += 1

print(f"  GBM documents: {gbm_total}")
print(f"    IDH wildtype confirmed: {gbm_idh_wt} ({100*gbm_idh_wt/max(gbm_total,1):.1f}%)")
print(f"    IDH missing:            {gbm_idh_missing}")
print(f"    IDH violation (mutated): {gbm_idh_violation}")

print(f"\n  Oligodendroglioma documents: {oligo_total}")
print(f"    1p/19q codel present: {oligo_codel_present} ({100*oligo_codel_present/max(oligo_total,1):.1f}%)")
print(f"    1p/19q codel missing: {oligo_codel_missing}")

print(f"\n  Astrocytoma IDH-mutant documents: {astro_idh_total}")
print(f"    IDH mutation confirmed: {astro_idh_correct} ({100*astro_idh_correct/max(astro_idh_total,1):.1f}%)")

# ---------------------------------------------------------------------------
# 7. Diagnosis-type breakdown
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. TUMOR CATEGORY BREAKDOWN")
print("=" * 60)

tumor_cats = Counter()
for doc in docs:
    fv = get_doc_field_values(doc)
    diag = " ".join(fv.get("diag_histologique", [])).lower()
    if "glioblastome" in diag or "gbm" in diag:
        tumor_cats["GBM"] += 1
    elif "oligodendrogliome" in diag or "oligodendroglioma" in diag:
        tumor_cats["Oligodendroglioma"] += 1
    elif "astrocytome" in diag:
        tumor_cats["Astrocytoma"] += 1
    elif "gliome" in diag:
        tumor_cats["Glioma (other)"] += 1
    elif "meningiome" in diag:
        tumor_cats["Meningioma"] += 1
    elif "ependymome" in diag:
        tumor_cats["Ependymoma"] += 1
    elif diag:
        tumor_cats[f"Other ({diag[:40]})"] += 1
    else:
        tumor_cats["No diagnosis"] += 1

for cat, count in tumor_cats.most_common():
    print(f"  {cat:40s}: {count:5d} ({100*count/len(docs):.1f}%)")

# ---------------------------------------------------------------------------
# 8. Per-doc-type annotation density
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("8. ANNOTATION DENSITY BY DOC TYPE")
print("=" * 60)

ents_by_type = defaultdict(list)
for doc in docs:
    _, dtype = parse_note_id(doc["note_id"])
    ents_by_type[dtype].append(len(doc.get("entities", [])))

for dtype in sorted(ents_by_type):
    s = stats_summary(ents_by_type[dtype])
    print(f"  {dtype:15s}: {s}")

# ---------------------------------------------------------------------------
# Save JSON summary
# ---------------------------------------------------------------------------
summary = {
    "total_docs": len(docs),
    "unique_patients": len(patient_ids),
    "doc_type_distribution": dict(doc_type_counter),
    "total_entities": total_entities,
    "entities_per_doc": stats_summary(entities_per_doc),
    "text_length_overall": stats_summary(all_lengths),
    "text_length_by_type": {k: stats_summary(v) for k, v in text_lengths_by_type.items()},
    "fields_present": len(fields_present),
    "fields_total": len(ALL_111_FIELDS),
    "missing_fields": sorted(missing_fields),
    "field_frequency_top30": [
        {"field": f, "docs": c, "pct": round(100*c/len(docs), 1)}
        for f, c in field_counter.most_common(30)
    ],
    "span_validation": {
        "total": total_spans,
        "valid": span_ok,
        "empty": span_empty,
        "out_of_bounds": span_oob,
        "start_ge_end": span_start_ge_end,
    },
    "who_checks": {
        "gbm_total": gbm_total,
        "gbm_idh_wt": gbm_idh_wt,
        "gbm_idh_violation": gbm_idh_violation,
        "oligo_total": oligo_total,
        "oligo_codel_present": oligo_codel_present,
        "astro_idh_total": astro_idh_total,
        "astro_idh_correct": astro_idh_correct,
    },
    "tumor_categories": dict(tumor_cats),
    "value_distributions": {
        field: dict(value_distributions[field].most_common(20))
        for field in KEY_FIELDS
    },
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nJSON summary saved to {OUTPUT_JSON}")
print("Done.")
