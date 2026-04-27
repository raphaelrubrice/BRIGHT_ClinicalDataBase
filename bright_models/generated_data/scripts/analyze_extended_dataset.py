#!/usr/bin/env python3
"""Analyze the extended dataset and compare with the original."""

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ORIGINAL = BASE / "data" / "generated_dataset.jsonl"
EXTENDED = BASE / "data" / "generated_dataset_extended.jsonl"
OUTPUT_JSON = BASE / "scripts" / "analysis_extended_summary.json"

NOTE_ID_RE = re.compile(r"synth-P(\d+)-(\w+)")

ALL_111_FIELDS = [
    "date_chir", "num_labo", "date_rcp", "dn_date", "date_deces",
    "sexe", "annee_de_naissance", "activite_professionnelle",
    "antecedent_tumoral", "ik_clinique",
    "diag_histologique", "diag_integre", "classification_oms", "grade",
    "tumeur_lateralite", "tumeur_position", "dominance_cerebrale",
    "exam_radio_date_decouverte", "contraste_1er_symptome",
    "prise_de_contraste", "oedeme_1er_symptome", "calcif_1er_symptome",
    "date_1er_symptome", "epilepsie_1er_symptome", "ceph_hic_1er_symptome",
    "deficit_1er_symptome", "cognitif_1er_symptome", "autre_trouble_1er_symptome",
    "epilepsie", "ceph_hic", "deficit", "cognitif", "autre_trouble",
    "histo_necrose", "histo_pec", "histo_mitoses", "aspect_cellulaire",
    "ihc_idh1", "ihc_atrx", "ihc_p53", "ihc_fgfr3", "ihc_braf",
    "ihc_gfap", "ihc_olig2", "ihc_ki67", "ihc_hist_h3k27m",
    "ihc_hist_h3k27me3", "ihc_egfr_hirsch", "ihc_mmr",
    "mol_idh1", "mol_idh2", "mol_mgmt", "mol_h3f3a", "mol_hist1h3b",
    "mol_tert", "mol_CDKN2A",
    "ch1p", "ch19q", "ch1p19q_codel", "ch7p", "ch7q",
    "ch10p", "ch10q", "ch9p", "ch9q",
    "mol_p53", "mol_atrx", "mol_cic", "mol_fubp1", "mol_fgfr1",
    "mol_egfr_mut", "mol_prkca", "mol_pten", "mol_braf",
    "ampli_egfr", "ampli_mdm2", "ampli_cdk4", "ampli_met", "ampli_mdm4",
    "fusion_fgfr", "fusion_ntrk", "fusion_autre",
    "type_chirurgie", "localisation_chir", "qualite_exerese", "chir_date",
    "chimios", "chimio_protocole", "chm_date_debut", "chm_date_fin", "chm_cycles",
    "rx_date_debut", "rx_date_fin", "rx_dose", "rx_fractionnement",
    "localisation_radiotherapie",
    "anti_epileptiques", "essai_therapeutique", "corticoides", "optune",
    "evol_clinique", "progress_clinique", "progress_radiologique",
    "reponse_radiologique", "date_progression",
    "neuroncologue", "neurochirurgien", "radiotherapeute",
    "anatomo_pathologiste",
    "infos_deces", "survie_globale",
]


def load_jsonl(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


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


def analyze_dataset(docs, label="Dataset"):
    """Run full analysis on a set of docs."""
    doc_type_counter = Counter()
    patient_ids = set()
    for doc in docs:
        m = NOTE_ID_RE.match(doc["note_id"])
        if m:
            doc_type_counter[m.group(2)] += 1
            patient_ids.add(m.group(1))

    # Text stats
    all_lengths = [len(d["note_text"]) for d in docs]
    text_by_type = defaultdict(list)
    for doc in docs:
        m = NOTE_ID_RE.match(doc["note_id"])
        if m:
            text_by_type[m.group(2)].append(len(doc["note_text"]))

    # Annotation stats
    total_entities = sum(len(d.get("entities", [])) for d in docs)
    entities_per_doc = [len(d.get("entities", [])) for d in docs]

    field_counter = Counter()
    field_total = Counter()
    for doc in docs:
        seen = set()
        for e in doc.get("entities", []):
            field_total[e["label"]] += 1
            seen.add(e["label"])
        for f in seen:
            field_counter[f] += 1

    # Value distributions for key fields
    KEY_FIELDS = ["sexe", "grade", "classification_oms", "tumeur_lateralite",
                  "type_chirurgie", "mol_idh1", "mol_mgmt", "ch1p19q_codel",
                  "diag_histologique", "chimio_protocole", "anti_epileptiques",
                  "corticoides", "tumeur_position"]
    value_dists = defaultdict(Counter)
    for doc in docs:
        for e in doc.get("entities", []):
            if e["label"] in KEY_FIELDS:
                value_dists[e["label"]][e["value"]] += 1

    # Span validation
    span_ok = span_empty = span_oob = span_bad = 0
    for doc in docs:
        text = doc["note_text"]
        tlen = len(text)
        for e in doc.get("entities", []):
            s, end = e["start"], e["end"]
            if s >= end:
                span_bad += 1
            elif end > tlen or s < 0:
                span_oob += 1
            elif text[s:end].strip() == "":
                span_empty += 1
            else:
                span_ok += 1

    # WHO checks
    def get_fv(doc):
        r = defaultdict(list)
        for e in doc.get("entities", []):
            r[e["label"]].append(e["value"].lower().strip())
        return r

    gbm_total = gbm_wt = gbm_viol = 0
    oligo_total = oligo_codel = 0
    for doc in docs:
        fv = get_fv(doc)
        diag = " ".join(fv.get("diag_histologique", [])).lower()
        if "glioblastome" in diag or "gbm" in diag or "glioblastoma" in diag:
            gbm_total += 1
            idh = fv.get("mol_idh1", [])
            if any("non" in v or "wt" in v or "sauvage" in v or "wild" in v for v in idh):
                gbm_wt += 1
            elif idh and not any("non" in v or "wt" in v or "sauvage" in v or "wild" in v for v in idh):
                gbm_viol += 1
        if "oligodendrogliome" in diag or "oligodendroglioma" in diag or "oligo" in diag:
            oligo_total += 1
            if fv.get("ch1p19q_codel"):
                oligo_codel += 1

    return {
        "total_docs": len(docs),
        "unique_patients": len(patient_ids),
        "doc_types": dict(doc_type_counter),
        "text_stats_overall": stats_summary(all_lengths),
        "text_stats_by_type": {k: stats_summary(v) for k, v in text_by_type.items()},
        "total_entities": total_entities,
        "entities_per_doc": stats_summary(entities_per_doc),
        "fields_present": len(set(field_counter.keys())),
        "fields_total": len(ALL_111_FIELDS),
        "missing_fields": sorted(set(ALL_111_FIELDS) - set(field_counter.keys())),
        "value_distributions": {
            field: dict(value_dists[field].most_common(25))
            for field in KEY_FIELDS if value_dists[field]
        },
        "span_validation": {
            "total": span_ok + span_empty + span_oob + span_bad,
            "valid": span_ok,
            "empty": span_empty,
            "oob": span_oob,
            "bad": span_bad,
        },
        "who_checks": {
            "gbm_total": gbm_total,
            "gbm_idh_wt": gbm_wt,
            "gbm_idh_violation": gbm_viol,
            "oligo_total": oligo_total,
            "oligo_codel_present": oligo_codel,
        },
    }


def print_comparison(orig, ext, aug_only):
    print("=" * 70)
    print("EXTENDED DATASET ANALYSIS, COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'Original':>12} {'Extended':>12} {'Aug-only':>12}")
    print("-" * 70)
    print(f"{'Total documents':<35} {orig['total_docs']:>12} {ext['total_docs']:>12} {aug_only['total_docs']:>12}")
    print(f"{'Unique patients':<35} {orig['unique_patients']:>12} {ext['unique_patients']:>12} {aug_only['unique_patients']:>12}")
    print(f"{'Total entities':<35} {orig['total_entities']:>12} {ext['total_entities']:>12} {aug_only['total_entities']:>12}")
    print(f"{'Fields present':<35} {orig['fields_present']:>12} {ext['fields_present']:>12} {aug_only['fields_present']:>12}")

    print(f"\n{'Entities/doc (mean)':<35} {orig['entities_per_doc']['mean']:>12} {ext['entities_per_doc']['mean']:>12} {aug_only['entities_per_doc']['mean']:>12}")
    print(f"{'Text length (mean chars)':<35} {orig['text_stats_overall']['mean']:>12} {ext['text_stats_overall']['mean']:>12} {aug_only['text_stats_overall']['mean']:>12}")

    print(f"\n{'Span valid':<35} {orig['span_validation']['valid']:>12} {ext['span_validation']['valid']:>12} {aug_only['span_validation']['valid']:>12}")
    print(f"{'Span errors':<35} {orig['span_validation']['empty'] + orig['span_validation']['oob'] + orig['span_validation']['bad']:>12} {ext['span_validation']['empty'] + ext['span_validation']['oob'] + ext['span_validation']['bad']:>12} {aug_only['span_validation']['empty'] + aug_only['span_validation']['oob'] + aug_only['span_validation']['bad']:>12}")

    print(f"\n{'GBM docs':<35} {orig['who_checks']['gbm_total']:>12} {ext['who_checks']['gbm_total']:>12} {aug_only['who_checks']['gbm_total']:>12}")
    print(f"{'GBM IDH-wt confirmed':<35} {orig['who_checks']['gbm_idh_wt']:>12} {ext['who_checks']['gbm_idh_wt']:>12} {aug_only['who_checks']['gbm_idh_wt']:>12}")
    print(f"{'GBM IDH violations':<35} {orig['who_checks']['gbm_idh_violation']:>12} {ext['who_checks']['gbm_idh_violation']:>12} {aug_only['who_checks']['gbm_idh_violation']:>12}")

    # Value variety comparison for key fields
    print("\n" + "=" * 70)
    print("VALUE VARIETY COMPARISON (unique values per field)")
    print("=" * 70)
    compare_fields = ["mol_idh1", "mol_mgmt", "type_chirurgie", "tumeur_lateralite",
                      "diag_histologique", "chimio_protocole", "tumeur_position",
                      "corticoides", "anti_epileptiques"]
    print(f"\n{'Field':<30} {'Orig uniq':>12} {'Ext uniq':>12}")
    print("-" * 55)
    for field in compare_fields:
        o_vals = len(orig["value_distributions"].get(field, {}))
        e_vals = len(ext["value_distributions"].get(field, {}))
        print(f"{field:<30} {o_vals:>12} {e_vals:>12}")

    # Show extended distributions for key fields
    print("\n" + "=" * 70)
    print("VALUE DISTRIBUTIONS IN EXTENDED DATASET (top 15)")
    print("=" * 70)
    for field in ["mol_idh1", "mol_mgmt", "type_chirurgie", "diag_histologique",
                  "chimio_protocole", "tumeur_lateralite", "tumeur_position"]:
        dist = ext["value_distributions"].get(field, {})
        total = sum(dist.values())
        if not dist:
            continue
        print(f"\n  {field} (n={total}):")
        for val, cnt in sorted(dist.items(), key=lambda x: -x[1])[:15]:
            print(f"    {val:50s}: {cnt:5d} ({100*cnt/total:5.1f}%)")

    # Doc type distribution
    print("\n" + "=" * 70)
    print("DOC TYPE DISTRIBUTION")
    print("=" * 70)
    print(f"\n{'Type':<20} {'Original':>10} {'Extended':>10} {'Aug-only':>10}")
    print("-" * 55)
    for dtype in ["consultation", "rcp", "anapath"]:
        o = orig["doc_types"].get(dtype, 0)
        e = ext["doc_types"].get(dtype, 0)
        a = aug_only["doc_types"].get(dtype, 0)
        print(f"{dtype:<20} {o:>10} {e:>10} {a:>10}")


def main():
    print("Loading original dataset...")
    orig_docs = load_jsonl(ORIGINAL)
    print(f"  {len(orig_docs)} documents")

    print("Loading extended dataset...")
    ext_docs = load_jsonl(EXTENDED)
    print(f"  {len(ext_docs)} documents")

    # Separate augmented-only docs (patient IDs >= 700)
    aug_docs = []
    for doc in ext_docs:
        m = NOTE_ID_RE.match(doc["note_id"])
        if m and int(m.group(1)) >= 700:
            aug_docs.append(doc)
    print(f"  {len(aug_docs)} augmented-only documents")

    print("\nAnalyzing original...")
    orig_stats = analyze_dataset(orig_docs, "Original")

    print("Analyzing extended...")
    ext_stats = analyze_dataset(ext_docs, "Extended")

    print("Analyzing augmented-only...")
    aug_stats = analyze_dataset(aug_docs, "Augmented")

    print_comparison(orig_stats, ext_stats, aug_stats)

    # Save JSON
    summary = {
        "original": orig_stats,
        "extended": ext_stats,
        "augmented_only": aug_stats,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nJSON summary saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
