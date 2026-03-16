#!/usr/bin/env python3
"""
profile_validation.py — Validation of synthetic neuro-oncology patient profiles.

Checks:
  1. Diagnosis proportions
  2. WHO 2021 classification constraints
  3. Value distributions for low-cardinality fields
  4. Date coherence
  5. Field coverage (111 fields)

Usage:
  python profile_validation.py <file1.json> [file2.json ...]
  python profile_validation.py profiles_dir/
"""

import json, sys, re
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path


# ═════════════════════════════════════════════════════════════════════════════
# WHO 2021 DETECTION HELPERS
#
# CRITICAL: substring ordering traps
#   "absence de codélétion" contains "codélétion" → check "absence" FIRST
#   "non délété" contains "délété"               → check "non" FIRST
#   "non muté" contains "muté"                   → check "non muté" FIRST
# ═════════════════════════════════════════════════════════════════════════════

def _is_positive_mutation(s):
    s = s.lower().strip()
    if not s or s in ("", "none"):
        return False
    if "non muté" in s or "non " in s or s == "wt":
        return False
    if "muté" in s or "r132" in s or "r172" in s:
        return True
    return False


def normalize_diag(profile):
    """Classify into 'gbm', 'astro_idh', 'oligo', or 'other'."""
    diag_h = (profile.get("diag_histologique") or "").lower()
    diag_i = (profile.get("diag_integre") or "").lower()
    combined = diag_h + " " + diag_i

    if "oligodendrogliome" in combined:
        return "oligo"
    if "glioblastome" in combined:
        return "gbm"
    # Specific subtypes containing "astrocytome" → "other" BEFORE catch-all
    if "pilocytique" in combined or "pilocytic" in combined:
        return "other"
    if "xanthoastrocytome" in combined or "pxa" in combined:
        return "other"
    if "sous-épendymaire" in combined or "subependym" in combined:
        return "other"
    # H3-mutated (may say "astrocytome" in diag_histologique)
    if "h3" in combined or "ligne médiane" in combined or "dipg" in combined:
        return "other"
    mol_h3 = str(profile.get("mol_h3f3a", "")).lower()
    ihc_h3 = str(profile.get("ihc_hist_h3k27m", "")).lower()
    if _is_positive_mutation(mol_h3) or "positif" in ihc_h3:
        return "other"
    # Other non-astro types
    for kw in ["épendymome", "ependym", "médulloblastome", "medullo",
               "gangliogliome", "craniopharyngiome"]:
        if kw in combined:
            return "other"
    # BRAF-mutated not labeled IDH-muté
    mol_braf = str(profile.get("mol_braf", "")).lower()
    ihc_braf = str(profile.get("ihc_braf", "")).lower()
    has_braf = (_is_positive_mutation(mol_braf) or "fusion" in mol_braf or "positif" in ihc_braf)
    if has_braf and "idh-muté" not in combined and "idh muté" not in combined:
        return "other"
    # Generic astrocytome catch-all
    if "astrocytome" in combined:
        return "astro_idh"
    return "other"


def has_1p19q_codel(p):
    """CRITICAL: check 'absence' BEFORE 'codélétion'."""
    codel = p.get("ch1p19q_codel")
    if codel is None:
        return None
    if isinstance(codel, bool):
        return codel
    s = str(codel).lower().strip()
    if "absence" in s or s in ("false", "0", "non"):
        return False
    if "codélétion" in s or s in ("true", "1", "oui"):
        return True
    return None


def is_idh_mutated(p):
    ihc = str(p.get("ihc_idh1", "")).lower()
    if "positif" in ihc:
        return True
    for field in ["mol_idh1", "mol_idh2"]:
        val = str(p.get(field, "")).lower()
        if _is_positive_mutation(val):
            return True
    return False


def is_atrx_lost(p):
    """'negatif'/'négatif' in IHC for ATRX = lost expression."""
    ihc = str(p.get("ihc_atrx", "")).lower()
    mol = str(p.get("mol_atrx", "")).lower()
    return ("perdu" in ihc or "lost" in ihc or "negatif" in ihc
            or _is_positive_mutation(mol))


def has_cdkn2a_deletion(p):
    """CRITICAL: check 'non délété' BEFORE 'délété'."""
    val = p.get("mol_CDKN2A")
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    s = str(val).lower().strip()
    if "non" in s or s in ("false", "0", "wt"):
        return False
    if "délété" in s or "homozygote" in s or s in ("true", "1"):
        return True
    return False


def has_tert_mutation(p):
    return _is_positive_mutation(str(p.get("mol_tert", "")))


def has_egfr_amplification(p):
    val = p.get("ampli_egfr")
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    s = str(val).lower()
    return "amplifié" in s or s in ("true", "1", "oui")


def has_plus7_minus10(p):
    ch7 = str(p.get("ch7p", "")).lower() + str(p.get("ch7q", "")).lower()
    ch10 = str(p.get("ch10p", "")).lower() + str(p.get("ch10q", "")).lower()
    return "gain" in ch7 and ("perte" in ch10 or "délétion" in ch10)


# ═════════════════════════════════════════════════════════════════════════════
# DATE PARSING
# ═════════════════════════════════════════════════════════════════════════════

def parse_date(val):
    if val is None:
        return None
    val = str(val).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(val, fmt).date()
        except ValueError:
            continue
    return None


def parse_year(val):
    if val is None:
        return None
    try:
        y = int(val)
        return y if 1900 <= y <= 2030 else None
    except (ValueError, TypeError):
        return None


# ═════════════════════════════════════════════════════════════════════════════
# 111 FIELDS
# ═════════════════════════════════════════════════════════════════════════════

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
    "ch1p", "ch19q", "ch1p19q_codel", "ch7p", "ch7q", "ch10p", "ch10q",
    "ch9p", "ch9q",
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
    "neuroncologue", "neurochirurgien", "radiotherapeute", "anatomo_pathologiste",
    "infos_deces", "survie_globale",
]

META_FIELDS = ["patient_id", "document_types"]


# ═════════════════════════════════════════════════════════════════════════════
# LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_profiles(paths):
    all_profiles = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            all_profiles.extend(data)
        elif isinstance(data, dict):
            all_profiles.append(data)
    return all_profiles


# ═════════════════════════════════════════════════════════════════════════════
# 1. PROPORTIONS
# ═════════════════════════════════════════════════════════════════════════════

def check_proportions(profiles):
    print("\n" + "=" * 70)
    print("1. DIAGNOSIS PROPORTIONS")
    print("=" * 70)
    cats = Counter(normalize_diag(p) for p in profiles)
    total = len(profiles)
    targets = {"gbm": 45, "astro_idh": 20, "oligo": 20, "other": 15}
    labels = {"gbm": "Glioblastome IDH-sauvage", "astro_idh": "Astrocytome IDH-muté",
              "oligo": "Oligodendrogliome", "other": "Autres (pilocytique, DIPG, NEC…)"}
    print(f"\n  Total: {total}\n")
    print(f"  {'Category':<40} {'N':>5} {'Actual':>7} {'Target':>7} {'Δ':>7}")
    print(f"  {'-'*40} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")
    for cat in ["gbm", "astro_idh", "oligo", "other"]:
        n = cats.get(cat, 0)
        pct = n / total * 100 if total else 0
        t = targets[cat]
        d = pct - t
        flag = " ⚠️" if abs(d) > 10 else ""
        print(f"  {labels[cat]:<40} {n:>5} {pct:>6.1f}% {t:>6.0f}% {d:>+6.1f}%{flag}")


# ═════════════════════════════════════════════════════════════════════════════
# 2. WHO 2021
# ═════════════════════════════════════════════════════════════════════════════

def check_oms2021(profiles):
    print("\n" + "=" * 70)
    print("2. WHO 2021 CLASSIFICATION CONSTRAINTS")
    print("=" * 70)
    errors = []
    for p in profiles:
        pid = p.get("patient_id", "?")
        cat = normalize_diag(p)
        grade = p.get("grade")
        idh_mut = is_idh_mutated(p)
        codel = has_1p19q_codel(p)
        atrx_lost = is_atrx_lost(p)
        cdkn2a_del = has_cdkn2a_deletion(p)

        if cat == "oligo":
            if not idh_mut:
                errors.append(f"  {pid}: Oligodendrogliome WITHOUT IDH mutation")
            if codel is False:
                errors.append(f"  {pid}: Oligodendrogliome WITHOUT 1p/19q codeletion")
            if codel is None:
                errors.append(f"  {pid}: Oligodendrogliome with 1p/19q codeletion NOT TESTED")
        if cat == "gbm" and idh_mut:
            errors.append(f"  {pid}: Glioblastome with IDH MUTATED (invalid under OMS 2021)")
        if cat == "astro_idh":
            if not idh_mut:
                errors.append(f"  {pid}: Astrocytome IDH-muté WITHOUT IDH mutation")
            if codel is True:
                errors.append(f"  {pid}: Astrocytome IDH-muté WITH 1p/19q codeletion (should be oligo)")
        if atrx_lost and codel is True:
            errors.append(f"  {pid}: ATRX lost AND 1p/19q codeletion (MUTUALLY EXCLUSIVE)")
        if cat == "astro_idh" and cdkn2a_del is True:
            if grade is not None and int(grade) != 4:
                errors.append(f"  {pid}: Astrocytome IDH-muté + CDKN2A del homozygote → grade should be 4, got {grade}")

        # Rule 6: IDH-wt astrocytome + GBM markers
        diag_h = (p.get("diag_histologique") or "").lower()
        if "astrocytome" in diag_h and cat == "astro_idh" and not idh_mut:
            if has_tert_mutation(p) or has_egfr_amplification(p) or has_plus7_minus10(p):
                errors.append(f"  {pid}: IDH-wt astrocytome with TERT/EGFR/+7-10 → should be GBM")

        # Rule 8: Treatment coherence
        proto = str(p.get("chimio_protocole", "")).lower()
        if cat == "oligo" and "stupp" in proto:
            errors.append(f"  {pid}: Oligodendrogliome with Stupp protocol (typically PCV)")
        if cat == "gbm" and "pcv" in proto and "stupp" not in proto:
            errors.append(f"  {pid}: Glioblastome with PCV protocol (typically Stupp)")

    if not errors:
        print(f"\n  ✅ All {len(profiles)} profiles pass OMS 2021 constraints.")
    else:
        print(f"\n  ❌ {len(errors)} constraint violation(s) found:\n")
        for e in errors:
            print(e)


# ═════════════════════════════════════════════════════════════════════════════
# 3. VALUE DISTRIBUTIONS
# ═════════════════════════════════════════════════════════════════════════════

def check_value_distributions(profiles):
    print("\n" + "=" * 70)
    print("3. VALUE DISTRIBUTIONS (fields with <15 unique values)")
    print("=" * 70)
    field_values = defaultdict(list)
    for p in profiles:
        for k, v in p.items():
            if k not in META_FIELDS:
                field_values[k].append(str(v) if not isinstance(v, list) else str(v))
    total = len(profiles)
    print(f"\n  Total: {total}\n")
    for field in sorted(field_values):
        vals = field_values[field]
        unique = set(vals)
        if len(unique) > 15:
            continue
        counter = Counter(vals)
        present = len(vals)
        print(f"  ── {field} ({present}/{total}) ──")
        for val, cnt in counter.most_common():
            print(f"      {val:<55} {cnt:>4} ({cnt/present*100:>5.1f}%)")
        print()


# ═════════════════════════════════════════════════════════════════════════════
# 4. DATES
# ═════════════════════════════════════════════════════════════════════════════

def check_dates(profiles):
    print("\n" + "=" * 70)
    print("4. DATE COHERENCE")
    print("=" * 70)
    errors, warnings = [], []
    for p in profiles:
        pid = p.get("patient_id", "?")
        by = parse_year(p.get("annee_de_naissance"))
        ds = parse_date(p.get("date_1er_symptome"))
        dr = parse_date(p.get("exam_radio_date_decouverte"))
        dc = parse_date(p.get("date_chir")) or parse_date(p.get("chir_date"))
        dcm_s = parse_date(p.get("chm_date_debut"))
        dcm_e = parse_date(p.get("chm_date_fin"))
        drx_s = parse_date(p.get("rx_date_debut"))
        drx_e = parse_date(p.get("rx_date_fin"))
        dp = parse_date(p.get("date_progression"))
        dd = parse_date(p.get("date_deces"))
        dn = parse_date(p.get("dn_date"))

        if by and ds and ds.year <= by:
            errors.append(f"  {pid}: symptoms ({ds}) <= birth ({by})")
        if ds and dr and dr < ds:
            errors.append(f"  {pid}: imaging ({dr}) BEFORE symptoms ({ds})")
        if ds and dc and dc < ds:
            errors.append(f"  {pid}: surgery ({dc}) BEFORE symptoms ({ds})")
        if dc and dcm_s and dcm_s < dc:
            warnings.append(f"  {pid}: chemo start ({dcm_s}) BEFORE surgery ({dc}) [néoadjuvant?]")
        if dc and drx_s and drx_s < dc:
            warnings.append(f"  {pid}: RT start ({drx_s}) BEFORE surgery ({dc}) [néoadjuvant?]")
        if dcm_s and dcm_e and dcm_e < dcm_s:
            errors.append(f"  {pid}: chemo end ({dcm_e}) BEFORE start ({dcm_s})")
        if drx_s and drx_e and drx_e < drx_s:
            errors.append(f"  {pid}: RT end ({drx_e}) BEFORE start ({drx_s})")
        lt = max(filter(None, [dcm_e, drx_e]), default=None)
        if lt and dp and dp < lt:
            warnings.append(f"  {pid}: progression ({dp}) BEFORE treatment end ({lt})")
        if dp and dd and dd < dp:
            errors.append(f"  {pid}: death ({dd}) BEFORE progression ({dp})")
        if dn:
            for label, d in [("date_chir", dc), ("date_deces", dd), ("date_progression", dp)]:
                if d and d > dn:
                    errors.append(f"  {pid}: {label} ({d}) AFTER last news ({dn})")
        if dd and dn and dd != dn:
            warnings.append(f"  {pid}: date_deces ({dd}) != dn_date ({dn})")
        survie = p.get("survie_globale")
        if survie is not None and ds and dd:
            try:
                actual = (dd.year - ds.year) * 12 + (dd.month - ds.month)
                if abs(actual - int(survie)) > 3:
                    warnings.append(f"  {pid}: survie_globale={survie}m vs computed ~{actual}m")
            except (ValueError, TypeError):
                pass

    print(f"\n  ❌ {len(errors)} error(s):")
    for e in errors:
        print(e)
    print(f"\n  ⚠️  {len(warnings)} warning(s):")
    for w in warnings:
        print(w)
    if not errors and not warnings:
        print(f"\n  ✅ All dates coherent.")


# ═════════════════════════════════════════════════════════════════════════════
# 5. FIELD COVERAGE
# ═════════════════════════════════════════════════════════════════════════════

def check_field_coverage(profiles):
    print("\n" + "=" * 70)
    print("5. FIELD COVERAGE (111 fields)")
    print("=" * 70)
    fc = Counter()
    for p in profiles:
        for f in ALL_111_FIELDS:
            if f in p and p[f] is not None:
                fc[f] += 1
    total = len(profiles)
    covered = sum(1 for f in ALL_111_FIELDS if fc[f] > 0)
    missing = [f for f in ALL_111_FIELDS if fc[f] == 0]
    print(f"\n  Total: {total} | Covered: {covered}/{len(ALL_111_FIELDS)} | Never seen: {len(missing)}")
    if missing:
        print(f"\n  ⚠️  Missing: {', '.join(missing)}")
    print(f"\n  {'Field':<40} {'N':>5} {'%':>6}")
    print(f"  {'-'*40} {'-'*5} {'-'*6}")
    for f in ALL_111_FIELDS:
        n = fc[f]
        pct = n / total * 100 if total else 0
        m = " ⚠️" if n == 0 else ""
        print(f"  {f:<40} {n:>5} {pct:>5.1f}%{m}")
    extra = set()
    for p in profiles:
        extra.update(p.keys())
    extra -= set(ALL_111_FIELDS) | set(META_FIELDS)
    if extra:
        print(f"\n  ℹ️  Extra fields (not in 111 spec):")
        for f in sorted(extra):
            print(f"      - {f} ({sum(1 for p in profiles if f in p)})")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(profiles):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(profiles)
    ids = [p.get("patient_id") for p in profiles]
    dup = total - len(set(ids))
    doc = Counter(dt for p in profiles for dt in (p.get("document_types") or []))
    sx = Counter(p.get("sexe") for p in profiles)
    dead = sum(1 for p in profiles if p.get("date_deces") is not None)
    print(f"\n  Profiles: {total} | Unique IDs: {len(set(ids))}" +
          (f" | ⚠️ {dup} duplicates" if dup else ""))
    print(f"  Sex: M={sx.get('M',0)}, F={sx.get('F',0)} | Deceased: {dead} ({dead/total*100:.1f}%)")
    print(f"  Documents: " + ", ".join(f"{dt}={n}" for dt, n in doc.most_common()))


# ═════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════════════

def _self_test():
    """Verify critical substring-ordering fixes."""
    assert has_1p19q_codel({"ch1p19q_codel": "absence de codélétion"}) == False
    assert has_1p19q_codel({"ch1p19q_codel": "codélétion confirmée"}) == True
    assert has_1p19q_codel({"ch1p19q_codel": True}) == True
    assert has_1p19q_codel({"ch1p19q_codel": False}) == False
    assert has_1p19q_codel({"ch1p19q_codel": "non"}) == False
    assert has_1p19q_codel({}) is None
    assert has_cdkn2a_deletion({"mol_CDKN2A": "non délété"}) == False
    assert has_cdkn2a_deletion({"mol_CDKN2A": "délété (homozygote)"}) == True
    assert is_idh_mutated({"mol_idh1": "non muté"}) == False
    assert is_idh_mutated({"mol_idh1": "muté (R132H)"}) == True
    assert normalize_diag({"diag_histologique": "astrocytome pilocytique",
                           "diag_integre": ""}) == "other"
    assert normalize_diag({"diag_histologique": "astrocytome",
                           "mol_h3f3a": "muté K27M"}) == "other"
    print("  ✅ Self-tests passed.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python profile_validation.py <file_or_dir> [...]")
        sys.exit(1)

    if "--self-test" in sys.argv:
        _self_test()
        return

    paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.json")))
        elif p.is_file() and p.suffix == ".json":
            paths.append(p)

    if not paths:
        print("No JSON files found.")
        sys.exit(1)

    print(f"Loading {len(paths)} file(s)...")
    profiles = load_profiles(paths)
    print(f"Loaded {len(profiles)} profiles.")

    _self_test()
    print_summary(profiles)
    check_proportions(profiles)
    check_oms2021(profiles)
    check_value_distributions(profiles)
    check_dates(profiles)
    check_field_coverage(profiles)

    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()