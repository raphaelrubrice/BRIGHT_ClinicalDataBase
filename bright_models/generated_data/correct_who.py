#!/usr/bin/env python3
"""
correct_who_2021.py — Fix WHO 2021 constraint violations in-place.

Usage:
  python correct_who_2021.py profiles_dir/
  python correct_who_2021.py profiles_dir/ --dry-run
  python correct_who_2021.py profiles_dir/ --no-backup
"""

import json, sys, shutil, re
from pathlib import Path
from collections import defaultdict


# ═════════════════════════════════════════════════════════════════════════════
# WHO 2021 DETECTION HELPERS (identical to profile_validation.py)
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
    diag_h = (profile.get("diag_histologique") or "").lower()
    diag_i = (profile.get("diag_integre") or "").lower()
    combined = diag_h + " " + diag_i
    if "oligodendrogliome" in combined:
        return "oligo"
    if "glioblastome" in combined:
        return "gbm"
    if "pilocytique" in combined or "pilocytic" in combined:
        return "other"
    if "xanthoastrocytome" in combined or "pxa" in combined:
        return "other"
    if "sous-épendymaire" in combined or "subependym" in combined:
        return "other"
    if "h3" in combined or "ligne médiane" in combined or "dipg" in combined:
        return "other"
    mol_h3 = str(profile.get("mol_h3f3a", "")).lower()
    ihc_h3 = str(profile.get("ihc_hist_h3k27m", "")).lower()
    if _is_positive_mutation(mol_h3) or "positif" in ihc_h3:
        return "other"
    for kw in ["épendymome", "ependym", "médulloblastome", "medullo",
               "gangliogliome", "craniopharyngiome"]:
        if kw in combined:
            return "other"
    mol_braf = str(profile.get("mol_braf", "")).lower()
    ihc_braf = str(profile.get("ihc_braf", "")).lower()
    has_braf = (_is_positive_mutation(mol_braf) or "fusion" in mol_braf or "positif" in ihc_braf)
    if has_braf and "idh-muté" not in combined and "idh muté" not in combined:
        return "other"
    if "astrocytome" in combined:
        return "astro_idh"
    return "other"


def has_1p19q_codel(p):
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
        if _is_positive_mutation(str(p.get(field, ""))):
            return True
    return False


def is_atrx_lost(p):
    ihc = str(p.get("ihc_atrx", "")).lower()
    mol = str(p.get("mol_atrx", "")).lower()
    return ("perdu" in ihc or "lost" in ihc or "negatif" in ihc
            or _is_positive_mutation(mol))


def has_cdkn2a_deletion(p):
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


def diag_explicitly_says_idh_mute(p):
    combined = ((p.get("diag_histologique") or "") + " " +
                (p.get("diag_integre") or "")).lower()
    return "idh-muté" in combined or "idh muté" in combined


# ═════════════════════════════════════════════════════════════════════════════
# FORMAT DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def _codel_false_fmt(p):
    v = p.get("ch1p19q_codel")
    if v is not None:
        s = str(v).lower()
        if v is False or s == "false":
            return False
        if s == "non":
            return "non"
        if "absence" in s:
            return "absence de codélétion"
    return False


def _codel_true_fmt(p):
    v = p.get("ch1p19q_codel")
    if v is not None:
        s = str(v).lower()
        if v is True or s == "true":
            return True
        if s == "oui":
            return "oui"
        if "codélétion" in s and "absence" not in s:
            return "codélétion confirmée"
    return True


def _chr_loss_fmt(p):
    for f in ["ch1p", "ch19q", "ch10q", "ch9p"]:
        v = str(p.get(f, "")).lower()
        if v and "non" not in v:
            if "délétion" in v:
                return "délétion"
            if "perte" in v:
                return "perte"
    return "perte"


def _atrx_maintained_fmt(p):
    return "conservé" if "conservé" in str(p.get("ihc_atrx", "")).lower() else "maintenu"


# ═════════════════════════════════════════════════════════════════════════════
# CORRECTION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def fix_astro_remove_codel(p):
    changes = {"ch1p19q_codel": _codel_false_fmt(p)}
    for f in ["ch1p", "ch19q"]:
        v = str(p.get(f, "")).lower()
        if v and ("perte" in v or "délétion" in v) and "non" not in v:
            changes[f] = None
    return changes


def fix_astro_cdkn2a_grade(p):
    changes = {"grade": 4}
    di = p.get("diag_integre", "")
    new = re.sub(r"grade\s+\d", "grade 4", di, flags=re.IGNORECASE)
    if new != di:
        changes["diag_integre"] = new
    return changes


def fix_oligo_add_codel(p):
    changes = {"ch1p19q_codel": _codel_true_fmt(p)}
    fmt = _chr_loss_fmt(p)
    if "ch1p" not in p:
        changes["ch1p"] = fmt
    if "ch19q" not in p:
        changes["ch19q"] = fmt
    return changes


def fix_astro_add_idh(p):
    if not diag_explicitly_says_idh_mute(p):
        return {}
    changes = {}
    if "positif" not in str(p.get("ihc_idh1", "")).lower():
        changes["ihc_idh1"] = "positif"
    mol = str(p.get("mol_idh1", "")).lower()
    if not _is_positive_mutation(mol):
        changes["mol_idh1"] = "muté (R132H)"
    return changes


def fix_oligo_atrx(p):
    changes = {}
    fmt = _atrx_maintained_fmt(p)
    ihc = str(p.get("ihc_atrx", "")).lower()
    if "perdu" in ihc or "negatif" in ihc:
        changes["ihc_atrx"] = fmt
    mol = str(p.get("mol_atrx", "")).lower()
    if _is_positive_mutation(mol):
        changes["mol_atrx"] = None
    return changes


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ═════════════════════════════════════════════════════════════════════════════

def correct_profile(p):
    corrections = []
    cat = normalize_diag(p)
    codel = has_1p19q_codel(p)

    if cat == "astro_idh" and codel is True:
        corrections.append(("astro_remove_codel", fix_astro_remove_codel(p)))

    if cat == "astro_idh" and has_cdkn2a_deletion(p) is True:
        g = p.get("grade")
        if g is not None and int(g) != 4:
            corrections.append(("astro_cdkn2a_grade4", fix_astro_cdkn2a_grade(p)))

    if cat == "oligo" and codel is None:
        corrections.append(("oligo_add_codel", fix_oligo_add_codel(p)))

    if cat == "astro_idh" and not is_idh_mutated(p):
        fx = fix_astro_add_idh(p)
        if fx:
            corrections.append(("astro_add_idh", fx))

    if cat == "oligo" and is_atrx_lost(p):
        fx = fix_oligo_atrx(p)
        if fx:
            corrections.append(("oligo_fix_atrx", fx))

    if cat == "other" and is_atrx_lost(p) and codel is True:
        corrections.append(("other_remove_codel", {"ch1p19q_codel": _codel_false_fmt(p)}))

    return corrections


def apply_corrections(profile, corrections):
    m = dict(profile)
    for _, changes in corrections:
        for field, val in changes.items():
            if val is None:
                m.pop(field, None)
            else:
                m[field] = val
    return m


def verify_profile(p):
    issues = []
    pid = p.get("patient_id", "?")
    cat = normalize_diag(p)
    codel = has_1p19q_codel(p)
    if cat == "oligo":
        if not is_idh_mutated(p):
            issues.append(f"{pid}: Oligo WITHOUT IDH")
        if codel is False:
            issues.append(f"{pid}: Oligo WITHOUT codel")
        if codel is None:
            issues.append(f"{pid}: Oligo codel NOT TESTED")
    if cat == "gbm" and is_idh_mutated(p):
        issues.append(f"{pid}: GBM with IDH mutated")
    if cat == "astro_idh":
        if not is_idh_mutated(p):
            issues.append(f"{pid}: Astro WITHOUT IDH")
        if codel is True:
            issues.append(f"{pid}: Astro WITH codel")
    if is_atrx_lost(p) and codel is True:
        issues.append(f"{pid}: ATRX+codel exclusive")
    if cat == "astro_idh" and has_cdkn2a_deletion(p) is True:
        g = p.get("grade")
        if g is not None and int(g) != 4:
            issues.append(f"{pid}: CDKN2A del grade={g}")
    return issues


# ═════════════════════════════════════════════════════════════════════════════
# FILE I/O
# ═════════════════════════════════════════════════════════════════════════════

def process_file(filepath, do_backup=True, dry_run=False):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    is_list = isinstance(data, list)
    profiles = data if is_list else [data]
    logs, verify_issues = [], []
    n_corrected = 0

    for i, p in enumerate(profiles):
        pid = p.get("patient_id", f"idx_{i}")
        corrs = correct_profile(p)
        if corrs:
            n_corrected += 1
            for rule, changes in corrs:
                for field, val in changes.items():
                    old = p.get(field, "<absent>")
                    logs.append(f"  {pid:>8} | {rule:<25} | {field}: {old!r} → {val!r}" if val is not None
                                else f"  {pid:>8} | {rule:<25} | {field}: {old!r} → REMOVED")
            corrected = apply_corrections(p, corrs)
            profiles[i] = corrected
            for issue in verify_profile(corrected):
                verify_issues.append(f"  ⚠️  {issue}")

    if n_corrected and not dry_run:
        if do_backup:
            shutil.copy2(filepath, filepath.with_suffix(".json.bak"))
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(profiles if is_list else profiles[0], f, ensure_ascii=False, indent=2)

    return len(profiles), n_corrected, logs, verify_issues


# ═════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════════════

def _self_test():
    assert has_1p19q_codel({"ch1p19q_codel": "absence de codélétion"}) == False
    assert has_1p19q_codel({"ch1p19q_codel": "codélétion confirmée"}) == True
    assert has_cdkn2a_deletion({"mol_CDKN2A": "non délété"}) == False
    assert has_cdkn2a_deletion({"mol_CDKN2A": "délété (homozygote)"}) == True
    assert is_idh_mutated({"mol_idh1": "non muté"}) == False
    assert is_idh_mutated({"mol_idh1": "muté (R132H)"}) == True
    assert normalize_diag({"diag_histologique": "astrocytome pilocytique"}) == "other"
    assert normalize_diag({"diag_histologique": "astrocytome", "mol_h3f3a": "muté K27M"}) == "other"
    print("  ✅ Self-tests passed.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python correct_who_2021.py <path> [--dry-run] [--no-backup]")
        sys.exit(1)

    _self_test()

    dry_run = "--dry-run" in sys.argv
    do_backup = "--no-backup" not in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    paths = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.json")))
        elif p.is_file() and p.suffix == ".json":
            paths.append(p)

    if not paths:
        print("No JSON files found.")
        sys.exit(1)

    mode = "DRY RUN" if dry_run else ("backup" if do_backup else "no backup")
    print(f"Processing {len(paths)} file(s) ({mode})\n")

    total_p, total_c = 0, 0
    all_logs, all_verify = [], []
    by_rule = defaultdict(int)

    for fp in paths:
        n, nc, logs, vis = process_file(fp, do_backup, dry_run)
        total_p += n
        total_c += nc
        all_logs.extend(logs)
        all_verify.extend(vis)
        for l in logs:
            parts = l.split("|")
            if len(parts) >= 2:
                by_rule[parts[1].strip()] += 1
        print(f"  {fp.name}: {'✅ ' + str(nc) + '/' + str(n) + ' corrected' if nc else '— ' + str(n) + ', ok'}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Scanned: {total_p} | Corrected: {total_c} | Changes: {len(all_logs)}")
    if by_rule:
        print(f"\n  By rule:")
        for r, c in sorted(by_rule.items(), key=lambda x: -x[1]):
            print(f"    {r:<30} {c:>4}")
    if all_logs:
        print(f"\n{'='*70}")
        print("CHANGES")
        print(f"{'='*70}\n")
        for l in all_logs:
            print(l)
    if all_verify:
        print(f"\n{'='*70}")
        print(f"POST-FIX: {len(all_verify)} remaining")
        print(f"{'='*70}")
        for v in all_verify:
            print(v)
    elif total_c:
        print(f"\n  ✅ All corrected profiles pass post-fix verification.")
    if dry_run:
        print(f"\n  ℹ️  DRY RUN — no files modified.")
    print(f"\n{'='*70}\nDONE\n{'='*70}")


if __name__ == "__main__":
    main()