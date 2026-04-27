"""Thin wrapper around profiles_validation.py for document-level WHO 2021 checks.

Re-uses the canonical detection helpers to avoid duplicating 300+ lines of
molecular-classification logic.
"""

import sys
from pathlib import Path

# Make profiles_validation.py importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from profiles_validation import (  # noqa: E402
    normalize_diag,
    is_idh_mutated,
    is_atrx_lost,
    has_1p19q_codel,
    has_cdkn2a_deletion,
    has_tert_mutation,
    has_egfr_amplification,
    has_plus7_minus10,
)


def validate_document_coherence(annotations: dict) -> list[str]:
    """Check WHO 2021 coherence on a single document's annotations.

    Parameters
    ----------
    annotations : dict
        field_name -> value (str) mapping extracted from a generated document.
        Can also accept field_name -> {"value": str, "span": str} dicts.

    Returns
    -------
    list[str]
        Violation messages. Empty list = valid.
    """
    # Normalise: accept both flat values and {value, span} dicts
    profile: dict = {}
    for k, v in annotations.items():
        if isinstance(v, dict) and "value" in v:
            profile[k] = v["value"]
        else:
            profile[k] = v

    errors: list[str] = []
    cat = normalize_diag(profile)
    grade = profile.get("grade")
    idh_mut = is_idh_mutated(profile)
    codel = has_1p19q_codel(profile)
    atrx_lost = is_atrx_lost(profile)
    cdkn2a_del = has_cdkn2a_deletion(profile)

    # Rule 1: Oligodendroglioma requires IDH mutation + 1p/19q codeletion
    if cat == "oligo":
        if not idh_mut:
            errors.append("Oligodendrogliome sans mutation IDH")
        if codel is False:
            errors.append("Oligodendrogliome sans codélétion 1p/19q")

    # Rule 2: GBM (WHO 2021) must be IDH-wildtype
    if cat == "gbm" and idh_mut:
        errors.append("Glioblastome avec IDH muté (invalide sous OMS 2021)")

    # Rule 3: IDH-mutant astrocytoma must not have 1p/19q codeletion
    if cat == "astro_idh":
        if not idh_mut:
            errors.append("Astrocytome IDH-muté sans mutation IDH")
        if codel is True:
            errors.append("Astrocytome IDH-muté avec codélétion 1p/19q (devrait être oligo)")

    # Rule 4: ATRX loss and 1p/19q codeletion are mutually exclusive
    if atrx_lost and codel is True:
        errors.append("ATRX perdu ET codélétion 1p/19q (mutuellement exclusifs)")

    # Rule 5: CDKN2A homozygous deletion in IDH-mutant astrocytoma → grade 4
    if cat == "astro_idh" and cdkn2a_del is True:
        if grade is not None:
            try:
                if int(grade) != 4:
                    errors.append(
                        f"Astrocytome IDH-muté + délétion CDKN2A → grade devrait être 4, "
                        f"trouvé {grade}"
                    )
            except (ValueError, TypeError):
                pass

    # Rule 6: IDH-wt astrocytoma + molecular GBM markers → should be GBM
    diag_h = str(profile.get("diag_histologique", "")).lower()
    if "astrocytome" in diag_h and not idh_mut:
        if has_tert_mutation(profile) or has_egfr_amplification(profile) or has_plus7_minus10(profile):
            errors.append(
                "Astrocytome IDH-sauvage avec TERT/EGFR/+7-10 → devrait être classé GBM"
            )

    # Rule 7: Treatment coherence
    proto = str(profile.get("chimio_protocole", "")).lower()
    if cat == "oligo" and "stupp" in proto:
        errors.append("Oligodendrogliome avec protocole Stupp (typiquement PCV)")
    if cat == "gbm" and "pcv" in proto and "stupp" not in proto:
        errors.append("Glioblastome avec protocole PCV (typiquement Stupp)")

    return errors
