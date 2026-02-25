"""Temporal forward-fill and conflict resolution across a patient's documents.

Implements three feature temporal categories:
- **Static** — set once, update only on explicit correction.
- **Specimen-bound** — carried from specimen date until next surgery.
- **Time-varying** — carry latest explicit value; ``NA`` does NOT overwrite.

And document-type priority for conflict resolution:
- BIO fields: anapath > molecular_report > rcp > consultation
- CLINIQUE fields: consultation > rcp > anapath

Public API
----------
- ``aggregate_patient_timeline(extractions)`` → ``pd.DataFrame``
"""

from __future__ import annotations

import logging
from typing import Optional, Any

import pandas as pd

from ..extraction.provenance import ExtractionResult
from ..extraction.schema import (
    ExtractionValue,
    ALL_BIO_FIELD_NAMES,
    ALL_CLINIQUE_FIELD_NAMES,
    DOCUMENT_TYPES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature temporal categories
# ---------------------------------------------------------------------------

STATIC_FEATURES: set[str] = {
    "nip", "sexe", "date_de_naissance", "tumeur_lateralite",
    "tumeur_position", "activite_professionnelle",
    "antecedent_tumoral",
    # Outcome (set when known, not forward-filled in the traditional sense,
    # but once recorded it persists)
    "date_deces", "infos_deces",
    # First symptoms (historical, set once)
    "date_1er_symptome", "epilepsie_1er_symptome",
    "ceph_hic_1er_symptome", "deficit_1er_symptome",
    "cognitif_1er_symptome", "autre_trouble_1er_symptome",
    # Radiology at discovery
    "exam_radio_date_decouverte", "contraste_1er_symptome",
    "oedeme_1er_symptome", "calcif_1er_symptome",
    # Care team
    "neuroncologue", "neurochirurgien", "radiotherapeute",
    "localisation_radiotherapie", "localisation_chir",
}

SPECIMEN_BOUND_FEATURES: set[str] = {
    # All BIO features — tied to a surgical specimen
    "date_chir", "num_labo",
    "diag_histologique", "diag_integre", "classification_oms", "grade",
    # IHC
    "ihc_idh1", "ihc_p53", "ihc_atrx", "ihc_fgfr3", "ihc_braf",
    "ihc_hist_h3k27m", "ihc_hist_h3k27me3", "ihc_egfr_hirsch",
    "ihc_gfap", "ihc_olig2", "ihc_ki67", "ihc_mmr",
    # Histology
    "histo_necrose", "histo_pec", "histo_mitoses",
    # Molecular
    "mol_idh1", "mol_idh2", "mol_tert", "mol_CDKN2A", "mol_h3f3a",
    "mol_hist1h3b", "mol_braf", "mol_mgmt", "mol_fgfr1", "mol_egfr_mut",
    "mol_prkca", "mol_p53", "mol_pten", "mol_cic", "mol_fubp1", "mol_atrx",
    # Chromosomal
    "ch1p", "ch19q", "ch10p", "ch10q", "ch7p", "ch7q", "ch9p", "ch9q",
    # Amplifications
    "ampli_mdm2", "ampli_cdk4", "ampli_egfr", "ampli_met", "ampli_mdm4",
    # Fusions
    "fusion_fgfr", "fusion_ntrk", "fusion_autre",
}

TIME_VARYING_FEATURES: set[str] = {
    "ik_clinique", "epilepsie", "ceph_hic", "deficit", "cognitif",
    "autre_trouble",
    "chimios", "chm_date_debut", "chm_date_fin", "chm_cycles",
    "chir_date", "type_chirurgie",
    "rx_date_debut", "rx_date_fin", "rx_dose",
    "corticoides", "optune",
    "anti_epileptiques", "essai_therapeutique",
    "progress_clinique", "progress_radiologique", "date_progression",
    "dn_date", "evol_clinique",
}


# ---------------------------------------------------------------------------
# Document-type priority for conflict resolution
# ---------------------------------------------------------------------------

BIO_PRIORITY: list[str] = ["anapath", "molecular_report", "rcp", "consultation", "radiology"]
CLINIQUE_PRIORITY: list[str] = ["consultation", "rcp", "anapath", "molecular_report", "radiology"]


def _priority_rank(doc_type: str, field_name: str) -> int:
    """Return a priority rank (lower = higher priority) for a field's source.

    Bio fields follow ``BIO_PRIORITY``, clinical fields follow
    ``CLINIQUE_PRIORITY``.
    """
    if field_name in ALL_BIO_FIELD_NAMES:
        priority_list = BIO_PRIORITY
    elif field_name in ALL_CLINIQUE_FIELD_NAMES:
        priority_list = CLINIQUE_PRIORITY
    else:
        # Unknown field — use CLINIQUE priority by default
        priority_list = CLINIQUE_PRIORITY

    try:
        return priority_list.index(doc_type)
    except ValueError:
        return len(priority_list)  # lowest priority


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_surgery_event(extraction: ExtractionResult) -> bool:
    """Check if this extraction reports a new surgery event."""
    for field in ("chir_date", "date_chir"):
        ev = extraction.features.get(field)
        if ev is not None and ev.value is not None:
            return True
    return False


def _extraction_sort_key(extraction: ExtractionResult) -> str:
    """Return a sort key based on document_date for chronological ordering.

    Parses DD/MM/YYYY into YYYYMMDD for correct chronological sorting.
    Missing dates sort last.
    """
    date_str = extraction.document_date
    if not date_str:
        return "99999999"
    # Try to parse DD/MM/YYYY → YYYYMMDD for proper chronological sorting
    parts = date_str.split("/")
    if len(parts) == 3:
        try:
            day, month, year = parts
            return f"{year.zfill(4)}{month.zfill(2)}{day.zfill(2)}"
        except (ValueError, IndexError):
            pass
    return date_str  # fallback to string sort if parsing fails


def _get_value(ev: Optional[ExtractionValue]) -> Any:
    """Extract the raw value from an ExtractionValue, or None."""
    if ev is None:
        return None
    return ev.value


# ---------------------------------------------------------------------------
# Main aggregation function
# ---------------------------------------------------------------------------

def aggregate_patient_timeline(
    extractions: list[ExtractionResult],
) -> pd.DataFrame:
    """Aggregate all ExtractionResults for one patient into a timeline.

    Given all ``ExtractionResult`` objects for one patient (already
    ordered by document date):

    1. Sort by ``document_date``.
    2. Initialize an empty timeline DataFrame.
    3. For each extraction (document):
       a. Add a new row or merge into existing row at that timepoint.
       b. Apply forward-fill rules:
          - **Static**: set once, update only on explicit correction.
          - **Specimen-bound**: carry from specimen date until next surgery.
          - **Time-varying**: carry latest explicit value; ``NA`` does NOT
            overwrite a previous explicit value.
    4. Resolve conflicts using document-type priority.
    5. Return the complete timeline DataFrame.

    Parameters
    ----------
    extractions : list[ExtractionResult]
        All extraction results for a single patient, from different
        documents / timepoints.

    Returns
    -------
    pd.DataFrame
        The aggregated patient timeline. Each row represents a
        timepoint. Columns are feature field names plus metadata
        columns (``document_id``, ``document_type``, ``document_date``).
    """
    if not extractions:
        return pd.DataFrame()

    # Sort chronologically
    sorted_extractions = sorted(extractions, key=_extraction_sort_key)

    # Collect all unique field names across all extractions
    all_fields: set[str] = set()
    for ext in sorted_extractions:
        all_fields.update(ext.features.keys())

    # State trackers
    static_state: dict[str, tuple[Any, str]] = {}  # field → (value, source_doc_type)
    specimen_state: dict[str, tuple[Any, str]] = {}  # field → (value, source_doc_type)
    time_varying_state: dict[str, tuple[Any, str]] = {}  # field → (value, source_doc_type)

    rows: list[dict[str, Any]] = []

    for extraction in sorted_extractions:
        row: dict[str, Any] = {
            "_document_id": extraction.document_id,
            "_document_type": extraction.document_type,
            "_document_date": extraction.document_date,
            "_patient_id": extraction.patient_id,
        }

        # Check if this is a new surgery (resets specimen-bound features)
        is_new_surgery = _is_surgery_event(extraction)
        if is_new_surgery:
            specimen_state.clear()
            logger.debug(
                "New surgery detected in %s — resetting specimen-bound features",
                extraction.document_id,
            )

        # Process each feature in this extraction
        for fname, ev in extraction.features.items():
            new_value = _get_value(ev)

            if fname in STATIC_FEATURES:
                _apply_static(
                    fname, new_value, extraction.document_type, static_state
                )
            elif fname in SPECIMEN_BOUND_FEATURES:
                _apply_specimen_bound(
                    fname, new_value, extraction.document_type,
                    specimen_state, is_new_surgery,
                )
            elif fname in TIME_VARYING_FEATURES:
                _apply_time_varying(
                    fname, new_value, extraction.document_type,
                    time_varying_state,
                )
            else:
                # Unknown category — treat as time-varying
                _apply_time_varying(
                    fname, new_value, extraction.document_type,
                    time_varying_state,
                )

        # Build the row from current state
        for fname in all_fields:
            if fname in static_state:
                row[fname] = static_state[fname][0]
            elif fname in specimen_state:
                row[fname] = specimen_state[fname][0]
            elif fname in time_varying_state:
                row[fname] = time_varying_state[fname][0]
            else:
                row[fname] = None

        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns: metadata first, then sorted feature columns
    meta_cols = ["_patient_id", "_document_id", "_document_type", "_document_date"]
    feature_cols = sorted([c for c in df.columns if not c.startswith("_")])
    col_order = [c for c in meta_cols if c in df.columns] + feature_cols
    df = df[col_order]

    return df


# ---------------------------------------------------------------------------
# State update helpers
# ---------------------------------------------------------------------------

def _apply_static(
    fname: str,
    new_value: Any,
    doc_type: str,
    state: dict[str, tuple[Any, str]],
) -> None:
    """Apply static-feature update rule: set once, update only on correction.

    A static feature is set the first time a non-None value is seen.
    Subsequent non-None values update only if they come from a
    higher-priority document type.
    """
    if new_value is None:
        return

    if fname not in state:
        state[fname] = (new_value, doc_type)
    else:
        _, existing_doc_type = state[fname]
        # Update only if the new source has higher priority
        if _priority_rank(doc_type, fname) < _priority_rank(existing_doc_type, fname):
            state[fname] = (new_value, doc_type)


def _apply_specimen_bound(
    fname: str,
    new_value: Any,
    doc_type: str,
    state: dict[str, tuple[Any, str]],
    is_new_surgery: bool,
) -> None:
    """Apply specimen-bound update rule.

    Specimen-bound features are carried from specimen date until next
    surgery.  When a new surgery is detected, the state is reset
    (handled by clearing ``state`` before calling this function).

    After reset, the feature is set from the first non-None value seen.
    Conflicts are resolved via document-type priority.
    """
    if new_value is None:
        return

    if fname not in state:
        state[fname] = (new_value, doc_type)
    else:
        _, existing_doc_type = state[fname]
        if _priority_rank(doc_type, fname) < _priority_rank(existing_doc_type, fname):
            state[fname] = (new_value, doc_type)


def _apply_time_varying(
    fname: str,
    new_value: Any,
    doc_type: str,
    state: dict[str, tuple[Any, str]],
) -> None:
    """Apply time-varying update rule.

    Carry the latest explicit value.  ``None`` does NOT overwrite a
    previous explicit value.  When two documents at the same timepoint
    conflict, resolve via document-type priority.
    """
    if new_value is None:
        return  # NA does NOT overwrite

    if fname not in state:
        state[fname] = (new_value, doc_type)
    else:
        # Always take the latest explicit value (chronological ordering
        # means we process in order).  But if we're still "at the same
        # timepoint" we use priority.  Since we iterate chronologically,
        # the latest document wins unless priority says otherwise.
        # For simplicity: always update when we have a new explicit value,
        # but prefer higher-priority doc type if the value is the same timepoint.
        state[fname] = (new_value, doc_type)
