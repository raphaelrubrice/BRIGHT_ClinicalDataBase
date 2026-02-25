"""Row duplication for documents reporting multiple treatment events.

Detects when a single document describes multiple distinct surgeries,
chemotherapy lines, radiotherapy courses, or progression events, and
splits into separate ``ExtractionResult`` rows.

Public API
----------
- ``detect_multiple_events(extraction)`` – Returns a list of
  ``ExtractionResult`` (length ≥ 1).  If only one event is found the
  original is returned unchanged (wrapped in a one-element list).
"""

from __future__ import annotations

import copy
import re
from dataclasses import replace
from typing import Optional

from ..extraction.provenance import ExtractionResult
from ..extraction.schema import ExtractionValue


# ---------------------------------------------------------------------------
# Feature groupings for duplication logic
# ---------------------------------------------------------------------------

# Fields that are shared across all rows (demographics, tumour location, etc.)
SHARED_FEATURES: set[str] = {
    # Demographics
    "nip", "date_de_naissance", "sexe", "activite_professionnelle",
    "antecedent_tumoral",
    # Care team
    "neuroncologue", "neurochirurgien", "radiotherapeute",
    "localisation_radiotherapie", "localisation_chir",
    # Tumour location
    "tumeur_lateralite", "tumeur_position",
    # Outcome (shared across timeline)
    "date_deces", "infos_deces",
    # Biological identifiers
    "num_labo",
    # Diagnosis (shared unless explicitly different per specimen)
    "diag_histologique", "diag_integre", "classification_oms", "grade",
    # First symptoms (historical, shared)
    "date_1er_symptome", "epilepsie_1er_symptome", "ceph_hic_1er_symptome",
    "deficit_1er_symptome", "cognitif_1er_symptome",
    "autre_trouble_1er_symptome",
    # Radiology at discovery
    "exam_radio_date_decouverte", "contraste_1er_symptome",
    "oedeme_1er_symptome", "calcif_1er_symptome",
    # All IHC / molecular / chromosomal / amplification / fusion (specimen-bound)
    "ihc_idh1", "ihc_p53", "ihc_atrx", "ihc_fgfr3", "ihc_braf",
    "ihc_hist_h3k27m", "ihc_hist_h3k27me3", "ihc_egfr_hirsch",
    "ihc_gfap", "ihc_olig2", "ihc_ki67", "ihc_mmr",
    "histo_necrose", "histo_pec", "histo_mitoses",
    "mol_idh1", "mol_idh2", "mol_tert", "mol_CDKN2A", "mol_h3f3a",
    "mol_hist1h3b", "mol_braf", "mol_mgmt", "mol_fgfr1", "mol_egfr_mut",
    "mol_prkca", "mol_p53", "mol_pten", "mol_cic", "mol_fubp1", "mol_atrx",
    "ch1p", "ch19q", "ch10p", "ch10q", "ch7p", "ch7q", "ch9p", "ch9q",
    "ampli_mdm2", "ampli_cdk4", "ampli_egfr", "ampli_met", "ampli_mdm4",
    "fusion_fgfr", "fusion_ntrk", "fusion_autre",
}

# Treatment-event-specific field groups
# Each group represents a distinct "event axis" that can trigger duplication.

SURGERY_EVENT_FIELDS: list[str] = [
    "chir_date", "type_chirurgie", "date_chir",
]

CHEMO_EVENT_FIELDS: list[str] = [
    "chimios", "chm_date_debut", "chm_date_fin", "chm_cycles",
]

RADIO_EVENT_FIELDS: list[str] = [
    "rx_date_debut", "rx_date_fin", "rx_dose",
]

PROGRESSION_EVENT_FIELDS: list[str] = [
    "date_progression", "progress_clinique", "progress_radiologique",
]

# Clinical state fields that accompany the timepoint
CLINICAL_STATE_FIELDS: list[str] = [
    "ik_clinique", "epilepsie", "ceph_hic", "deficit", "cognitif",
    "autre_trouble", "anti_epileptiques", "essai_therapeutique",
    "corticoides", "optune",
    "dn_date", "evol_clinique",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_feature_value(extraction: ExtractionResult, field: str) -> Optional[str]:
    """Return the raw value of a feature, or None if absent."""
    ev = extraction.features.get(field)
    if ev is None or ev.value is None:
        return None
    return str(ev.value)


def _parse_multiple_values(value_str: str) -> list[str]:
    """Split a concatenated value string into individual items.

    Clinical documents sometimes list multiple dates or treatments
    separated by commas, semicolons, " et ", " puis ", or slashes.
    """
    if not value_str:
        return []
    # Split on common delimiters (NOT slash — dates use DD/MM/YYYY)
    parts = re.split(r"[;,]|\bet\b|\bpuis\b", value_str, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def _count_distinct_dates(extraction: ExtractionResult, date_field: str) -> list[str]:
    """Extract distinct date values from a feature.

    A feature may contain multiple dates concatenated as a single string
    (e.g. ``"01/03/2020, 15/09/2021"``).
    """
    raw = _get_feature_value(extraction, date_field)
    if raw is None:
        return []
    parts = _parse_multiple_values(raw)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# Core duplication logic
# ---------------------------------------------------------------------------

def _create_event_row(
    base: ExtractionResult,
    event_features: dict[str, ExtractionValue],
    event_index: int,
    event_type: str,
) -> ExtractionResult:
    """Create a new ExtractionResult for a specific event occurrence.

    Copies all shared features from *base* and overlays *event_features*
    for the specific event.
    """
    new_result = replace(
        base,
        features=dict(base.features),  # shallow copy
        extraction_log=list(base.extraction_log),
        flagged_for_review=list(base.flagged_for_review),
    )

    # Keep only shared features + the event-specific features
    new_features: dict[str, ExtractionValue] = {}

    # Copy shared features
    for fname, fval in base.features.items():
        if fname in SHARED_FEATURES:
            new_features[fname] = fval

    # Overlay event-specific features
    for fname, fval in event_features.items():
        new_features[fname] = fval

    new_result.features = new_features
    new_result.add_log(
        f"Row duplicated: event {event_index + 1} ({event_type}) "
        f"from document {base.document_id}"
    )

    return new_result


def _detect_surgery_events(
    extraction: ExtractionResult,
) -> list[dict[str, ExtractionValue]]:
    """Detect multiple surgery events from chir_date / date_chir fields."""
    dates: list[str] = []
    for field in ("chir_date", "date_chir"):
        dates.extend(_count_distinct_dates(extraction, field))

    # Deduplicate across both fields
    seen: set[str] = set()
    unique_dates: list[str] = []
    for d in dates:
        if d not in seen:
            seen.add(d)
            unique_dates.append(d)

    if len(unique_dates) <= 1:
        return []

    # Build one event dict per surgery date
    events: list[dict[str, ExtractionValue]] = []
    for date_val in unique_dates:
        event: dict[str, ExtractionValue] = {}
        # Set the surgery date
        event["chir_date"] = ExtractionValue(
            value=date_val,
            extraction_tier="rule",
            source_span=date_val,
        )
        event["date_chir"] = ExtractionValue(
            value=date_val,
            extraction_tier="rule",
            source_span=date_val,
        )
        # Copy type_chirurgie from original (same for all unless LLM split it)
        if "type_chirurgie" in extraction.features:
            event["type_chirurgie"] = extraction.features["type_chirurgie"]

        # Copy clinical state fields from original
        for f in CLINICAL_STATE_FIELDS:
            if f in extraction.features:
                event[f] = extraction.features[f]

        events.append(event)

    return events


def _detect_chemo_events(
    extraction: ExtractionResult,
) -> list[dict[str, ExtractionValue]]:
    """Detect multiple chemotherapy line events."""
    dates = _count_distinct_dates(extraction, "chm_date_debut")
    if len(dates) <= 1:
        return []

    # Try to split chimios names as well
    chemo_names = _parse_multiple_values(
        _get_feature_value(extraction, "chimios") or ""
    )

    events: list[dict[str, ExtractionValue]] = []
    for i, date_val in enumerate(dates):
        event: dict[str, ExtractionValue] = {}
        event["chm_date_debut"] = ExtractionValue(
            value=date_val,
            extraction_tier="rule",
            source_span=date_val,
        )
        # Assign chemo name if we have as many names as dates
        if len(chemo_names) == len(dates):
            event["chimios"] = ExtractionValue(
                value=chemo_names[i],
                extraction_tier="rule",
                source_span=chemo_names[i],
            )
        elif "chimios" in extraction.features:
            event["chimios"] = extraction.features["chimios"]

        # Copy other chemo fields from original
        for f in ("chm_date_fin", "chm_cycles"):
            if f in extraction.features:
                event[f] = extraction.features[f]

        # Copy clinical state fields
        for f in CLINICAL_STATE_FIELDS:
            if f in extraction.features:
                event[f] = extraction.features[f]

        events.append(event)

    return events


def _detect_radio_events(
    extraction: ExtractionResult,
) -> list[dict[str, ExtractionValue]]:
    """Detect multiple radiotherapy course events."""
    dates = _count_distinct_dates(extraction, "rx_date_debut")
    if len(dates) <= 1:
        return []

    events: list[dict[str, ExtractionValue]] = []
    for date_val in dates:
        event: dict[str, ExtractionValue] = {}
        event["rx_date_debut"] = ExtractionValue(
            value=date_val,
            extraction_tier="rule",
            source_span=date_val,
        )
        # Copy dose / end date from original
        for f in ("rx_date_fin", "rx_dose"):
            if f in extraction.features:
                event[f] = extraction.features[f]

        # Copy clinical state
        for f in CLINICAL_STATE_FIELDS:
            if f in extraction.features:
                event[f] = extraction.features[f]

        events.append(event)

    return events


def _detect_progression_events(
    extraction: ExtractionResult,
) -> list[dict[str, ExtractionValue]]:
    """Detect multiple progression events."""
    dates = _count_distinct_dates(extraction, "date_progression")
    if len(dates) <= 1:
        return []

    events: list[dict[str, ExtractionValue]] = []
    for date_val in dates:
        event: dict[str, ExtractionValue] = {}
        event["date_progression"] = ExtractionValue(
            value=date_val,
            extraction_tier="rule",
            source_span=date_val,
        )
        # Copy progression flags from original
        for f in ("progress_clinique", "progress_radiologique"):
            if f in extraction.features:
                event[f] = extraction.features[f]

        # Copy clinical state
        for f in CLINICAL_STATE_FIELDS:
            if f in extraction.features:
                event[f] = extraction.features[f]

        events.append(event)

    return events


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_multiple_events(
    extraction: ExtractionResult,
) -> list[ExtractionResult]:
    """Check if the document reports multiple distinct treatment events.

    Duplication triggers (checked in priority order):
    - Multiple distinct surgery dates (``chir_date`` / ``date_chir``)
    - Multiple chemotherapy lines (``chimios`` + ``chm_date_debut``)
    - Multiple radiotherapy courses (``rx_date_debut`` + ``rx_dose``)
    - Multiple progression events (``date_progression``)

    For each additional event, a duplicate ``ExtractionResult`` is created
    with:
    - **Shared features** (demographics, tumour characteristics) copied.
    - **Treatment-specific features** isolated to the respective row.

    Parameters
    ----------
    extraction : ExtractionResult
        The extraction result from a single document.

    Returns
    -------
    list[ExtractionResult]
        A list of one or more ExtractionResults.  If no duplication is
        needed, the original is returned wrapped in a single-element list.
    """
    # Try each event type.  The first type that yields multiple events
    # triggers duplication.  We do *not* combine multiple event types
    # in a single pass — that would lead to a combinatorial explosion.
    for detector, event_type in [
        (_detect_surgery_events, "surgery"),
        (_detect_chemo_events, "chemotherapy"),
        (_detect_radio_events, "radiotherapy"),
        (_detect_progression_events, "progression"),
    ]:
        events = detector(extraction)
        if events:
            rows = [
                _create_event_row(extraction, ev, i, event_type)
                for i, ev in enumerate(events)
            ]
            return rows

    # No duplication needed — return original as-is
    return [extraction]
