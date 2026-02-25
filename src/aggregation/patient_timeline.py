"""End-to-end patient timeline builder.

Orchestrates extraction, row duplication, and temporal aggregation
to produce a complete patient timeline DataFrame.

Public API
----------
- ``build_patient_timeline(patient_id, documents, pipeline)`` → ``pd.DataFrame``
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ..extraction.pipeline import ExtractionPipeline
from ..extraction.provenance import ExtractionResult
from .row_duplicator import detect_multiple_events
from .temporal_aggregation import aggregate_patient_timeline

logger = logging.getLogger(__name__)


def build_patient_timeline(
    patient_id: str,
    documents: list[dict],
    pipeline: ExtractionPipeline,
) -> pd.DataFrame:
    """Build a complete patient timeline from a list of documents.

    End-to-end pipeline for one patient:

    1. **Extract** all documents using the :class:`ExtractionPipeline`.
    2. **Detect and duplicate** rows for multiple treatment events
       within a single document.
    3. **Aggregate** into a temporal timeline with forward-fill and
       conflict resolution.
    4. Return the final ``pd.DataFrame``.

    Parameters
    ----------
    patient_id : str
        Pseudonymised patient identifier.
    documents : list[dict]
        List of document dicts.  Each dict must have:
        - ``"text"`` (str) — the full document text
        - ``"document_id"`` (str, optional) — unique ID
        - ``"document_date"`` (str, optional) — date in DD/MM/YYYY format
    pipeline : ExtractionPipeline
        A configured extraction pipeline instance.

    Returns
    -------
    pd.DataFrame
        The aggregated patient timeline. Each row is a timepoint.
        Columns are feature field names plus metadata columns
        (``_patient_id``, ``_document_id``, ``_document_type``,
        ``_document_date``).
    """
    if not documents:
        logger.warning("No documents provided for patient %s", patient_id)
        return pd.DataFrame()

    # ── Step 1: Extract all documents ────────────────────────────────────
    logger.info(
        "Extracting %d documents for patient %s", len(documents), patient_id
    )

    all_extractions: list[ExtractionResult] = []

    for doc in documents:
        text = doc.get("text", "")
        doc_id = doc.get("document_id", "")
        doc_date = doc.get("document_date")

        if not text.strip():
            logger.warning(
                "Skipping empty document %s for patient %s", doc_id, patient_id
            )
            continue

        extraction = pipeline.extract_document(
            text=text,
            document_id=doc_id,
            patient_id=patient_id,
        )

        # Override document_date if provided in the document dict
        # (the pipeline may extract its own, but the caller's value
        # takes precedence as it may come from metadata).
        if doc_date is not None:
            extraction.document_date = doc_date

        all_extractions.append(extraction)

    if not all_extractions:
        logger.warning(
            "No valid extractions for patient %s", patient_id
        )
        return pd.DataFrame()

    # ── Step 2: Row duplication for multiple events ──────────────────────
    logger.info("Detecting multiple events across %d extractions", len(all_extractions))

    expanded_extractions: list[ExtractionResult] = []
    for extraction in all_extractions:
        duplicated = detect_multiple_events(extraction)
        expanded_extractions.extend(duplicated)

    logger.info(
        "After row duplication: %d rows (from %d original extractions)",
        len(expanded_extractions),
        len(all_extractions),
    )

    # ── Step 3: Temporal aggregation ────────────────────────────────────
    logger.info("Aggregating patient timeline")
    timeline = aggregate_patient_timeline(expanded_extractions)

    logger.info(
        "Patient %s timeline: %d rows × %d columns",
        patient_id,
        len(timeline),
        len(timeline.columns),
    )

    return timeline


def build_patient_timeline_from_extractions(
    patient_id: str,
    extractions: list[ExtractionResult],
) -> pd.DataFrame:
    """Build a patient timeline from pre-computed ExtractionResults.

    This is a convenience function for when extractions have already been
    performed (e.g., in batch mode or when testing).  It skips the
    extraction step and directly applies row duplication and temporal
    aggregation.

    Parameters
    ----------
    patient_id : str
        Pseudonymised patient identifier.
    extractions : list[ExtractionResult]
        Pre-computed extraction results for the patient.

    Returns
    -------
    pd.DataFrame
        The aggregated patient timeline.
    """
    if not extractions:
        return pd.DataFrame()

    # Row duplication
    expanded: list[ExtractionResult] = []
    for ext in extractions:
        expanded.extend(detect_multiple_events(ext))

    # Temporal aggregation
    return aggregate_patient_timeline(expanded)
