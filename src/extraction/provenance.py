"""Extraction result dataclass with provenance and audit trail.

Defines ``ExtractionResult`` which bundles extracted features with
document metadata, source spans, and extraction decisions log.

Public API
----------
- ``ExtractionResult`` – Full extraction result for one document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .schema import ExtractionValue


@dataclass
class ExtractionResult:
    """Full extraction result for a single document.

    Bundles the extracted features with document-level metadata,
    section information, an audit trail of pipeline decisions,
    and a list of fields flagged for human review.

    Attributes
    ----------
    document_id : str
        Unique identifier for the document.
    document_type : str
        Classified document type (one of ``anapath``, ``molecular_report``,
        ``consultation``, ``rcp``, ``radiology``).
    document_date : str or None
        Extracted consultation/report date in ``DD/MM/YYYY`` format.
    patient_id : str
        Pseudonymised patient identifier.
    features : dict[str, ExtractionValue]
        Mapping ``field_name → ExtractionValue`` for all extracted fields.
    sections_detected : list[str]
        Names of sections detected in the document.
    extraction_log : list[str]
        Audit trail of pipeline decisions (human-readable messages).
    flagged_for_review : list[str]
        Field names that need human review (out-of-vocab, missing
        source span, low confidence, etc.).
    classification_confidence : float
        Confidence score of the document type classification.
    classification_is_ambiguous : bool
        Whether the document type classification was ambiguous.
    tier1_count : int
        Number of features extracted by Tier 1 (rule-based).
    tier2_count : int
        Number of features extracted by Tier 2 (LLM).
    total_extraction_time_ms : float
        Total time spent on extraction in milliseconds.
    """

    document_id: str = ""
    document_type: str = ""
    document_date: Optional[str] = None
    patient_id: str = ""
    features: dict[str, ExtractionValue] = field(default_factory=dict)
    sections_detected: list[str] = field(default_factory=list)
    extraction_log: list[str] = field(default_factory=list)
    flagged_for_review: list[str] = field(default_factory=list)
    classification_confidence: float = 0.0
    classification_is_ambiguous: bool = False
    tier1_count: int = 0
    tier2_count: int = 0
    total_extraction_time_ms: float = 0.0

    # -- Convenience helpers -------------------------------------------------

    def add_log(self, message: str) -> None:
        """Append a message to the extraction audit log."""
        self.extraction_log.append(message)

    def flag_field(self, field_name: str) -> None:
        """Mark a field as needing human review."""
        if field_name not in self.flagged_for_review:
            self.flagged_for_review.append(field_name)

    def update_flagged_from_features(self) -> None:
        """Scan ``self.features`` and update ``flagged_for_review``.

        Any feature whose ``ExtractionValue.flagged`` attribute is ``True``
        is added to the ``flagged_for_review`` list.
        """
        for fname, ev in self.features.items():
            if ev.flagged and fname not in self.flagged_for_review:
                self.flagged_for_review.append(fname)

    def summary(self) -> dict:
        """Return a concise summary dict (useful for logging / UI)."""
        return {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "document_date": self.document_date,
            "patient_id": self.patient_id,
            "total_features": len(self.features),
            "tier1_count": self.tier1_count,
            "tier2_count": self.tier2_count,
            "flagged_count": len(self.flagged_for_review),
            "sections": self.sections_detected,
            "extraction_time_ms": round(self.total_extraction_time_ms, 1),
        }

    def get_values_dict(self) -> dict[str, Optional[str | int | float]]:
        """Return a flat dict of ``field_name → value`` (no provenance)."""
        return {
            fname: ev.value
            for fname, ev in self.features.items()
        }
