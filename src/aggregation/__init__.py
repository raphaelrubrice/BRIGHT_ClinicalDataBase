"""Patient-level aggregation: row duplication, temporal forward-fill, and timeline building.

Public API
----------
- :func:`detect_multiple_events` — Split extractions with multiple treatment events
- :func:`aggregate_patient_timeline` — Temporal forward-fill and conflict resolution
- :func:`build_patient_timeline` — End-to-end patient timeline builder
- :func:`build_patient_timeline_from_extractions` — Timeline from pre-computed extractions
"""

from .row_duplicator import detect_multiple_events
from .temporal_aggregation import aggregate_patient_timeline
from .patient_timeline import build_patient_timeline, build_patient_timeline_from_extractions

__all__ = [
    "detect_multiple_events",
    "aggregate_patient_timeline",
    "build_patient_timeline",
    "build_patient_timeline_from_extractions",
]
