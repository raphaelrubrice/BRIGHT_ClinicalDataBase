"""Tests for src/aggregation/patient_timeline.py — end-to-end patient timeline.

Covers:
- Building timeline from a single document
- Building timeline from multiple documents
- Row duplication + temporal aggregation integration
- Empty document list
- build_patient_timeline_from_extractions convenience function
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from src.aggregation.patient_timeline import (
    build_patient_timeline,
    build_patient_timeline_from_extractions,
)
from src.extraction.provenance import ExtractionResult
from src.extraction.schema import ExtractionValue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extraction(
    doc_id: str = "doc_1",
    doc_type: str = "consultation",
    doc_date: str = "01/01/2024",
    patient_id: str = "patient_1",
    **features,
) -> ExtractionResult:
    """Create an ExtractionResult with the given features."""
    feat_dict: dict[str, ExtractionValue] = {}
    for fname, fval in features.items():
        if isinstance(fval, ExtractionValue):
            feat_dict[fname] = fval
        elif fval is None:
            feat_dict[fname] = ExtractionValue(value=None, extraction_tier="rule")
        else:
            feat_dict[fname] = ExtractionValue(
                value=fval, extraction_tier="rule", source_span=str(fval)
            )
    return ExtractionResult(
        document_id=doc_id,
        document_type=doc_type,
        document_date=doc_date,
        patient_id=patient_id,
        features=feat_dict,
    )


def _mock_pipeline():
    """Create a mock ExtractionPipeline."""
    pipeline = MagicMock()
    return pipeline


# ---------------------------------------------------------------------------
# Tests: build_patient_timeline_from_extractions
# ---------------------------------------------------------------------------

class TestBuildTimelineFromExtractions:
    """Test the convenience function that takes pre-computed extractions."""

    def test_single_document(self):
        ext = _make_extraction(
            sexe="M",
            ik_clinique=90,
            epilepsie="oui",
        )
        df = build_patient_timeline_from_extractions("patient_1", [ext])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df["sexe"].iloc[0] == "M"
        assert df["ik_clinique"].iloc[0] == 90

    def test_multiple_documents(self):
        ext1 = _make_extraction(
            doc_id="d1", doc_date="01/01/2024",
            sexe="M", ik_clinique=90,
        )
        ext2 = _make_extraction(
            doc_id="d2", doc_date="01/06/2024",
            ik_clinique=70,
        )
        df = build_patient_timeline_from_extractions("patient_1", [ext1, ext2])
        assert len(df) == 2
        assert df["ik_clinique"].iloc[0] == 90
        assert df["ik_clinique"].iloc[1] == 70
        # Static feature carried forward
        assert df["sexe"].iloc[1] == "M"

    def test_with_row_duplication(self):
        ext = _make_extraction(
            chir_date="01/03/2020, 15/09/2021",
            sexe="M",
        )
        df = build_patient_timeline_from_extractions("patient_1", [ext])
        # Should have 2 rows due to row duplication
        assert len(df) == 2

    def test_empty_extractions(self):
        df = build_patient_timeline_from_extractions("patient_1", [])
        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestBuildPatientTimeline:
    """Test the full pipeline function (with mocked extraction pipeline)."""

    def test_empty_documents(self):
        pipeline = _mock_pipeline()
        df = build_patient_timeline("patient_1", [], pipeline)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_single_document(self):
        pipeline = _mock_pipeline()

        # Configure mock to return a specific extraction
        ext = _make_extraction(sexe="M", ik_clinique=90)
        pipeline.extract_document.return_value = ext

        documents = [
            {"text": "Compte-rendu de consultation...", "document_id": "doc_1"},
        ]
        df = build_patient_timeline("patient_1", documents, pipeline)
        assert len(df) == 1
        assert df["sexe"].iloc[0] == "M"
        pipeline.extract_document.assert_called_once()

    def test_document_date_override(self):
        """Caller-provided document_date should override pipeline-extracted date."""
        pipeline = _mock_pipeline()

        ext = _make_extraction(doc_date="01/01/2024", sexe="M")
        pipeline.extract_document.return_value = ext

        documents = [
            {
                "text": "Some text",
                "document_id": "doc_1",
                "document_date": "15/03/2024",
            },
        ]
        df = build_patient_timeline("patient_1", documents, pipeline)
        # The date should be overridden
        assert df["_document_date"].iloc[0] == "15/03/2024"

    def test_skips_empty_documents(self):
        pipeline = _mock_pipeline()

        ext = _make_extraction(sexe="M")
        pipeline.extract_document.return_value = ext

        documents = [
            {"text": "", "document_id": "empty_doc"},
            {"text": "   ", "document_id": "whitespace_doc"},
            {"text": "Real document text", "document_id": "real_doc"},
        ]
        df = build_patient_timeline("patient_1", documents, pipeline)
        # Only the real document should be processed
        assert pipeline.extract_document.call_count == 1

    def test_multiple_documents_integration(self):
        pipeline = _mock_pipeline()

        ext1 = _make_extraction(
            doc_id="d1", doc_date="01/01/2024", doc_type="consultation",
            sexe="M", ik_clinique=90,
        )
        ext2 = _make_extraction(
            doc_id="d2", doc_date="01/06/2024", doc_type="consultation",
            ik_clinique=70,
        )

        pipeline.extract_document.side_effect = [ext1, ext2]

        documents = [
            {"text": "First consultation...", "document_id": "d1"},
            {"text": "Second consultation...", "document_id": "d2"},
        ]
        df = build_patient_timeline("patient_1", documents, pipeline)
        assert len(df) == 2
        assert pipeline.extract_document.call_count == 2

    def test_metadata_columns(self):
        pipeline = _mock_pipeline()
        ext = _make_extraction(
            doc_id="d1", doc_date="01/01/2024", doc_type="consultation",
            patient_id="patient_1", sexe="M",
        )
        pipeline.extract_document.return_value = ext

        documents = [{"text": "Some text", "document_id": "d1"}]
        df = build_patient_timeline("patient_1", documents, pipeline)

        assert "_patient_id" in df.columns
        assert "_document_id" in df.columns
        assert "_document_type" in df.columns
        assert "_document_date" in df.columns
