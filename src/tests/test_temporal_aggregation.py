"""Tests for src/aggregation/temporal_aggregation.py — forward-fill and conflict resolution.

Covers:
- Static feature forward-fill (set once, update only on correction)
- Specimen-bound feature reset when a new surgery occurs
- Time-varying features: NA does NOT overwrite previous explicit values
- Conflict resolution via document-type priority
- Chronological ordering
- Multiple documents from the same patient
- Synthetic 5-document patient test case
"""

import pytest
import pandas as pd

from src.aggregation.temporal_aggregation import (
    aggregate_patient_timeline,
    STATIC_FEATURES,
    SPECIMEN_BOUND_FEATURES,
    TIME_VARYING_FEATURES,
    BIO_PRIORITY,
    CLINIQUE_PRIORITY,
    _priority_rank,
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


# ---------------------------------------------------------------------------
# Tests: Priority ranking
# ---------------------------------------------------------------------------

class TestPriorityRank:
    """Test document-type priority resolution."""

    def test_bio_field_priority(self):
        # anapath > molecular_report > rcp > consultation
        assert _priority_rank("anapath", "ihc_idh1") < _priority_rank("molecular_report", "ihc_idh1")
        assert _priority_rank("molecular_report", "ihc_idh1") < _priority_rank("rcp", "ihc_idh1")
        assert _priority_rank("rcp", "ihc_idh1") < _priority_rank("consultation", "ihc_idh1")

    def test_clinique_field_priority(self):
        # consultation > rcp > anapath
        assert _priority_rank("consultation", "ik_clinique") < _priority_rank("rcp", "ik_clinique")
        assert _priority_rank("rcp", "ik_clinique") < _priority_rank("anapath", "ik_clinique")

    def test_unknown_doc_type(self):
        # Unknown types get the lowest priority
        rank = _priority_rank("unknown_type", "ihc_idh1")
        assert rank == len(BIO_PRIORITY)


# ---------------------------------------------------------------------------
# Tests: Static features
# ---------------------------------------------------------------------------

class TestStaticFeatureForwardFill:
    """Static features should be set once and carried forward."""

    def test_set_once_and_persist(self):
        ext1 = _make_extraction(doc_id="d1", doc_date="01/01/2024", sexe="M")
        ext2 = _make_extraction(doc_id="d2", doc_date="01/06/2024")

        df = aggregate_patient_timeline([ext1, ext2])
        assert len(df) == 2
        # sexe should be M in both rows
        assert df["sexe"].iloc[0] == "M"
        assert df["sexe"].iloc[1] == "M"

    def test_na_does_not_overwrite(self):
        ext1 = _make_extraction(doc_id="d1", doc_date="01/01/2024", sexe="M")
        ext2 = _make_extraction(doc_id="d2", doc_date="01/06/2024", sexe=None)

        df = aggregate_patient_timeline([ext1, ext2])
        # sexe should remain M (NA doesn't overwrite)
        assert df["sexe"].iloc[0] == "M"
        assert df["sexe"].iloc[1] == "M"

    def test_higher_priority_updates(self):
        # consultation has higher priority for clinical fields
        ext1 = _make_extraction(
            doc_id="d1", doc_date="01/01/2024", doc_type="rcp",
            tumeur_lateralite="droite",
        )
        ext2 = _make_extraction(
            doc_id="d2", doc_date="01/06/2024", doc_type="consultation",
            tumeur_lateralite="gauche",
        )

        df = aggregate_patient_timeline([ext1, ext2])
        # consultation has higher priority for clinical fields
        assert df["tumeur_lateralite"].iloc[1] == "gauche"

    def test_lower_priority_does_not_update(self):
        ext1 = _make_extraction(
            doc_id="d1", doc_date="01/01/2024", doc_type="consultation",
            tumeur_lateralite="gauche",
        )
        ext2 = _make_extraction(
            doc_id="d2", doc_date="01/06/2024", doc_type="radiology",
            tumeur_lateralite="droite",
        )

        df = aggregate_patient_timeline([ext1, ext2])
        # consultation has higher priority → stays "gauche"
        assert df["tumeur_lateralite"].iloc[1] == "gauche"


# ---------------------------------------------------------------------------
# Tests: Specimen-bound features
# ---------------------------------------------------------------------------

class TestSpecimenBoundFeatures:
    """Specimen-bound features carry from specimen date until next surgery."""

    def test_carry_forward_without_surgery(self):
        ext1 = _make_extraction(
            doc_id="d1", doc_date="01/01/2024", doc_type="anapath",
            ihc_idh1="positif",
            grade=4,
        )
        ext2 = _make_extraction(
            doc_id="d2", doc_date="01/06/2024", doc_type="consultation",
        )

        df = aggregate_patient_timeline([ext1, ext2])
        # specimen-bound features carry forward
        assert df["ihc_idh1"].iloc[0] == "positif"
        assert df["ihc_idh1"].iloc[1] == "positif"
        assert df["grade"].iloc[0] == 4
        assert df["grade"].iloc[1] == 4

    def test_reset_on_new_surgery(self):
        ext1 = _make_extraction(
            doc_id="d1", doc_date="01/01/2024", doc_type="anapath",
            ihc_idh1="positif",
        )
        # A new surgery event
        ext2 = _make_extraction(
            doc_id="d2", doc_date="01/06/2024", doc_type="anapath",
            chir_date="01/06/2024",
            ihc_idh1="negatif",
        )

        df = aggregate_patient_timeline([ext1, ext2])
        assert df["ihc_idh1"].iloc[0] == "positif"
        # After new surgery, the new value takes over
        assert df["ihc_idh1"].iloc[1] == "negatif"

    def test_reset_clears_previous_specimen_values(self):
        ext1 = _make_extraction(
            doc_id="d1", doc_date="01/01/2024", doc_type="anapath",
            ihc_idh1="positif",
            mol_tert="mute",
        )
        # New surgery only sets ihc_idh1, not mol_tert
        ext2 = _make_extraction(
            doc_id="d2", doc_date="01/06/2024", doc_type="anapath",
            chir_date="01/06/2024",
            ihc_idh1="negatif",
        )

        df = aggregate_patient_timeline([ext1, ext2])
        # After surgery reset, mol_tert should be None (cleared)
        assert df["ihc_idh1"].iloc[1] == "negatif"
        assert df["mol_tert"].iloc[1] is None or pd.isna(df["mol_tert"].iloc[1])


# ---------------------------------------------------------------------------
# Tests: Time-varying features
# ---------------------------------------------------------------------------

class TestTimeVaryingFeatures:
    """Time-varying features carry latest explicit value; NA doesn't overwrite."""

    def test_latest_value_wins(self):
        ext1 = _make_extraction(
            doc_id="d1", doc_date="01/01/2024",
            ik_clinique=90,
        )
        ext2 = _make_extraction(
            doc_id="d2", doc_date="01/06/2024",
            ik_clinique=70,
        )

        df = aggregate_patient_timeline([ext1, ext2])
        assert df["ik_clinique"].iloc[0] == 90
        assert df["ik_clinique"].iloc[1] == 70

    def test_na_does_not_overwrite_explicit(self):
        ext1 = _make_extraction(
            doc_id="d1", doc_date="01/01/2024",
            ik_clinique=90,
        )
        ext2 = _make_extraction(
            doc_id="d2", doc_date="01/06/2024",
            ik_clinique=None,
        )

        df = aggregate_patient_timeline([ext1, ext2])
        assert df["ik_clinique"].iloc[0] == 90
        assert df["ik_clinique"].iloc[1] == 90  # carried forward, not overwritten

    def test_epilepsie_carries_forward(self):
        ext1 = _make_extraction(
            doc_id="d1", doc_date="01/01/2024",
            epilepsie="oui",
        )
        ext2 = _make_extraction(
            doc_id="d2", doc_date="01/06/2024",
        )

        df = aggregate_patient_timeline([ext1, ext2])
        assert df["epilepsie"].iloc[0] == "oui"
        assert df["epilepsie"].iloc[1] == "oui"


# ---------------------------------------------------------------------------
# Tests: Conflict resolution
# ---------------------------------------------------------------------------

class TestConflictResolution:
    """Conflict resolution should follow document-type priority."""

    def test_bio_field_anapath_wins(self):
        ext1 = _make_extraction(
            doc_id="d1", doc_date="01/01/2024", doc_type="consultation",
            ihc_idh1="negatif",
        )
        ext2 = _make_extraction(
            doc_id="d2", doc_date="01/01/2024", doc_type="anapath",
            ihc_idh1="positif",
        )

        df = aggregate_patient_timeline([ext1, ext2])
        # anapath has higher priority for bio fields
        assert df["ihc_idh1"].iloc[-1] == "positif"

    def test_clinique_field_consultation_wins(self):
        ext1 = _make_extraction(
            doc_id="d1", doc_date="01/01/2024", doc_type="anapath",
            ik_clinique=60,
        )
        ext2 = _make_extraction(
            doc_id="d2", doc_date="01/01/2024", doc_type="consultation",
            ik_clinique=80,
        )

        df = aggregate_patient_timeline([ext1, ext2])
        # Consultation has higher priority for clinical fields
        # Since time-varying takes latest value, the consultation value (processed after)
        # should be present
        assert df["ik_clinique"].iloc[-1] == 80


# ---------------------------------------------------------------------------
# Tests: Chronological ordering
# ---------------------------------------------------------------------------

class TestChronologicalOrdering:
    """Documents should be sorted by date regardless of input order."""

    def test_out_of_order_input(self):
        ext1 = _make_extraction(doc_id="d1", doc_date="01/06/2024", sexe="M")
        ext2 = _make_extraction(doc_id="d2", doc_date="01/01/2024", sexe="F")

        df = aggregate_patient_timeline([ext1, ext2])
        # Should be sorted chronologically
        assert df["_document_date"].iloc[0] == "01/01/2024"
        assert df["_document_date"].iloc[1] == "01/06/2024"

    def test_missing_dates_sort_last(self):
        ext1 = _make_extraction(doc_id="d1", doc_date="01/01/2024", sexe="M")
        ext2 = ExtractionResult(
            document_id="d2",
            document_type="consultation",
            patient_id="patient_1",
            features={"sexe": ExtractionValue(value="M", extraction_tier="rule")},
        )

        df = aggregate_patient_timeline([ext2, ext1])
        assert df["_document_id"].iloc[0] == "d1"  # has a date → first


# ---------------------------------------------------------------------------
# Tests: Empty and edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for temporal aggregation."""

    def test_empty_list(self):
        df = aggregate_patient_timeline([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_single_extraction(self):
        ext = _make_extraction(sexe="M", ik_clinique=90)
        df = aggregate_patient_timeline([ext])
        assert len(df) == 1
        assert df["sexe"].iloc[0] == "M"
        assert df["ik_clinique"].iloc[0] == 90

    def test_metadata_columns_present(self):
        ext = _make_extraction(sexe="M")
        df = aggregate_patient_timeline([ext])
        assert "_patient_id" in df.columns
        assert "_document_id" in df.columns
        assert "_document_type" in df.columns
        assert "_document_date" in df.columns


# ---------------------------------------------------------------------------
# Tests: Synthetic 5-document patient scenario
# ---------------------------------------------------------------------------

class TestSyntheticPatientTimeline:
    """A synthetic 5-document patient test case produces the expected timeline.

    Documents (in chronological order):
    1. Consultation (01/2024) — initial assessment, demographics, first symptoms
    2. Anapath (02/2024) — biopsy results, IHC, molecular
    3. RCP (03/2024) — treatment decision
    4. Consultation (06/2024) — follow-up, new IK, progression
    5. Anapath (09/2024) — re-surgery, new specimen results
    """

    def test_full_timeline(self):
        ext1 = _make_extraction(
            doc_id="consult_1", doc_date="15/01/2024", doc_type="consultation",
            sexe="M",
            date_de_naissance="01/05/1960",
            tumeur_lateralite="gauche",
            epilepsie="oui",
            ik_clinique=90,
            date_1er_symptome="01/12/2023",
        )
        ext2 = _make_extraction(
            doc_id="anapath_1", doc_date="10/02/2024", doc_type="anapath",
            ihc_idh1="positif",
            mol_tert="mute",
            grade=3,
            ch1p="perte",
            ch19q="perte",
            chir_date="10/02/2024",
            date_chir="10/02/2024",
        )
        ext3 = _make_extraction(
            doc_id="rcp_1", doc_date="01/03/2024", doc_type="rcp",
            chimios="Temozolomide",
            chm_date_debut="15/03/2024",
            rx_dose="60",
            rx_date_debut="15/03/2024",
        )
        ext4 = _make_extraction(
            doc_id="consult_2", doc_date="15/06/2024", doc_type="consultation",
            ik_clinique=70,
            epilepsie="oui",
            progress_clinique="non",
        )
        ext5 = _make_extraction(
            doc_id="anapath_2", doc_date="01/09/2024", doc_type="anapath",
            chir_date="01/09/2024",
            date_chir="01/09/2024",
            ihc_idh1="positif",
            mol_tert="wt",
            grade=4,
        )

        df = aggregate_patient_timeline([ext1, ext2, ext3, ext4, ext5])

        assert len(df) == 5

        # ── Row 0 (consult_1): demographics set ──
        assert df["sexe"].iloc[0] == "M"
        assert df["ik_clinique"].iloc[0] == 90
        assert df["epilepsie"].iloc[0] == "oui"

        # ── Row 1 (anapath_1): specimen-bound features set ──
        assert df["ihc_idh1"].iloc[1] == "positif"
        assert df["mol_tert"].iloc[1] == "mute"
        assert df["grade"].iloc[1] == 3
        # Static features carried forward
        assert df["sexe"].iloc[1] == "M"

        # ── Row 2 (rcp_1): treatment info, specimen carries ──
        assert df["ihc_idh1"].iloc[2] == "positif"  # specimen carried
        assert df["grade"].iloc[2] == 3  # specimen carried
        assert df["sexe"].iloc[2] == "M"  # static carried

        # ── Row 3 (consult_2): IK updated ──
        assert df["ik_clinique"].iloc[3] == 70
        assert df["sexe"].iloc[3] == "M"
        # Specimen features still carried
        assert df["ihc_idh1"].iloc[3] == "positif"

        # ── Row 4 (anapath_2): new surgery → specimen reset ──
        assert df["ihc_idh1"].iloc[4] == "positif"  # new value
        assert df["mol_tert"].iloc[4] == "wt"  # new specimen value
        assert df["grade"].iloc[4] == 4  # new grade
        # Static features still present
        assert df["sexe"].iloc[4] == "M"
        # ch1p was only from first specimen, reset
        assert df.get("ch1p", pd.Series([None])).iloc[4] is None or pd.isna(
            df.get("ch1p", pd.Series([None])).iloc[4]
        )
