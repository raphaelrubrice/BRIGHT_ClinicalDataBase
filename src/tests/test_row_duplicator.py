"""Tests for src/aggregation/row_duplicator.py — multiple event row splitting.

Covers:
- Single event → no duplication
- Multiple surgery dates → 2+ rows
- Multiple chemo lines → 2+ rows
- Multiple radiotherapy courses → 2+ rows
- Multiple progression events → 2+ rows
- Shared features correctly copied to all rows
- Event-specific features correctly isolated
- Edge cases: empty features, single date, null values
"""

import pytest

from src.aggregation.row_duplicator import (
    detect_multiple_events,
    _parse_multiple_values,
    _count_distinct_dates,
    SHARED_FEATURES,
    SURGERY_EVENT_FIELDS,
    CHEMO_EVENT_FIELDS,
)
from src.extraction.provenance import ExtractionResult
from src.extraction.schema import ExtractionValue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extraction(**features: dict) -> ExtractionResult:
    """Create an ExtractionResult with the given features."""
    feat_dict: dict[str, ExtractionValue] = {}
    for fname, fval in features.items():
        if isinstance(fval, ExtractionValue):
            feat_dict[fname] = fval
        else:
            feat_dict[fname] = ExtractionValue(value=fval, extraction_tier="rule")
    return ExtractionResult(
        document_id="doc_test",
        document_type="consultation",
        document_date="01/01/2024",
        patient_id="patient_test",
        features=feat_dict,
    )


def _ev(value, tier="rule", span=None) -> ExtractionValue:
    """Shorthand to create an ExtractionValue."""
    return ExtractionValue(
        value=value,
        extraction_tier=tier,
        source_span=span or str(value),
    )


# ---------------------------------------------------------------------------
# Tests: _parse_multiple_values
# ---------------------------------------------------------------------------

class TestParseMultipleValues:
    """Test the helper that splits multi-value strings."""

    def test_comma_separated(self):
        result = _parse_multiple_values("01/03/2020, 15/09/2021")
        assert result == ["01/03/2020", "15/09/2021"]

    def test_semicolon_separated(self):
        result = _parse_multiple_values("Temozolomide; Bevacizumab")
        assert result == ["Temozolomide", "Bevacizumab"]

    def test_et_separated(self):
        result = _parse_multiple_values("Temozolomide et Bevacizumab")
        assert result == ["Temozolomide", "Bevacizumab"]

    def test_puis_separated(self):
        result = _parse_multiple_values("Temozolomide puis PCV")
        assert result == ["Temozolomide", "PCV"]

    def test_single_value(self):
        result = _parse_multiple_values("01/03/2020")
        assert result == ["01/03/2020"]

    def test_empty_string(self):
        result = _parse_multiple_values("")
        assert result == []

    def test_none_safe(self):
        result = _parse_multiple_values(None)
        assert result == []

    def test_slash_not_split(self):
        """Slashes are NOT treated as delimiters (dates use DD/MM/YYYY)."""
        result = _parse_multiple_values("01/03/2020")
        assert result == ["01/03/2020"]


class TestCountDistinctDates:
    """Test extracting distinct date values from a feature."""

    def test_single_date(self):
        ext = _make_extraction(chir_date="01/03/2020")
        dates = _count_distinct_dates(ext, "chir_date")
        assert dates == ["01/03/2020"]

    def test_multiple_dates_comma(self):
        ext = _make_extraction(chir_date="01/03/2020, 15/09/2021")
        dates = _count_distinct_dates(ext, "chir_date")
        assert dates == ["01/03/2020", "15/09/2021"]

    def test_duplicates_removed(self):
        ext = _make_extraction(chir_date="01/03/2020, 01/03/2020, 15/09/2021")
        dates = _count_distinct_dates(ext, "chir_date")
        assert dates == ["01/03/2020", "15/09/2021"]

    def test_missing_field(self):
        ext = _make_extraction(sexe="M")
        dates = _count_distinct_dates(ext, "chir_date")
        assert dates == []

    def test_null_value(self):
        ext = _make_extraction(chir_date=None)
        dates = _count_distinct_dates(ext, "chir_date")
        assert dates == []


# ---------------------------------------------------------------------------
# Tests: detect_multiple_events
# ---------------------------------------------------------------------------

class TestSingleEventNoduplication:
    """When only one event is detected, no duplication occurs."""

    def test_no_features(self):
        ext = _make_extraction()
        result = detect_multiple_events(ext)
        assert len(result) == 1
        assert result[0] is ext  # same object, no copy

    def test_single_surgery(self):
        ext = _make_extraction(
            chir_date="01/03/2020",
            type_chirurgie="biopsie",
            sexe="M",
            nip="12345",
        )
        result = detect_multiple_events(ext)
        assert len(result) == 1

    def test_single_chemo(self):
        ext = _make_extraction(
            chimios="Temozolomide",
            chm_date_debut="01/04/2020",
        )
        result = detect_multiple_events(ext)
        assert len(result) == 1

    def test_single_radio(self):
        ext = _make_extraction(
            rx_date_debut="01/05/2020",
            rx_dose="60",
        )
        result = detect_multiple_events(ext)
        assert len(result) == 1

    def test_single_progression(self):
        ext = _make_extraction(
            date_progression="01/06/2020",
            progress_clinique="oui",
        )
        result = detect_multiple_events(ext)
        assert len(result) == 1


class TestMultipleSurgeries:
    """Documents with multiple surgery dates should be split."""

    def test_two_surgeries(self):
        ext = _make_extraction(
            chir_date="01/03/2020, 15/09/2021",
            type_chirurgie="biopsie",
            sexe="M",
            nip="12345",
            tumeur_lateralite="gauche",
        )
        result = detect_multiple_events(ext)
        assert len(result) == 2

        # Both rows should have shared features
        for row in result:
            assert "sexe" in row.features
            assert row.features["sexe"].value == "M"
            assert "nip" in row.features
            assert row.features["nip"].value == "12345"
            assert "tumeur_lateralite" in row.features
            assert row.features["tumeur_lateralite"].value == "gauche"

        # Each row should have the correct surgery date
        dates = [row.features["chir_date"].value for row in result]
        assert "01/03/2020" in dates
        assert "15/09/2021" in dates

    def test_surgery_audit_log(self):
        ext = _make_extraction(chir_date="01/03/2020, 15/09/2021")
        result = detect_multiple_events(ext)
        assert len(result) == 2
        for i, row in enumerate(result):
            assert any("Row duplicated" in msg for msg in row.extraction_log)
            assert any(f"event {i + 1}" in msg for msg in row.extraction_log)

    def test_three_surgeries(self):
        ext = _make_extraction(
            chir_date="01/01/2019, 15/06/2020, 01/12/2021"
        )
        result = detect_multiple_events(ext)
        assert len(result) == 3


class TestMultipleChemoLines:
    """Documents with multiple chemo start dates should be split."""

    def test_two_chemo_lines(self):
        ext = _make_extraction(
            chimios="Temozolomide, Bevacizumab",
            chm_date_debut="01/04/2020, 01/01/2021",
            nip="12345",
        )
        result = detect_multiple_events(ext)
        assert len(result) == 2

        # Check that chemo names are correctly assigned
        names = [row.features.get("chimios") for row in result]
        name_values = [n.value for n in names if n]
        assert "Temozolomide" in name_values
        assert "Bevacizumab" in name_values

    def test_chemo_with_mismatched_names(self):
        """If #names ≠ #dates, all rows get the original chimios value."""
        ext = _make_extraction(
            chimios="Temozolomide + Bevacizumab",
            chm_date_debut="01/04/2020, 01/01/2021, 01/06/2022",
        )
        result = detect_multiple_events(ext)
        assert len(result) == 3


class TestMultipleRadiotherapy:
    """Documents with multiple radiotherapy start dates should be split."""

    def test_two_radio_courses(self):
        ext = _make_extraction(
            rx_date_debut="01/05/2020, 01/11/2021",
            rx_dose="60",
            nip="12345",
        )
        result = detect_multiple_events(ext)
        assert len(result) == 2

        # Both rows should have the dose (copied from original)
        for row in result:
            assert "rx_dose" in row.features
            assert "nip" in row.features


class TestMultipleProgressions:
    """Documents with multiple progression dates should be split."""

    def test_two_progressions(self):
        ext = _make_extraction(
            date_progression="01/06/2020, 01/02/2022",
            progress_clinique="oui",
        )
        result = detect_multiple_events(ext)
        assert len(result) == 2

        # Both should have progression flags
        for row in result:
            if "progress_clinique" in row.features:
                assert row.features["progress_clinique"].value == "oui"


class TestDuplicationPriority:
    """Surgery duplication takes priority over chemo, etc."""

    def test_surgery_takes_priority_over_chemo(self):
        ext = _make_extraction(
            chir_date="01/03/2020, 15/09/2021",
            chm_date_debut="01/04/2020, 01/01/2021",
        )
        result = detect_multiple_events(ext)
        # Surgery detected first → splits on surgery dates
        dates = [row.features.get("chir_date") for row in result]
        # Should have exactly 2 rows from surgery duplication
        assert len(result) == 2
        assert all(d is not None for d in dates)


class TestSharedFeaturesCopied:
    """Verify that all shared features are correctly propagated."""

    def test_all_shared_features_present(self):
        # Create extraction with representative shared features
        ext = _make_extraction(
            sexe="M",
            nip="12345",
            date_de_naissance="01/01/1970",
            tumeur_lateralite="gauche",
            tumeur_position="frontal",
            ihc_idh1="positif",
            mol_tert="mute",
            ch1p="perte",
            ampli_egfr="oui",
            fusion_fgfr="non",
            chir_date="01/03/2020, 15/09/2021",
        )
        result = detect_multiple_events(ext)
        assert len(result) == 2

        for row in result:
            # Check representative shared features
            assert row.features.get("sexe") is not None
            assert row.features["sexe"].value == "M"
            assert row.features.get("ihc_idh1") is not None
            assert row.features["ihc_idh1"].value == "positif"
            assert row.features.get("mol_tert") is not None
            assert row.features["mol_tert"].value == "mute"


class TestEdgeCases:
    """Edge cases for the duplicator."""

    def test_empty_features(self):
        ext = ExtractionResult(
            document_id="empty",
            document_type="consultation",
            features={},
        )
        result = detect_multiple_events(ext)
        assert len(result) == 1
        assert result[0] is ext

    def test_all_null_values(self):
        ext = _make_extraction(
            chir_date=None,
            chm_date_debut=None,
            rx_date_debut=None,
            date_progression=None,
        )
        result = detect_multiple_events(ext)
        assert len(result) == 1

    def test_document_metadata_preserved(self):
        ext = _make_extraction(chir_date="01/03/2020, 15/09/2021")
        ext.document_id = "doc_123"
        ext.document_type = "consultation"
        ext.patient_id = "patient_456"
        ext.document_date = "01/03/2020"

        result = detect_multiple_events(ext)
        for row in result:
            assert row.document_id == "doc_123"
            assert row.document_type == "consultation"
            assert row.patient_id == "patient_456"
            assert row.document_date == "01/03/2020"
