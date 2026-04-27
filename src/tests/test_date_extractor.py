"""Tests for DateExtractor, context-aware date extraction via eds.dates."""

import pytest
pytest.importorskip("edsnlp", reason="requires edsnlp (install via setup.sh)")

from src.extraction.date_extractor import (
    DateExtractor,
    _extract_consult_date_from_text,
    _get_context,
    _parse_raw_consult_date,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def extractor():
    """Shared DateExtractor instance (loads edsnlp pipeline once)."""
    return DateExtractor()


# ── Unit tests: consultation date regex ───────────────────────────────────

class TestConsultDateRegex:
    def test_consultation_du(self):
        text = "Consultation du 15/03/2024 à l'hôpital."
        assert _extract_consult_date_from_text(text) == "15/03/2024"

    def test_paris_le(self):
        text = "Paris, le 02/01/2023\nCompte rendu..."
        assert _extract_consult_date_from_text(text) == "02/01/2023"

    def test_date_reception(self):
        text = "Date de réception: 28/11/2022\nExamen..."
        assert _extract_consult_date_from_text(text) == "28/11/2022"

    def test_no_match(self):
        text = "Le patient est vu pour suivi."
        assert _extract_consult_date_from_text(text) is None


class TestParseRawConsultDate:
    def test_dmy(self):
        assert _parse_raw_consult_date("15/03/2024") == "15/03/2024"

    def test_ymd(self):
        assert _parse_raw_consult_date("2024/03/15") == "15/03/2024"

    def test_text_month(self):
        assert _parse_raw_consult_date("15 mars 2024") == "15/03/2024"

    def test_invalid(self):
        assert _parse_raw_consult_date("hello") is None


# ── Unit tests: context extraction ────────────────────────────────────────

class TestGetContext:
    def test_basic_context(self):
        text = "Le patient est décédé le 15/03/2024 à l'hôpital."
        # date span is "15/03/2024" at positions 25-35
        ctx = _get_context(text, 25, 35, [(25, 35)], half_window=20)
        assert "décédé" in ctx.lower()

    def test_masks_other_dates(self):
        text = "Opéré le 01/02/2023, décédé le 15/03/2024 à domicile."
        # date1 = "01/02/2023" at 9-19, date2 = "15/03/2024" at 31-41
        all_spans = [(9, 19), (31, 41)]
        ctx = _get_context(text, 31, 41, all_spans, half_window=30)
        # Context for date2 should not contain date1 text
        assert "01/02/2023" not in ctx


# ── Integration tests: DateExtractor.extract ──────────────────────────────

class TestDateExtractorFR:
    """French date extraction tests."""

    def test_deces_basic(self, extractor):
        text = "Le patient est décédé le 15 mars 2024 à l'hôpital Pitié-Salpêtrière."
        result = extractor.extract(
            text, feature_subset=["date_deces"], language="fr"
        )
        assert "date_deces" in result
        assert result["date_deces"].value == "15/03/2024"
        assert result["date_deces"].extraction_tier == "rule"

    def test_chirurgie(self, extractor):
        text = "Le patient a été opéré le 10/06/2023 pour exérèse tumorale."
        result = extractor.extract(
            text, feature_subset=["date_chir"], language="fr"
        )
        assert "date_chir" in result
        assert result["date_chir"].value == "10/06/2023"

    def test_chimio_dates(self, extractor):
        text = (
            "Début de chimiothérapie le 01/09/2023. "
            "Dernière cure le 15/12/2023."
        )
        result = extractor.extract(
            text,
            feature_subset=["chm_date_debut", "chm_date_fin"],
            language="fr",
        )
        assert "chm_date_debut" in result
        assert result["chm_date_debut"].value == "01/09/2023"

    def test_multiple_dates_correct_assignment(self, extractor):
        text = (
            "Consultation du 05/01/2024. "
            "Le patient a été opéré le 10/06/2023 pour craniotomie. "
            "Décédé le 20/03/2024."
        )
        result = extractor.extract(
            text,
            feature_subset=["date_chir", "date_deces"],
            language="fr",
            consultation_date="05/01/2024",
        )
        # Consultation date should be excluded
        for ev in result.values():
            assert ev.value != "05/01/2024"
        # Surgery and death dates should be assigned
        if "date_chir" in result:
            assert result["date_chir"].value == "10/06/2023"
        if "date_deces" in result:
            assert result["date_deces"].value == "20/03/2024"

    def test_no_context_returns_empty(self, extractor):
        text = "15/03/2024"
        result = extractor.extract(
            text, feature_subset=["date_deces"], language="fr"
        )
        # Isolated date with absolutely no context → should not be assigned
        assert "date_deces" not in result

    def test_rcp(self, extractor):
        text = "Dossier discuté en RCP le 12/04/2023."
        result = extractor.extract(
            text, feature_subset=["date_rcp"], language="fr"
        )
        assert "date_rcp" in result
        assert result["date_rcp"].value == "12/04/2023"

    def test_radiotherapy(self, extractor):
        text = "Début de radiothérapie le 15/07/2023. Fin RT le 30/08/2023."
        result = extractor.extract(
            text,
            feature_subset=["rx_date_debut", "rx_date_fin"],
            language="fr",
        )
        assert "rx_date_debut" in result
        assert result["rx_date_debut"].value == "15/07/2023"

    def test_empty_feature_subset(self, extractor):
        text = "Décédé le 15/03/2024."
        result = extractor.extract(text, feature_subset=[], language="fr")
        assert result == {}


class TestDateExtractorEN:
    """English date extraction tests."""

    def test_death_en(self, extractor):
        text = "The patient died on 15/03/2024 at the hospital."
        result = extractor.extract(
            text, feature_subset=["date_deces"], language="en"
        )
        assert "date_deces" in result
        assert result["date_deces"].value == "15/03/2024"

    def test_surgery_en(self, extractor):
        text = "The patient was operated on 10/06/2023 for tumor resection."
        result = extractor.extract(
            text, feature_subset=["date_chir"], language="en"
        )
        assert "date_chir" in result
        assert result["date_chir"].value == "10/06/2023"

    def test_tumor_board_en(self, extractor):
        text = "Case discussed at tumor board on 12/04/2023."
        result = extractor.extract(
            text, feature_subset=["date_rcp"], language="en"
        )
        assert "date_rcp" in result


class TestDateExtractorConsultExclusion:
    """Tests for consultation date exclusion."""

    def test_auto_detect_consultation_date(self, extractor):
        text = (
            "Consultation du 05/01/2024. "
            "Le patient est décédé le 20/03/2024."
        )
        result = extractor.extract(
            text,
            feature_subset=["date_deces"],
            language="fr",
            consultation_date=None,  # should auto-detect "05/01/2024"
        )
        assert "date_deces" in result
        assert result["date_deces"].value == "20/03/2024"

    def test_explicit_consultation_date(self, extractor):
        text = (
            "Le 05/01/2024, le patient consulte. "
            "Décédé le 20/03/2024."
        )
        result = extractor.extract(
            text,
            feature_subset=["date_deces"],
            language="fr",
            consultation_date="05/01/2024",
        )
        assert "date_deces" in result
        assert result["date_deces"].value == "20/03/2024"


class TestDateExtractorConfidence:
    """Tests for confidence scoring."""

    def test_confidence_range(self, extractor):
        text = "Le patient est décédé le 15/03/2024 à l'hôpital."
        result = extractor.extract(
            text, feature_subset=["date_deces"], language="fr"
        )
        if "date_deces" in result:
            assert 0.0 <= result["date_deces"].confidence <= 1.0

    def test_greedy_no_double_assign(self, extractor):
        """A single date span should only be assigned to one field."""
        text = "Le patient est décédé le 15/03/2024 à domicile."
        result = extractor.extract(
            text,
            feature_subset=["date_deces", "date_chir", "dn_date"],
            language="fr",
        )
        # Only one field should get this date
        assigned_values = [ev.value for ev in result.values()]
        if "15/03/2024" in assigned_values:
            assert assigned_values.count("15/03/2024") == 1
