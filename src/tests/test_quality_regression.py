"""Quality regression tests for the fixes introduced in Implementation Plan 2.

Four test classes, each covering one root-cause fix:
  - TestDateContextAssignment  (Phase 0.1)
  - TestPseudoTokenRejection   (Phase 0.5)
  - TestConsultationBioRouting  (Phase 1.1)
  - TestPseudoBirthdate         (Phase 0.3)
"""

import re
import pytest

from src.extraction.rule_extraction import (
    _assign_dates_by_context,
    _PAT_PSEUDO_BIRTHDATE,
    extract_dates,
    run_rule_extraction,
)
from src.extraction.schema import (
    ExtractionValue,
    get_extractable_fields,
)
from src.extraction.validation import _PSEUDO_TOKEN_RE, _is_reasonable_date


# ═══════════════════════════════════════════════════════════════════════════
# Date context assignment  (Phase 0.1)
# ═══════════════════════════════════════════════════════════════════════════

class TestDateContextAssignment:
    """Verify that dates are assigned to the correct field based on keywords."""

    def test_birth_date_keyword(self):
        """'né(e) le' context → annee_de_naissance."""
        text = "Monsieur X, né le 15/06/1977, 47 ans"
        dates = extract_dates(text)
        assigned = _assign_dates_by_context(dates, ["annee_de_naissance", "chir_date"], text)
        assert "annee_de_naissance" in assigned
        assert assigned["annee_de_naissance"].value == "1977"
        assert "chir_date" not in assigned  # Should NOT contaminate

    def test_surgery_date_keyword(self):
        """'opéré le' context → chir_date."""
        text = "Patient opéré le 20/03/2024 au bloc central"
        dates = extract_dates(text)
        assigned = _assign_dates_by_context(dates, ["chir_date", "annee_de_naissance"], text)
        assert "chir_date" in assigned
        assert assigned["chir_date"].value == "20/03/2024"
        assert "annee_de_naissance" not in assigned

    def test_multi_date_no_cross_contamination(self):
        """Multiple dates with different contexts get separate field assignments."""
        text = (
            "Patient née le 12/04/1980. "
            "Chirurgie réalisée (exérèse) le 01/02/2024. "
            "Début radiothérapie le 15/03/2024."
        )
        dates = extract_dates(text)
        fields = ["annee_de_naissance", "chir_date", "rx_date_debut"]
        assigned = _assign_dates_by_context(dates, fields, text)

        assert assigned.get("annee_de_naissance", ExtractionValue()).value == "1980"
        assert assigned.get("chir_date", ExtractionValue()).value == "01/02/2024"
        assert assigned.get("rx_date_debut", ExtractionValue()).value == "15/03/2024"

    def test_unmatched_dates_are_left_for_llm(self):
        """Dates without keyword context are NOT assigned (left for Tier 2)."""
        text = "Consultation du 15/11/2024"
        dates = extract_dates(text)
        assigned = _assign_dates_by_context(dates, ["chir_date", "rx_date_debut"], text)
        # "consultation du" doesn't match surgery or radiotherapy keywords
        assert len(assigned) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Pseudo-token rejection  (Phase 0.5)
# ═══════════════════════════════════════════════════════════════════════════

class TestPseudoTokenRejection:
    """Verify the pseudo-token regex correctly identifies pseudonymised values."""

    def test_rejects_nom_token(self):
        assert _PSEUDO_TOKEN_RE.search("[NOM_A3F1B2]") is not None

    def test_rejects_hopital_token(self):
        assert _PSEUDO_TOKEN_RE.search("[HOPITAL_1234AB]") is not None

    def test_rejects_prenom_token_no_brackets(self):
        assert _PSEUDO_TOKEN_RE.search("PRENOM_deadbeef") is not None

    def test_preserves_normal_value(self):
        """Normal clinical values should NOT be rejected."""
        assert _PSEUDO_TOKEN_RE.search("Mathon") is None
        assert _PSEUDO_TOKEN_RE.search("positif") is None
        assert _PSEUDO_TOKEN_RE.search("12/06/2024") is None

    def test_reasonable_date_validation(self):
        """Dates outside 1900–current+1 should be rejected."""
        assert _is_reasonable_date("12/06/2024") is True
        assert _is_reasonable_date("01/01/1850") is False
        assert _is_reasonable_date("15/03/2030") is False


# ═══════════════════════════════════════════════════════════════════════════
# Consultation bio routing  (Phase 1.1)
# ═══════════════════════════════════════════════════════════════════════════

class TestConsultationBioRouting:
    """Verify consultation documents can now extract bio fields."""

    def test_consultation_includes_diagnosis(self):
        fields = get_extractable_fields("consultation")
        for f in ["diag_histologique", "diag_integre", "classification_oms", "grade"]:
            assert f in fields, f"Expected '{f}' in consultation extractable fields"

    def test_consultation_includes_ihc(self):
        fields = get_extractable_fields("consultation")
        for f in ["ihc_idh1", "ihc_p53", "ihc_atrx", "ihc_olig2", "ihc_gfap", "ihc_ki67"]:
            assert f in fields, f"Expected '{f}' in consultation extractable fields"

    def test_consultation_includes_molecular(self):
        fields = get_extractable_fields("consultation")
        for f in ["mol_idh1", "mol_idh2", "mol_tert", "mol_mgmt", "mol_CDKN2A"]:
            assert f in fields, f"Expected '{f}' in consultation extractable fields"

    def test_radiology_excludes_bio(self):
        """Radiology documents should NOT include bio-only fields."""
        fields = get_extractable_fields("radiology")
        assert "fusion_fgfr" not in fields
        assert "mol_idh1" not in fields


# ═══════════════════════════════════════════════════════════════════════════
# Pseudonymized birthdate  (Phase 0.3)
# ═══════════════════════════════════════════════════════════════════════════

class TestPseudoBirthdate:
    """Verify detection and handling of pseudonymised birthdates (YYYY-??-??)."""

    def test_regex_matches_ne_le_pattern(self):
        m = _PAT_PSEUDO_BIRTHDATE.search("Patient né le 1977-??-??")
        assert m is not None
        assert m.group(1) == "1977"

    def test_regex_matches_nee_le_pattern(self):
        m = _PAT_PSEUDO_BIRTHDATE.search("Patiente née le 1985-??-??")
        assert m is not None
        assert m.group(1) == "1985"

    def test_regex_matches_date_de_naissance(self):
        m = _PAT_PSEUDO_BIRTHDATE.search("date de naissance : 1990-??-??")
        assert m is not None
        assert m.group(1) == "1990"
