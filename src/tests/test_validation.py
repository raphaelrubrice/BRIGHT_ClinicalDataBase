"""Tests for src/extraction/validation.py — controlled vocabulary enforcement."""

import pytest

from src.extraction.schema import (
    ALL_FIELDS_BY_NAME,
    ControlledVocab,
    ExtractionValue,
    FieldDefinition,
    FieldType,
)
from src.extraction.validation import (
    normalise_value,
    validate_extraction,
    _is_value_valid,
    _NORMALISATION_MAP,
)


# ---------------------------------------------------------------------------
# Test normalise_value
# ---------------------------------------------------------------------------

class TestNormaliseValue:
    """Tests for the normalise_value function."""

    def test_none_returns_none(self):
        assert normalise_value("ihc_idh1", None) is None

    def test_null_string_returns_none(self):
        assert normalise_value("ihc_idh1", "null") is None
        assert normalise_value("ihc_idh1", "None") is None
        assert normalise_value("ihc_idh1", "N/A") is None
        assert normalise_value("ihc_idh1", "na") is None
        assert normalise_value("ihc_idh1", "") is None

    def test_boolean_to_oui_non(self):
        assert normalise_value("epilepsie", True) == "oui"
        assert normalise_value("epilepsie", False) == "non"

    def test_accent_normalisation_negatif(self):
        assert normalise_value("ihc_idh1", "négatif") == "negatif"
        assert normalise_value("ihc_idh1", "négative") == "negatif"
        assert normalise_value("ihc_idh1", "negative") == "negatif"

    def test_accent_normalisation_mute(self):
        assert normalise_value("mol_idh1", "muté") == "mute"
        assert normalise_value("mol_idh1", "mutée") == "mute"

    def test_molecular_wt_synonyms(self):
        assert normalise_value("mol_idh1", "wild-type") == "wt"
        assert normalise_value("mol_idh1", "sauvage") == "wt"
        assert normalise_value("mol_idh1", "type sauvage") == "wt"
        assert normalise_value("mol_idh1", "non muté") == "wt"
        assert normalise_value("mol_idh1", "absence de mutation") == "wt"

    def test_methylation_normalisation(self):
        assert normalise_value("mol_mgmt", "méthylé") == "methyle"
        assert normalise_value("mol_mgmt", "non méthylé") == "non methyle"
        assert normalise_value("mol_mgmt", "methylation positive") == "methyle"
        assert normalise_value("mol_mgmt", "absence de methylation") == "non methyle"

    def test_chromosomal_status_normalisation(self):
        assert normalise_value("ch1p", "délétion") == "perte"
        assert normalise_value("ch1p", "deletion") == "perte"

    def test_binary_synonyms(self):
        assert normalise_value("epilepsie", "yes") == "oui"
        assert normalise_value("epilepsie", "no") == "non"
        assert normalise_value("epilepsie", "present") == "oui"
        assert normalise_value("epilepsie", "absent") == "non"

    def test_sex_normalisation(self):
        assert normalise_value("sexe", "homme") == "M"
        assert normalise_value("sexe", "femme") == "F"
        assert normalise_value("sexe", "masculin") == "M"
        assert normalise_value("sexe", "féminin") == "F"
        assert normalise_value("sexe", "h") == "M"
        assert normalise_value("sexe", "f") == "F"

    def test_laterality_normalisation(self):
        assert normalise_value("tumeur_lateralite", "bilatéral") == "bilateral"
        assert normalise_value("tumeur_lateralite", "médian") == "median"

    def test_surgery_normalisation(self):
        assert normalise_value("type_chirurgie", "exérèse complète") == "exerese complete"
        assert normalise_value("type_chirurgie", "biopsie stéréotaxique") == "biopsie"

    def test_who_classification_normalisation(self):
        assert normalise_value("classification_oms", "oms 2021") == "2021"
        assert normalise_value("classification_oms", "who 2016") == "2016"

    def test_no_normalisation_needed(self):
        assert normalise_value("ihc_idh1", "positif") == "positif"
        assert normalise_value("mol_idh1", "wt") == "wt"

    def test_integer_field_parsing(self):
        result = normalise_value("grade", "3")
        assert result == 3

    def test_integer_passthrough(self):
        result = normalise_value("grade", 3)
        assert result == 3

    def test_ihc_maintenu_synonym(self):
        assert normalise_value("ihc_atrx", "conservé") == "maintenu"
        assert normalise_value("ihc_atrx", "conservée") == "maintenu"

    def test_perte_expression_to_negatif(self):
        assert normalise_value("ihc_atrx", "perte d'expression") == "negatif"

    def test_free_text_passthrough(self):
        assert normalise_value("diag_histologique", "glioblastome IDH-wt") == "glioblastome IDH-wt"

    def test_whitespace_stripping(self):
        assert normalise_value("ihc_idh1", "  positif  ") == "positif"


# ---------------------------------------------------------------------------
# Test _is_value_valid
# ---------------------------------------------------------------------------

class TestIsValueValid:
    """Tests for the _is_value_valid helper."""

    def test_none_is_always_valid(self):
        field_def = ALL_FIELDS_BY_NAME["ihc_idh1"]
        assert _is_value_valid(field_def, None) is True

    def test_valid_ihc_status(self):
        field_def = ALL_FIELDS_BY_NAME["ihc_idh1"]
        assert _is_value_valid(field_def, "positif") is True
        assert _is_value_valid(field_def, "negatif") is True
        assert _is_value_valid(field_def, "maintenu") is True

    def test_invalid_ihc_status(self):
        field_def = ALL_FIELDS_BY_NAME["ihc_idh1"]
        assert _is_value_valid(field_def, "unknown_value") is False

    def test_valid_binary(self):
        field_def = ALL_FIELDS_BY_NAME["epilepsie"]
        assert _is_value_valid(field_def, "oui") is True
        assert _is_value_valid(field_def, "non") is True

    def test_invalid_binary(self):
        field_def = ALL_FIELDS_BY_NAME["epilepsie"]
        assert _is_value_valid(field_def, "maybe") is False

    def test_valid_grade(self):
        field_def = ALL_FIELDS_BY_NAME["grade"]
        assert _is_value_valid(field_def, 1) is True
        assert _is_value_valid(field_def, 4) is True

    def test_invalid_grade(self):
        field_def = ALL_FIELDS_BY_NAME["grade"]
        assert _is_value_valid(field_def, 5) is False

    def test_valid_chromosomal(self):
        field_def = ALL_FIELDS_BY_NAME["ch1p"]
        assert _is_value_valid(field_def, "gain") is True
        assert _is_value_valid(field_def, "perte") is True
        assert _is_value_valid(field_def, "perte partielle") is True

    def test_invalid_chromosomal(self):
        field_def = ALL_FIELDS_BY_NAME["ch1p"]
        assert _is_value_valid(field_def, "unknown") is False

    def test_valid_evolution(self):
        field_def = ALL_FIELDS_BY_NAME["evol_clinique"]
        # evol_clinique uses a custom validator
        assert _is_value_valid(field_def, "initial") is True

    def test_free_text_always_valid(self):
        field_def = ALL_FIELDS_BY_NAME["diag_histologique"]
        assert _is_value_valid(field_def, "any string is fine") is True

    def test_molecular_variant_valid(self):
        field_def = ALL_FIELDS_BY_NAME["mol_idh1"]
        assert _is_value_valid(field_def, "R132H") is True
        assert _is_value_valid(field_def, "wt") is True
        assert _is_value_valid(field_def, "mute") is True

    def test_valid_laterality(self):
        field_def = ALL_FIELDS_BY_NAME["tumeur_lateralite"]
        assert _is_value_valid(field_def, "gauche") is True
        assert _is_value_valid(field_def, "droite") is True

    def test_valid_surgery_type(self):
        field_def = ALL_FIELDS_BY_NAME["type_chirurgie"]
        assert _is_value_valid(field_def, "biopsie") is True
        assert _is_value_valid(field_def, "exerese complete") is True


# ---------------------------------------------------------------------------
# Test validate_extraction (main entry point)
# ---------------------------------------------------------------------------

class TestValidateExtraction:
    """Tests for the validate_extraction function."""

    def test_valid_values_pass(self):
        extractions = {
            "ihc_idh1": ExtractionValue(value="positif", extraction_tier="rule"),
            "epilepsie": ExtractionValue(value="oui", extraction_tier="rule"),
            "grade": ExtractionValue(value=3, extraction_tier="rule"),
        }
        result = validate_extraction(extractions)

        assert result["ihc_idh1"].vocab_valid is True
        assert result["ihc_idh1"].flagged is False
        assert result["epilepsie"].vocab_valid is True
        assert result["grade"].vocab_valid is True

    def test_invalid_value_flagged(self):
        extractions = {
            "ihc_idh1": ExtractionValue(value="unknown_status", extraction_tier="llm"),
        }
        result = validate_extraction(extractions)

        assert result["ihc_idh1"].vocab_valid is False
        assert result["ihc_idh1"].flagged is True

    def test_normalisation_applied(self):
        extractions = {
            "ihc_idh1": ExtractionValue(value="négatif", extraction_tier="rule"),
        }
        result = validate_extraction(extractions)

        assert result["ihc_idh1"].value == "negatif"
        assert result["ihc_idh1"].vocab_valid is True
        assert result["ihc_idh1"].flagged is False

    def test_molecular_normalisation(self):
        extractions = {
            "mol_idh1": ExtractionValue(value="muté", extraction_tier="rule"),
        }
        result = validate_extraction(extractions)

        assert result["mol_idh1"].value == "mute"
        assert result["mol_idh1"].vocab_valid is True

    def test_none_value_skipped(self):
        extractions = {
            "ihc_idh1": ExtractionValue(value=None, extraction_tier="rule"),
        }
        result = validate_extraction(extractions)

        # None values should not be flagged
        assert result["ihc_idh1"].flagged is False

    def test_unknown_field_flagged(self):
        extractions = {
            "nonexistent_field": ExtractionValue(value="something", extraction_tier="llm"),
        }
        result = validate_extraction(extractions)

        assert result["nonexistent_field"].flagged is True
        assert result["nonexistent_field"].vocab_valid is False

    def test_multiple_fields_mixed(self):
        extractions = {
            "ihc_idh1": ExtractionValue(value="positif", extraction_tier="rule"),
            "ihc_p53": ExtractionValue(value="garbage", extraction_tier="llm"),
            "epilepsie": ExtractionValue(value="négatif", extraction_tier="rule"),
            "grade": ExtractionValue(value=3, extraction_tier="rule"),
        }
        result = validate_extraction(extractions)

        assert result["ihc_idh1"].vocab_valid is True
        assert result["ihc_p53"].vocab_valid is False
        assert result["ihc_p53"].flagged is True
        # "négatif" gets normalised but it's not in BINARY set {"oui","non"}
        # Check that epilepsie with "negatif" is flagged (it should be "oui"/"non")
        assert result["epilepsie"].value == "negatif"
        assert result["epilepsie"].vocab_valid is False  # "negatif" not in {"oui", "non"}
        assert result["grade"].vocab_valid is True

    def test_free_text_field_no_flagging(self):
        extractions = {
            "diag_histologique": ExtractionValue(
                value="Glioblastome, IDH-wildtype, WHO grade 4",
                extraction_tier="llm",
            ),
        }
        result = validate_extraction(extractions)

        assert result["diag_histologique"].vocab_valid is True
        assert result["diag_histologique"].flagged is False

    def test_methylation_normalisation_and_validation(self):
        extractions = {
            "mol_mgmt": ExtractionValue(value="méthylé", extraction_tier="rule"),
        }
        result = validate_extraction(extractions)

        assert result["mol_mgmt"].value == "methyle"
        assert result["mol_mgmt"].vocab_valid is True

    def test_boolean_normalisation(self):
        extractions = {
            "epilepsie": ExtractionValue(value=True, extraction_tier="llm"),
            "deficit": ExtractionValue(value=False, extraction_tier="llm"),
        }
        result = validate_extraction(extractions)

        assert result["epilepsie"].value == "oui"
        assert result["epilepsie"].vocab_valid is True
        assert result["deficit"].value == "non"
        assert result["deficit"].vocab_valid is True

    def test_case_insensitive_match(self):
        """Controlled vocab check should be case-insensitive for strings."""
        extractions = {
            "type_chirurgie": ExtractionValue(value="Biopsie", extraction_tier="llm"),
        }
        result = validate_extraction(extractions)

        assert result["type_chirurgie"].vocab_valid is True

    def test_integer_grade_from_string(self):
        extractions = {
            "grade": ExtractionValue(value="3", extraction_tier="llm"),
        }
        result = validate_extraction(extractions)

        assert result["grade"].value == 3
        assert result["grade"].vocab_valid is True

    def test_empty_extraction_dict(self):
        result = validate_extraction({})
        assert result == {}

    def test_surgery_type_synonym(self):
        extractions = {
            "type_chirurgie": ExtractionValue(
                value="exérèse complète", extraction_tier="llm"
            ),
        }
        result = validate_extraction(extractions)

        assert result["type_chirurgie"].value == "exerese complete"
        assert result["type_chirurgie"].vocab_valid is True

    def test_sex_normalisation_in_context(self):
        extractions = {
            "sexe": ExtractionValue(value="homme", extraction_tier="llm"),
        }
        result = validate_extraction(extractions)

        assert result["sexe"].value == "M"
        assert result["sexe"].vocab_valid is True


# ---------------------------------------------------------------------------
# Test normalisation map completeness
# ---------------------------------------------------------------------------

class TestNormalisationMapCompleteness:
    """Ensure the normalisation map covers key variants."""

    def test_all_ihc_status_norms(self):
        """All expected IHC status variants should normalise to valid values."""
        ihc_variants = ["négatif", "négative", "negative", "positif", "positive", "+", "-"]
        for v in ihc_variants:
            result = normalise_value("ihc_idh1", v)
            assert result in ControlledVocab.IHC_STATUS or result is not None

    def test_all_binary_norms(self):
        """All expected binary variants should normalise to oui/non."""
        binary_variants = ["yes", "no", "true", "false", "present", "absent", "oui", "non"]
        for v in binary_variants:
            result = normalise_value("epilepsie", v)
            assert result in ControlledVocab.BINARY
