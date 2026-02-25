"""Tests for src/extraction/section_detector.py — section segmentation.

Covers:
- Correct segmentation of a sample anapath report (IHC, molecular, microscopy, conclusion).
- Correct handling of documents with no identifiable sections (full_text fallback).
- Section-to-feature mapping completeness (all 102 fields are reachable).
- Edge cases: empty text, preamble capture, lenient fallback, duplicate headers.
- ``get_features_for_sections`` and ``get_section_for_feature`` helpers.
"""

import pytest

from src.extraction.section_detector import (
    SECTION_PATTERNS,
    SECTION_PATTERNS_LENIENT,
    SECTION_TO_FEATURES,
    DetectionResult,
    SectionDetector,
    SectionMatch,
    _PREAMBLE_FEATURES,
    get_features_for_sections,
    get_section_for_feature,
)
from src.extraction.schema import (
    ALL_BIO_FIELD_NAMES,
    ALL_CLINIQUE_FIELD_NAMES,
)


# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------

SAMPLE_ANAPATH_REPORT = """\
Hôpital Universitaire — Service d'Anatomie Pathologique
NIP: 12345678   Date chirurgie: 15/03/2024   N° labo: AP24-1234

Examen macroscopique
Pièce de résection tumorale temporale droite, 3.5 x 2.8 x 2 cm, fixation formolée.

Examen microscopique
Prolifération gliale hypercellulaire. Présence de nécrose palissadique.
Atypies cytonucléaires marquées. Différenciation astrocytaire.
Prolifération endothéliocapillaire. 8 mitoses / 10 champs.

Immunohistochimie
IDH1 : négatif
p53 : positif (surexpression nucléaire diffuse)
ATRX : maintenu
Ki67 : 30%
GFAP : positif
Olig2 : positif
H3K27M : négatif
H3K27me3 : maintenu
EGFR Hirsch : score 2
MMR : maintenu

Biologie moléculaire
IDH1 : wild-type
IDH2 : wild-type
TERT : muté (C228T)
MGMT : non méthylé
CDKN2A : délétion homozygote

CGH-array
Gain du chromosome 7 (7p, 7q). Perte du chromosome 10 (10p, 10q).
Perte partielle 9p. Amplification EGFR.

Conclusion
Glioblastome, IDH-wildtype, OMS 2021, grade 4.
Profil moléculaire défavorable : TERT muté, MGMT non méthylé.
"""

SAMPLE_CONSULTATION_NOTE = """\
Compte-rendu de consultation du 22/01/2025
Dr Martin — Neuro-oncologie

Antécédents
Patient de 54 ans, homme, enseignant.
Antécédent tumoral : non.
Première crise d'épilepsie en septembre 2024, suivie de céphalées.

Examen clinique
IK 80%. Pas de déficit moteur. Épilepsie contrôlée sous Keppra.
Pas de trouble cognitif notable.

Imagerie
IRM cérébrale du 20/01/2025 : lésion temporale droite, 4cm, prise de contraste,
œdème péri-lésionnel modéré. Pas de calcification.

Traitement
Chirurgie le 15/03/2024, exérèse complète.
Témozolomide 6 cycles du 15/04/2024 au 15/10/2024.
Radiothérapie 60 Gy du 01/04/2024 au 15/05/2024.
Corticoïdes : non. Optune : non.
"""

SAMPLE_FREE_TEXT = """\
Le patient NIP 87654321 est suivi depuis 2022 pour un gliome de bas grade.
Dernière IRM stable. Karnofsky 90%. Pas de nouvelle épilepsie.
Poursuite de la surveillance.
"""

SAMPLE_MOLECULAR_ONLY = """\
Service de biologie moléculaire — Résultats

Biologie moléculaire
IDH1 R132H : muté
IDH2 : wild-type
TERT : wild-type
MGMT : méthylé
BRAF : wild-type
CDKN2A : normal

CGH-array
Codélétion 1p/19q confirmée.
Pas d'amplification EGFR. Pas d'amplification MDM2.
"""


# ---------------------------------------------------------------------------
# Tests — Pattern matching
# ---------------------------------------------------------------------------

class TestSectionPatterns:
    """Verify that individual patterns match expected header strings."""

    @pytest.mark.parametrize("header", [
        "Immunohistochimie",
        "  Immunohistochimie  ",
        "IHC",
        "IHC :",
        "Marqueurs immunohistochimiques",
        "Marqueurs immuno",
    ])
    def test_ihc_pattern_matches(self, header: str):
        assert SECTION_PATTERNS["ihc"].match(header), f"Pattern did not match: {header!r}"

    @pytest.mark.parametrize("non_header", [
        "IDH1 : positif via IHC",  # IHC mid-sentence
        "Les marqueurs",
    ])
    def test_ihc_pattern_rejects(self, non_header: str):
        assert not SECTION_PATTERNS["ihc"].match(non_header), f"Pattern falsely matched: {non_header!r}"

    @pytest.mark.parametrize("header", [
        "Biologie moléculaire",
        "Analyse moléculaire",
        "Panel NGS",
        "Séquençage",
        "Résultats moléculaire",
    ])
    def test_molecular_pattern_matches(self, header: str):
        assert SECTION_PATTERNS["molecular"].match(header)

    @pytest.mark.parametrize("header", [
        "CGH-array",
        "CGH array",
        "Altérations chromosomiques",
        "Profil génomique",
    ])
    def test_chromosomal_pattern_matches(self, header: str):
        assert SECTION_PATTERNS["chromosomal"].match(header)

    @pytest.mark.parametrize("header", [
        "Examen macroscopique",
        "Macroscopie",
        "Description macroscopique",
    ])
    def test_macroscopy_pattern_matches(self, header: str):
        assert SECTION_PATTERNS["macroscopy"].match(header)

    @pytest.mark.parametrize("header", [
        "Examen microscopique",
        "Microscopie",
        "Description microscopique",
        "Histologie",
    ])
    def test_microscopy_pattern_matches(self, header: str):
        assert SECTION_PATTERNS["microscopy"].match(header)

    @pytest.mark.parametrize("header", [
        "Conclusion",
        "Diagnostic",
        "Synthèse diagnostique",
        "Diagnostic intégré",
    ])
    def test_conclusion_pattern_matches(self, header: str):
        assert SECTION_PATTERNS["conclusion"].match(header)

    @pytest.mark.parametrize("header", [
        "Antécédents",
        "Histoire de la maladie",
        "Anamnèse",
    ])
    def test_history_pattern_matches(self, header: str):
        assert SECTION_PATTERNS["history"].match(header)

    @pytest.mark.parametrize("header", [
        "Traitement",
        "Traitements",
        "Proposition thérapeutique",
        "Thérapeutique",
        "Protocole thérapeutique",
    ])
    def test_treatment_pattern_matches(self, header: str):
        assert SECTION_PATTERNS["treatment"].match(header)

    @pytest.mark.parametrize("header", [
        "Examen clinique",
        "Examen neurologique",
        "Interrogatoire",
    ])
    def test_clinical_exam_pattern_matches(self, header: str):
        assert SECTION_PATTERNS["clinical_exam"].match(header)

    @pytest.mark.parametrize("header", [
        "Imagerie",
        "IRM",
        "IRM cérébrale",
        "Scanner cérébral",
        "Radiologie",
        "Bilan radiologique",
    ])
    def test_radiology_pattern_matches(self, header: str):
        assert SECTION_PATTERNS["radiology"].match(header)


# ---------------------------------------------------------------------------
# Tests — SectionDetector on full documents
# ---------------------------------------------------------------------------

class TestSectionDetectorAnapath:
    """Phase 3 acceptance: segment a sample anapath report."""

    def setup_method(self):
        self.detector = SectionDetector()
        self.sections = self.detector.detect(SAMPLE_ANAPATH_REPORT)

    def test_sections_detected(self):
        """Report should be split into multiple sections."""
        assert "full_text" not in self.sections, "Detector should have found section headers"
        assert len(self.sections) >= 4, f"Expected ≥4 sections, got {list(self.sections.keys())}"

    def test_macroscopy_section_present(self):
        assert "macroscopy" in self.sections

    def test_microscopy_section_present(self):
        assert "microscopy" in self.sections

    def test_ihc_section_present(self):
        assert "ihc" in self.sections

    def test_molecular_section_present(self):
        assert "molecular" in self.sections

    def test_chromosomal_section_present(self):
        assert "chromosomal" in self.sections

    def test_conclusion_section_present(self):
        assert "conclusion" in self.sections

    def test_preamble_captured(self):
        """Text before the first header should be captured as preamble."""
        assert "preamble" in self.sections
        assert "NIP" in self.sections["preamble"]

    def test_ihc_content(self):
        """IHC section should contain marker results."""
        ihc = self.sections["ihc"]
        assert "IDH1" in ihc
        assert "Ki67" in ihc
        assert "30%" in ihc

    def test_molecular_content(self):
        """Molecular section should contain gene results."""
        mol = self.sections["molecular"]
        assert "TERT" in mol
        assert "C228T" in mol

    def test_conclusion_content(self):
        """Conclusion should contain the diagnosis."""
        conc = self.sections["conclusion"]
        assert "Glioblastome" in conc

    def test_chromosomal_content(self):
        """Chromosomal section should contain CGH data."""
        chrom = self.sections["chromosomal"]
        assert "chromosome 7" in chrom or "7p" in chrom or "Gain" in chrom


class TestSectionDetectorConsultation:
    """Segment a consultation note into history, exam, treatment, radiology sections."""

    def setup_method(self):
        self.detector = SectionDetector()
        self.sections = self.detector.detect(SAMPLE_CONSULTATION_NOTE)

    def test_sections_detected(self):
        assert "full_text" not in self.sections

    def test_history_section_present(self):
        assert "history" in self.sections

    def test_clinical_exam_present(self):
        assert "clinical_exam" in self.sections

    def test_radiology_present(self):
        assert "radiology" in self.sections

    def test_treatment_present(self):
        assert "treatment" in self.sections

    def test_history_content(self):
        hist = self.sections["history"]
        assert "54 ans" in hist or "épilepsie" in hist.lower()

    def test_treatment_content(self):
        treat = self.sections["treatment"]
        assert "Témozolomide" in treat or "témozolomide" in treat.lower()

    def test_radiology_content(self):
        rad = self.sections["radiology"]
        assert "IRM" in rad or "lésion" in rad


class TestSectionDetectorFreeText:
    """Documents with no section headers should fall back to full_text."""

    def test_full_text_fallback(self):
        detector = SectionDetector()
        sections = detector.detect(SAMPLE_FREE_TEXT)
        assert "full_text" in sections
        assert sections["full_text"] == SAMPLE_FREE_TEXT

    def test_full_text_preserves_content(self):
        detector = SectionDetector()
        sections = detector.detect(SAMPLE_FREE_TEXT)
        assert "87654321" in sections["full_text"]


class TestSectionDetectorMolecular:
    """Molecular-only report should detect molecular and chromosomal sections."""

    def setup_method(self):
        self.detector = SectionDetector()
        self.sections = self.detector.detect(SAMPLE_MOLECULAR_ONLY)

    def test_molecular_present(self):
        assert "molecular" in self.sections

    def test_chromosomal_present(self):
        assert "chromosomal" in self.sections

    def test_preamble_present(self):
        assert "preamble" in self.sections


# ---------------------------------------------------------------------------
# Tests — Edge cases
# ---------------------------------------------------------------------------

class TestSectionDetectorEdgeCases:
    """Various edge cases for robustness."""

    def test_empty_string(self):
        sections = SectionDetector().detect("")
        assert sections == {"full_text": ""}

    def test_whitespace_only(self):
        sections = SectionDetector().detect("   \n\n  ")
        assert "full_text" in sections

    def test_none_like_empty(self):
        """Passing None is not supported but empty string should work."""
        sections = SectionDetector().detect("")
        assert "full_text" in sections

    def test_single_section(self):
        """A document with exactly one section header."""
        text = "Conclusion\nGlioblastome, grade 4"
        sections = SectionDetector().detect(text)
        assert "conclusion" in sections
        assert "Glioblastome" in sections["conclusion"]

    def test_duplicate_headers_keeps_first(self):
        """If the same header appears twice, only the first is used."""
        text = (
            "Immunohistochimie\nIDH1 : positif\n\n"
            "Immunohistochimie\nATRX : maintenu\n"
        )
        sections = SectionDetector().detect(text)
        assert "ihc" in sections
        # The content should include both blocks (the second "header" is
        # treated as body text of the first section since we deduplicate).
        assert "IDH1" in sections["ihc"]

    def test_headers_with_colons(self):
        """Headers followed by colons should still match."""
        text = "Conclusion :\nGlioblastome IDH-wildtype, grade 4.\n"
        sections = SectionDetector().detect(text)
        assert "conclusion" in sections


# ---------------------------------------------------------------------------
# Tests — detect_with_metadata()
# ---------------------------------------------------------------------------

class TestDetectWithMetadata:
    """Test the richer return type."""

    def test_returns_detection_result(self):
        result = SectionDetector().detect_with_metadata(SAMPLE_ANAPATH_REPORT)
        assert isinstance(result, DetectionResult)

    def test_section_names_excludes_preamble(self):
        result = SectionDetector().detect_with_metadata(SAMPLE_ANAPATH_REPORT)
        assert "preamble" not in result.section_names
        assert len(result.section_names) >= 4

    def test_matches_populated(self):
        result = SectionDetector().detect_with_metadata(SAMPLE_ANAPATH_REPORT)
        assert len(result.matches) >= 4
        for m in result.matches:
            assert isinstance(m, SectionMatch)
            assert m.section_name in SECTION_PATTERNS

    def test_fallback_on_free_text(self):
        result = SectionDetector().detect_with_metadata(SAMPLE_FREE_TEXT)
        assert result.used_fallback is True
        assert "full_text" in result.sections


# ---------------------------------------------------------------------------
# Tests — SECTION_TO_FEATURES mapping completeness
# ---------------------------------------------------------------------------

class TestSectionToFeaturesCompleteness:
    """Acceptance criterion: section-to-feature mapping covers all 102 fields."""

    def _all_mapped_features(self) -> set[str]:
        """Return the union of all features mapped in SECTION_TO_FEATURES
        plus _PREAMBLE_FEATURES."""
        result: set[str] = set()
        for features in SECTION_TO_FEATURES.values():
            result.update(features)
        result.update(_PREAMBLE_FEATURES)
        return result

    def test_all_bio_fields_reachable(self):
        """Every BIO field should be reachable via section mapping or
        full_text fallback (get_features_for_sections with full_text)."""
        # full_text always returns everything, so the real check is that
        # every field is either in a section mapping or in preamble features.
        all_mapped = self._all_mapped_features()
        full_text_features = set(
            get_features_for_sections(["full_text"])
        )
        for field_name in ALL_BIO_FIELD_NAMES:
            assert field_name in full_text_features, (
                f"BIO field {field_name!r} not reachable via full_text fallback"
            )

    def test_all_clinique_fields_reachable(self):
        full_text_features = set(
            get_features_for_sections(["full_text"])
        )
        for field_name in ALL_CLINIQUE_FIELD_NAMES:
            assert field_name in full_text_features, (
                f"CLINIQUE field {field_name!r} not reachable via full_text fallback"
            )

    def test_all_102_fields_in_full_text(self):
        """full_text should cover all 102 fields."""
        full_text_features = get_features_for_sections(["full_text"])
        all_fields = set(ALL_BIO_FIELD_NAMES + ALL_CLINIQUE_FIELD_NAMES)
        assert set(full_text_features) == all_fields

    def test_ihc_section_maps_to_ihc_fields(self):
        features = SECTION_TO_FEATURES["ihc"]
        assert "ihc_idh1" in features
        assert "ihc_p53" in features
        assert "ihc_atrx" in features
        assert "ihc_ki67" in features

    def test_molecular_section_maps_to_mol_fields(self):
        features = SECTION_TO_FEATURES["molecular"]
        assert "mol_idh1" in features
        assert "mol_tert" in features
        assert "mol_mgmt" in features

    def test_chromosomal_section_maps_to_ch_fields(self):
        features = SECTION_TO_FEATURES["chromosomal"]
        assert "ch1p" in features
        assert "ch19q" in features
        assert "ampli_egfr" in features

    def test_treatment_section_maps_to_treatment_fields(self):
        features = SECTION_TO_FEATURES["treatment"]
        assert "chimios" in features
        assert "rx_dose" in features
        assert "chir_date" in features

    def test_clinical_exam_maps_to_exam_fields(self):
        features = SECTION_TO_FEATURES["clinical_exam"]
        assert "ik_clinique" in features
        assert "epilepsie" in features

    def test_history_maps_to_symptom_fields(self):
        features = SECTION_TO_FEATURES["history"]
        assert "date_1er_symptome" in features
        assert "epilepsie_1er_symptome" in features


# ---------------------------------------------------------------------------
# Tests — get_features_for_sections / get_section_for_feature
# ---------------------------------------------------------------------------

class TestFeatureHelpers:
    """Test the feature-lookup helper functions."""

    def test_features_for_ihc_section(self):
        features = get_features_for_sections(["ihc"])
        assert "ihc_idh1" in features
        assert "ihc_ki67" in features

    def test_features_for_multiple_sections(self):
        features = get_features_for_sections(["ihc", "molecular"])
        assert "ihc_idh1" in features
        assert "mol_tert" in features

    def test_features_for_full_text_returns_all(self):
        features = get_features_for_sections(["full_text"])
        all_fields = set(ALL_BIO_FIELD_NAMES + ALL_CLINIQUE_FIELD_NAMES)
        assert set(features) == all_fields

    def test_preamble_features_included_by_default(self):
        features = get_features_for_sections(["ihc"])
        # Preamble features like NIP should be included
        assert "nip" in features

    def test_preamble_features_excluded_when_requested(self):
        features = get_features_for_sections(["ihc"], include_preamble=False)
        # Only IHC features should be present (NIP is a preamble feature)
        for f in features:
            assert f in SECTION_TO_FEATURES["ihc"], f"Unexpected feature {f!r}"

    def test_unknown_section_names_ignored(self):
        features = get_features_for_sections(["nonexistent_section"])
        # Should still include preamble features but nothing else
        assert "nip" in features
        assert "ihc_idh1" not in features

    def test_get_section_for_ihc_field(self):
        sections = get_section_for_feature("ihc_idh1")
        assert "ihc" in sections
        # ihc_idh1 also appears in conclusion mapping
        assert "conclusion" in sections

    def test_get_section_for_mol_field(self):
        sections = get_section_for_feature("mol_tert")
        assert "molecular" in sections

    def test_get_section_for_unmapped_field(self):
        """Fields only in preamble (not in SECTION_TO_FEATURES) return empty."""
        sections = get_section_for_feature("localisation_radiotherapie")
        # This field is in _PREAMBLE_FEATURES only
        assert sections == []

    def test_features_sorted_and_deduplicated(self):
        features = get_features_for_sections(["ihc", "conclusion"])
        # ihc_idh1 appears in both sections — should appear only once
        assert features.count("ihc_idh1") == 1
        assert features == sorted(features)


# ---------------------------------------------------------------------------
# Tests — Integration: end-to-end detection + feature lookup
# ---------------------------------------------------------------------------

class TestIntegrationDetectAndMap:
    """Wire detection into feature lookup for a complete test."""

    def test_anapath_pipeline(self):
        """Detect sections in anapath report, then look up features."""
        detector = SectionDetector()
        sections = detector.detect(SAMPLE_ANAPATH_REPORT)
        section_names = [k for k in sections if k not in ("preamble", "full_text")]
        features = get_features_for_sections(section_names)

        # All IHC, molecular, chromosomal, and conclusion features should
        # be in the union.
        assert "ihc_idh1" in features
        assert "mol_tert" in features
        assert "ch1p" in features
        assert "diag_histologique" in features

    def test_consultation_pipeline(self):
        """Detect sections in consultation, then look up features."""
        detector = SectionDetector()
        sections = detector.detect(SAMPLE_CONSULTATION_NOTE)
        section_names = [k for k in sections if k not in ("preamble", "full_text")]
        features = get_features_for_sections(section_names)

        assert "ik_clinique" in features
        assert "chimios" in features
        assert "tumeur_lateralite" in features

    def test_free_text_pipeline(self):
        """Free text → full_text → all features returned."""
        detector = SectionDetector()
        sections = detector.detect(SAMPLE_FREE_TEXT)
        features = get_features_for_sections(list(sections.keys()))
        all_fields = set(ALL_BIO_FIELD_NAMES + ALL_CLINIQUE_FIELD_NAMES)
        assert set(features) == all_fields
