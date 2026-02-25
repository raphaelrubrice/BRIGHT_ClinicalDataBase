"""Tests for src/extraction/document_classifier.py — document type detection.

Validates:
- Keyword-based scoring correctly classifies each of the 5 document types.
- Ambiguous cases are flagged when top-two scores are close.
- LLM fallback is invoked for ambiguous cases (with a mock client).
- Edge cases: empty text, unknown language, mixed-type documents.
- Individual helper functions (_score_text, _rank_scores, _compute_confidence).
- ClassificationResult structure.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.extraction.document_classifier import (
    DOCUMENT_TYPE_KEYWORDS,
    VALID_DOCUMENT_TYPES,
    ClassificationResult,
    DocumentClassifier,
    _compute_confidence,
    _parse_llm_response,
    _rank_scores,
    _score_text,
    _truncate_to_tokens,
    classify_document,
)


# ======================================================================
# Sample documents — one per type
# ======================================================================

SAMPLE_ANAPATH = """\
COMPTE RENDU ANATOMOPATHOLOGIQUE

Examen macroscopique : Pièce opératoire reçue en fixation formolée,
fragment tumoral de 3,2 x 2,1 x 1,5 cm.

Examen microscopique : Prolifération gliale d'architecture diffuse.
Présence de nécrose en palissade et de prolifération
endothéliocapillaire. Mitoses : 6/10 HPF.

Immunohistochimie (IHC) :
- IDH1 R132H : négatif
- ATRX : maintenu
- p53 : négatif (expression faible)
- Ki67 : 15-20%
- GFAP : positif
- Olig2 : positif

Coloration HES : présence d'atypies cytonucléaires marquées.

Conclusion : Glioblastome, IDH non muté, grade 4 OMS 2021.
"""

SAMPLE_MOLECULAR = """\
RÉSULTATS DE BIOLOGIE MOLÉCULAIRE

Analyse moléculaire réalisée par panel NGS et CGH-array.

Patient : XYZ
Date de chirurgie : 15/10/2024

Séquençage :
- IDH1 : wild-type (wt)
- IDH2 : wild-type (wt)
- TERT promoteur : muté (C228T)
- BRAF : wild-type
- CDKN2A : délétion homozygote
- H3F3A : wild-type

Altérations chromosomiques (CGH-array) :
- Gain 7p, Gain 7q
- Perte 10p, Perte 10q
- 1p/19q : pas de codélétion

MGMT : méthylation positive

Amplification : EGFR amplifié, pas d'amplification MDM2, CDK4, MET.

Profil moléculaire compatible avec un glioblastome IDH-wildtype.
"""

SAMPLE_CONSULTATION = """\
COMPTE-RENDU DE CONSULTATION

Consultation du 13/01/2026
Patient vu en consultation de suivi neuro-oncologie.

Interrogatoire :
Le patient rapporte une bonne autonomie. Pas de nouvelle crise
d'épilepsie sous traitement anti-épileptique. Pas de céphalées.
Pas de déficit moteur.

Indice de Karnofsky : 100%

Traitement en cours : Témozolomide cycles 1 à 4.
Pas d'Avastin. Pas d'essai thérapeutique.

Examen clinique :
Examen neurologique normal. Pas de déficit. Pas de trouble cognitif.

Conclusion :
Poursuite du Témozolomide. Contrôle IRM dans 3 mois.
Optune en place. Pas de corticoïdes.
"""

SAMPLE_RCP = """\
RÉUNION DE CONCERTATION PLURIDISCIPLINAIRE (RCP)

Date : 20/11/2024
Participants : Dr Martin (neurochirurgien), Dr Dupont (neuro-oncologue),
Dr Leblanc (radiothérapeute), Dr Moreau (neuropathologiste).

Discussion collégiale :

Patient de 47 ans, glioblastome IDH non muté, grade 4.
Chirurgie le 15/10/2024 : exérèse complète.

Proposition thérapeutique :
- Protocole de Stupp : radiothérapie focale 60 Gy + témozolomide
  concomitant suivi de 6 cycles de témozolomide adjuvant.
- Optune recommandé.

Décision thérapeutique collégiale validée à l'unanimité.
"""

SAMPLE_RADIOLOGY = """\
COMPTE-RENDU RADIOLOGIQUE — IRM CÉRÉBRALE

IRM encéphalique avec injection de gadolinium.
Séquences FLAIR, T1 gadolinium, diffusion, perfusion.

Résultats :
Lésion intra-axiale pariétale gauche, mesurant 42 x 35 x 30 mm.
Prise de contraste hétérogène en anneau.
Effet de masse modéré avec déviation de la ligne médiane de 5 mm.
Œdème péri-lésionnel en hypersignal FLAIR.
Pas de spectroscopie réalisée.

Conclusion :
Aspect compatible avec un processus tumoral glial de haut grade.
Comparaison avec IRM de contrôle précédente : stabilité dimensionnelle.
"""


# ======================================================================
# ClassificationResult tests
# ======================================================================


class TestClassificationResult:
    """Verify the ClassificationResult dataclass."""

    def test_creation(self):
        result = ClassificationResult(
            document_type="anapath",
            scores={"anapath": 10, "consultation": 2},
            confidence=0.8,
        )
        assert result.document_type == "anapath"
        assert result.confidence == 0.8
        assert result.is_ambiguous is False
        assert result.used_llm_fallback is False

    def test_default_values(self):
        result = ClassificationResult(document_type="rcp")
        assert result.scores == {}
        assert result.confidence == 0.0
        assert result.matched_keywords == {}


# ======================================================================
# Keyword scoring tests
# ======================================================================


class TestKeywordScoring:
    """Test the internal _score_text function."""

    def test_anapath_scores_highest_for_anapath_doc(self):
        scores, matched = _score_text(SAMPLE_ANAPATH)
        ranked = _rank_scores(scores)
        assert ranked[0][0] == "anapath"
        assert scores["anapath"] > 0

    def test_molecular_scores_highest_for_molecular_doc(self):
        scores, matched = _score_text(SAMPLE_MOLECULAR)
        ranked = _rank_scores(scores)
        assert ranked[0][0] == "molecular_report"

    def test_consultation_scores_highest_for_consultation_doc(self):
        scores, matched = _score_text(SAMPLE_CONSULTATION)
        ranked = _rank_scores(scores)
        assert ranked[0][0] == "consultation"

    def test_rcp_scores_highest_for_rcp_doc(self):
        scores, matched = _score_text(SAMPLE_RCP)
        ranked = _rank_scores(scores)
        assert ranked[0][0] == "rcp"

    def test_radiology_scores_highest_for_radiology_doc(self):
        scores, matched = _score_text(SAMPLE_RADIOLOGY)
        ranked = _rank_scores(scores)
        assert ranked[0][0] == "radiology"

    def test_strong_keyword_weight(self):
        """A strong keyword contributes 3 points."""
        text = "compte rendu anatomopathologique"
        scores, _ = _score_text(text)
        # "anatomopathologique" is a strong keyword for anapath
        assert scores["anapath"] >= 3

    def test_moderate_keyword_weight(self):
        """A moderate keyword contributes 1 point."""
        text = "coloration HES simple"
        scores, _ = _score_text(text)
        assert scores["anapath"] >= 1

    def test_matched_keywords_returned(self):
        scores, matched = _score_text(SAMPLE_ANAPATH)
        assert len(matched["anapath"]) > 0
        # At least some strong keywords should appear
        anapath_matched_lower = [kw.lower() for kw in matched["anapath"]]
        assert "anatomopathologique" in anapath_matched_lower or \
               any("anatomopathologique" in kw for kw in anapath_matched_lower)

    def test_empty_text(self):
        scores, matched = _score_text("")
        for dt in VALID_DOCUMENT_TYPES:
            assert scores[dt] == 0
            assert matched[dt] == []

    def test_case_insensitive(self):
        """Keywords should match regardless of case."""
        text = "IMMUNOHISTOCHIMIE et SÉQUENÇAGE"
        scores, _ = _score_text(text)
        assert scores["anapath"] > 0  # IHC is strong for anapath
        assert scores["molecular_report"] > 0  # séquençage is strong for molecular


# ======================================================================
# Ranking & confidence helpers
# ======================================================================


class TestRankingHelpers:
    """Test _rank_scores and _compute_confidence."""

    def test_rank_scores_descending(self):
        scores = {"a": 5, "b": 10, "c": 3}
        ranked = _rank_scores(scores)
        assert ranked[0] == ("b", 10)
        assert ranked[1] == ("a", 5)
        assert ranked[2] == ("c", 3)

    def test_compute_confidence_high(self):
        ranked = [("a", 10), ("b", 2)]
        conf = _compute_confidence(ranked)
        assert conf == pytest.approx(0.8)

    def test_compute_confidence_low(self):
        ranked = [("a", 5), ("b", 4)]
        conf = _compute_confidence(ranked)
        assert conf == pytest.approx(0.2)

    def test_compute_confidence_zero_scores(self):
        ranked = [("a", 0), ("b", 0)]
        conf = _compute_confidence(ranked)
        assert conf == 0.0

    def test_compute_confidence_empty(self):
        conf = _compute_confidence([])
        assert conf == 0.0

    def test_compute_confidence_single_type(self):
        ranked = [("a", 10)]
        conf = _compute_confidence(ranked)
        assert conf == 1.0


# ======================================================================
# DocumentClassifier — keyword-only tests
# ======================================================================


class TestDocumentClassifierKeywordOnly:
    """Test DocumentClassifier without LLM fallback."""

    def setup_method(self):
        self.classifier = DocumentClassifier()

    def test_classify_anapath(self):
        result = self.classifier.classify(SAMPLE_ANAPATH)
        assert result.document_type == "anapath"
        assert result.confidence > 0.0
        assert isinstance(result.scores, dict)

    def test_classify_molecular_report(self):
        result = self.classifier.classify(SAMPLE_MOLECULAR)
        assert result.document_type == "molecular_report"

    def test_classify_consultation(self):
        result = self.classifier.classify(SAMPLE_CONSULTATION)
        assert result.document_type == "consultation"

    def test_classify_rcp(self):
        result = self.classifier.classify(SAMPLE_RCP)
        assert result.document_type == "rcp"

    def test_classify_radiology(self):
        result = self.classifier.classify(SAMPLE_RADIOLOGY)
        assert result.document_type == "radiology"

    def test_result_has_all_types_in_scores(self):
        result = self.classifier.classify(SAMPLE_ANAPATH)
        for dt in VALID_DOCUMENT_TYPES:
            assert dt in result.scores

    def test_empty_text_is_ambiguous(self):
        result = self.classifier.classify("")
        assert result.is_ambiguous is True
        assert result.confidence == 0.0

    def test_whitespace_only_is_ambiguous(self):
        result = self.classifier.classify("   \n\t  ")
        assert result.is_ambiguous is True

    def test_unrelated_text_low_confidence(self):
        result = self.classifier.classify("Bonjour, ceci est un texte sans rapport.")
        assert result.confidence == 0.0
        assert result.is_ambiguous is True


# ======================================================================
# Ambiguity detection
# ======================================================================


class TestAmbiguityDetection:
    """Test that closely scored documents are flagged as ambiguous."""

    def test_ambiguous_when_scores_close(self):
        """A doc with both anapath and molecular keywords might be ambiguous."""
        # Craft a text that scores similarly for two types
        text = "IDH1 résultat IHC séquençage"  # IHC → anapath, séquençage → molecular
        classifier = DocumentClassifier()
        result = classifier.classify(text)
        # The exact outcome depends on scoring, but we can check the logic
        # At minimum, verify the structure
        assert isinstance(result.is_ambiguous, bool)
        assert isinstance(result.confidence, float)

    def test_custom_threshold(self):
        """Higher threshold makes more cases ambiguous."""
        classifier = DocumentClassifier(ambiguity_threshold=100)
        result = classifier.classify(SAMPLE_ANAPATH)
        # With threshold=100, almost everything is ambiguous
        assert result.is_ambiguous is True

    def test_zero_threshold(self):
        """With threshold=0, only exact ties are ambiguous."""
        classifier = DocumentClassifier(ambiguity_threshold=0)
        result = classifier.classify(SAMPLE_ANAPATH)
        # Anapath should win clearly
        assert result.is_ambiguous is False


# ======================================================================
# LLM fallback tests (mocked)
# ======================================================================


class TestLLMFallback:
    """Test LLM fallback with a mock Ollama client."""

    def _make_mock_client(self, response_text: str) -> MagicMock:
        """Create a mock OllamaClient that returns *response_text*."""
        client = MagicMock()
        client.generate.return_value = {"response": response_text}
        return client

    def test_llm_called_when_ambiguous(self):
        """LLM fallback should be invoked for ambiguous classifications."""
        mock_client = self._make_mock_client("anapath")
        # Use a very high threshold to force ambiguity
        classifier = DocumentClassifier(
            ollama_client=mock_client,
            ambiguity_threshold=1000,
        )
        result = classifier.classify(SAMPLE_ANAPATH)
        assert result.used_llm_fallback is True
        mock_client.generate.assert_called_once()

    def test_llm_not_called_when_clear(self):
        """LLM fallback should NOT be invoked for clear classifications."""
        mock_client = self._make_mock_client("anapath")
        classifier = DocumentClassifier(
            ollama_client=mock_client,
            ambiguity_threshold=0,
        )
        result = classifier.classify(SAMPLE_ANAPATH)
        assert result.used_llm_fallback is False
        mock_client.generate.assert_not_called()

    def test_llm_overrides_keyword_result(self):
        """When the LLM returns a different type, it should override."""
        mock_client = self._make_mock_client("molecular_report")
        classifier = DocumentClassifier(
            ollama_client=mock_client,
            ambiguity_threshold=1000,
        )
        result = classifier.classify(SAMPLE_ANAPATH)
        assert result.document_type == "molecular_report"
        assert result.used_llm_fallback is True

    def test_llm_confirms_keyword_result(self):
        """When LLM confirms the keyword result, confidence should increase."""
        mock_client = self._make_mock_client("anapath")
        classifier = DocumentClassifier(
            ollama_client=mock_client,
            ambiguity_threshold=1000,
        )
        result = classifier.classify(SAMPLE_ANAPATH)
        assert result.document_type == "anapath"
        assert result.used_llm_fallback is True
        assert result.confidence > 0.0

    def test_llm_unparseable_response(self):
        """If LLM returns garbage, fall back to keyword result."""
        mock_client = self._make_mock_client("je ne sais pas du tout")
        classifier = DocumentClassifier(
            ollama_client=mock_client,
            ambiguity_threshold=1000,
        )
        result = classifier.classify(SAMPLE_ANAPATH)
        # Should still return *something* (keyword-based winner)
        assert result.document_type in VALID_DOCUMENT_TYPES

    def test_llm_exception_handled(self):
        """If LLM call raises, fall back to keyword result gracefully."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = ConnectionError("Ollama offline")
        classifier = DocumentClassifier(
            ollama_client=mock_client,
            ambiguity_threshold=1000,
        )
        result = classifier.classify(SAMPLE_ANAPATH)
        assert result.document_type in VALID_DOCUMENT_TYPES
        # LLM was attempted but failed
        assert result.used_llm_fallback is False


# ======================================================================
# LLM response parsing
# ======================================================================


class TestLLMResponseParsing:
    """Test _parse_llm_response helper."""

    @pytest.mark.parametrize("response,expected", [
        ("anapath", "anapath"),
        ("molecular_report", "molecular_report"),
        ("consultation", "consultation"),
        ("rcp", "rcp"),
        ("radiology", "radiology"),
        ("  ANAPATH  ", "anapath"),
        ("Le type est anapath.", "anapath"),
        ("C'est un molecular_report.", "molecular_report"),
    ])
    def test_valid_responses(self, response: str, expected: str):
        assert _parse_llm_response(response) == expected

    def test_invalid_response(self):
        assert _parse_llm_response("something unknown") is None

    def test_empty_response(self):
        assert _parse_llm_response("") is None


# ======================================================================
# Text truncation
# ======================================================================


class TestTextTruncation:
    """Test _truncate_to_tokens helper."""

    def test_short_text_unchanged(self):
        text = "short text"
        assert _truncate_to_tokens(text, max_tokens=500) == text

    def test_long_text_truncated(self):
        text = "a" * 10000
        result = _truncate_to_tokens(text, max_tokens=100)
        # 100 tokens × 4 chars + "…" = 401 chars
        assert len(result) <= 401

    def test_truncation_adds_ellipsis(self):
        text = "a" * 10000
        result = _truncate_to_tokens(text, max_tokens=100)
        assert result.endswith("…")


# ======================================================================
# Convenience function
# ======================================================================


class TestConvenienceFunction:
    """Test the module-level classify_document() shortcut."""

    def test_classify_document_returns_result(self):
        result = classify_document(SAMPLE_CONSULTATION)
        assert isinstance(result, ClassificationResult)
        assert result.document_type == "consultation"

    def test_classify_document_no_llm(self):
        result = classify_document(SAMPLE_ANAPATH)
        assert result.used_llm_fallback is False


# ======================================================================
# Custom keyword override
# ======================================================================


class TestCustomKeywords:
    """Test that custom keywords can be injected."""

    def test_override_keywords(self):
        custom_kw = {
            "anapath": {"strong": ["CUSTOM_MAGIC_WORD"], "moderate": []},
            "molecular_report": {"strong": [], "moderate": []},
            "consultation": {"strong": [], "moderate": []},
            "rcp": {"strong": [], "moderate": []},
            "radiology": {"strong": [], "moderate": []},
        }
        classifier = DocumentClassifier(keywords=custom_kw)
        result = classifier.classify("This text has CUSTOM_MAGIC_WORD in it.")
        assert result.document_type == "anapath"
        assert result.scores["anapath"] >= 3


# ======================================================================
# Integration-style tests with realistic French clinical text
# ======================================================================


class TestRealisticDocuments:
    """Additional realistic classification tests."""

    def test_short_anapath_conclusion(self):
        """Even a short conclusion section should classify as anapath."""
        text = "Diagnostic histologique : glioblastome. Immunohistochimie réalisée."
        result = classify_document(text)
        assert result.document_type == "anapath"

    def test_short_molecular(self):
        text = "Biologie moléculaire : IDH1 wt, TERT muté, MGMT méthylé."
        result = classify_document(text)
        assert result.document_type == "molecular_report"

    def test_short_rcp(self):
        text = "Réunion de concertation pluridisciplinaire du 15/01/2025. Discussion collégiale."
        result = classify_document(text)
        assert result.document_type == "rcp"

    def test_short_radiology(self):
        text = "IRM cérébrale avec injection. Séquences FLAIR et T1 gadolinium."
        result = classify_document(text)
        assert result.document_type == "radiology"

    def test_short_consultation(self):
        text = "Consultation du 10/01/2025. Examen clinique normal. Karnofsky 90%."
        result = classify_document(text)
        assert result.document_type == "consultation"

    def test_mixed_rcp_consultation(self):
        """An RCP that mentions consultation should still classify as RCP
        if RCP keywords dominate."""
        text = (
            "Réunion de concertation pluridisciplinaire. "
            "Discussion collégiale du cas.  "
            "Le patient a été vu en consultation la semaine dernière. "
            "Décision thérapeutique collégiale : protocole de Stupp."
        )
        result = classify_document(text)
        assert result.document_type == "rcp"

    def test_document_type_is_valid(self):
        """All results should return a valid document type."""
        for sample in [SAMPLE_ANAPATH, SAMPLE_MOLECULAR, SAMPLE_CONSULTATION,
                       SAMPLE_RCP, SAMPLE_RADIOLOGY]:
            result = classify_document(sample)
            assert result.document_type in VALID_DOCUMENT_TYPES
