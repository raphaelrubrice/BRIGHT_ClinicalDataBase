"""Tests for src/extraction/negation.py — EDS-NLP assertion annotation."""

import pytest

from src.extraction.negation import (
    AnnotatedSpan,
    AssertionAnnotator,
    _has_pattern_near_span,
    _NEGATION_PATTERNS,
    _HYPOTHESIS_PATTERNS,
    _HISTORY_PATTERNS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def annotator():
    """Return an annotator using the regex backend (no edsnlp dependency)."""
    return AssertionAnnotator(use_edsnlp=False)


# ---------------------------------------------------------------------------
# AnnotatedSpan dataclass
# ---------------------------------------------------------------------------

class TestAnnotatedSpan:
    """Basic tests for the AnnotatedSpan dataclass."""

    def test_defaults(self):
        span = AnnotatedSpan(text="test", start=0, end=4)
        assert span.is_negated is False
        assert span.is_hypothesis is False
        assert span.is_history is False
        assert span.label == ""

    def test_fields(self):
        span = AnnotatedSpan(
            text="épilepsie",
            start=10,
            end=19,
            label="epilepsie",
            is_negated=True,
        )
        assert span.text == "épilepsie"
        assert span.is_negated is True


# ---------------------------------------------------------------------------
# Negation detection (regex backend)
# ---------------------------------------------------------------------------

class TestNegation:
    """Tests for negation detection with the regex fallback."""

    def test_pas_de_negation(self, annotator):
        text = "On note une pas d'épilepsie chez ce patient."
        idx = text.index("épilepsie")
        spans = [(idx, idx + len("épilepsie"), "epilepsie")]
        results = annotator.annotate(text, spans)
        assert len(results) == 1
        assert results[0].is_negated is True

    def test_absence_de_negation(self, annotator):
        text = "Absence de déficit neurologique."
        idx = text.index("déficit")
        spans = [(idx, idx + len("déficit"), "deficit")]
        results = annotator.annotate(text, spans)
        assert len(results) == 1
        assert results[0].is_negated is True

    def test_sans_negation(self, annotator):
        text = "Sans céphalées ni nausées."
        idx = text.index("céphalées")
        spans = [(idx, idx + len("céphalées"), "cephalees")]
        results = annotator.annotate(text, spans)
        assert len(results) == 1
        assert results[0].is_negated is True

    def test_aucun_negation(self, annotator):
        text = "Aucune épilepsie retrouvée."
        idx = text.index("épilepsie")
        spans = [(idx, idx + len("épilepsie"), "epilepsie")]
        results = annotator.annotate(text, spans)
        assert len(results) == 1
        assert results[0].is_negated is True

    def test_non_negation(self, annotator):
        text = "MGMT : non méthylé"
        idx = text.index("méthylé")
        spans = [(idx, idx + len("méthylé"), "mgmt")]
        results = annotator.annotate(text, spans)
        assert len(results) == 1
        assert results[0].is_negated is True

    def test_positive_no_negation(self, annotator):
        text = "Le patient présente une épilepsie depuis 2021."
        idx = text.index("épilepsie")
        spans = [(idx, idx + len("épilepsie"), "epilepsie")]
        results = annotator.annotate(text, spans)
        assert len(results) == 1
        assert results[0].is_negated is False

    def test_positive_ihc(self, annotator):
        text = "IDH1 : positif"
        idx = text.index("positif")
        spans = [(idx, idx + len("positif"), "ihc_idh1")]
        results = annotator.annotate(text, spans)
        assert len(results) == 1
        assert results[0].is_negated is False

    def test_multiple_spans(self, annotator):
        text = "Pas d'épilepsie. Déficit moteur droit."
        idx1 = text.index("épilepsie")
        idx2 = text.index("Déficit")
        spans = [
            (idx1, idx1 + len("épilepsie"), "epilepsie"),
            (idx2, idx2 + len("Déficit"), "deficit"),
        ]
        results = annotator.annotate(text, spans)
        assert len(results) == 2
        assert results[0].is_negated is True
        assert results[1].is_negated is False


# ---------------------------------------------------------------------------
# Hypothesis detection (regex backend)
# ---------------------------------------------------------------------------

class TestHypothesis:
    """Tests for hypothesis detection."""

    def test_possible_hypothesis(self, annotator):
        text = "Possible glioblastome frontal droit."
        idx = text.index("glioblastome")
        spans = [(idx, idx + len("glioblastome"), "diag")]
        results = annotator.annotate(text, spans)
        assert len(results) == 1
        assert results[0].is_hypothesis is True

    def test_suspicion_hypothesis(self, annotator):
        text = "Suspicion de récidive tumorale."
        idx = text.index("récidive")
        spans = [(idx, idx + len("récidive"), "recidive")]
        results = annotator.annotate(text, spans)
        assert len(results) == 1
        assert results[0].is_hypothesis is True

    def test_a_confirmer(self, annotator):
        text = "Diagnostic à confirmer par IHC."
        idx = text.index("Diagnostic")
        spans = [(idx, idx + len("Diagnostic"), "diag")]
        results = annotator.annotate(text, spans)
        assert len(results) == 1
        assert results[0].is_hypothesis is True

    def test_no_hypothesis(self, annotator):
        text = "Glioblastome confirmé histologiquement."
        idx = text.index("Glioblastome")
        spans = [(idx, idx + len("Glioblastome"), "diag")]
        results = annotator.annotate(text, spans)
        assert len(results) == 1
        assert results[0].is_hypothesis is False


# ---------------------------------------------------------------------------
# History detection (regex backend)
# ---------------------------------------------------------------------------

class TestHistory:
    """Tests for history detection."""

    def test_antecedent(self, annotator):
        text = "Antécédents : hypertension artérielle, diabète."
        idx = text.index("hypertension")
        spans = [(idx, idx + len("hypertension"), "hta")]
        results = annotator.annotate(text, spans)
        assert len(results) == 1
        assert results[0].is_history is True

    def test_en_year(self, annotator):
        text = "Chirurgie en 2018 pour gliome frontal."
        idx = text.index("gliome")
        spans = [(idx, idx + len("gliome"), "diag")]
        results = annotator.annotate(text, spans)
        assert len(results) == 1
        assert results[0].is_history is True

    def test_no_history(self, annotator):
        text = "IRM cérébrale réalisée ce jour."
        idx = text.index("IRM")
        spans = [(idx, idx + len("IRM"), "irm")]
        results = annotator.annotate(text, spans)
        assert len(results) == 1
        assert results[0].is_history is False


# ---------------------------------------------------------------------------
# Convenience: detect_negation()
# ---------------------------------------------------------------------------

class TestDetectNegation:
    """Tests for the convenience ``detect_negation`` method."""

    def test_negated_target(self, annotator):
        assert annotator.detect_negation("Pas d'épilepsie.", "épilepsie") is True

    def test_non_negated_target(self, annotator):
        assert annotator.detect_negation("Épilepsie depuis 2020.", "Épilepsie") is False

    def test_target_not_found(self, annotator):
        assert annotator.detect_negation("Aucun signe.", "épilepsie") is False


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

class TestBackendSelection:
    """Tests for the backend selection logic."""

    def test_regex_backend(self):
        ann = AssertionAnnotator(use_edsnlp=False)
        assert ann.backend == "regex"

    def test_edsnlp_backend_or_fallback(self):
        """If edsnlp is installed, backend is 'edsnlp'; otherwise 'regex'."""
        ann = AssertionAnnotator(use_edsnlp=True)
        assert ann.backend in ("edsnlp", "regex")
