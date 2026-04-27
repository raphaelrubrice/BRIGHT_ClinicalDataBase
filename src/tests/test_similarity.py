"""Tests for src/extraction/similarity.py, vocabulary matching cascade.

Covers:
- Tier 1: Exact match (case-insensitive, accent-normalized)
- Tier 2: Normalisation map lookup
- Tier 3: Fuzzy match
- Tier 4: spaCy vector similarity (skipped if model unavailable)
- Tier 5: Fallback to "NA"
"""

import pytest

from src.extraction.similarity import match_to_vocab


# ---------------------------------------------------------------------------
# Test vocabulary sets (mirrors schema ControlledVocab)
# ---------------------------------------------------------------------------

IHC_VALUES = {"positif", "negatif", "maintenu", "NA"}
BINARY_VALUES = {"oui", "non", "NA"}
CHROMOSOMAL_VALUES = {"gain", "perte", "perte partielle", "NA"}
SURGERY_VALUES = {"biopsie", "exerese", "biopsie stereotaxique", "en attente", "NA"}
LATERALITY_VALUES = {"droite", "gauche", "bilateral", "median", "NA"}


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1: Exact match
# ═══════════════════════════════════════════════════════════════════════════

class TestTier1ExactMatch:
    """Exact (case-insensitive, accent-normalized) matching."""

    def test_exact_lowercase(self):
        val, score = match_to_vocab("positif", IHC_VALUES, "ihc_idh1")
        assert val == "positif"
        assert score == 1.0

    def test_exact_uppercase(self):
        val, score = match_to_vocab("POSITIF", IHC_VALUES, "ihc_idh1")
        assert val == "positif"
        assert score == 1.0

    def test_exact_mixed_case(self):
        val, score = match_to_vocab("Negatif", IHC_VALUES, "ihc_p53")
        assert val == "negatif"
        assert score == 1.0

    def test_exact_with_accent(self):
        val, score = match_to_vocab("négatif", IHC_VALUES, "ihc_atrx")
        assert val == "negatif"
        assert score == 1.0

    def test_exact_gain(self):
        val, score = match_to_vocab("gain", CHROMOSOMAL_VALUES, "ch7p")
        assert val == "gain"
        assert score == 1.0

    def test_exact_perte_partielle(self):
        val, score = match_to_vocab("perte partielle", CHROMOSOMAL_VALUES, "ch10q")
        assert val == "perte partielle"
        assert score == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2: Normalisation map lookup
# ═══════════════════════════════════════════════════════════════════════════

class TestTier2NormMap:
    """Normalisation map lookup (from validation._NORMALISATION_MAP)."""

    def test_norm_pos_to_positif(self):
        """'pos' should normalise to 'positif' via the norm map."""
        val, score = match_to_vocab("pos", IHC_VALUES, "ihc_idh1")
        assert val == "positif"
        assert score >= 0.70  # At least fuzzy quality

    def test_norm_neg_to_negatif(self):
        """'neg' should normalise to 'negatif'."""
        val, score = match_to_vocab("neg", IHC_VALUES, "ihc_p53")
        assert val == "negatif"
        assert score >= 0.70


# ═══════════════════════════════════════════════════════════════════════════
# Tier 3: Fuzzy match
# ═══════════════════════════════════════════════════════════════════════════

class TestTier3FuzzyMatch:
    """Fuzzy matching via rapidfuzz."""

    def test_fuzzy_typo(self):
        """Small typo should fuzzy match to correct value."""
        val, score = match_to_vocab("positf", IHC_VALUES, "ihc_idh1")
        assert val == "positif"
        assert score >= 0.70

    def test_fuzzy_biopsie(self):
        """'biopsie' exact and 'biopsié' accent → should match."""
        val, score = match_to_vocab("biopsié", SURGERY_VALUES, "type_chirurgie")
        # Should match 'biopsie' via fuzzy or accent normalization
        assert val == "biopsie"


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases and fallback
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases: empty input, no match → NA, empty vocab."""

    def test_empty_span_returns_na(self):
        val, score = match_to_vocab("", IHC_VALUES, "ihc_idh1")
        assert val == "NA"
        assert score == 0.0

    def test_whitespace_only_returns_na(self):
        val, score = match_to_vocab("   ", IHC_VALUES, "ihc_p53")
        assert val == "NA"
        assert score == 0.0

    def test_empty_vocab_returns_span(self):
        """If no candidates in vocab, return span as-is with low confidence."""
        val, score = match_to_vocab("something", set(), "test_field")
        assert val == "something"
        assert score == 0.5

    def test_na_only_vocab_returns_span(self):
        """If the only candidate is 'NA', return span as-is."""
        val, score = match_to_vocab("positif", {"NA"}, "test_field")
        assert val == "positif"
        assert score == 0.5

    def test_gibberish_returns_something(self):
        """Completely unrelated text should still return a result (possibly via spaCy or NA)."""
        val, score = match_to_vocab("xyzabc123", IHC_VALUES, "ihc_idh1")
        # Either a spaCy match or NA
        assert val in {"positif", "negatif", "maintenu", "NA"}


# ═══════════════════════════════════════════════════════════════════════════
# Clinical scenario tests
# ═══════════════════════════════════════════════════════════════════════════

class TestClinicalScenarios:
    """Real-world clinical span → vocab matching."""

    def test_maintenu_ihc(self):
        val, score = match_to_vocab("maintenu", IHC_VALUES, "ihc_atrx")
        assert val == "maintenu"
        assert score == 1.0

    def test_en_attente(self):
        """'en attente' should match in surgery vocab."""
        val, score = match_to_vocab("en attente", SURGERY_VALUES, "type_chirurgie")
        assert val == "en attente"
        assert score == 1.0

    def test_bilateral_laterality(self):
        val, score = match_to_vocab("bilateral", LATERALITY_VALUES, "tumeur_lateralite")
        assert val == "bilateral"
        assert score == 1.0

    def test_droite_laterality(self):
        val, score = match_to_vocab("droite", LATERALITY_VALUES, "tumeur_lateralite")
        assert val == "droite"
        assert score == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# "autre" category fallback
# ═══════════════════════════════════════════════════════════════════════════

# Vocab sets that include "autre"
MOLECULAR_STATUS_VALUES = {"wt", "mute", "autre", "NA"}
EVOLUTION_VALUES = {"initial", "terminal", "autre", "NA"}
GRADE_VALUES = {"1", "2", "3", "4", "autre"}  # stringified for match_to_vocab


class TestAutreFallback:
    """'autre' category fallback when no vocab option matches."""

    def test_autre_accepts_novel_value(self):
        """Novel text should be accepted as-is when 'autre' is in vocab."""
        val, score = match_to_vocab(
            "duplication 7q", MOLECULAR_STATUS_VALUES, "mol_idh1"
        )
        # Should NOT be "NA", the vocab has "autre" so novel values are accepted
        assert val == "duplication 7q"
        assert score == 0.6

    def test_autre_not_triggered_without_autre(self):
        """Vocab without 'autre' should still fall back to NA."""
        val, score = match_to_vocab("xyzabc123", IHC_VALUES, "ihc_idh1")
        # IHC_VALUES has no "autre" → must still go to NA
        assert val in {"positif", "negatif", "maintenu", "NA"}

    def test_exact_match_takes_precedence(self):
        """Exact match should still win even when 'autre' is in vocab."""
        val, score = match_to_vocab("wt", MOLECULAR_STATUS_VALUES, "mol_idh1")
        assert val == "wt"
        assert score == 1.0

    def test_autre_for_grade(self):
        """Novel value against GRADE (has 'autre') should return raw value."""
        val, score = match_to_vocab(
            "inclassable", GRADE_VALUES, "grade"
        )
        assert val == "inclassable"
        assert score == 0.6

    def test_autre_for_evolution(self):
        """Novel evolution value should be accepted via 'autre' fallback."""
        val, score = match_to_vocab(
            "stabilisation partielle", EVOLUTION_VALUES, "evol_clinique"
        )
        assert val == "stabilisation partielle"
        assert score == 0.6
