"""Tests for src/extraction/text_normalisation.py."""

import pytest

from src.extraction.text_normalisation import (
    normalise_text,
    normalise,
    expand_abbreviations,
    fuzzy_match,
)


class TestNormaliseText:
    """Document-level NFC normalisation and sanitisation."""

    def test_nfc_accent_normalisation(self):
        """Composed vs decomposed accents should both produce NFC."""
        # e + combining acute = é (NFC)
        decomposed = "e\u0301"
        result = normalise_text(decomposed)
        assert result == "\u00e9"

    def test_nbsp_replaced(self):
        """Non-breaking spaces should become regular spaces."""
        text = "mot1\u00a0mot2\u202fmot3"
        assert "\u00a0" not in normalise_text(text)
        assert "\u202f" not in normalise_text(text)
        assert "mot1 mot2 mot3" == normalise_text(text)

    def test_typographic_quotes(self):
        """Typographic quotes should become ASCII quotes."""
        text = "\u201cbonjour\u201d et \u2018salut\u2019"
        result = normalise_text(text)
        assert "\u201c" not in result
        assert '"bonjour" et \'salut\'' == result

    def test_em_dash_replaced(self):
        """Em/en dashes should become hyphens."""
        text = "value\u2013range\u2014end"
        result = normalise_text(text)
        assert result == "value-range-end"

    def test_control_chars_stripped(self):
        """Zero-width and control characters should be removed."""
        text = "abc\u200bdef\ufeff"
        result = normalise_text(text)
        assert result == "abcdef"

    def test_multi_space_collapsed(self):
        """Multiple spaces should collapse to one."""
        text = "word1   word2     word3"
        assert normalise_text(text) == "word1 word2 word3"

    def test_empty_string(self):
        result = normalise_text("")
        assert result == ""

    def test_utf8_safety(self):
        """Valid French text should pass through cleanly."""
        text = "Résection complète à l'hémisphère droit"
        result = normalise_text(text)
        assert "Résection" in result
        assert "complète" in result


class TestNormalise:
    """Accent-stripping + lowercase for regex matching."""

    def test_accent_stripping(self):
        assert normalise("éàê") == "eae"

    def test_lowercase(self):
        assert normalise("HELLO World") == "hello world"

    def test_whitespace_collapse(self):
        assert normalise("  a   b  ") == "a b"

    def test_combined(self):
        """Full normalisation: accent + case + space."""
        text = "Résection   Complète"
        assert normalise(text) == "resection complete"

    def test_empty(self):
        assert normalise("") == ""


class TestExpandAbbreviations:
    """Abbreviation expansion pre-processing."""

    def test_tmz(self):
        result = expand_abbreviations("TMZ 6 cycles")
        assert "temozolomide 6 cycles" == result

    def test_gtr(self):
        result = expand_abbreviations("GTR réalisé")
        assert "exerese complete" in result

    def test_case_sensitive_ik(self):
        """IK should expand but 'ik' lowercase should not."""
        result = expand_abbreviations("IK à 80%")
        assert "indice de Karnofsky" in result

    def test_no_partial_match(self):
        """TMZ in a longer word should not expand."""
        result = expand_abbreviations("aTMZb")
        assert "temozolomide" not in result


class TestFuzzyMatch:
    """Levenshtein-based fuzzy matching for short tokens (Phase C1)."""

    def test_fuzzy_gene_exact(self):
        """Exact match should return the entry."""
        assert fuzzy_match("IDH1", ["IDH1", "BRAF", "TERT"]) == "IDH1"

    def test_fuzzy_gene_hyphenated(self):
        """IDH-1 should match IDH1 (close Levenshtein distance)."""
        assert fuzzy_match("IDH-1", ["IDH1", "IDH2", "BRAF"]) == "IDH1"

    def test_fuzzy_drug_typo(self):
        """Common drug typo should match canonical name."""
        vocab = ["temozolomide", "bevacizumab", "lomustine"]
        assert fuzzy_match("temozolimide", vocab) == "temozolomide"

    def test_fuzzy_short_token_rejected(self):
        """Tokens shorter than 4 chars should be rejected."""
        assert fuzzy_match("A", ["ABC", "DEF"]) is None
        assert fuzzy_match("AB", ["ABC", "DEF"]) is None

    def test_fuzzy_long_token_rejected(self):
        """Tokens longer than 12 chars should be rejected."""
        long_token = "a" * 13
        assert fuzzy_match(long_token, ["aaaaaaaaaaaaa"]) is None

    def test_fuzzy_low_similarity(self):
        """Very dissimilar token should return None."""
        assert fuzzy_match("WXYZ", ["IDH1", "BRAF", "TERT"]) is None

    def test_fuzzy_empty_inputs(self):
        """Empty token or vocab should return None."""
        assert fuzzy_match("", ["IDH1"]) is None
        assert fuzzy_match("IDH1", []) is None

    def test_fuzzy_case_insensitive(self):
        """Matching should be case-insensitive."""
        assert fuzzy_match("idh1", ["IDH1", "BRAF"]) == "IDH1"
