"""Tests for src/extraction/llm_extraction.py — Tier 2 LLM extraction.

Tests use ``unittest.mock`` to mock the OllamaClient, avoiding the
need for a running Ollama server. The tests verify:
- LLM extraction skips already-extracted fields
- Correct grouping of remaining fields
- Prompt selection and section text selection
- Response parsing into ExtractionValue objects
- Source span validation (exact, fuzzy, and missing)
- Handling of malformed LLM responses
- Value normalisation
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.llm_extraction import (
    _determine_groups_for_features,
    _normalise_llm_value,
    _normalise_whitespace,
    _parse_llm_response,
    _select_section_text,
    run_llm_extraction,
    validate_source_spans,
)
from src.extraction.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponse,
)
from src.extraction.schema import ExtractionValue


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client() -> MagicMock:
    """Return a mocked OllamaClient."""
    client = MagicMock(spec=OllamaClient)
    client.model = "qwen3:4b"
    return client


def _make_ollama_response(
    parsed_json: dict | None = None,
    content: str = "",
) -> OllamaResponse:
    """Create an OllamaResponse with the given JSON."""
    if parsed_json and not content:
        content = json.dumps(parsed_json)
    return OllamaResponse(
        content=content,
        parsed_json=parsed_json,
        model="qwen3:4b",
    )


def _make_extraction_value(value: str, tier: str = "rule") -> ExtractionValue:
    """Create a simple ExtractionValue."""
    return ExtractionValue(
        value=value,
        source_span=f"some span for {value}",
        extraction_tier=tier,
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# _determine_groups_for_features
# ---------------------------------------------------------------------------

class TestDetermineGroups:
    """Test grouping of remaining fields by feature group."""

    def test_ihc_fields(self):
        remaining = {"ihc_idh1", "ihc_p53", "ihc_atrx"}
        groups = _determine_groups_for_features(remaining)
        assert "ihc" in groups
        assert set(groups["ihc"]) == remaining

    def test_molecular_fields(self):
        remaining = {"mol_idh1", "mol_tert"}
        groups = _determine_groups_for_features(remaining)
        assert "molecular" in groups
        assert set(groups["molecular"]) == remaining

    def test_mixed_fields(self):
        remaining = {"ihc_idh1", "mol_idh1", "ch1p"}
        groups = _determine_groups_for_features(remaining)
        assert "ihc" in groups
        assert "molecular" in groups
        assert "chromosomal" in groups

    def test_empty_remaining(self):
        groups = _determine_groups_for_features(set())
        assert groups == {}

    def test_unknown_fields_not_grouped(self):
        remaining = {"completely_unknown_field_xyz"}
        groups = _determine_groups_for_features(remaining)
        assert groups == {}


# ---------------------------------------------------------------------------
# _select_section_text
# ---------------------------------------------------------------------------

class TestSelectSectionText:
    """Test section text selection for feature groups."""

    def test_direct_section_match(self):
        sections = {
            "ihc": "IDH1 : positif, ATRX : maintenu",
            "conclusion": "Glioblastome grade 4",
        }
        result = _select_section_text(sections, "ihc", "full text")
        assert "IDH1" in result

    def test_fallback_to_full_text_section(self):
        sections = {"full_text": "tout le document"}
        result = _select_section_text(sections, "ihc", "backup")
        assert result == "tout le document"

    def test_fallback_to_full_text_argument(self):
        sections = {}
        result = _select_section_text(sections, "ihc", "le texte complet")
        assert result == "le texte complet"

    def test_empty_section_skipped(self):
        sections = {"ihc": "   ", "full_text": "fallback text"}
        result = _select_section_text(sections, "ihc", "backup")
        assert result == "fallback text"


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------

class TestParseLLMResponse:
    """Test LLM response parsing."""

    def test_standard_response(self):
        response = _make_ollama_response(parsed_json={
            "values": {
                "ihc_idh1": "positif",
                "ihc_p53": "negatif",
                "ihc_atrx": None,
            },
            "_source": {
                "ihc_idh1": "IDH1 : positif",
                "ihc_p53": "p53 négatif",
            },
        })
        result = _parse_llm_response(
            response, "ihc", ["ihc_idh1", "ihc_p53", "ihc_atrx"]
        )
        assert "ihc_idh1" in result
        assert result["ihc_idh1"].value == "positif"
        assert result["ihc_idh1"].source_span == "IDH1 : positif"
        assert result["ihc_idh1"].extraction_tier == "llm"
        assert "ihc_p53" in result
        assert result["ihc_p53"].value == "negatif"
        # ihc_atrx was None → should be skipped
        assert "ihc_atrx" not in result

    def test_null_json_response(self):
        response = _make_ollama_response(
            parsed_json=None,
            content="Invalid JSON from model",
        )
        result = _parse_llm_response(response, "ihc", ["ihc_idh1"])
        assert result == {}

    def test_flat_response_without_values_key(self):
        """Some models might return flat JSON without 'values' nesting."""
        response = _make_ollama_response(parsed_json={
            "ihc_idh1": "positif",
            "ihc_p53": "negatif",
        })
        result = _parse_llm_response(
            response, "ihc", ["ihc_idh1", "ihc_p53"]
        )
        assert "ihc_idh1" in result
        assert result["ihc_idh1"].value == "positif"

    def test_only_target_fields_extracted(self):
        response = _make_ollama_response(parsed_json={
            "values": {
                "ihc_idh1": "positif",
                "ihc_p53": "negatif",
                "extra_field": "should be ignored",
            },
            "_source": {},
        })
        result = _parse_llm_response(response, "ihc", ["ihc_idh1"])
        assert "ihc_idh1" in result
        assert "ihc_p53" not in result
        assert "extra_field" not in result


# ---------------------------------------------------------------------------
# _normalise_llm_value
# ---------------------------------------------------------------------------

class TestNormaliseLLMValue:
    """Test value normalisation for LLM outputs."""

    def test_none(self):
        assert _normalise_llm_value("ihc_idh1", None) is None

    def test_null_string(self):
        assert _normalise_llm_value("ihc_idh1", "null") is None
        assert _normalise_llm_value("ihc_idh1", "None") is None
        assert _normalise_llm_value("ihc_idh1", "N/A") is None

    def test_empty_string(self):
        assert _normalise_llm_value("ihc_idh1", "") is None

    def test_boolean_true(self):
        assert _normalise_llm_value("epilepsie", True) == "oui"

    def test_boolean_false(self):
        assert _normalise_llm_value("epilepsie", False) == "non"

    def test_integer(self):
        result = _normalise_llm_value("grade", 3)
        assert result == 3

    def test_french_accent_normalisation(self):
        assert _normalise_llm_value("ihc_idh1", "négatif") == "negatif"
        assert _normalise_llm_value("mol_idh1", "muté") == "mute"
        assert _normalise_llm_value("mol_mgmt", "méthylé") == "methyle"
        assert _normalise_llm_value("mol_mgmt", "non méthylé") == "non methyle"

    def test_plain_value_passthrough(self):
        assert _normalise_llm_value("ihc_idh1", "positif") == "positif"
        assert _normalise_llm_value("mol_idh1", "wt") == "wt"
        assert _normalise_llm_value("diag_histologique", "glioblastome") == "glioblastome"

    def test_whitespace_stripped(self):
        assert _normalise_llm_value("ihc_idh1", "  positif  ") == "positif"


# ---------------------------------------------------------------------------
# run_llm_extraction
# ---------------------------------------------------------------------------

class TestRunLLMExtraction:
    """Test the main run_llm_extraction function."""

    def test_skips_when_all_extracted(self, mock_client):
        already = {
            "ihc_idh1": _make_extraction_value("positif"),
            "ihc_p53": _make_extraction_value("negatif"),
        }
        result = run_llm_extraction(
            text="some text",
            sections={"full_text": "some text"},
            feature_subset=["ihc_idh1", "ihc_p53"],
            already_extracted=already,
            client=mock_client,
        )
        assert result == {}
        mock_client.generate.assert_not_called()

    def test_only_extracts_remaining_fields(self, mock_client):
        already = {"ihc_idh1": _make_extraction_value("positif")}

        mock_client.generate.return_value = _make_ollama_response(
            parsed_json={
                "values": {"ihc_p53": "negatif", "ihc_atrx": "maintenu"},
                "_source": {"ihc_p53": "p53 négatif"},
            }
        )

        result = run_llm_extraction(
            text="IDH1 positif. p53 négatif. ATRX maintenu.",
            sections={"ihc": "IDH1 positif. p53 négatif. ATRX maintenu."},
            feature_subset=["ihc_idh1", "ihc_p53", "ihc_atrx"],
            already_extracted=already,
            client=mock_client,
        )

        # ihc_idh1 was already extracted — should not be in results
        assert "ihc_idh1" not in result
        # ihc_p53 and ihc_atrx should be extracted
        assert "ihc_p53" in result
        assert result["ihc_p53"].value == "negatif"

    def test_handles_ollama_error_gracefully(self, mock_client):
        mock_client.generate.side_effect = OllamaConnectionError("down")

        result = run_llm_extraction(
            text="some text",
            sections={"full_text": "some text"},
            feature_subset=["ihc_idh1"],
            already_extracted={},
            client=mock_client,
        )

        # Should return empty dict on error, not raise
        assert result == {}

    def test_handles_malformed_json(self, mock_client):
        mock_client.generate.return_value = _make_ollama_response(
            parsed_json=None,
            content="not json at all",
        )

        result = run_llm_extraction(
            text="some text",
            sections={"full_text": "some text"},
            feature_subset=["ihc_idh1"],
            already_extracted={},
            client=mock_client,
        )

        assert result == {}

    def test_multiple_groups(self, mock_client):
        """Test extraction across multiple feature groups."""
        call_count = 0

        def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_ollama_response(parsed_json={
                    "values": {"ihc_idh1": "positif"},
                    "_source": {"ihc_idh1": "IDH1 positif"},
                })
            else:
                return _make_ollama_response(parsed_json={
                    "values": {"mol_tert": "mute"},
                    "_source": {"mol_tert": "TERT muté"},
                })

        mock_client.generate.side_effect = mock_generate

        result = run_llm_extraction(
            text="IDH1 positif. TERT muté.",
            sections={"full_text": "IDH1 positif. TERT muté."},
            feature_subset=["ihc_idh1", "mol_tert"],
            already_extracted={},
            client=mock_client,
        )

        assert "ihc_idh1" in result
        assert "mol_tert" in result


# ---------------------------------------------------------------------------
# validate_source_spans
# ---------------------------------------------------------------------------

class TestValidateSourceSpans:
    """Test source span validation."""

    def test_exact_match_not_flagged(self):
        extractions = {
            "ihc_idh1": ExtractionValue(
                value="positif",
                source_span="IDH1 : positif",
                extraction_tier="llm",
            ),
        }
        original = "Immunohistochimie : IDH1 : positif, p53 : négatif"
        result = validate_source_spans(extractions, original)
        assert result["ihc_idh1"].flagged is False

    def test_whitespace_normalised_match(self):
        extractions = {
            "ihc_idh1": ExtractionValue(
                value="positif",
                source_span="IDH1  :  positif",  # Extra spaces
                extraction_tier="llm",
            ),
        }
        original = "IDH1 : positif"
        result = validate_source_spans(extractions, original)
        assert result["ihc_idh1"].flagged is False

    def test_case_insensitive_match(self):
        extractions = {
            "ihc_idh1": ExtractionValue(
                value="positif",
                source_span="idh1 : POSITIF",
                extraction_tier="llm",
            ),
        }
        original = "IDH1 : positif"
        result = validate_source_spans(extractions, original)
        assert result["ihc_idh1"].flagged is False

    def test_fabricated_span_flagged(self):
        extractions = {
            "ihc_idh1": ExtractionValue(
                value="positif",
                source_span="IDH1 montre un résultat positif confirmé",
                extraction_tier="llm",
            ),
        }
        original = "Pas d'IDH1 dans ce texte"
        result = validate_source_spans(extractions, original)
        assert result["ihc_idh1"].flagged is True

    def test_missing_span_llm_flagged(self):
        extractions = {
            "ihc_idh1": ExtractionValue(
                value="positif",
                source_span=None,
                extraction_tier="llm",
            ),
        }
        original = "some text"
        result = validate_source_spans(extractions, original)
        assert result["ihc_idh1"].flagged is True

    def test_missing_span_rule_not_flagged(self):
        extractions = {
            "ihc_idh1": ExtractionValue(
                value="positif",
                source_span=None,
                extraction_tier="rule",
            ),
        }
        original = "some text"
        result = validate_source_spans(extractions, original)
        assert result["ihc_idh1"].flagged is False

    def test_fuzzy_match_above_threshold(self):
        extractions = {
            "ihc_idh1": ExtractionValue(
                value="positif",
                source_span="IDH1 immunohistochimie résultat positif",
                extraction_tier="llm",
            ),
        }
        # Most words appear but in slightly different context
        original = "L'IDH1 en immunohistochimie montre un résultat positif pour le patient"
        result = validate_source_spans(extractions, original, fuzzy_threshold=0.8)
        assert result["ihc_idh1"].flagged is False

    def test_empty_span_skipped(self):
        extractions = {
            "ihc_idh1": ExtractionValue(
                value="positif",
                source_span="   ",
                extraction_tier="llm",
            ),
        }
        original = "some text"
        result = validate_source_spans(extractions, original)
        assert result["ihc_idh1"].flagged is True

    def test_multiple_extractions(self):
        extractions = {
            "ihc_idh1": ExtractionValue(
                value="positif",
                source_span="IDH1 : positif",
                extraction_tier="llm",
            ),
            "ihc_p53": ExtractionValue(
                value="negatif",
                source_span="COMPLETELY FABRICATED SPAN",
                extraction_tier="llm",
            ),
        }
        original = "Résultats IHC : IDH1 : positif, p53 négatif"
        result = validate_source_spans(extractions, original)
        assert result["ihc_idh1"].flagged is False
        assert result["ihc_p53"].flagged is True


# ---------------------------------------------------------------------------
# _normalise_whitespace
# ---------------------------------------------------------------------------

class TestNormaliseWhitespace:
    """Test whitespace normalisation helper."""

    def test_collapse_spaces(self):
        assert _normalise_whitespace("  a   b  c  ") == "a b c"

    def test_collapse_newlines(self):
        assert _normalise_whitespace("a\n\nb\tc") == "a b c"

    def test_empty_string(self):
        assert _normalise_whitespace("") == ""

    def test_already_normalised(self):
        assert _normalise_whitespace("hello world") == "hello world"
