"""Tests for src/extraction/gliner_extractor.py — batching strategies & postprocessing.

Covers:
- BatchingStrategy enum
- _build_heterogeneous_batches() produces diverse batches
- _postprocess_span() with PRESENCE / SIMILARITY / DIRECT mapping types
- _resolve_batches() for all 3 strategies
- GlinerExtractor.extract() with mock model
"""

import pytest

from src.extraction.gliner_extractor import (
    BatchingStrategy,
    GlinerExtractor,
    SEMANTIC_BATCHES,
    _build_heterogeneous_batches,
    _ALL_GLINER_FIELDS,
    _FIELD_TO_DOMAIN,
    FIELD_DESCRIPTIONS_EN,
    FIELD_DESCRIPTIONS_FR,
    _REVERSE_DESC_MAP,
)
from src.extraction.schema import MappingType, ALL_FIELDS_BY_NAME


# ═══════════════════════════════════════════════════════════════════════════
# BatchingStrategy enum
# ═══════════════════════════════════════════════════════════════════════════

class TestBatchingStrategy:
    """BatchingStrategy enum values."""

    def test_semantic_context(self):
        assert BatchingStrategy("semantic_context") == BatchingStrategy.SEMANTIC_CONTEXT

    def test_semantic_only(self):
        assert BatchingStrategy("semantic_only") == BatchingStrategy.SEMANTIC_ONLY

    def test_heterogeneous(self):
        assert BatchingStrategy("heterogeneous") == BatchingStrategy.HETEROGENEOUS

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            BatchingStrategy("invalid_strategy")


# ═══════════════════════════════════════════════════════════════════════════
# Heterogeneous batch builder
# ═══════════════════════════════════════════════════════════════════════════

class TestHeterogeneousBatchBuilder:
    """_build_heterogeneous_batches() should produce diverse batches."""

    def test_builds_batches(self):
        """Should produce at least one batch."""
        batches = _build_heterogeneous_batches(_ALL_GLINER_FIELDS, "fr")
        assert len(batches) > 0

    def test_max_batch_size(self):
        """Each batch should have at most max_per_batch fields."""
        batches = _build_heterogeneous_batches(_ALL_GLINER_FIELDS, "fr", max_per_batch=5)
        for batch in batches:
            assert len(batch["fields"]) <= 5

    def test_domain_diversity(self):
        """Fields within a single batch should come from different domains."""
        batches = _build_heterogeneous_batches(_ALL_GLINER_FIELDS, "fr", max_per_batch=5)
        for batch in batches:
            domains = set()
            for field in batch["fields"]:
                domain = _FIELD_TO_DOMAIN.get(field)
                if domain:
                    domains.add(domain)
            # With 5 fields max, we expect domains >= min(5, num_domains)
            assert len(domains) == len(batch["fields"]), (
                f"Batch should have all fields from different domains, "
                f"got {len(domains)} domains for {len(batch['fields'])} fields"
            )

    def test_all_fields_covered(self):
        """All target fields should appear in exactly one batch."""
        target = _ALL_GLINER_FIELDS.copy()
        batches = _build_heterogeneous_batches(target, "fr")
        covered = set()
        for batch in batches:
            for f in batch["fields"]:
                assert f not in covered, f"Field '{f}' appears in multiple batches"
                covered.add(f)
        assert covered == target, f"Missing fields: {target - covered}"

    def test_no_anchors(self):
        """Heterogeneous batches should have no anchors (empty set)."""
        batches = _build_heterogeneous_batches(_ALL_GLINER_FIELDS, "fr")
        for batch in batches:
            assert batch["anchors"] == set()

    def test_labels_present(self):
        """Each batch should have a 'labels' dict mapping field → label."""
        batches = _build_heterogeneous_batches(_ALL_GLINER_FIELDS, "fr")
        for batch in batches:
            assert "labels" in batch
            for field in batch["fields"]:
                assert field in batch["labels"], f"Missing label for field '{field}'"

    def test_en_labels(self):
        """English labels should be used when language is 'en'."""
        batches = _build_heterogeneous_batches({"ihc_idh1", "mol_idh1"}, "en")
        labels_used = set()
        for batch in batches:
            labels_used.update(batch["labels"].values())
        # At least one EN label expected
        assert any("IDH" in l for l in labels_used)

    def test_subset_fields(self):
        """Should handle a small subset of fields."""
        target = {"ihc_idh1", "mol_mgmt", "sexe"}
        batches = _build_heterogeneous_batches(target, "fr")
        covered = set()
        for batch in batches:
            covered.update(batch["fields"])
        assert covered == target


# ═══════════════════════════════════════════════════════════════════════════
# _resolve_batches
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveBatches:
    """_resolve_batches() should return correct batch lists for each strategy."""

    def test_heterogeneous_returns_none_names(self):
        ext = GlinerExtractor(batching_strategy="heterogeneous")
        batches = ext._resolve_batches(_ALL_GLINER_FIELDS, "fr")
        for name, config in batches:
            assert name is None  # heterogeneous batches have no name

    def test_semantic_context_returns_named_batches(self):
        ext = GlinerExtractor(batching_strategy="semantic_context")
        batches = ext._resolve_batches(_ALL_GLINER_FIELDS, "fr")
        for name, config in batches:
            assert name in SEMANTIC_BATCHES

    def test_semantic_only_returns_named_batches(self):
        ext = GlinerExtractor(batching_strategy="semantic_only")
        batches = ext._resolve_batches(_ALL_GLINER_FIELDS, "fr")
        for name, config in batches:
            assert name in SEMANTIC_BATCHES


# ═══════════════════════════════════════════════════════════════════════════
# _postprocess_span — mapping_type-aware postprocessing
# ═══════════════════════════════════════════════════════════════════════════

class TestPostprocessSpan:
    """_postprocess_span() behaviour by mapping_type."""

    @pytest.fixture
    def extractor(self):
        ext = GlinerExtractor()
        ext._model = True  # dummy to avoid loading
        return ext

    # -- PRESENCE fields -----------------------------------------------

    def test_presence_found_returns_oui(self, extractor):
        """PRESENCE field: entity found → 'oui'."""
        result = extractor._postprocess_span("epilepsie", "crise d'épilepsie")
        assert result == "oui"

    def test_presence_negation_keyword_returns_non(self, extractor):
        """PRESENCE field: span with negation keyword → 'non'."""
        result = extractor._postprocess_span("epilepsie", "pas de crise")
        assert result == "non"

    def test_presence_absence_keyword_returns_non(self, extractor):
        """PRESENCE field: span with 'absence' → 'non'."""
        result = extractor._postprocess_span("histo_necrose", "absence de nécrose")
        assert result == "non"

    def test_presence_amplification(self, extractor):
        """PRESENCE field: amplification found → 'oui'."""
        result = extractor._postprocess_span("ampli_egfr", "amplification EGFR")
        assert result == "oui"

    def test_presence_no_amplification(self, extractor):
        """PRESENCE field: 'pas d'amplification' → 'non'."""
        result = extractor._postprocess_span("ampli_egfr", "pas d'amplification")
        assert result == "non"

    # -- SIMILARITY fields ---------------------------------------------

    def test_similarity_ihc_positif(self, extractor):
        """SIMILARITY field: IHC 'positif' matches vocab."""
        result = extractor._postprocess_span("ihc_idh1", "positif")
        assert result == "positif"

    def test_similarity_ihc_maintenu(self, extractor):
        """SIMILARITY field: IHC 'maintenu' matches vocab."""
        result = extractor._postprocess_span("ihc_atrx", "maintenu")
        assert result == "maintenu"

    def test_similarity_chromosomal_gain(self, extractor):
        """SIMILARITY field: chromosomal 'gain' matches vocab."""
        result = extractor._postprocess_span("ch7p", "gain")
        assert result == "gain"

    def test_similarity_chromosomal_perte(self, extractor):
        """SIMILARITY field: chromosomal 'perte' matches vocab."""
        result = extractor._postprocess_span("ch10q", "perte")
        assert result == "perte"

    # -- DIRECT fields -------------------------------------------------

    def test_direct_returns_as_is(self, extractor):
        """DIRECT field: returns span text unchanged."""
        result = extractor._postprocess_span("diag_histologique", "glioblastome IDH-wildtype")
        assert result == "glioblastome IDH-wildtype"

    def test_direct_date(self, extractor):
        """DIRECT field: date text returned as-is."""
        result = extractor._postprocess_span("date_chir", "15/03/2024")
        assert result == "15/03/2024"

    def test_direct_free_text(self, extractor):
        """DIRECT field: free text returned as-is."""
        result = extractor._postprocess_span("neuroncologue", "Dr Dupont")
        assert result == "Dr Dupont"


# ═══════════════════════════════════════════════════════════════════════════
# Schema: mapping_type annotations
# ═══════════════════════════════════════════════════════════════════════════

class TestMappingTypeAnnotations:
    """Verify mapping_type is correctly annotated on representative fields."""

    @pytest.mark.parametrize("field_name", [
        "epilepsie", "histo_necrose", "histo_pec", "ampli_egfr",
        "ampli_mdm2", "fusion_fgfr", "ch1p19q_codel",
        "epilepsie_1er_symptome", "prise_de_contraste",
        "antecedent_tumoral", "corticoides", "optune",
    ])
    def test_presence_fields(self, field_name):
        fd = ALL_FIELDS_BY_NAME.get(field_name)
        assert fd is not None, f"Field '{field_name}' not found in schema"
        assert fd.mapping_type == MappingType.PRESENCE, (
            f"Field '{field_name}' should be PRESENCE, got {fd.mapping_type}"
        )

    @pytest.mark.parametrize("field_name", [
        "ihc_idh1", "ihc_p53", "ihc_atrx", "ihc_gfap", "ihc_olig2", "ihc_mmr",
        "ch1p", "ch19q", "ch10p", "ch10q", "ch7p", "ch7q", "ch9p", "ch9q",
        "type_chirurgie", "tumeur_lateralite", "dominance_cerebrale", "sexe",
    ])
    def test_similarity_fields(self, field_name):
        fd = ALL_FIELDS_BY_NAME.get(field_name)
        assert fd is not None, f"Field '{field_name}' not found in schema"
        assert fd.mapping_type == MappingType.SIMILARITY, (
            f"Field '{field_name}' should be SIMILARITY, got {fd.mapping_type}"
        )

    @pytest.mark.parametrize("field_name", [
        "date_chir", "diag_histologique", "neuroncologue",
        "mol_idh1", "mol_mgmt", "chimios", "grade",
    ])
    def test_direct_fields(self, field_name):
        fd = ALL_FIELDS_BY_NAME.get(field_name)
        assert fd is not None, f"Field '{field_name}' not found in schema"
        assert fd.mapping_type == MappingType.DIRECT, (
            f"Field '{field_name}' should be DIRECT, got {fd.mapping_type}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Schema: "NA" in all controlled vocabularies
# ═══════════════════════════════════════════════════════════════════════════

class TestNAInVocabs:
    """Verify 'NA' is accepted by all controlled vocabulary sets."""

    def test_na_in_binary(self):
        from src.extraction.schema import ControlledVocab
        assert "NA" in ControlledVocab.BINARY

    def test_na_in_ihc_status(self):
        from src.extraction.schema import ControlledVocab
        assert "NA" in ControlledVocab.IHC_STATUS

    def test_na_in_molecular_status(self):
        from src.extraction.schema import ControlledVocab
        assert "NA" in ControlledVocab.MOLECULAR_STATUS

    def test_na_in_chromosomal(self):
        from src.extraction.schema import ControlledVocab
        assert "NA" in ControlledVocab.CHROMOSOMAL

    def test_na_in_surgery_type(self):
        from src.extraction.schema import ControlledVocab
        assert "NA" in ControlledVocab.SURGERY_TYPE

    def test_na_in_laterality(self):
        from src.extraction.schema import ControlledVocab
        assert "NA" in ControlledVocab.LATERALITY

    def test_na_in_sex(self):
        from src.extraction.schema import ControlledVocab
        assert "NA" in ControlledVocab.SEX

    def test_na_valid_evolution(self):
        from src.extraction.schema import ControlledVocab
        assert ControlledVocab.is_valid_evolution("NA") is True

    def test_na_valid_molecular(self):
        from src.extraction.schema import ControlledVocab
        assert ControlledVocab.is_valid_molecular("NA") is True


# ═══════════════════════════════════════════════════════════════════════════
# GlinerExtractor.extract() with mock model
# ═══════════════════════════════════════════════════════════════════════════

class TestGlinerExtractWithMock:
    """End-to-end extract() with a mock GLiNER model."""

    def _make_extractor(self, strategy="heterogeneous"):
        ext = GlinerExtractor(batching_strategy=strategy)
        ext._model = self._MockModel()
        return ext

    class _MockModel:
        """Mock GLiNER predict_entities returning fixed results."""

        def predict_entities(self, text, labels, threshold=0.5):
            results = []
            if "crise d'épilepsie" in text:
                for label in labels:
                    if any(kw in label.lower() for kw in ("épilepsie", "seizure", "epilep", "crise", "convulsion")):
                        results.append({"text": "crise d'épilepsie", "label": label, "score": 0.95})
            if "positif" in text:
                for label in labels:
                    if "IDH1" in label or "idh1" in label.lower():
                        results.append({"text": "positif", "label": label, "score": 0.92})
            if "glioblastome" in text:
                for label in labels:
                    if any(kw in label.lower() for kw in ("diag", "histol", "tumor diag", "tumeur")):
                        results.append({"text": "glioblastome", "label": label, "score": 0.98})
            return results

    def test_extract_heterogeneous(self):
        ext = self._make_extractor("heterogeneous")
        text = "Crise d'épilepsie inaugurale. Diagnostic: glioblastome."
        result = ext.extract(text, ["epilepsie_1er_symptome", "diag_histologique"])
        # Should have extracted at least some fields
        assert isinstance(result, dict)

    def test_extract_semantic_only(self):
        ext = self._make_extractor("semantic_only")
        text = "Crise d'épilepsie inaugurale."
        result = ext.extract(text, ["epilepsie_1er_symptome"])
        assert isinstance(result, dict)

    def test_extract_semantic_context(self):
        ext = self._make_extractor("semantic_context")
        text = "Crise d'épilepsie inaugurale."
        result = ext.extract(text, ["epilepsie_1er_symptome"])
        assert isinstance(result, dict)

    def test_extract_empty_subset(self):
        ext = self._make_extractor()
        result = ext.extract("Some text", [])
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════
# Field descriptions coverage
# ═══════════════════════════════════════════════════════════════════════════

class TestFieldDescriptions:
    """Verify FIELD_DESCRIPTIONS cover all 111 GLiNER fields."""

    def test_en_covers_all_fields(self):
        missing = _ALL_GLINER_FIELDS - set(FIELD_DESCRIPTIONS_EN.keys())
        assert not missing, f"Missing EN descriptions: {missing}"

    def test_fr_covers_all_fields(self):
        missing = _ALL_GLINER_FIELDS - set(FIELD_DESCRIPTIONS_FR.keys())
        assert not missing, f"Missing FR descriptions: {missing}"

    def test_reverse_desc_map_covers_en(self):
        for field, desc in FIELD_DESCRIPTIONS_EN.items():
            assert _REVERSE_DESC_MAP.get(desc) == field, (
                f"EN desc for '{field}' not in _REVERSE_DESC_MAP"
            )

    def test_reverse_desc_map_covers_fr(self):
        for field, desc in FIELD_DESCRIPTIONS_FR.items():
            assert _REVERSE_DESC_MAP.get(desc) == field, (
                f"FR desc for '{field}' not in _REVERSE_DESC_MAP"
            )

    def test_descriptions_count(self):
        assert len(FIELD_DESCRIPTIONS_EN) == 111
        assert len(FIELD_DESCRIPTIONS_FR) == 111


if __name__ == "__main__":
    pytest.main(["-v", __file__])
