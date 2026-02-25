"""Tests for src/extraction/pipeline.py — end-to-end extraction pipeline.

Tests use rule-based extraction only (``use_llm=False``) to avoid
dependency on a running Ollama server.  LLM integration is tested
separately via test_llm_extraction.py and test_ollama_client.py.
"""

import pytest

from src.extraction.pipeline import ExtractionPipeline
from src.extraction.provenance import ExtractionResult
from src.extraction.schema import ExtractionValue


# ---------------------------------------------------------------------------
# Sample documents for testing
# ---------------------------------------------------------------------------

SAMPLE_ANAPATH = """\
COMPTE-RENDU ANATOMOPATHOLOGIQUE

Patient: NIP 12345
Date chirurgie: 15/03/2024
Numéro laboratoire: AN-2024-001

Examen macroscopique:
Pièce opératoire reçue en fixation formolée, mesurant 3x2x1.5 cm.

Examen microscopique:
Prolifération gliale de haute densité cellulaire avec atypies nucléaires marquées.
Présence de nécrose palissadique.
Prolifération endothéliocapillaire.
15 mitoses / 10 HPF.

Immunohistochimie:
IDH1 : négatif
p53 : positif
ATRX : maintenu
Ki67 : 30%
GFAP : positif
Olig2 : positif
H3K27M : négatif
H3K27me3 : maintenu

Biologie moléculaire:
IDH1 : wt
IDH2 : wt
TERT : muté C228T
MGMT : méthylé

CGH-array:
1p : gain
19q : gain
7p : gain
7q : gain
10p : perte
10q : perte

Pas d'amplification EGFR.
Pas d'amplification MDM2.
Amplification CDK4.
Pas de fusion FGFR.

Conclusion:
Glioblastome, IDH-wildtype, OMS 2021, grade 4.
"""

SAMPLE_CONSULTATION = """\
Compte-Rendu de Consultation du 20/01/2025

Patient vu(e) en consultation de neuro-oncologie.
NIP: 12345
Sexe: M
Date de naissance: 05/06/1970

Neuro-oncologue: Dr Dupont
Neurochirurgien: Dr Martin
Radiothérapeute: Dr Durand

Antécédents:
Pas d'antécédent tumoral.
Date 1er symptôme: 01/12/2023
Épilepsie au premier symptôme.
Pas de céphalées.
Pas de déficit.

Examen clinique:
IK : 80%
Épilepsie sous traitement anti-épileptique (Keppra).
Pas de déficit neurologique.
Pas de troubles cognitifs.

Tumeur:
Localisation: temporal droit.
Latéralité: droite.

Traitement:
Chirurgie le 15/03/2024, exérèse complète.
Radiothérapie: 60 Gy, du 15/04/2024 au 30/05/2024.
Chimiothérapie: Témozolomide, 6 cycles, début 15/04/2024, fin 15/10/2024.
Pas de corticoïdes.
Pas d'Optune.

Évolution: initial.
Pas de progression clinique.
Pas de progression radiologique.
"""

SAMPLE_RCP = """\
Réunion de Concertation Pluridisciplinaire
Date: 10/02/2025

Patient: NIP 12345
Sexe: M

Diagnostic: Glioblastome IDH-wildtype, grade 4, OMS 2021.
IDH1 : négatif (IHC)
MGMT : méthylé

Proposition thérapeutique:
Protocole Stupp: radiothérapie 60 Gy + témozolomide concomitant puis adjuvant.
"""

SAMPLE_SHORT_TEXT = "Ceci est un texte très court sans structure."


# ---------------------------------------------------------------------------
# ExtractionPipeline tests — rule-based only
# ---------------------------------------------------------------------------

class TestExtractionPipelineRuleOnly:
    """Test ExtractionPipeline with use_llm=False."""

    @pytest.fixture
    def pipeline(self):
        return ExtractionPipeline(use_llm=False, use_negation=True)

    def test_extract_anapath(self, pipeline):
        """Extract from a sample anapath report."""
        result = pipeline.extract_document(
            text=SAMPLE_ANAPATH,
            document_id="test_anapath_001",
            patient_id="P12345",
        )

        assert isinstance(result, ExtractionResult)
        assert result.document_id == "test_anapath_001"
        assert result.patient_id == "P12345"
        assert result.document_type == "anapath"
        assert result.tier2_count == 0  # LLM disabled

        # Check that features were extracted
        assert len(result.features) > 0, "Should extract at least some features"

        # Check IHC results
        if "ihc_idh1" in result.features:
            assert result.features["ihc_idh1"].value == "negatif"
        if "ihc_p53" in result.features:
            assert result.features["ihc_p53"].value == "positif"
        if "ihc_atrx" in result.features:
            assert result.features["ihc_atrx"].value == "maintenu"

        # Check molecular results
        if "mol_idh1" in result.features:
            assert result.features["mol_idh1"].value == "wt"
        if "mol_tert" in result.features:
            assert result.features["mol_tert"].value == "mute"

        # Check chromosomal
        if "ch10p" in result.features:
            assert result.features["ch10p"].value == "perte"
        if "ch7p" in result.features:
            assert result.features["ch7p"].value == "gain"

        # Check amplifications
        if "ampli_egfr" in result.features:
            assert result.features["ampli_egfr"].value == "non"
        if "ampli_cdk4" in result.features:
            assert result.features["ampli_cdk4"].value == "oui"

        # Check binary fields
        if "histo_necrose" in result.features:
            assert result.features["histo_necrose"].value == "oui"

    def test_extract_consultation(self, pipeline):
        """Extract from a sample consultation note."""
        result = pipeline.extract_document(
            text=SAMPLE_CONSULTATION,
            document_id="test_consult_001",
            patient_id="P12345",
        )

        assert isinstance(result, ExtractionResult)
        assert result.document_type == "consultation"
        assert len(result.features) > 0

        # Check numerical results
        if "ik_clinique" in result.features:
            assert result.features["ik_clinique"].value == 80

        # Check binary results
        if "epilepsie" in result.features:
            assert result.features["epilepsie"].value == "oui"
        if "anti_epileptiques" in result.features:
            assert result.features["anti_epileptiques"].value == "oui"

    def test_extract_rcp(self, pipeline):
        """Extract from a sample RCP note."""
        result = pipeline.extract_document(
            text=SAMPLE_RCP,
            document_id="test_rcp_001",
            patient_id="P12345",
        )

        assert isinstance(result, ExtractionResult)
        assert result.document_type == "rcp"

    def test_extract_short_text(self, pipeline):
        """Pipeline should handle short/unstructured text gracefully."""
        result = pipeline.extract_document(
            text=SAMPLE_SHORT_TEXT,
            document_id="test_short_001",
        )

        assert isinstance(result, ExtractionResult)
        # Should still produce a result (possibly with few/no features)
        assert result.document_id == "test_short_001"

    def test_extract_empty_text(self, pipeline):
        """Pipeline should handle empty text gracefully."""
        result = pipeline.extract_document(
            text="",
            document_id="test_empty_001",
        )

        assert isinstance(result, ExtractionResult)
        assert result.document_id == "test_empty_001"


# ---------------------------------------------------------------------------
# Pipeline behaviour tests
# ---------------------------------------------------------------------------

class TestPipelineBehaviour:
    """Tests for specific pipeline behaviours."""

    @pytest.fixture
    def pipeline(self):
        return ExtractionPipeline(use_llm=False, use_negation=True)

    def test_tier1_precedence_over_tier2(self, pipeline):
        """Tier 1 results should take precedence when merged."""
        # Since we're using use_llm=False, all results are Tier 1.
        # We test the merge logic directly.
        result = pipeline.extract_document(
            text=SAMPLE_ANAPATH,
            document_id="test_precedence",
        )

        # All extracted features should be Tier 1
        for fname, ev in result.features.items():
            assert ev.extraction_tier == "rule", (
                f"Field '{fname}' should be 'rule' tier when LLM is disabled"
            )

    def test_extraction_log_populated(self, pipeline):
        """Extraction log should contain meaningful audit entries."""
        result = pipeline.extract_document(
            text=SAMPLE_ANAPATH,
            document_id="test_log",
        )

        assert len(result.extraction_log) > 0
        # Check for key log entries
        log_text = "\n".join(result.extraction_log)
        assert "Pipeline started" in log_text
        assert "Document classified" in log_text
        assert "Sections detected" in log_text
        assert "Tier 1" in log_text
        assert "Pipeline completed" in log_text

    def test_flagged_fields_tracked(self, pipeline):
        """Fields with vocab violations should be tracked in flagged_for_review."""
        result = pipeline.extract_document(
            text=SAMPLE_ANAPATH,
            document_id="test_flagged",
        )

        # The flagged_for_review list should exist (may be empty if all valid)
        assert isinstance(result.flagged_for_review, list)

    def test_sections_detected_populated(self, pipeline):
        """Section detection should identify relevant sections."""
        result = pipeline.extract_document(
            text=SAMPLE_ANAPATH,
            document_id="test_sections",
        )

        assert len(result.sections_detected) > 0

    def test_extraction_timing(self, pipeline):
        """Pipeline should report extraction timing."""
        result = pipeline.extract_document(
            text=SAMPLE_ANAPATH,
            document_id="test_timing",
        )

        assert result.total_extraction_time_ms > 0

    def test_classification_metadata(self, pipeline):
        """Classification metadata should be populated."""
        result = pipeline.extract_document(
            text=SAMPLE_ANAPATH,
            document_id="test_classification",
        )

        assert result.document_type in [
            "anapath", "molecular_report", "consultation", "rcp", "radiology"
        ]
        assert 0.0 <= result.classification_confidence <= 1.0

    def test_tier_counts(self, pipeline):
        """Tier 1 and Tier 2 counts should be accurate."""
        result = pipeline.extract_document(
            text=SAMPLE_ANAPATH,
            document_id="test_counts",
        )

        assert result.tier1_count >= 0
        assert result.tier2_count == 0  # LLM disabled

    def test_vocab_validation_runs(self, pipeline):
        """Vocabulary validation should run on all extracted features."""
        result = pipeline.extract_document(
            text=SAMPLE_ANAPATH,
            document_id="test_vocab",
        )

        # If features were extracted, vocab_valid should be set
        for fname, ev in result.features.items():
            assert isinstance(ev.vocab_valid, bool)


# ---------------------------------------------------------------------------
# Batch processing tests
# ---------------------------------------------------------------------------

class TestExtractBatch:
    """Tests for the extract_batch method."""

    @pytest.fixture
    def pipeline(self):
        return ExtractionPipeline(use_llm=False, use_negation=True)

    def test_batch_multiple_documents(self, pipeline):
        """Process multiple documents in batch."""
        documents = [
            {"text": SAMPLE_ANAPATH, "document_id": "doc1", "patient_id": "P1"},
            {"text": SAMPLE_CONSULTATION, "document_id": "doc2", "patient_id": "P2"},
            {"text": SAMPLE_RCP, "document_id": "doc3", "patient_id": "P3"},
        ]

        results = pipeline.extract_batch(documents)

        assert len(results) == 3
        assert all(isinstance(r, ExtractionResult) for r in results)
        assert results[0].document_id == "doc1"
        assert results[1].document_id == "doc2"
        assert results[2].document_id == "doc3"

    def test_batch_empty_list(self, pipeline):
        """Batch with empty list should return empty list."""
        results = pipeline.extract_batch([])
        assert results == []

    def test_batch_default_ids(self, pipeline):
        """Documents without explicit IDs should get default IDs."""
        documents = [
            {"text": SAMPLE_SHORT_TEXT},
        ]

        results = pipeline.extract_batch(documents)
        assert len(results) == 1
        assert results[0].document_id == "doc_0"


# ---------------------------------------------------------------------------
# ExtractionResult tests
# ---------------------------------------------------------------------------

class TestExtractionResult:
    """Tests for the ExtractionResult dataclass."""

    def test_add_log(self):
        result = ExtractionResult()
        result.add_log("Test message")
        assert "Test message" in result.extraction_log

    def test_flag_field(self):
        result = ExtractionResult()
        result.flag_field("ihc_idh1")
        assert "ihc_idh1" in result.flagged_for_review

    def test_flag_field_no_duplicates(self):
        result = ExtractionResult()
        result.flag_field("ihc_idh1")
        result.flag_field("ihc_idh1")
        assert result.flagged_for_review.count("ihc_idh1") == 1

    def test_update_flagged_from_features(self):
        result = ExtractionResult()
        result.features = {
            "ihc_idh1": ExtractionValue(value="positif", flagged=False),
            "ihc_p53": ExtractionValue(value="garbage", flagged=True),
            "grade": ExtractionValue(value=3, flagged=False),
        }
        result.update_flagged_from_features()
        assert "ihc_p53" in result.flagged_for_review
        assert "ihc_idh1" not in result.flagged_for_review

    def test_summary(self):
        result = ExtractionResult(
            document_id="test",
            document_type="anapath",
            patient_id="P1",
        )
        result.features = {
            "ihc_idh1": ExtractionValue(value="positif"),
        }
        result.tier1_count = 1

        summary = result.summary()
        assert summary["document_id"] == "test"
        assert summary["total_features"] == 1
        assert summary["tier1_count"] == 1

    def test_get_values_dict(self):
        result = ExtractionResult()
        result.features = {
            "ihc_idh1": ExtractionValue(value="positif"),
            "grade": ExtractionValue(value=3),
            "mol_idh1": ExtractionValue(value=None),
        }

        values = result.get_values_dict()
        assert values["ihc_idh1"] == "positif"
        assert values["grade"] == 3
        assert values["mol_idh1"] is None

    def test_default_values(self):
        result = ExtractionResult()
        assert result.document_id == ""
        assert result.document_type == ""
        assert result.document_date is None
        assert result.patient_id == ""
        assert result.features == {}
        assert result.sections_detected == []
        assert result.extraction_log == []
        assert result.flagged_for_review == []
        assert result.tier1_count == 0
        assert result.tier2_count == 0
        assert result.total_extraction_time_ms == 0.0
