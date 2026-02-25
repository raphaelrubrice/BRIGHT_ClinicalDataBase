"""Tests for src/evaluation/gold_standard.py — gold standard loading."""

import sys
from pathlib import Path

import pytest

# Ensure src/ is on the path for imports
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from evaluation.gold_standard import (
    GoldAnnotation,
    GoldDocument,
    load_gold_document,
    load_gold_standard,
    load_manifest,
    compare_extraction,
    DEFAULT_GOLD_DIR,
)


class TestGoldAnnotation:
    def test_string_match_case_insensitive(self):
        ann = GoldAnnotation(field_name="ihc_idh1", value="negatif")
        assert ann.matches("negatif")
        assert ann.matches("Negatif")
        assert ann.matches("NEGATIF")
        assert not ann.matches("positif")

    def test_int_match(self):
        ann = GoldAnnotation(field_name="grade", value=4)
        assert ann.matches(4)
        assert not ann.matches(3)

    def test_none_handling(self):
        ann = GoldAnnotation(field_name="ihc_braf", value=None)
        assert ann.matches(None)
        assert not ann.matches("positif")

        ann2 = GoldAnnotation(field_name="ihc_braf", value="positif")
        assert not ann2.matches(None)


class TestLoadGoldStandard:
    def test_load_all_documents(self):
        docs = load_gold_standard(DEFAULT_GOLD_DIR)
        assert len(docs) == 8
        for doc in docs:
            assert doc.patient_id
            assert doc.document_id
            assert doc.n_annotations > 0

    def test_load_manifest(self):
        manifest = load_manifest(DEFAULT_GOLD_DIR)
        assert manifest["total_entries"] == 8
        assert len(manifest["entries"]) == 8

    def test_load_single_document(self):
        filepath = DEFAULT_GOLD_DIR / "8003373720_initial.json"
        doc = load_gold_document(filepath)
        assert doc.document_id == "8003373720_initial"
        assert doc.patient_id == "8003373720"
        assert doc.evol_clinique == "initial"
        assert doc.has_bio_annotations is True
        assert doc.has_clinique_annotations is True
        assert doc.n_annotations >= 80

    def test_specific_annotations(self):
        filepath = DEFAULT_GOLD_DIR / "8003373720_initial.json"
        doc = load_gold_document(filepath)

        # BIO fields
        assert doc.get("ihc_idh1").value == "negatif"
        assert doc.get("grade").value == 4
        assert doc.get("mol_tert").value == "mute"
        assert doc.get("mol_braf").value == "V600E"
        assert doc.get("ampli_cdk4").value == "oui"

        # CLINIQUE fields
        assert doc.get("sexe").value == "M"
        assert doc.get("chimios").value == "Temozolomide"
        assert doc.get("type_chirurgie").value == "exerese complete"

    def test_clinique_only_document(self):
        """8003100840_P1 has no BIO annotations."""
        filepath = DEFAULT_GOLD_DIR / "8003100840_P1.json"
        doc = load_gold_document(filepath)
        assert doc.has_bio_annotations is False
        assert doc.has_clinique_annotations is True

    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            load_gold_standard("/nonexistent/path")


class TestCompareExtraction:
    def test_perfect_match(self):
        doc = GoldDocument(
            document_id="test",
            patient_id="123",
            date_chir=None,
            evol_clinique=None,
            has_bio_annotations=True,
            has_clinique_annotations=False,
            annotations={
                "grade": GoldAnnotation("grade", 4),
                "ihc_idh1": GoldAnnotation("ihc_idh1", "negatif"),
            },
        )
        predicted = {"grade": 4, "ihc_idh1": "negatif"}
        results = compare_extraction(doc, predicted)
        assert results["grade"]["category"] == "TP"
        assert results["ihc_idh1"]["category"] == "TP"

    def test_missed_field(self):
        doc = GoldDocument(
            document_id="test",
            patient_id="123",
            date_chir=None,
            evol_clinique=None,
            has_bio_annotations=True,
            has_clinique_annotations=False,
            annotations={
                "grade": GoldAnnotation("grade", 4),
            },
        )
        predicted = {}
        results = compare_extraction(doc, predicted)
        assert results["grade"]["category"] == "FN"

    def test_hallucinated_field(self):
        doc = GoldDocument(
            document_id="test",
            patient_id="123",
            date_chir=None,
            evol_clinique=None,
            has_bio_annotations=False,
            has_clinique_annotations=False,
            annotations={},
        )
        predicted = {"grade": 4}
        results = compare_extraction(doc, predicted)
        assert results["grade"]["category"] == "FP"

    def test_wrong_value(self):
        doc = GoldDocument(
            document_id="test",
            patient_id="123",
            date_chir=None,
            evol_clinique=None,
            has_bio_annotations=True,
            has_clinique_annotations=False,
            annotations={
                "grade": GoldAnnotation("grade", 4),
            },
        )
        predicted = {"grade": 3}
        results = compare_extraction(doc, predicted)
        assert results["grade"]["category"] == "FP"
        assert results["grade"]["match"] is False

    def test_unwraps_dict_values(self):
        doc = GoldDocument(
            document_id="test",
            patient_id="123",
            date_chir=None,
            evol_clinique=None,
            has_bio_annotations=True,
            has_clinique_annotations=False,
            annotations={
                "grade": GoldAnnotation("grade", 4),
            },
        )
        predicted = {"grade": {"value": 4, "confidence": 0.95}}
        results = compare_extraction(doc, predicted)
        assert results["grade"]["category"] == "TP"
