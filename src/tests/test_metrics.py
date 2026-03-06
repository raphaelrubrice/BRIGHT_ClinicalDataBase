import pytest
import pandas as pd
from src.extraction.schema import ExtractionValue
from src.evaluation.metrics import compute_per_feature_metrics, compute_aggregate_metrics

class TestMetrics:

    def test_per_feature_metrics(self):
        predicted = {
            "feat_tp": ExtractionValue(value="match"),
            "feat_fp": ExtractionValue(value="hallucinated"),
            "feat_alt": ExtractionValue(value="wrong"),
            "feat_tn": ExtractionValue(value=None)
        }
        
        ground_truth = {
            "feat_tp": {"value": "match"},
            "feat_fn": {"value": "missed"},
            "feat_alt": {"value": "correct_val"}
        }
        
        metrics = compute_per_feature_metrics(predicted, ground_truth)
        
        assert metrics["feat_tp"]["TP"] == 1
        assert metrics["feat_fp"]["FP_hallucination"] == 1
        assert metrics["feat_fn"]["FN_omission"] == 1
        assert metrics["feat_alt"]["alteration"] == 1
        assert metrics["feat_tn"]["TN"] == 1

    def test_hallucination_rate(self):
        doc1_metrics = {
            "feat1": {"TP": 1, "TN": 0, "FP_hallucination": 0, "FN_omission": 0, "alteration": 0},
        }
        doc2_metrics = {
            "feat1": {"TP": 0, "TN": 1, "FP_hallucination": 1, "FN_omission": 0, "alteration": 0},
        }
        
        df = compute_aggregate_metrics([doc1_metrics, doc2_metrics])
        
        # Hallucination rate = FP_h / (FP_h + TN) = 1 / (1 + 1) = 0.5
        assert df.loc["feat1", "Hallucination_Rate"] == 0.5

    def test_aggregate_metrics(self):
        doc1_metrics = {
            "feat1": {"TP": 1, "TN": 0, "FP_hallucination": 0, "FN_omission": 0, "alteration": 0, "extraction_tier": "llm"},
        }
        doc2_metrics = {
            "feat1": {"TP": 0, "TN": 0, "FP_hallucination": 1, "FN_omission": 0, "alteration": 0, "extraction_tier": "llm"},
        }
        doc3_metrics = {
            "feat1": {"TP": 0, "TN": 0, "FP_hallucination": 0, "FN_omission": 0, "alteration": 1, "extraction_tier": "llm"},
        }
        
        df = compute_aggregate_metrics([doc1_metrics, doc2_metrics, doc3_metrics])
        
        # FP = 1 + 0.5 = 1.5, TP = 1 => Precision = 1 / 2.5 = 0.4
        # FN = 0 + 0.5 = 0.5, TP = 1 => Recall = 1 / 1.5 = 0.666...
        
        assert df.loc["feat1", "TP"] == 1
        assert df.loc["feat1", "FP"] == 1.5
        assert df.loc["feat1", "FN"] == 0.5
        assert round(df.loc["feat1", "Precision"], 4) == 0.4000
        assert round(df.loc["feat1", "Recall"], 4) == 0.6667

    def test_normalize_int_vs_string(self):
        predicted = {"val": ExtractionValue(value=4)}
        ground_truth = {"val": {"value": "4"}}
        metrics = compute_per_feature_metrics(predicted, ground_truth)
        assert metrics["val"]["TP"] == 1
        
        predicted = {"val": ExtractionValue(value=90)}
        ground_truth = {"val": {"value": "90"}}
        metrics = compute_per_feature_metrics(predicted, ground_truth)
        assert metrics["val"]["TP"] == 1

    def test_normalize_na(self):
        predicted = {"val": None}
        ground_truth = {"val": {"value": "NA"}}
        metrics = compute_per_feature_metrics(predicted, ground_truth)
        assert metrics["val"]["TN"] == 1
        
        ground_truth = {"val": {"value": " na "}}
        metrics = compute_per_feature_metrics(predicted, ground_truth)
        assert metrics["val"]["TN"] == 1
        
        ground_truth = {"val": {"value": "Na"}}
        metrics = compute_per_feature_metrics(predicted, ground_truth)
        assert metrics["val"]["TN"] == 1

    def test_normalize_date(self):
        predicted = {"val": ExtractionValue(value="01/04/2010")}
        ground_truth = {"val": {"value": "avr-10"}}
        metrics = compute_per_feature_metrics(predicted, ground_truth)
        assert metrics["val"]["TP"] == 1
        
        predicted = {"val": ExtractionValue(value="01/03/2025")}
        ground_truth = {"val": {"value": "mars-25"}}
        metrics = compute_per_feature_metrics(predicted, ground_truth)
        assert metrics["val"]["TP"] == 1
        
        predicted = {"val": ExtractionValue(value="2008")}
        ground_truth = {"val": {"value": "2008"}}
        metrics = compute_per_feature_metrics(predicted, ground_truth)
        assert metrics["val"]["TP"] == 1
        
        predicted = {"val": ExtractionValue(value="12 mars 2021")}
        ground_truth = {"val": {"value": "12.03.2021"}}
        metrics = compute_per_feature_metrics(predicted, ground_truth)
        assert metrics["val"]["TP"] == 1

    def test_fuzzy_matching(self):
        # Eligible field
        predicted = {"diag_integre": ExtractionValue(value="glioblastome")}
        ground_truth = {"diag_integre": {"value": "gliolastome"}}
        metrics = compute_per_feature_metrics(predicted, ground_truth)
        assert metrics["diag_integre"]["TP"] == 1
        
        # Ineligible field
        predicted = {"nip": ExtractionValue(value="123456")}
        ground_truth = {"nip": {"value": "123457"}}
        metrics = compute_per_feature_metrics(predicted, ground_truth)
        assert metrics["nip"]["alteration"] == 1
        
    def test_normalize_bool_and_nulls(self):
        predicted = {"val1": ExtractionValue(value="oui"), "val2": ExtractionValue(value="non")}
        ground_truth = {"val1": {"value": True}, "val2": {"value": False}}
        metrics = compute_per_feature_metrics(predicted, ground_truth)
        assert metrics["val1"]["TP"] == 1
        assert metrics["val2"]["TP"] == 1
        
        predicted = {"val3": ExtractionValue(value=""), "val4": ExtractionValue(value="null")}
        ground_truth = {"val3": {"value": "none"}, "val4": {"value": "NA"}}
        metrics = compute_per_feature_metrics(predicted, ground_truth)
        assert metrics["val3"]["TN"] == 1
        assert metrics["val4"]["TN"] == 1
        
    def test_normalize_numeric_strings(self):
        predicted = {"val": ExtractionValue(value="4.0")}
        ground_truth = {"val": {"value": "4"}}
        metrics = compute_per_feature_metrics(predicted, ground_truth)
        assert metrics["val"]["TP"] == 1

    def test_normalize_chimios_order(self):
        predicted = {"chimios": ExtractionValue(value="bevacizumab + temozolomide")}
        ground_truth = {"chimios": {"value": "Temozolomide + bevacizumab"}}
        metrics = compute_per_feature_metrics(predicted, ground_truth)
        assert metrics["chimios"]["TP"] == 1
