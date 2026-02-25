import pytest
import tempfile
import pandas as pd
from pathlib import Path

from src.evaluation.benchmark import run_benchmark
from src.extraction.pipeline import ExtractionPipeline
from src.extraction.schema import ExtractionValue
from src.extraction.provenance import ExtractionResult
from src.evaluation.gold_standard import save_gold_standard

class DummyPipeline(ExtractionPipeline):
    def __init__(self):
        # Override init to avoid loading LLM models in test
        pass
        
    def extract_document(self, text: str, document_id: str = "", patient_id: str = "") -> ExtractionResult:
        res = ExtractionResult(document_id=document_id, patient_id=patient_id)
        if text == "match":
            res.features["f1"] = ExtractionValue(value="val1")
        elif text == "hallucinate":
            res.features["f1"] = ExtractionValue(value="fake_val")
        return res

def test_run_benchmark():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        gs_dir = tmp_path / "gs"
        out_dir = tmp_path / "out"
        
        doc1 = {"document_id": "d1", "raw_text": "match", "annotations": {"f1": {"value": "val1"}}}
        doc2 = {"document_id": "d2", "raw_text": "hallucinate", "annotations": {"f1": {"value": "real_val"}}}
        
        save_gold_standard(doc1, gs_dir / "d1.json")
        save_gold_standard(doc2, gs_dir / "d2.json")
        
        pipeline = DummyPipeline()
        df = run_benchmark(gs_dir, pipeline, out_dir)
        
        assert len(df) == 1
        assert df.loc["f1", "TP"] == 1
        assert df.loc["f1", "alteration"] == 1
        
        assert (out_dir / "benchmark_metrics.csv").exists()
        assert (out_dir / "error_analysis.csv").exists()
