import json
import tempfile
from pathlib import Path
from src.evaluation.gold_standard import load_gold_standard, save_gold_standard

def test_load_save_gold_standard():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        doc1 = {"document_id": "doc1", "annotations": {"feat1": {"value": 1}}}
        doc2 = {"document_id": "doc2", "annotations": {"feat2": {"value": "test"}}}
        
        save_gold_standard(doc1, tmp_path / "doc1.json")
        save_gold_standard(doc2, tmp_path / "doc2.json")
        
        loaded = load_gold_standard(tmp_path)
        assert len(loaded) == 2
        
        ids = [d["document_id"] for d in loaded]
        assert "doc1" in ids
        assert "doc2" in ids
        
        # Test missing directory
        assert len(load_gold_standard(tmp_path / "missing")) == 0
