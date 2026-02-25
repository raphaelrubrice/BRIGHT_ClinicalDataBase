import pandas as pd
from pathlib import Path
from typing import Any

from src.extraction.pipeline import ExtractionPipeline
from src.evaluation.gold_standard import load_gold_standard
from src.evaluation.metrics import compute_per_feature_metrics, compute_aggregate_metrics

def run_benchmark(
    gold_standard_dir: str,
    pipeline: ExtractionPipeline,
    output_dir: str
) -> pd.DataFrame:
    """
    Run the extraction pipeline on the gold standard dataset and compute metrics.
    
    Parameters
    ----------
    gold_standard_dir : str
        Directory containing the gold standard JSON files.
    pipeline : ExtractionPipeline
        The instantiated extraction pipeline.
    output_dir : str
        Directory to save the resulting CSV reports.
        
    Returns
    -------
    pd.DataFrame
        The aggregated metrics DataFrame.
    """
    docs = load_gold_standard(gold_standard_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    error_analysis = []
    
    for doc in docs:
        doc_id = doc.get("document_id", "")
        patient_id = doc.get("patient_id", "")
        text = doc.get("raw_text", "")  # Fallback empty string if omitted in gold truth file
        
        result = pipeline.extract_document(text, document_id=doc_id, patient_id=patient_id)
        
        gt_annotations = doc.get("annotations", {})
        
        metrics = compute_per_feature_metrics(result.features, gt_annotations)
        all_metrics.append(metrics)
        
        for feat, counts in metrics.items():
            if counts["FP_hallucination"] or counts["FN_omission"] or counts["alteration"]:
                p_val = result.features[feat].value if feat in result.features else None
                gt_val = gt_annotations.get(feat, {}).get("value") if isinstance(gt_annotations.get(feat), dict) else gt_annotations.get(feat)
                error_type = "hallucination" if counts["FP_hallucination"] else ("omission" if counts["FN_omission"] else "alteration")
                
                error_analysis.append({
                    "document_id": doc_id,
                    "patient_id": patient_id,
                    "feature": feat,
                    "error_type": error_type,
                    "predicted": p_val,
                    "ground_truth": gt_val
                })
                
    df_metrics = compute_aggregate_metrics(all_metrics)
    
    df_metrics.to_csv(output_path / "benchmark_metrics.csv")
    if error_analysis:
        pd.DataFrame(error_analysis).to_csv(output_path / "error_analysis.csv", index=False)
    else:
        # Create an empty error analysis CSV if no errors
        pd.DataFrame(columns=["document_id", "patient_id", "feature", "error_type", "predicted", "ground_truth"]).to_csv(output_path / "error_analysis.csv", index=False)
        
    return df_metrics
