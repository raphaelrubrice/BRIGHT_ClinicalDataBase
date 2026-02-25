from typing import Any
import pandas as pd
from src.extraction.schema import ExtractionValue

def compute_per_feature_metrics(
    predicted: dict[str, ExtractionValue],
    ground_truth: dict[str, Any]
) -> dict[str, dict[str, int]]:
    """
    Compare a single document's predicted features against the ground truth.
    
    Parameters
    ----------
    predicted : dict[str, ExtractionValue]
        The features dictionary from an ExtractionResult.
    ground_truth : dict[str, Any]
        The ground truth annotations dictionary.
        
    Returns
    -------
    dict[str, dict[str, int]]
        Metrics counts per feature: TP, TN, FP_hallucination, FN_omission, alteration.
    """
    results = {}
    all_features = set(predicted.keys()) | set(ground_truth.keys())
    
    for feature in all_features:
        p_val = predicted[feature].value if feature in predicted and predicted[feature] else None
        
        gt_entry = ground_truth.get(feature)
        if isinstance(gt_entry, dict):
            g_val = gt_entry.get("value")
        else:
            g_val = gt_entry
            
        tp = tn = fp_hallucination = fn_omission = alteration = 0
        
        def normalize(v):
            if isinstance(v, str):
                return v.strip().lower()
            return v
            
        p_val_norm = normalize(p_val)
        g_val_norm = normalize(g_val)
        
        if p_val_norm == g_val_norm:
            if p_val_norm is not None:
                tp = 1
            else:
                tn = 1
        else:
            if g_val_norm is None:
                fp_hallucination = 1
            elif p_val_norm is None:
                fn_omission = 1
            else:
                alteration = 1
                
        results[feature] = {
            "TP": tp,
            "TN": tn,
            "FP_hallucination": fp_hallucination,
            "FN_omission": fn_omission,
            "alteration": alteration
        }
        
    return results

def compute_aggregate_metrics(all_results: list[dict[str, dict[str, int]]]) -> pd.DataFrame:
    """
    Aggregate per-feature metrics across all documents and compute overall metrics.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by feature name with columns for TP, TN, FP, FN, P, R, F1, and rates.
    """
    aggregated = {}
    for doc_result in all_results:
        for feature, counts in doc_result.items():
            if feature not in aggregated:
                aggregated[feature] = {"TP": 0, "TN": 0, "FP_hallucination": 0, "FN_omission": 0, "alteration": 0}
            for k, v in counts.items():
                aggregated[feature][k] += v
                
    records = []
    for feature, counts in aggregated.items():
        tp = counts["TP"]
        tn = counts["TN"]
        fp_h = counts["FP_hallucination"]
        fn_o = counts["FN_omission"]
        alt = counts["alteration"]
        
        fp_total = fp_h + alt
        fn_total = fn_o + alt
        
        precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
        recall = tp / (tp + fn_total) if (tp + fn_total) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        hallucination_rate = fp_h / (fp_h + tn) if (fp_h + tn) > 0 else 0.0
        
        actual_positives = tp + fn_o + alt
        alteration_rate = alt / actual_positives if actual_positives > 0 else 0.0
        omission_rate = fn_o / actual_positives if actual_positives > 0 else 0.0
        
        records.append({
            "feature": feature,
            "TP": tp,
            "TN": tn,
            "FP": fp_total,
            "FN": fn_total,
            "FP_hallucination": fp_h,
            "FN_omission": fn_o,
            "alteration": alt,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Hallucination_Rate": hallucination_rate,
            "Alteration_Rate": alteration_rate,
            "Omission_Rate": omission_rate
        })
        
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index("feature")
    return df
