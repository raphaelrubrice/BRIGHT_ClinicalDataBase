import re
import pandas as pd
from pathlib import Path
from typing import Any

from ..extraction.pipeline import ExtractionPipeline
from ..extraction.schema import get_extractable_fields, ALL_FIELDS_BY_NAME, FieldType
from .gold_standard import load_gold_standard
from .metrics import (
    compute_per_feature_metrics,
    compute_aggregate_metrics,
    compute_category_metrics,
    compute_tier_category_metrics,
)

# ---------------------------------------------------------------------------
# Per-Extractor Reporting Helper
# ---------------------------------------------------------------------------

def _generate_extractor_reports(all_data: list[dict], output_dir: str):
    """Generate detailed F1/TP/FP/FN/examples and row-by-row comparisons for each specific extractor."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Extractors: date, controlled, rules, eds
    extractors = ["date", "controlled", "rules", "eds"]
    
    for ext_name in extractors:
        all_metrics_ext = []
        comparison_rows = []
        
        for p_data in all_data:
            doc_id = p_data["document_id"]
            patient_id = p_data["patient_id"]
            text = p_data["text"]
            res = p_data["result"]
            gt_annotations = p_data["gt_annotations"]
            
            # Access the specific dictionary output depending on the extractor
            if ext_name == "date":
                p_dict = res.date_results
            elif ext_name == "controlled":
                p_dict = res.controlled_results
            elif ext_name == "rules":
                p_dict = res.rule_results
            elif ext_name == "eds":
                p_dict = res.eds_results
            else:
                p_dict = {}

            # The fields this extractor actually predicted for this document
            predicted_fields = set(p_dict.keys())
            
            # To get TP, TN, FP, FN over this subset, we must subset the GT
            # ONLY to the fields the extractor was active on, OR the fields it *should* have been active on.
            # But we don't know easily what fields it *should* have been active on 
            # if we just look at p_dict.keys(). 
            # We can use the extractable subset AND predicted subset to compute the metrics.
            
            ext_gt = {k: v for k, v in gt_annotations.items() if k in predicted_fields or k in gt_annotations}
            
            # Temporary metrics dict for this extractor on this doc
            doc_metrics = compute_per_feature_metrics(p_dict, ext_gt)
            
            # Filter the doc_metrics to ONLY include fields that were either:
            # 1. Predicted by this specific extractor
            # 2. Part of this extractor's domain (e.g. date fields for date extractor) but missed
            # A good heuristic: if it's in p_dict (a prediction), it belongs here.
            # For omissions, we must figure out the domain.
            
            domain_fields = set(predicted_fields)
            if ext_name == "date":
                # Add all valid date fields in GT
                domain_fields.update({f for f in ext_gt if f in _DATE_FIELDS})
            elif ext_name == "controlled":
                from ..extraction.controlled_vocab_data import CONTROLLED_REGISTRY_FR
                domain_fields.update({f for f in ext_gt if f in CONTROLLED_REGISTRY_FR})
            elif ext_name == "rules":
                # Rules handle all non-date fields
                domain_fields.update({f for f in ext_gt if f not in _DATE_FIELDS})
            elif ext_name == "eds":
                # EDS handles the fallback for non-dates
                domain_fields.update({f for f in ext_gt if f not in _DATE_FIELDS})
            
            # Keep only the metrics for the domain fields
            filtered_doc_metrics = {f: m for f, m in doc_metrics.items() if f in domain_fields}
            if filtered_doc_metrics:
                all_metrics_ext.append(filtered_doc_metrics)

            # Build comparison rows for this document
            for f in domain_fields:
                ev_str = str(p_dict[f].value) if f in p_dict and p_dict[f].value is not None else ""
                gt_entry = gt_annotations.get(f)
                gt_str = str(gt_entry.get("value") if isinstance(gt_entry, dict) else gt_entry) if gt_entry else ""
                
                # Determine status
                status = "Match"
                if ev_str == gt_str:
                    if not ev_str:
                        status = "TN" # Both empty
                else: # Mismatch
                    m_dict = doc_metrics.get(f, {})
                    if m_dict.get("FP_hallucination", 0):
                        status = "Hallucination"
                    elif m_dict.get("FN_omission", 0):
                        status = "Omission"
                    elif m_dict.get("alteration", 0):
                        status = "Alteration"
                    else:
                        status = "Mismatch"

                comparison_rows.append({
                    "document_id": doc_id,
                    "patient_id": patient_id,
                    "feature": f,
                    "predicted": ev_str,
                    "ground_truth": gt_str,
                    "status": status
                })

        # Generate performance_{extractor}.csv
        if all_metrics_ext:
            df_perf = compute_aggregate_metrics(all_metrics_ext)
            # Add exactly 5 examples per feature
            # we iterate over comparison_rows to find examples of predictions where ev_str != ""
            examples_dict = {f: [] for f in df_perf.index}
            for row in comparison_rows:
                f = row["feature"]
                val = row["predicted"]
                if f in examples_dict and val and len(examples_dict[f]) < 5:
                    if val not in examples_dict[f]:
                        examples_dict[f].append(val)
            
            df_perf["Examples"] = [", ".join(examples_dict[f]) for f in df_perf.index]
            df_perf.to_csv(out_path / f"performance_{ext_name}.csv")
            
        # Generate comparison_{extractor}.csv
        if comparison_rows:
            df_comp = pd.DataFrame(comparison_rows)
            df_comp.to_csv(out_path / f"comparison_{ext_name}.csv", index=False)


# ---------------------------------------------------------------------------
# Pseudo-token regex (mirrors the one in llm_extraction.py)
# ---------------------------------------------------------------------------
_PSEUDO_TOKEN_RE = re.compile(
    r'\[?(NOM|PRENOM|TEL|MAIL|ADDRESS|HOPITAL|VILLE|IPP|DATE|ZIP|SSID|NDA)_[A-Fa-f0-9]+\]?'
)

# Date fields (used for date-mismatch classification)
_DATE_FIELDS: set[str] = {
    f.name for f in ALL_FIELDS_BY_NAME.values() if f.field_type == FieldType.DATE
}


def _classify_error_cause(
    feature: str,
    error_type: str,
    predicted: Any,
    ground_truth: Any,
    extractable: set[str],
    text: str,
) -> str:
    """Heuristically classify the root cause of an extraction error.

    Returns one of: ``pseudo_token``, ``date_mismatch``, ``routing_omission``,
    ``truncation``, ``hallucination``, ``format_mismatch``.
    """
    p_str = str(predicted) if predicted is not None else ""
    gt_str = str(ground_truth) if ground_truth is not None else ""

    # 1. Pseudo-token: predicted value contains a pseudonymisation token
    if p_str and _PSEUDO_TOKEN_RE.search(p_str):
        return "pseudo_token"

    # 2. Routing omission: feature was not in the extractable set
    if error_type == "omission" and feature not in extractable:
        return "routing_omission"

    # 3. Date mismatch: both pred and gt are date-like but don't match
    if feature in _DATE_FIELDS and error_type == "alteration":
        return "date_mismatch"

    # 4. Truncation: ground truth value exists in text but was likely
    #    beyond the truncation window (simple heuristic: gt appears
    #    only in the last 30% of text)
    if error_type == "omission" and gt_str and text:
        pos = text.lower().find(gt_str.lower())
        if pos != -1 and pos > len(text) * 0.7:
            return "truncation"

    # 5. Format mismatch: values are semantically similar (case/accent)
    if error_type == "alteration" and p_str and gt_str:
        if p_str.strip().lower().replace("é", "e").replace("è", "e") == \
           gt_str.strip().lower().replace("é", "e").replace("è", "e"):
            return "format_mismatch"

    # 6. Int/string format mismatch
    if error_type == "alteration" and p_str and gt_str:
        if p_str.strip() == gt_str.strip():
            return "type_mismatch"

    # 7. Default: hallucination (FP) or generic omission/alteration
    return "hallucination" if error_type in ("hallucination", "omission") else "format_mismatch"


def run_benchmark(
    gold_standard_dir: str,
    pipeline: ExtractionPipeline,
    output_dir: str,
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

    all_metrics: list[dict] = []
    error_analysis: list[dict] = []
    
    # Store all parsed document data & results to compute per-extractor reports efficiently later
    all_docs_data: list[dict] = []
    
    tier1_total = 0

    for doc in docs:
        doc_id = doc.get("document_id", "")
        patient_id = doc.get("patient_id", "")
        text = doc.get("raw_text", "")

        result = pipeline.extract_document(text, document_id=doc_id, patient_id=patient_id)
        
        tier1_total += result.tier1_count

        gt_annotations = doc.get("annotations", {})

        # Filter ground truth to only score features extractable for this doc type
        try:
            extractable = set(get_extractable_fields(result.document_type))
            gt_annotations = {k: v for k, v in gt_annotations.items() if k in extractable}
        except (ValueError, AttributeError):
            extractable = set()  # Unknown doc type, score all features

        metrics = compute_per_feature_metrics(result.features, gt_annotations)
        all_metrics.append(metrics)

        for feat, counts in metrics.items():
            if counts["FP_hallucination"] or counts["FN_omission"] or counts["alteration"]:
                p_val = result.features[feat].value if feat in result.features else None
                gt_entry = gt_annotations.get(feat)
                gt_val = gt_entry.get("value") if isinstance(gt_entry, dict) else gt_entry
                error_type = (
                    "hallucination" if counts["FP_hallucination"]
                    else ("omission" if counts["FN_omission"] else "alteration")
                )

                cause = _classify_error_cause(
                    feature=feat,
                    error_type=error_type,
                    predicted=p_val,
                    ground_truth=gt_val,
                    extractable=extractable,
                    text=text,
                )

                error_analysis.append({
                    "document_id": doc_id,
                    "patient_id": patient_id,
                    "feature": feat,
                    "error_type": error_type,
                    "cause": cause,
                    "predicted": p_val,
                    "ground_truth": gt_val,
                })

        all_docs_data.append({
            "document_id": doc_id,
            "patient_id": patient_id,
            "text": text,
            "result": result,
            "gt_annotations": gt_annotations
        })

    # ── Aggregate metrics ──
    df_metrics = compute_aggregate_metrics(all_metrics)
    df_metrics.to_csv(output_path / "benchmark_metrics.csv")

    # ── Per-category metrics ──
    if not df_metrics.empty:
        df_cat = compute_category_metrics(df_metrics)
        df_cat.to_csv(output_path / "category_metrics.csv")
        
        df_tier = compute_tier_category_metrics(df_metrics)
        if not df_tier.empty:
            df_tier.to_csv(output_path / "tier_category_metrics.csv")

    # ── Error analysis ──
    error_cols = ["document_id", "patient_id", "feature", "error_type", "cause", "predicted", "ground_truth"]
    if error_analysis:
        pd.DataFrame(error_analysis).to_csv(output_path / "error_analysis.csv", index=False)
    else:
        pd.DataFrame(columns=error_cols).to_csv(output_path / "error_analysis.csv", index=False)

    # ── Per-Extractor Reports ──
    _generate_extractor_reports(all_docs_data, output_dir)

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Extraction provenance summary across all benchmark documents:")
    logger.info("  Tier 1 (Rule/EDS): %d", tier1_total)

    return df_metrics
