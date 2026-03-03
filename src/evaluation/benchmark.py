import re
import pandas as pd
from pathlib import Path
from typing import Any

from src.extraction.pipeline import ExtractionPipeline
from src.extraction.schema import get_extractable_fields, ALL_FIELDS_BY_NAME, FieldType
from src.evaluation.gold_standard import load_gold_standard
from src.evaluation.metrics import (
    compute_per_feature_metrics,
    compute_aggregate_metrics,
    compute_category_metrics,
)

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

    # 6. Default: hallucination (FP) or generic omission/alteration
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

    for doc in docs:
        doc_id = doc.get("document_id", "")
        patient_id = doc.get("patient_id", "")
        text = doc.get("raw_text", "")

        result = pipeline.extract_document(text, document_id=doc_id, patient_id=patient_id)

        gt_annotations = doc.get("annotations", {})

        # Filter ground truth to only score features extractable for this doc type
        try:
            extractable = set(get_extractable_fields(result.document_type))
            gt_annotations = {k: v for k, v in gt_annotations.items() if k in extractable}
        except (ValueError, AttributeError):
            extractable = set()  # Unknown doc type — score all features

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

    # ── Aggregate metrics ──
    df_metrics = compute_aggregate_metrics(all_metrics)
    df_metrics.to_csv(output_path / "benchmark_metrics.csv")

    # ── Per-category metrics ──
    if not df_metrics.empty:
        df_cat = compute_category_metrics(df_metrics)
        df_cat.to_csv(output_path / "category_metrics.csv")

    # ── Error analysis ──
    error_cols = ["document_id", "patient_id", "feature", "error_type", "cause", "predicted", "ground_truth"]
    if error_analysis:
        pd.DataFrame(error_analysis).to_csv(output_path / "error_analysis.csv", index=False)
    else:
        pd.DataFrame(columns=error_cols).to_csv(output_path / "error_analysis.csv", index=False)

    return df_metrics
