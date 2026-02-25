import os
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.extraction.pipeline import ExtractionPipeline
from src.evaluation.benchmark import run_benchmark
from src.aggregation.patient_timeline import build_patient_timeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(r"C:\Users\rapha\OneDrive\Bureau\MVA\BRIGHT\test_annotated")
DB_PATH = BASE_DIR / "RE MAJ Infos cliniques Braincap" / "clinical_db_pseudo_only.csv"
CLI_ANN_PATH = BASE_DIR / "ANNOTATIONS_RE MAJ Infos cliniques Braincap" / "REQ_CLINIQUE.csv"
BIO_ANN_PATH = BASE_DIR / "ANNOTATIONS_RE MAJ Infos cliniques Braincap" / "REQ_BIO.csv"
OUTPUT_DIR = Path("demo_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_transposed_annotations(path: Path) -> dict[str, list[dict[str, str]]]:
    """Load a transposed annotation CSV, preserving duplicate NIP columns.

    The annotation files are transposed: rows are features, columns are
    documents identified by NIP (patient ID) in the header.  The same NIP
    can appear multiple times (one column per document for that patient).

    Returns
    -------
    dict[str, list[dict[str, str]]]
        Mapping from NIP → list of annotation dicts (one per column),
        in the order they appear in the file.  Each annotation dict maps
        feature_name → value (strings, empty values omitted).
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        rows = list(csv.reader(f, delimiter=';'))

    if not rows:
        return {}

    header = rows[0]
    nip_columns = header[1:]  # skip the label column ('nip')

    # Build per-column annotation dicts
    column_annotations: list[tuple[str, dict]] = []
    for col_idx, nip in enumerate(nip_columns):
        nip = nip.strip()
        if not nip:
            continue
        column_annotations.append((nip, {}))

    for row in rows[1:]:
        if not row:
            continue
        feature_name = row[0].strip()
        if not feature_name:
            continue
        for col_idx, (nip, ann_dict) in enumerate(column_annotations):
            cell_idx = col_idx + 1  # offset by the feature-name column
            if cell_idx < len(row):
                val = row[cell_idx].strip()
                if val:
                    ann_dict[feature_name] = val

    # Group by NIP, preserving order
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for nip, ann_dict in column_annotations:
        result[nip].append(ann_dict)

    return dict(result)


def merge_annotation_lists(
    cli: dict[str, list[dict[str, str]]],
    bio: dict[str, list[dict[str, str]]],
) -> dict[str, list[dict[str, str]]]:
    """Merge CLINIQUE and BIO annotation lists per NIP.

    For each NIP, the two files may have different numbers of columns.
    We merge positionally: the first CLINIQUE column is merged with the
    first BIO column, etc.  If one list is longer, the extra entries
    are kept as-is.

    Returns
    -------
    dict[str, list[dict[str, str]]]
        Mapping from NIP → list of merged annotation dicts.
    """
    all_nips = set(cli.keys()) | set(bio.keys())
    merged: dict[str, list[dict[str, str]]] = {}

    for nip in all_nips:
        cli_list = cli.get(nip, [])
        bio_list = bio.get(nip, [])
        max_len = max(len(cli_list), len(bio_list))
        nip_merged = []
        for i in range(max_len):
            entry: dict[str, str] = {}
            if i < len(bio_list):
                entry.update(bio_list[i])
            if i < len(cli_list):
                entry.update(cli_list[i])  # clinical overrides bio on conflicts
            nip_merged.append(entry)
        merged[nip] = nip_merged

    return merged


def main():
    logger.info("Starting Full Demo Pipeline Test")

    if not DB_PATH.exists() or not CLI_ANN_PATH.exists() or not BIO_ANN_PATH.exists():
        logger.error("Missing necessary data files in test_annotated directory.")
        return

    # ------------------------------------------------------------------
    # 1. Load ground truth annotations (preserving per-column structure)
    # ------------------------------------------------------------------
    logger.info("Loading ground truth annotations...")
    cli_annotations = load_transposed_annotations(CLI_ANN_PATH)
    bio_annotations = load_transposed_annotations(BIO_ANN_PATH)
    merged_annotations = merge_annotation_lists(cli_annotations, bio_annotations)

    total_ann_cols = sum(len(v) for v in merged_annotations.values())
    logger.info(
        "Loaded annotations for %d patients (%d total annotation columns).",
        len(merged_annotations), total_ann_cols,
    )

    # ------------------------------------------------------------------
    # 2. Load documents database
    # ------------------------------------------------------------------
    logger.info("Loading documents database...")
    docs_db = pd.read_csv(DB_PATH, sep=',', dtype=str)
    docs_db = docs_db.dropna(subset=['PSEUDO'])
    logger.info("Found %d text documents.", len(docs_db))

    # ------------------------------------------------------------------
    # 3. Match documents to annotation columns (positional per NIP)
    # ------------------------------------------------------------------
    # Group DB docs by IPP, keeping original DID order
    docs_db = docs_db.sort_values('DID', key=lambda s: pd.to_numeric(s, errors='coerce'))
    docs_by_ipp: dict[str, list[dict]] = defaultdict(list)
    for _, row in docs_db.iterrows():
        docs_by_ipp[row['IPP']].append(row.to_dict())

    # ------------------------------------------------------------------
    # 4. Initialise pipeline
    # ------------------------------------------------------------------
    logger.info("Initializing ExtractionPipeline...")
    pipeline = ExtractionPipeline(ollama_model="qwen3:0.6b")

    # ------------------------------------------------------------------
    # 5. Build gold-standard JSONs and run benchmark
    # ------------------------------------------------------------------
    with TemporaryDirectory() as temp_dir:
        temp_gs_dir = Path(temp_dir)

        documents_for_timeline: list[dict] = []
        doc_count = 0
        skipped = 0

        for ipp, doc_rows in docs_by_ipp.items():
            ann_list = merged_annotations.get(ipp, [])
            if not ann_list:
                logger.debug("No annotations for IPP %s — skipping.", ipp)
                skipped += len(doc_rows)
                continue

            n_match = min(len(doc_rows), len(ann_list))
            if len(doc_rows) != len(ann_list):
                logger.warning(
                    "IPP %s: %d DB docs but %d annotation columns — matching first %d.",
                    ipp, len(doc_rows), len(ann_list), n_match,
                )

            for idx in range(n_match):
                row = doc_rows[idx]
                annotations = ann_list[idx]

                doc_id = str(row.get('DID', f'doc_{doc_count}'))
                patient_id = str(ipp)
                text = str(row.get('PSEUDO', ''))

                if not text.strip():
                    skipped += 1
                    continue

                # Wrap annotation values in {"value": ...} for gold_standard format
                gs_annotations = {k: {"value": v} for k, v in annotations.items()}

                gs_data = {
                    "document_id": doc_id,
                    "patient_id": patient_id,
                    "raw_text": text,
                    "annotations": gs_annotations,
                }

                with open(temp_gs_dir / f"{doc_id}.json", 'w', encoding='utf-8') as f:
                    json.dump(gs_data, f, indent=4, ensure_ascii=False)

                documents_for_timeline.append({
                    "document_id": doc_id,
                    "document_date": str(row.get('CONSULT_DATE', '')),
                    "patient_id": patient_id,
                    "text": text,
                })
                doc_count += 1

        if doc_count == 0:
            logger.warning("No matching documents found with annotations.")
            return

        logger.info(
            "Prepared %d documents for benchmark testing (%d skipped).",
            doc_count, skipped,
        )

        logger.info("Running evaluation benchmark...")
        benchmark_metrics = run_benchmark(str(temp_gs_dir), pipeline, str(OUTPUT_DIR))

        logger.info("Benchmark completed. Metrics saved to %s/benchmark_metrics.csv", OUTPUT_DIR)
        logger.info("\nOverall Benchmark Head:\n%s", benchmark_metrics.head(10))

    # ------------------------------------------------------------------
    # 6. Test timeline building for one patient
    # ------------------------------------------------------------------
    sample_patient = documents_for_timeline[0]["patient_id"]
    logger.info("Building timeline for sample patient: %s", sample_patient)

    patient_docs = [d for d in documents_for_timeline if d["patient_id"] == sample_patient]
    try:
        timeline_df = build_patient_timeline(sample_patient, patient_docs, pipeline)
        logger.info("Timeline generation test successful.")
        timeline_df.to_csv(OUTPUT_DIR / f"timeline_{sample_patient}.csv", index=False)
    except Exception as e:
        logger.error("Timeline generation failed: %s", e)


if __name__ == "__main__":
    main()
