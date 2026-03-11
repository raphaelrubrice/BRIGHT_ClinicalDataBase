from __future__ import annotations

import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd

from src.extraction.pipeline import ExtractionPipeline
from src.aggregation.patient_timeline import build_patient_timeline
from src.extraction.schema import ALL_BIO_FIELD_NAMES, ALL_CLINIQUE_FIELD_NAMES

logger = logging.getLogger(__name__)

def run_extraction_cli(
    db_path: Path,
    output_dir: Path,
    use_gliner: bool = True,
    batching_strategy: str = "heterogeneous",
    parallel_workers: int = 1,
) -> None:
    """Run full extraction and timeline creation purely via CLI logic.
    
    This function:
    1. Loads the document DB (requires 'PSEUDO' column).
    2. Initializes ExtractionPipeline.
    3. Builds per-patient timeline.
    4. Aggregates them into bio.csv and clinique.csv databases.
    """
    logger.info("Initializing extraction CLI.")
    
    if not db_path.exists():
        logger.error(f"Database file not found at {db_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    docs_db = pd.read_csv(db_path, sep=',', dtype=str)
    if 'PSEUDO' not in docs_db.columns:
        logger.error(f"'PSEUDO' column not found in database {db_path}")
        return
        
    docs_db = docs_db.dropna(subset=['PSEUDO'])
    logger.info(f"Found {len(docs_db)} text documents to process.")

    # Group DB docs by IPP
    docs_db = docs_db.sort_values('DID', key=lambda s: pd.to_numeric(s, errors='coerce'))
    docs_by_ipp: dict[str, list[dict]] = defaultdict(list)
    for _, row in docs_db.iterrows():
        docs_by_ipp[str(row['IPP'])].append(row.to_dict())

    # Initialise pipeline
    logger.info("Initializing ExtractionPipeline "
                f"(GLiNER={use_gliner}, batching={batching_strategy}, n_jobs={parallel_workers})...")
                
    pipeline_kwargs = {
        "use_gliner": use_gliner,
        "use_eds": True,
        "use_negation": True,
        "batching_strategy": batching_strategy,
        "verbose": True,
        "n_jobs": parallel_workers,
    }
    pipeline = ExtractionPipeline(**pipeline_kwargs)

    all_timelines = []

    for ipp, doc_rows in docs_by_ipp.items():
        logger.info(f"Processing patient IPP {ipp} ({len(doc_rows)} documents)...")
        documents_for_timeline = []
        for row in doc_rows:
            doc_id = str(row.get('DID', ''))
            text = str(row.get('PSEUDO', ''))
            if not text.strip():
                continue
            documents_for_timeline.append({
                "document_id": doc_id,
                "document_date": str(row.get('CONSULT_DATE', '')),
                "patient_id": ipp,
                "text": text,
            })
            
        try:
            timeline_df = build_patient_timeline(ipp, documents_for_timeline, pipeline)
            if not timeline_df.empty:
                all_timelines.append(timeline_df)
        except Exception as e:
            logger.error(f"Timeline generation failed for IPP {ipp}: {e}")

    if not all_timelines:
        logger.warning("No timeline data generated across all patients.")
        return

    final_df = pd.concat(all_timelines, ignore_index=True)
    logger.info(f"Final aggregated extraction DB size: {final_df.shape[0]} rows.")
    
    # Dump full combined dataset
    combined_path = output_dir / "extraction_full.csv"
    final_df.to_csv(combined_path, index=False)
    
    # Split bio / clinique
    metadata_cols = ["_patient_id", "_document_id", "_document_type", "_document_date", "time_index"]
    
    # Make sure we only grab columns that actually exist in the dataframe
    avail_cols = set(final_df.columns)
    
    bio_cols = [c for c in metadata_cols if c in avail_cols] + \
               [c for c in ALL_BIO_FIELD_NAMES if c in avail_cols]
    clinique_cols = [c for c in metadata_cols if c in avail_cols] + \
                    [c for c in ALL_CLINIQUE_FIELD_NAMES if c in avail_cols]

    bio_df = final_df[bio_cols]
    clinique_df = final_df[clinique_cols]

    bio_path = output_dir / "bio.csv"
    clinique_path = output_dir / "clinique.csv"
    
    bio_df.to_csv(bio_path, index=False)
    clinique_df.to_csv(clinique_path, index=False)
    
    logger.info(f"Successfully wrote outputs to {output_dir}:")
    logger.info(f"  - {combined_path.name}")
    logger.info(f"  - {bio_path.name}")
    logger.info(f"  - {clinique_path.name}")
