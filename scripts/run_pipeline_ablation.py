#!/usr/bin/env python3
"""Transparent ablation runner — executes the pipeline on gold-standard documents
and saves comprehensive step-by-step logs.

Usage
-----
    # Rules-only mode (EDSExtractor + regex rules, no HF models)
    python scripts/run_pipeline_ablation.py --mode rule --out-dir tmp/rule_run

    # ML-only mode (HFExtractor on all fields, no regex rules)
    python scripts/run_pipeline_ablation.py --mode ml --out-dir tmp/ml_run

    # Rules + ML mode (default hybrid)
    python scripts/run_pipeline_ablation.py --mode both --out-dir tmp/both_run

    # Ablate specific HF groups
    python scripts/run_pipeline_ablation.py --mode ml --groups diagnosis ihc molecular

    # Use locally cached models
    python scripts/run_pipeline_ablation.py --mode ml --local-model-dir /path/to/models

Options
-------
    --mode              rule | ml | both  (required)
    --data-dir          Path to directory containing gold-standard JSON files.
                        Defaults to data/gold_standard/
    --out-dir           Where to write results JSON + log file. Created if needed.
    --max-docs          Cap number of documents processed (useful for quick tests).
    --no-negation       Disable negation detection (for ablation).
    --jobs              Number of parallel workers (-1 = auto, 1 = sequential).
                        Ignored when --mode is ml or both (HF batch forces sequential).
    --groups            Subset of HF model groups to enable (default: all 10).
    --local-model-dir   Directory containing locally cached bright-eds-{group} subdirs.

Output
------
The script writes two files to --out-dir:

1. ``results_<mode>_<timestamp>.json`` — one record per document with:
   - Final features (field → value)
   - Intermediate results per branch (date_results, controlled_results,
     rule_results, eds_results, hf_results, rules_merged)
   - Extraction log (all pipeline events)
   - Flagged fields

2. ``run_<mode>_<timestamp>.log`` — full DEBUG-level log capturing every
   intermediate extraction dict printed by ``transparent=True``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.pipeline import ExtractionPipeline  # noqa: E402
from src.extraction.provenance import ExtractionResult  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GOLD_STANDARD_DIR = PROJECT_ROOT / "data" / "gold_standard"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pipeline(
    mode: str,
    use_negation: bool,
    n_jobs: int,
    enabled_groups: list[str] | None = None,
    local_model_dir: Path | None = None,
) -> ExtractionPipeline:
    """Instantiate the right pipeline for the requested ablation mode."""
    common = dict(
        use_negation=use_negation,
        transparent=True,           # always enabled — step-level logging goes to DEBUG
        verbose=True,               # step headers go to DEBUG too
        n_jobs=n_jobs,
        enabled_groups=enabled_groups,
        local_model_dir=local_model_dir,
    )
    if mode == "rule":
        return ExtractionPipeline(use_rules=True, use_eds=False, **common)
    if mode == "ml":
        return ExtractionPipeline(use_rules=False, use_eds=True, **common)
    if mode == "both":
        return ExtractionPipeline(use_rules=True, use_eds=True, **common)
    raise ValueError(f"Unknown mode: {mode!r}")


def _load_docs(data_dir: Path, max_docs: int | None) -> list[dict]:
    """Load all *.json gold-standard files from *data_dir*, skipping manifest."""
    docs: list[dict] = []
    for path in sorted(data_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logging.warning("Skipping %s: %s", path.name, exc)
    if max_docs is not None:
        docs = docs[:max_docs]
    return docs


def _ev_to_dict(ev) -> dict | None:
    """Serialise an ExtractionValue to a plain dict (None-safe)."""
    if ev is None:
        return None
    return {
        "value": ev.value,
        "source_span": ev.source_span,
        "source_span_start": ev.source_span_start,
        "source_span_end": ev.source_span_end,
        "confidence": ev.confidence,
        "extraction_tier": ev.extraction_tier,
        "section": ev.section,
        "vocab_valid": ev.vocab_valid,
        "flagged": ev.flagged,
    }


def _result_to_record(result: ExtractionResult) -> dict:
    """Convert an ExtractionResult to a JSON-serialisable record."""
    def _dict_of_ev(d: dict) -> dict:
        return {k: _ev_to_dict(v) for k, v in d.items()}

    return {
        "document_id": result.document_id,
        "patient_id": result.patient_id,
        "document_type": result.document_type,
        "document_date": result.document_date,
        "sections_detected": result.sections_detected,
        "tier1_count": result.tier1_count,
        "total_extraction_time_ms": round(result.total_extraction_time_ms, 1),
        "flagged_for_review": result.flagged_for_review,
        # Final decided features
        "features": _dict_of_ev(result.features),
        # Per-branch intermediate results
        "date_results": _dict_of_ev(result.date_results),
        "controlled_results": _dict_of_ev(result.controlled_results),
        "rule_results": _dict_of_ev(result.rule_results),
        "eds_results": _dict_of_ev(result.eds_results),    # EDSExtractor (RULES branch)
        "rules_merged": _dict_of_ev(result.rules_merged),
        "hf_results": _dict_of_ev(result.hf_results),      # HFExtractor  (ML branch)
        # Full audit trail
        "extraction_log": result.extraction_log,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transparent ablation runner for the BRIGHT extraction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["rule", "ml", "both"], required=True,
                        help="Pipeline ablation mode.")
    parser.add_argument("--data-dir", type=Path, default=GOLD_STANDARD_DIR,
                        help="Directory containing gold-standard JSON files.")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/pipeline_ablation"),
                        help="Output directory for results and log.")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Maximum number of documents to process.")
    parser.add_argument("--no-negation", action="store_true",
                        help="Disable negation detection.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel workers (-1=auto, 1=sequential). Ignored for ml/both modes.")
    parser.add_argument("--groups", nargs="+", default=None,
                        metavar="GROUP",
                        help="HF model groups to enable (default: all 10). "
                             "E.g.: --groups diagnosis ihc molecular")
    parser.add_argument("--local-model-dir", type=Path, default=None,
                        help="Directory containing locally cached bright-eds-{group} subdirs.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Logging setup — file (DEBUG) + console (INFO)
    # ------------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.out_dir / f"run_{args.mode}_{ts}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)-7s] %(name)s: %(message)s")
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger("ablation")

    logger.info("=" * 70)
    logger.info("BRIGHT Pipeline Ablation Runner")
    logger.info("  mode        : %s", args.mode)
    logger.info("  negation    : %s", not args.no_negation)
    logger.info("  data_dir    : %s", args.data_dir)
    logger.info("  out_dir     : %s", args.out_dir)
    logger.info("  log_file    : %s", log_path)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Load documents
    # ------------------------------------------------------------------
    if not args.data_dir.exists():
        logger.error("data-dir does not exist: %s", args.data_dir)
        sys.exit(1)

    docs = _load_docs(args.data_dir, args.max_docs)
    if not docs:
        logger.error("No documents found in %s", args.data_dir)
        sys.exit(1)
    logger.info("Loaded %d document(s) from %s", len(docs), args.data_dir)

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    logger.info("Initialising pipeline (mode=%s)…", args.mode)
    pipeline = _build_pipeline(
        mode=args.mode,
        use_negation=not args.no_negation,
        n_jobs=args.jobs,
        enabled_groups=args.groups,
        local_model_dir=args.local_model_dir,
    )
    logger.info("Pipeline ready.")

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------
    records: list[dict] = []
    for i, doc in enumerate(docs):
        doc_id = doc.get("document_id", f"doc_{i}")
        patient_id = doc.get("patient_id", "")
        text = doc.get("raw_text", "")
        consultation_date = doc.get("date_chir")  # use surgery date as reference

        logger.info(
            "─── Document %d / %d  id=%s  patient=%s ───",
            i + 1, len(docs), doc_id, patient_id,
        )

        try:
            result = pipeline.extract_document(
                text=text,
                document_id=doc_id,
                patient_id=patient_id,
                consultation_date=consultation_date,
            )
        except Exception as exc:
            logger.error("  FAILED: %s", exc, exc_info=True)
            result = ExtractionResult(document_id=doc_id, patient_id=patient_id)
            result.add_log(f"Pipeline failed: {exc}")

        records.append(_result_to_record(result))

        # Inline summary to console
        logger.info(
            "  → type=%-15s  %d features  %d flagged  %.0fms",
            result.document_type,
            len(result.features),
            len(result.flagged_for_review),
            result.total_extraction_time_ms,
        )
        if result.flagged_for_review:
            logger.info("  → flagged: %s", result.flagged_for_review)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = args.out_dir / f"results_{args.mode}_{ts}.json"
    out_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("=" * 70)
    logger.info("Saved %d results → %s", len(records), out_path)
    logger.info("Full debug log   → %s", log_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
