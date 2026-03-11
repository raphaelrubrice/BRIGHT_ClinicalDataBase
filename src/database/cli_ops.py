from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List

import pandas as pd

from src.database.ops import (
    init_db,
    append_rows_locked,
    load_db,
    DEFAULT_COLUMNS,
    extract_IPP_from_document,
    extract_IPP_from_path,
)
from src.database.pseudonymizer import TextPseudonymizer
from src.database.security import get_or_create_salt_file
from src.database.text_extraction import TextExtractor
from src.database.utils import resolve_eds_model_path, prepare_eds_registry

logger = logging.getLogger(__name__)

def run_pseudonymization_cli(
    db_path: Path,
    pdf_paths: List[Path],
    eds_path: Optional[Path] = None,
    make_pseudo_only: bool = True,
    chunk_size: int = 1,
) -> None:
    """Run pseudonymization pipeline purely via CLI logic.
    
    This function:
    1. Initializes or loads the DB and pseudonymization model.
    2. Extracts text from PDFs.
    3. Pseudonymizes text and commits rows per chunk.
    4. Optionally saves a `_pseudo_only.csv` duplicate.
    """
    logger.info("Initializing pseudonymization CLI.")

    if not db_path.exists():
        logger.info(f"DB not found at {db_path}. Creating new DB.")
        init_db(db_path, columns=DEFAULT_COLUMNS)
    
    df_db = load_db(db_path)
    salt = get_or_create_salt_file(db_path)

    # Initialize model
    artifacts_path = resolve_eds_model_path(str(eds_path) if eds_path else None)
    prepare_eds_registry(artifacts_path.parent)

    try:
        pseudonymizer = TextPseudonymizer(
            model_path=str(artifacts_path),
            auto_update=False,
            secret_salt=salt,
        )
    except Exception as e:
        logger.error(f"Failed to load EDS model: {e}")
        return

    extractor = TextExtractor()

    candidates: list[dict] = []
    skipped = 0
    errors = 0

    # 1. Extraction Phase
    for p in pdf_paths:
        try:
            text = extractor.pdf_to_text(p)
            if not text.strip():
                raise ValueError("Empty extracted text")

            try:
                ipp = str(int(extract_IPP_from_document(text)))
            except Exception:
                ipp = str(int(extract_IPP_from_path(p)))

            # Check duplicate
            is_dup = not df_db.empty and ((df_db["IPP"] == ipp) & (df_db["DOCUMENT"] == text)).any()
            if is_dup:
                skipped += 1
            else:
                candidates.append({
                    "path": p,
                    "ipp": ipp,
                    "document": text
                })
        except Exception as e:
            logger.error(f"[Extraction] {p.name}: {e}")
            errors += 1

    if not candidates:
        logger.info(f"No documents to commit. Skipped duplicates: {skipped}, Errors: {errors}")
        return
        
    logger.info(f"Extract Phase Done. {len(candidates)} candidates, {skipped} skipped, {errors} errors.")

    # 2. Pseudonymization Phase
    pending_rows = []
    pending_names = []
    committed_files = []

    def flush_pending():
        if not pending_rows: return
        try:
            append_rows_locked(db_path, pd.DataFrame(pending_rows))
            committed_files.extend(pending_names)
        except Exception:
            for row_dict, name in zip(pending_rows, pending_names):
                try:
                    append_rows_locked(db_path, pd.DataFrame([row_dict]))
                    committed_files.append(name)
                except Exception as row_err:
                    logger.error(f"[Commit] {name}: {row_err}")

    for i, item in enumerate(candidates, start=1):
        p = item["path"]
        try:
            pseudo = pseudonymizer.pseudonymize(
                item["document"],
                ipp=item["ipp"],
                keep_practitioner_names=True,
            )

            pending_rows.append({
                "IPP": item["ipp"],
                "SOURCE_FILE": str(p.resolve()),
                "DOCUMENT": item["document"],
                "PSEUDO": pseudo,
                "ORDER": 1,
            })
            pending_names.append(p.name)

            if len(pending_rows) >= chunk_size:
                flush_pending()
                pending_rows.clear()
                pending_names.clear()

        except Exception as e:
            logger.error(f"[Pseudonymization] {p.name}: {e}")

    flush_pending()

    logger.info(f"Committed {len(committed_files)} documents to {db_path}.")

    if make_pseudo_only:
        pseudo_only_path = db_path.with_name(f"{db_path.stem}_pseudo_only{db_path.suffix}")
        df_full = load_db(db_path)
        if "DOCUMENT" in df_full.columns:
            df_full = df_full.drop(columns=["DOCUMENT"])
        df_full.to_csv(pseudo_only_path, index=True)
        logger.info(f"Saved pseudo-only copy to {pseudo_only_path}.")
