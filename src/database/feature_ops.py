"""Feature extraction integration with the CSV database.

Bridges the extraction pipeline (``src.extraction.pipeline``) with the
existing database operations (``src.database.ops``).  Provides helpers to:

- Run extraction on a document and store results in the database.
- Add the 102 feature columns to the database schema.
- Convert an ``ExtractionResult`` into a flat dict of column values.

Public API
----------
- ``FEATURE_COLUMNS``            – List of all 102 feature column names.
- ``EXTENDED_COLUMNS``           – ``DEFAULT_COLUMNS`` + feature columns.
- ``extraction_result_to_row()`` – Convert ExtractionResult to a flat row dict.
- ``extract_and_store()``        – Extract features and append to the DB.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.database.ops import DEFAULT_COLUMNS, load_db, save_db, init_db
from src.extraction.pipeline import ExtractionPipeline
from src.extraction.provenance import ExtractionResult
from src.extraction.schema import (
    ALL_BIO_FIELD_NAMES,
    ALL_CLINIQUE_FIELD_NAMES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extended column schema
# ---------------------------------------------------------------------------

# All 102 feature field names as database columns
FEATURE_COLUMNS: list[str] = sorted(
    set(ALL_BIO_FIELD_NAMES + ALL_CLINIQUE_FIELD_NAMES)
)

# Metadata columns added alongside features
_META_COLUMNS: list[str] = [
    "DOC_TYPE",            # Classified document type
    "DOC_DATE",            # Extracted document date
    "EXTRACTION_TIER1",    # Count of Tier 1 extractions
    "EXTRACTION_TIER2",    # Count of Tier 2 extractions
    "EXTRACTION_FLAGGED",  # Count of flagged fields
    "EXTRACTION_TIME_MS",  # Total extraction time
]

EXTENDED_COLUMNS: list[str] = (
    DEFAULT_COLUMNS + _META_COLUMNS + FEATURE_COLUMNS
)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def extraction_result_to_row(result: ExtractionResult) -> dict:
    """Convert an ``ExtractionResult`` into a flat dict for a DataFrame row.

    Returns a dict where:
    - Standard DB columns (IPP, etc.) are set to empty strings.
    - Feature columns contain the extracted value (or ``None``).
    - Metadata columns contain extraction statistics.
    """
    row: dict = {}

    # Metadata columns
    row["DOC_TYPE"] = result.document_type
    row["DOC_DATE"] = result.document_date or ""
    row["EXTRACTION_TIER1"] = result.tier1_count
    row["EXTRACTION_TIER2"] = result.tier2_count
    row["EXTRACTION_FLAGGED"] = len(result.flagged_for_review)
    row["EXTRACTION_TIME_MS"] = round(result.total_extraction_time_ms, 1)

    # Feature columns
    for fname in FEATURE_COLUMNS:
        ev = result.features.get(fname)
        if ev is not None and ev.value is not None:
            row[fname] = ev.value
        else:
            row[fname] = None

    return row


# ---------------------------------------------------------------------------
# Database integration
# ---------------------------------------------------------------------------

def init_extended_db(path: str | Path) -> Path:
    """Initialise a database with the extended column schema.

    Creates a new CSV database with the standard columns plus all 102
    feature columns and extraction metadata columns.
    """
    return init_db(path, columns=EXTENDED_COLUMNS)


def extend_existing_db(db_path: str | Path) -> pd.DataFrame:
    """Add feature columns to an existing database.

    Loads the database, adds any missing feature/metadata columns
    (with NaN), and saves it back.

    Returns the updated DataFrame.
    """
    df = load_db(db_path)

    for col in _META_COLUMNS + FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    save_db(df, db_path)
    logger.info(
        "Extended database at '%s' with %d new columns.",
        db_path,
        len(_META_COLUMNS) + len(FEATURE_COLUMNS),
    )
    return df


def extract_and_store(
    db_path: str | Path,
    text: str,
    document_id: str = "",
    patient_id: str = "",
    pipeline: Optional[ExtractionPipeline] = None,
    use_llm: bool = True,
) -> ExtractionResult:
    """Run extraction on a document and store results in the database.

    Parameters
    ----------
    db_path : str or Path
        Path to the CSV database.
    text : str
        The full document text.
    document_id : str
        Unique document identifier.
    patient_id : str
        Patient identifier.
    pipeline : ExtractionPipeline, optional
        Pre-configured pipeline instance. If ``None``, a new one is
        created with default settings.
    use_llm : bool
        Whether to enable LLM extraction (passed to pipeline if
        creating a new one).

    Returns
    -------
    ExtractionResult
        The extraction result (also stored in the database).
    """
    if pipeline is None:
        pipeline = ExtractionPipeline(use_llm=use_llm)

    # Run extraction
    result = pipeline.extract_document(
        text=text,
        document_id=document_id,
        patient_id=patient_id,
    )

    # Convert to row
    row_data = extraction_result_to_row(result)

    # Load existing DB and extend if needed
    db_path = Path(db_path)
    if not db_path.exists():
        init_extended_db(db_path)

    df = load_db(db_path)

    # Add missing columns
    for col in _META_COLUMNS + FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Build the new row
    new_row = {col: row_data.get(col) for col in df.columns if col in row_data}
    new_row["IPP"] = patient_id
    new_row["DOCUMENT"] = text[:500]  # Store truncated text reference
    new_row["PSEUDO"] = ""
    new_row["SOURCE_FILE"] = document_id
    new_row["ORDER"] = 1

    # Append
    new_idx = df.index.max() + 1 if len(df) > 0 else 0
    new_df = pd.DataFrame([new_row], index=[new_idx])
    new_df.index.name = "DID"

    df = pd.concat([df, new_df])
    df.index.name = "DID"
    save_db(df, db_path)

    logger.info(
        "Stored extraction result for document '%s' (patient '%s') "
        "in database '%s'.",
        document_id,
        patient_id,
        db_path,
    )

    return result
