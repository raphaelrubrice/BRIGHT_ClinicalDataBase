from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.database.security import get_or_create_salt_file

try:
    import portalocker
except ImportError:
    portalocker = None


DEFAULT_COLUMNS = ["PID", "SOURCE_FILE", "DOCUMENT", "PSEUDO", "ORDER"]

# Helpers
def _normalize_pid(pid) -> str:
    """
    Canonical PID representation. Prevents 1 vs 1.0 vs "1".
    """
    if pd.isna(pid):
        return ""
    s = str(pid).strip()
    # Collapse float-like integers: "1.0" -> "1"
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def _next_did(df: pd.DataFrame) -> int:
    if df.shape[0] == 0:
        return 0
    try:
        return int(pd.to_numeric(df.index).max()) + 1
    except Exception:
        return df.shape[0]


def _require_columns(df: pd.DataFrame, rows: pd.DataFrame) -> None:
    missing = set(df.columns) - set(rows.columns)
    if missing:
        raise ValueError(f"Missing required DB columns in new rows: {sorted(missing)}")

def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=True)
    tmp.replace(path)

# DB funcs
def init_db(path: str | Path, columns: Optional[list[str]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if columns is None:
        columns = DEFAULT_COLUMNS

    df = pd.DataFrame({col: [] for col in columns})
    df.index.name = "DID"
    _atomic_write_csv(df, path)
    # Ensure a persistent pseudonymization salt is created for this DB
    get_or_create_salt_file(path)
    return path


def load_db(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")

    df = pd.read_csv(path, index_col=0)
    df.index.name = "DID"

    # Normalize PID to string (prevents 1 vs 1.0 vs "1")
    if "PID" in df.columns:
        df["PID"] = df["PID"].apply(_normalize_pid)

    # ORDER numeric
    if "ORDER" in df.columns and len(df) > 0:
        df["ORDER"] = pd.to_numeric(df["ORDER"], errors="coerce").astype("Int64")

    return df

def save_db(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(df, path)


def append_rows_locked(db_path: str | Path, new_rows: pd.DataFrame) -> None:
    """
    Insert new rows with ORDER semantics (per PID) using optional lock + atomic write.
    new_rows must contain: PID, DOCUMENT, PSEUDO, ORDER (ORDER may be empty for brand-new PID).
    """
    db_path = Path(db_path)

    # Ensure DB has a persistent pseudonymization salt (created on first write if missing)
    get_or_create_salt_file(db_path)

    if portalocker is None:
        df = load_db(db_path)
        df2 = insert_documents_with_order(df, new_rows)
        save_db(df2, db_path)
        return

    lock_path = db_path.with_suffix(db_path.suffix + ".lock")
    with portalocker.Lock(str(lock_path), timeout=10):
        df = load_db(db_path)
        df2 = insert_documents_with_order(df, new_rows)
        save_db(df2, db_path)


def insert_documents_with_order(df: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Correct ORDER insertion without any post-hoc 'repair' pass:

    For each PID:
      - Define the current sequence deterministically as existing rows sorted by (ORDER, DID).
        (If the DB already has duplicates, this defines the baseline unambiguously.)
      - Normalize that baseline sequence to ORDER=1..N in-memory.
      - Insert new documents one by one into that sequence at the requested position:
          ORDER in [1, len(sequence)+1]
        If PID has no existing docs, ORDER is assigned sequentially starting at 1,
        unless the user provided explicit ORDER (optional decision; below we ignore provided ORDER for new PID).
      - After all inserts for that PID, write ORDER=1..N according to the final sequence.

    The resulting ORDER is unique and contiguous per PID by construction.
    The requested insertion position is preserved exactly.
    """
    _require_columns(df, new_rows)

    out = df.copy()
    out.index.name = "DID"

    # Normalize PID + ORDER types in both DB and input
    out["PID"] = out["PID"].apply(_normalize_pid)
    out["ORDER"] = pd.to_numeric(out["ORDER"], errors="coerce").astype("Int64")

    new_rows = new_rows.copy()
    new_rows["PID"] = new_rows["PID"].apply(_normalize_pid)
    new_rows["ORDER"] = pd.to_numeric(new_rows["ORDER"], errors="coerce").astype("Int64")

    # Preserve input order (important when multiple inserts target same ORDER)
    new_rows["_INPUT_POS"] = range(len(new_rows))

    next_did = _next_did(out)

    for pid, batch in new_rows.groupby("PID", sort=False):
        batch = batch.sort_values("_INPUT_POS")  # apply inserts in UI order

        existing = out.loc[out["PID"] == pid].copy()

        # Build baseline sequence (deterministic): sort by (ORDER, DID)
        # If ORDER is missing/NaN in existing, refuse: the baseline is ill-defined.
        if not existing.empty:
            if existing["ORDER"].isna().any():
                raise ValueError(
                    f"DB has missing ORDER values for PID={pid}. "
                    "Cannot safely perform ordered insertion until fixed."
                )

            existing = existing.sort_values(["ORDER"])
            # Tie-break duplicates by DID (index), for determinism
            existing = existing.sort_index(kind="stable") if existing.index.is_monotonic_increasing else existing.sort_index(kind="stable")
            # More explicit deterministic ordering:
            existing = existing.sort_values(["ORDER"], kind="stable")
            existing = existing.loc[existing.sort_values(["ORDER"], kind="stable").index]
            # Build the ordered DID list using (ORDER, DID)
            existing = existing.reset_index().sort_values(["ORDER", "DID"], kind="stable").set_index("DID")
            sequence = list(existing.index)
        else:
            sequence = []

        # In-memory normalization of baseline to 1..N (not a “repair pass” — it defines the baseline sequence)
        # We only do this for internal consistency before insertion.
        # We will write final ORDER values at the end from the final sequence.
        # (No DB write happens until the very end.)
        # Insert each new row
        if len(sequence) == 0:
            # New PID: assign ORDER sequentially in input order, ignore provided ORDER
            for i, (_, row) in enumerate(batch.iterrows(), start=1):
                did = next_did
                next_did += 1

                row_dict = {col: row[col] for col in out.columns}
                row_dict["PID"] = pid
                row_dict["ORDER"] = i

                out.loc[did, out.columns] = pd.Series(row_dict)
                sequence.append(did)

        else:
            # Existing PID: ORDER is required for each inserted doc
            for _, row in batch.iterrows():
                if pd.isna(row["ORDER"]):
                    raise ValueError(f"ORDER is required when PID already exists (PID={pid}).")

                requested = int(row["ORDER"])
                # Allow insertion at end: len(sequence)+1
                if not (1 <= requested <= len(sequence) + 1):
                    raise ValueError(
                        f"Invalid ORDER for PID={pid}: {requested}. "
                        f"Must be between 1 and {len(sequence) + 1}."
                    )

                did = next_did
                next_did += 1

                row_dict = {col: row[col] for col in out.columns}
                row_dict["PID"] = pid
                # Temporary; final ORDER assigned after sequence finalized
                row_dict["ORDER"] = pd.NA

                out.loc[did, out.columns] = pd.Series(row_dict)

                # Insert DID at the requested 1-based position
                sequence.insert(requested - 1, did)

        # Write final ORDER deterministically from final sequence
        for i, did in enumerate(sequence, start=1):
            out.at[did, "ORDER"] = i

    out.drop(columns=["_INPUT_POS"], errors="ignore", inplace=True)
    out.index.name = "DID"
    return out


