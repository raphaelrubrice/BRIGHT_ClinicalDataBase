from __future__ import annotations

from pathlib import Path
from typing import Optional

import re
import pandas as pd

from src.database.security import get_or_create_salt_file

try:
    import portalocker
except ImportError:
    portalocker = None


DEFAULT_COLUMNS = ["IPP", "SOURCE_FILE", "DOCUMENT", "PSEUDO", "ORDER"]

# Helpers
def _normalize_ipp(ipp) -> str:
    """
    Canonical IPP representation. Prevents 1 vs 1.0 vs "1".
    """
    if pd.isna(ipp):
        return ""
    s = str(ipp).strip()
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

    # Normalize IPP to string (prevents 1 vs 1.0 vs "1")
    if "IPP" in df.columns:
        df["IPP"] = df["IPP"].apply(_normalize_ipp)

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
    Insert new rows with ORDER semantics (per IPP) using optional lock + atomic write.
    new_rows must contain: IPP, DOCUMENT, PSEUDO, ORDER (ORDER may be empty for brand-new IPP).
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


# def insert_documents_with_order(df: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
#     """
#     Correct ORDER insertion without any post-hoc 'repair' pass:

#     For each IPP:
#       - Define the current sequence deterministically as existing rows sorted by (ORDER, DID).
#         (If the DB already has duplicates, this defines the baseline unambiguously.)
#       - Normalize that baseline sequence to ORDER=1..N in-memory.
#       - Insert new documents one by one into that sequence at the requested position:
#           ORDER in [1, len(sequence)+1]
#         If IPP has no existing docs, ORDER is assigned sequentially starting at 1,
#         unless the user provided explicit ORDER (optional decision; below we ignore provided ORDER for new IPP).
#       - After all inserts for that IPP, write ORDER=1..N according to the final sequence.

#     The resulting ORDER is unique and contiguous per IPP by construction.
#     The requested insertion position is preserved exactly.
#     """
#     _require_columns(df, new_rows)

#     out = df.copy()
#     out.index.name = "DID"

#     # Normalize IPP + ORDER types in both DB and input
#     out["IPP"] = out["IPP"].apply(_normalize_ipp)
#     out["ORDER"] = pd.to_numeric(out["ORDER"], errors="coerce").astype("Int64")

#     new_rows = new_rows.copy()
#     new_rows["IPP"] = new_rows["IPP"].apply(_normalize_ipp)
#     new_rows["ORDER"] = pd.to_numeric(new_rows["ORDER"], errors="coerce").astype("Int64")

#     # Preserve input order (important when multiple inserts target same ORDER)
#     new_rows["_INPUT_POS"] = range(len(new_rows))

#     next_did = _next_did(out)

#     for ipp, batch in new_rows.groupby("IPP", sort=False):
#         batch = batch.sort_values("_INPUT_POS")  # apply inserts in UI order

#         existing = out.loc[out["IPP"] == ipp].copy()

#         # Build baseline sequence (deterministic): sort by (ORDER, DID)
#         # If ORDER is missing/NaN in existing, refuse: the baseline is ill-defined.
#         if not existing.empty:
#             if existing["ORDER"].isna().any():
#                 raise ValueError(
#                     f"DB has missing ORDER values for IPP={ipp}. "
#                     "Cannot safely perform ordered insertion until fixed."
#                 )

#             existing = existing.sort_values(["ORDER"])
#             # Tie-break duplicates by DID (index), for determinism
#             existing = existing.sort_index(kind="stable") if existing.index.is_monotonic_increasing else existing.sort_index(kind="stable")
#             # More explicit deterministic ordering:
#             existing = existing.sort_values(["ORDER"], kind="stable")
#             existing = existing.loc[existing.sort_values(["ORDER"], kind="stable").index]
#             # Build the ordered DID list using (ORDER, DID)
#             existing = existing.reset_index().sort_values(["ORDER", "DID"], kind="stable").set_index("DID")
#             sequence = list(existing.index)
#         else:
#             sequence = []

#         # In-memory normalization of baseline to 1..N (not a “repair pass” — it defines the baseline sequence)
#         # We only do this for internal consistency before insertion.
#         # We will write final ORDER values at the end from the final sequence.
#         # (No DB write happens until the very end.)
#         # Insert each new row
#         if len(sequence) == 0:
#             # New IPP: assign ORDER sequentially in input order, ignore provided ORDER
#             for i, (_, row) in enumerate(batch.iterrows(), start=1):
#                 did = next_did
#                 next_did += 1

#                 row_dict = {col: row[col] for col in out.columns}
#                 row_dict["IPP"] = ipp
#                 row_dict["ORDER"] = i

#                 out.loc[did, out.columns] = pd.Series(row_dict)
#                 sequence.append(did)

#         else:
#             # Existing IPP: ORDER is required for each inserted doc
#             for _, row in batch.iterrows():
#                 if pd.isna(row["ORDER"]):
#                     raise ValueError(f"ORDER is required when IPP already exists (IPP={ipp}).")

#                 requested = int(row["ORDER"])
#                 # Allow insertion at end: len(sequence)+1
#                 if not (1 <= requested <= len(sequence) + 1):
#                     raise ValueError(
#                         f"Invalid ORDER for IPP={ipp}: {requested}. "
#                         f"Must be between 1 and {len(sequence) + 1}."
#                     )

#                 did = next_did
#                 next_did += 1

#                 row_dict = {col: row[col] for col in out.columns}
#                 row_dict["IPP"] = ipp
#                 # Temporary; final ORDER assigned after sequence finalized
#                 row_dict["ORDER"] = pd.NA

#                 out.loc[did, out.columns] = pd.Series(row_dict)

#                 # Insert DID at the requested 1-based position
#                 sequence.insert(requested - 1, did)

#         # Write final ORDER deterministically from final sequence
#         for i, did in enumerate(sequence, start=1):
#             out.at[did, "ORDER"] = i

#     out.drop(columns=["_INPUT_POS"], errors="ignore", inplace=True)
#     out.index.name = "DID"
#     return out

def extract_consult_date_num(text):
    matches = re.search(r"(c|C)onsultation.du.[0-9]{2}\/[0-9]{2}\/[0-9]{4}", text, re.MULTILINE)
    if matches is None:
        matches = re.search(r"(c|C)onsultation.du.[0-9]{4}\/[0-9]{2}\/[0-9]{2}", text, re.MULTILINE)
        if matches is None:
            raise ValueError("Unable to find a date in supported formats (DD/MM/YYYY or MM/DD/YYYY or YYYY/DD/MM or YYYY/MM/DD) for at least one row.")
    date = matches.group(0)[16:] # assuming first one is the last 
    num_list = sorted([int(num) for num in date.split("/")], reverse=True)

    if num_list[1] > 12:
        out_list = [str(num_list[0]), str(num_list[2]), str(num_list[1])]
        out_list = [el if len(el) >= 2 else '0' + el for el in out_list]
        return int(''.join(out_list))
    out_list = [str(num_list[0]), str(num_list[1]), str(num_list[2])]
    out_list = [el if len(el) >= 2 else '0' + el for el in out_list]
    return int(''.join(out_list))

# def extract_consult_date(text):
#     matches = re.search(r"(c|C)onsultation.du.[0-9]{2}\/[0-9]{2}\/[0-9]{4}", text, re.MULTILINE)
#     if matches is None:
#         matches = re.search(r"(c|C)onsultation.du.[0-9]{4}\/[0-9]{2}\/[0-9]{2}", text, re.MULTILINE)
#         if matches is None:
#             raise ValueError("Unable to find a date in supported formats (DD/MM/YYYY or MM/DD/YYYY or YYYY/DD/MM or YYYY/MM/DD) for at least one row.")
#     date = matches.group(0)[16:] # assuming first one is the last 
#     num_list = sorted([int(num) for num in date.split("/")], reverse=True)
#     return '/'.join([str(el) for el in num_list])

# def extract_consult_date(text):
#     # Mapping French month names to numbers
#     # Keys should be lowercase to handle case-insensitivity
#     months_map = {
#         "janvier": 1, "février": 2, "mars": 3, "avril": 4,
#         "mai": 5, "juin": 6, "juillet": 7, "août": 8,
#         "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
#         # Handling potential encoding/spelling variations
#         "fevrier": 2, "aout": 8, "decembre": 12
#     }

#     # The exact regex pattern provided
#     pattern = r"(((c|C)onsultation.du.)|(Paris, le ))(([0-9]{2}\/[0-9]{2}\/[0-9]{4})|([0-9]{2} [a-zA-Z]+ [0-9]{4}))"
    
#     matches = re.search(pattern, text, re.MULTILINE)

#     if matches is None:
#         raise ValueError("Unable to find a date in supported formats for at least one row.")

#     # Group 5 contains the actual date string (ignoring the "Consultation..." or "Paris..." prefix)
#     # It will contain either "DD/MM/YYYY" or "DD Month YYYY"
#     raw_date = matches.group(5)
    
#     num_list = []

#     # Logic to handle "DD/MM/YYYY"
#     if "/" in raw_date:
#         num_list = [int(num) for num in raw_date.split("/")]

#     # Logic to handle "DD Month YYYY" (e.g., "17 Janvier 2023")
#     else:
#         parts = raw_date.split() # Splits by space
#         day = int(parts[0])
#         year = int(parts[2])
#         month_str = parts[1].lower()
        
#         if month_str in months_map:
#             month = months_map[month_str]
#         else:
#             raise ValueError(f"Could not map month name '{parts[1]}' to a number.")
            
#         num_list = [day, month, year]

#     # Preserving your original return logic: Sort descending (Year/Max/Min)
#     num_list = sorted(num_list, reverse=True)
    
#     return '/'.join([str(el) for el in num_list])

def extract_consult_date_num(text):
    # Mapping French month names to numbers
    months_map = {
        "janvier": 1, "février": 2, "mars": 3, "avril": 4,
        "mai": 5, "juin": 6, "juillet": 7, "août": 8,
        "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
        # Handling potential encoding/spelling variations
        "fevrier": 2, "aout": 8, "decembre": 12
    }

    # New regex pattern supporting "Consultation du" or "Paris, le"
    # Group 5 contains the date string
    pattern = r"(((c|C)onsultation.du.)|(Paris, le ))(([0-9]{2}\/[0-9]{2}\/[0-9]{4})|([0-9]{2} [a-zA-Z]+ [0-9]{4}))"
    
    matches = re.search(pattern, text, re.MULTILINE)

    if matches is None:
        raise ValueError("Unable to find a date in supported formats for at least one row.")

    raw_date = matches.group(5)
    num_list = []

    # Logic to handle "DD/MM/YYYY"
    if "/" in raw_date:
        num_list = [int(num) for num in raw_date.split("/")]

    # Logic to handle "DD Month YYYY" (e.g., "17 Janvier 2023")
    else:
        parts = raw_date.split() # Splits by space
        day = int(parts[0])
        year = int(parts[2])
        month_str = parts[1].lower()
        
        if month_str in months_map:
            month = months_map[month_str]
        else:
            raise ValueError(f"Could not map month name '{parts[1]}' to a number.")
            
        num_list = [day, month, year]

    # Sort descending (Year, Max(M,D), Min(M,D))
    num_list = sorted(num_list, reverse=True)

    # Original logic to determine Day/Month order and format as YYYYMMDD
    # If the second largest number is > 12, it must be the Day -> [Year, Day, Month]
    # We want output [Year, Month, Day]
    if num_list[1] > 12:
        out_list = [str(num_list[0]), str(num_list[2]), str(num_list[1])]
    else:
        # Otherwise assume [Year, Month, Day]
        out_list = [str(num_list[0]), str(num_list[1]), str(num_list[2])]

    # Pad with leading zeros where necessary
    out_list = [el if len(el) >= 2 else '0' + el for el in out_list]
    
    # Return integer
    return int(''.join(out_list))

def insert_documents_with_order(df: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """
    1) Extract which IPPs are already in db and those which are new but with more than one line
    2) Add new rows to the DB
    3) For identified rows in 1), fetch all documents and do for each IPP:
        - extract consultation date
        - rank DIDs based on dates
        - Assign updated ORDER field based on rank
    """
    base_unique_ipp = set(df["IPP"].unique())
    newrows_unique_ipp = set(new_rows["IPP"].unique())

    common_ipp = base_unique_ipp.intersection(newrows_unique_ipp)
    truly_new_ipp = newrows_unique_ipp - common_ipp

    if set(df.columns) == set(new_rows.columns):
        index_list = list(df.index)
        new_idx = index_list + [max(index_list, default=0)+1+i for i in range(new_rows.shape[0])]
        concat_db = pd.concat([df, new_rows], axis=0)
        concat_db.index = new_idx
        

        if "CONSULT_DATE" not in concat_db.columns:
            concat_db["CONSULT_DATE_NUM"] = concat_db["DOCUMENT"].apply(extract_consult_date_num)
            concat_db["CONSULT_DATE"] = concat_db["DOCUMENT"].apply(extract_consult_date)
    else:
        raise IndexError(f"Columns do not match, current = {df.columns}, new rows = {new_rows.columns}")
    
    # no need to check order if its the only row with that IPP
    ipp_to_check = common_ipp.union(truly_new_ipp)
    ipp_to_check = [ipp for ipp in ipp_to_check 
                    if len(concat_db[concat_db["IPP"] == ipp]) > 1]

    # Inspect order
    for ipp in ipp_to_check:
        mask = concat_db["IPP"] == ipp
        date_col = concat_db[mask]["CONSULT_DATE_NUM"].to_list()
        ordered_dates = sorted(date_col)
        new_order = [int(ordered_dates.index(row_date))+1 for row_date in date_col]
        concat_db.loc[mask, "ORDER"] = new_order
    
    concat_db.index.name = "DID"
    return concat_db

def extract_IPP_from_path(path):
    file_name = Path(path).name if isinstance(str, path) else path.name
    matches = re.search(r"8[0-9]{9}", file_name)
    if matches is None:
        raise ValueError(f"No IPP found in file {path}. Must contain a 10 digit id starting by 8.")
    return int(matches.group(0))

def extract_IPP_from_document(text):
    """
    Retrieves the IPP from the INS/NIR line.
    """
    matches = re.search(r" 8[0-9]{9}", text)
    if matches is None:
        raise ValueError(f"No IPP found in document.")
    return int(matches.group(0)[2:-2])

def ensure_correct_IPP(df):
    """
    Enforces that the IPP column match what is in the document.
    """
    df["IPP"] = df["DOCUMENT"].apply(extract_IPP_from_document)
    return df

def extract_ORDER_from_path(path):
    file_name = Path(path).name if isinstance(str, path) else path.name
    if "_cs.pdf" in file_name:
        return 2
    else:
        return 1
    
if __name__ == "__main__":
    test_df = test_df = pd.DataFrame({"IPP": [0,0,1], "ORDER": [2,1,1], "DOCUMENT":["""Références : ALE/ALE
Compte-Rendu de Consultation du 01/12/2025
Madame LAURENGE ep. LEPRINCE Alice, née le 18/05/1989, âgée de 36 ans, a été vue en
consultation.""", """Références : ALE/ALE
Compte-Rendu de Consultation du 2024/12/20
Madame LAURENGE ep. LEPRINCE Alice, née le 18/05/1989, âgée de 36 ans, a été vue en
consultation.""", """Références : ALE/ALE
Compte-Rendu de consultation du 2025/31/11
Madame LAURENGE ep. LEPRINCE Alice, née le 18/05/1989, âgée de 36 ans, a été vue en
consultation."""]})
    new_df = pd.DataFrame({"IPP": [0,1,1,2], "ORDER": [1,1,1,1], "DOCUMENT":["""Références : ALE/ALE
Compte-Rendu de Consultation du 31/12/2024
Madame LAURENGE ep. LEPRINCE Alice, née le 18/05/1989, âgée de 36 ans, a été vue en
consultation.""", """Références : ALE/ALE
Compte-Rendu de Consultation du 2012/12/20
Madame LAURENGE ep. LEPRINCE Alice, née le 18/05/1989, âgée de 36 ans, a été vue en
consultation.""", """Références : ALE/ALE
Compte-Rendu de consultation du 2025/31/12
Madame LAURENGE ep. LEPRINCE Alice, née le 18/05/1989, âgée de 36 ans, a été vue en
consultation.""","""Références : ALE/ALE
Compte-Rendu de consultation du 2004/01/12
Madame LAURENGE ep. LEPRINCE Alice, née le 18/05/1989, âgée de 36 ans, a été vue en
consultation."""]})

    print("DB BEFORE\n", test_df)
    new_db = insert_documents_with_order(test_df, new_df)
    print("DB AFTER\n", new_db)
    
    
