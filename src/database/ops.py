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

def extract_consult_date(text, return_num=False):
    # Mapping French month names to numbers
    months_map = {
        "janvier": 1, "février": 2, "mars": 3, "avril": 4,
        "mai": 5, "juin": 6, "juillet": 7, "août": 8,
        "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
        # Handling potential encoding/spelling variations
        "fevrier": 2, "aout": 8, "decembre": 12
    }

    # UPDATED REGEX PATTERN
    # The (?i:) flag at the start handles case-insensitivity for the prefix
    pattern = r"(?i:((consultation.+du(|.))|(Paris, le )))(((([0-9]{4})|([0-9]{2}))\/[0-9]{2}\/(([0-9]{4})|([0-9]{2})))|([0-9]{2} \D+ [0-9]{4}))"
    
    matches = re.search(pattern, text, re.MULTILINE)

    if matches is None:
        raise ValueError("Unable to find a date in supported formats for at least one row.")

    # UDPATED GROUP INDEX
    # Old regex date was group 5. In this new regex structure:
    # Group 1: ((consultation.du.)|(Paris, le ))
    # Group 2: (consultation.du.)
    # Group 3: (|.)
    # Group 4: (Paris, le )
    # Group 5: The date string
    raw_date = matches.group(5)

    num_list = []

    # Logic to handle "DD/MM/YYYY" or similar numeric formats
    if "/" in raw_date:
        num_list = [int(num) for num in raw_date.split("/")]
    
    # Logic to handle "DD Month YYYY" (e.g., "17 Janvier 2023")
    else:
        parts = raw_date.split()  # Splits by space
        day = int(parts[0])
        year = int(parts[2])
        month_str = parts[1].lower()
        
        if month_str in months_map:
            month = months_map[month_str]
        else:
            raise ValueError(f"Could not map month name '{parts[1]}' to a number.")
            
        num_list = [day, month, year]

    # Sort descending (Year, Max(M,D), Min(M,D))
    # This ensures Year is index 0.
    num_list = sorted(num_list, reverse=True)

    # Determine Day/Month order and format as YYYYMMDD string components
    # If the second largest number is > 12, it must be the Day -> [Year, Day, Month]
    # We want output format [Year, Month, Day]
    if num_list[1] > 12:
        out_list = [str(num_list[0]), str(num_list[2]), str(num_list[1])]
    else:
        # Otherwise assume [Year, Month, Day]
        out_list = [str(num_list[0]), str(num_list[1]), str(num_list[2])]

    # Pad with leading zeros where necessary for the numeric return
    out_list = [el if len(el) >= 2 else '0' + el for el in out_list]
    
    # Return integer format (YYYYMMDD)
    if return_num:
        return int(''.join(out_list))

    # Return string format (Sorted: YYYY/MM/DD or YYYY/DD/MM based on the sort logic above)
    return '/'.join([str(el) for el in num_list])

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

    if "CONSULT_DATE" not in df.columns:
        df = df.copy()  # Don't modify the original
        df["CONSULT_DATE_NUM"] = df["DOCUMENT"].apply(extract_consult_date, return_num=True)
        df["CONSULT_DATE"] = df["DOCUMENT"].apply(extract_consult_date, return_num=False)

    # FIXED: Add CONSULT_DATE columns to new_rows BEFORE checking column equality
    # This ensures new_rows always has the same columns as df after the first chunk
    if "CONSULT_DATE" not in new_rows.columns:
        new_rows = new_rows.copy()  # Don't modify the original
        new_rows["CONSULT_DATE_NUM"] = new_rows["DOCUMENT"].apply(extract_consult_date, return_num=True)
        new_rows["CONSULT_DATE"] = new_rows["DOCUMENT"].apply(extract_consult_date, return_num=False)
    
    # Now check column equality
    if set(df.columns) == set(new_rows.columns):
        index_list = list(df.index)
        new_idx = index_list + [max(index_list, default=0)+1+i for i in range(new_rows.shape[0])]
        if df.shape[0] >= 1:
            concat_db = pd.concat([df, new_rows], axis=0)
        else:
            concat_db = new_rows
        concat_db.index = new_idx
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
    matches = re.search(r" 8[0-9]{9} ", text)
    if matches is None:
        raise ValueError(f"No IPP found in document.")
    return int(matches.group(0)[1:-1])

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
Compte-Rendu de CONSULTATION ONCO-SOMMEILDU01/12/2025
Madame LAURENGE ep. LEPRINCE Alice, née le 18/05/1989, âgée de 36 ans, a été vue en
consultation.""", """Références : ALE/ALE
Compte-Rendu de Consultation du 2024/12/20
Madame LAURENGE ep. LEPRINCE Alice, née le 18/05/1989, âgée de 36 ans, a été vue en
consultation.""", """Références : ALE/ALE
Paris, le 11 Décembre 2025
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
    
    
