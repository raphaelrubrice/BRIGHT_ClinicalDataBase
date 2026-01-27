from __future__ import annotations

import base64
import sqlite3
import secrets
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path


META_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def generate_salt(num_bytes: int = 32) -> str:
    """
    Generates a cryptographically secure salt.
    Returns a URL-safe base64 string.
    """
    raw = secrets.token_bytes(num_bytes)
    return base64.urlsafe_b64encode(raw).decode("ascii")

def get_or_create_salt(conn: sqlite3.Connection) -> str:
    """
    Ensures the DB has a persistent pseudonymization salt and returns it.
    This is atomic under SQLite when using a transaction.
    """
    conn.execute(META_TABLE_SQL)

    row = conn.execute(
        "SELECT value FROM meta WHERE key = ?",
        ("pseudonymization_salt",),
    ).fetchone()

    if row and row[0]:
        return row[0]

    salt = generate_salt()

    # Use INSERT OR IGNORE to be safe if two app instances race.
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value, created_at) VALUES (?, ?, ?)",
        ("pseudonymization_salt", salt, _utc_now_iso()),
    )

    # Re-read to ensure we return the persisted value even if another writer won.
    row2 = conn.execute(
        "SELECT value FROM meta WHERE key = ?",
        ("pseudonymization_salt",),
    ).fetchone()

    if not row2 or not row2[0]:
        raise RuntimeError("Failed to create or retrieve pseudonymization salt from DB.")

    return row2[0]



# ---------------------------------------------------------------------------
# CSV-backed DB salt storage (sidecar file)
# ---------------------------------------------------------------------------

SALT_SIDECAR_SUFFIX = ".pseudonym_salt"
SALT_SIDECAR_ENCODING = "utf-8"

def _salt_sidecar_path(db_path: str | Path) -> Path:
    """Return the sidecar file path used to persist the pseudonymization salt for a CSV DB."""
    p = Path(db_path)
    return p.with_suffix(p.suffix + SALT_SIDECAR_SUFFIX)

def get_or_create_salt_file(db_path: str | Path) -> str:
    """
    Get or create a persistent pseudonymization salt for a CSV database.

    Storage:
      - Sidecar file next to the CSV: <db>.csv.pseudonym_salt

    Returns:
      The salt as a URL-safe base64 string.
    """
    sidecar = _salt_sidecar_path(db_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)

    if sidecar.exists():
        salt = sidecar.read_text(encoding=SALT_SIDECAR_ENCODING).strip()
        if not salt:
            raise RuntimeError(f"Salt sidecar exists but is empty: {sidecar}")
        return salt

    salt = generate_salt()

    # Atomic write: write to temp file then replace
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp.write_text(salt + "\n", encoding=SALT_SIDECAR_ENCODING)
    tmp.replace(sidecar)

    return salt

import hashlib
from scipy.stats import hypergeom

class OptimizedOPE:
    def __init__(self, key, out_range_size=2**32):
        self.key = key.encode()
        self.out_range_size = out_range_size

    def _get_random_split(self, in_size, out_size, out_split_point, seed_context):
        """
        Returns 'x': the number of input items that fall into the lower half 
        of the output space.
        
        in_size (M): Total remaining input points (population).
        out_size (N): Total remaining output points.
        out_split_point (n): The size of the lower output chunk we are checking.
        """
        
        # 1. Deterministic Randomness
        # Create a seed specific to this exact step in the tree
        seed_str = f"{self.key}:{seed_context}"
        h = hashlib.sha256(seed_str.encode()).hexdigest()
        
        # Convert hash to a float between 0.0 and 1.0 (Uniform source)
        uniform_random = int(h, 16) / (2**256)
        
        # 2. Hypergeometric Sampling (Inverse Transform Sampling)
        # "ppf" is the Percent Point Function (Inverse CDF).
        # It maps our uniform random number to the Hypergeometric distribution.
        # Params: (Total Outputs, Total Inputs, Size of Lower Output Chunk)
        x = hypergeom.ppf(uniform_random, out_size, in_size, out_split_point)
        
        return int(x)

    def encrypt(self, value, in_size=3000):
        """
        Encrypts a value 'a' where 0 <= a < in_size.
        """
        # Start with full ranges
        in_low, in_high = 0, in_size
        out_low, out_high = 0, self.out_range_size
        
        # Context to ensure unique hash per recursion level
        depth = 0
        
        while out_high - out_low > 1:
            if in_high == in_low:
                # We narrowed down to one input, but we continue 
                # to narrow the output to maintain distribution properties
                pass 

            # 1. Pick the middle of the OUTPUT range
            out_mid = (out_low + out_high) // 2
            
            # Size of the lower output chunk
            current_out_lower_size = out_mid - out_low
            current_out_total = out_high - out_low
            current_in_total = in_high - in_low
            
            # 2. Determine how many input points (x) land in this lower chunk
            # using the Hypergeometric distribution.
            if current_in_total > 0:
                x = self._get_random_split(
                    current_in_total, 
                    current_out_total, 
                    current_out_lower_size, 
                    depth
                )
            else:
                x = 0 # No inputs left to distribute
            
            # The split point in the input domain relative to current window
            in_split = in_low + x
            
            # 3. Navigate the tree
            if value < in_split:
                # Go Left
                out_high = out_mid
                in_high = in_split
            else:
                # Go Right
                out_low = out_mid
                in_low = in_split
            
            depth += 1
            
            # Termination: If our input range is size 1 (just our target),
            # we technically found it
            if in_high - in_low == 1:
                 # In a real OPE, we might continue to define the bits,
                 # but essentially we have isolated the block.
                 pass

        return out_low
