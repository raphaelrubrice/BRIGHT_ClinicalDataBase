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
