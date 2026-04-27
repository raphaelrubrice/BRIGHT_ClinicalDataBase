"""Text normalisation utilities for clinical document extraction.

Provides two levels of normalisation:

- ``normalise_text(text)``, NFC normalisation, strip control characters,
  fix typographic quotes/dashes, ensure valid UTF-8.  Applied at document
  ingestion time so downstream processing always gets clean text.

- ``normalise(text)``, Accent-stripping + lowercase + whitespace collapse.
  Used *only* for regex matching; extracted values reference the NFC original.

Public API
----------
- ``normalise_text``
- ``normalise``
- ``expand_abbreviations``
- ``fuzzy_match``
"""

from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# Step 1: Document-level normalisation (NFC + sanitisation)
# ---------------------------------------------------------------------------

# Non-breaking and special spaces → regular space
_SPECIAL_SPACES = str.maketrans({
    "\u00a0": " ",   # NBSP
    "\u202f": " ",   # Narrow NBSP
    "\u2007": " ",   # Figure space
    "\u2009": " ",   # Thin space
    "\u200a": " ",   # Hair space
})

# Typographic quotes → ASCII equivalents
_TYPOGRAPHIC_QUOTES = str.maketrans({
    "\u2018": "'",   # Left single
    "\u2019": "'",   # Right single
    "\u201c": '"',   # Left double
    "\u201d": '"',   # Right double
    "\u00ab": '"',   # «
    "\u00bb": '"',   # »
})

# Em/en dash → hyphen
_DASHES = str.maketrans({
    "\u2013": "-",   # En dash
    "\u2014": "-",   # Em dash
})

# Zero-width and control characters to strip
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\ufeff]"
)

# Collapse multiple spaces (but preserve newlines)
_MULTI_SPACE_RE = re.compile(r" {2,}")


def normalise_text(text: str) -> str:
    """Apply document-level normalisation.

    Steps (applied in order):
    1. Unicode NFC normalisation
    2. Replace special spaces with regular space
    3. Replace typographic quotes with ASCII equivalents
    4. Replace em/en dashes with hyphens
    5. Strip zero-width and control characters
    6. Collapse multiple consecutive spaces
    7. Encode/decode UTF-8 as safety net

    Parameters
    ----------
    text : str
        Raw text extracted from a document.

    Returns
    -------
    str
        Cleaned, NFC-normalised text safe for downstream processing.
    """
    if not text:
        return text

    text = unicodedata.normalize("NFC", text)
    text = text.translate(_SPECIAL_SPACES)
    text = text.translate(_TYPOGRAPHIC_QUOTES)
    text = text.translate(_DASHES)
    text = _CONTROL_CHAR_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    return text


# ---------------------------------------------------------------------------
# Step 2: Accent-stripping normalisation (for regex matching)
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Strip accents, lowercase, and collapse whitespace.

    Used to create a normalised copy of text for pattern matching.
    Source span positions should reference the original (NFC) text.

    Parameters
    ----------
    text : str
        Input text (ideally already NFC-normalised).

    Returns
    -------
    str
        Lowercased, accent-free, single-spaced text.
    """
    if not text:
        return text

    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return _WHITESPACE_RE.sub(" ", stripped).strip().lower()


# ---------------------------------------------------------------------------
# Step 3: Abbreviation expansion (for pre-processing before matching)
# ---------------------------------------------------------------------------

_ABBREVIATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bTMZ\b", re.IGNORECASE), "temozolomide"),
    (re.compile(r"\bRT\b"), "radiotherapie"),  # case-sensitive, avoid "rt" in words
    (re.compile(r"\bGBM\b", re.IGNORECASE), "glioblastome"),
    (re.compile(r"\bGTR\b", re.IGNORECASE), "exerese complete"),
    (re.compile(r"\bSTR\b", re.IGNORECASE), "exerese partielle"),
    (re.compile(r"\bBST\b", re.IGNORECASE), "biopsie stereotaxique"),
    (re.compile(r"\bIK\b"), "indice de Karnofsky"),  # case-sensitive
    (re.compile(r"\bKPS\b", re.IGNORECASE), "indice de Karnofsky"),
    (re.compile(r"\bPEC\b"), "proliferation endotheliocapillaire"),
    (re.compile(r"\bHGG\b", re.IGNORECASE), "gliome de haut grade"),
    (re.compile(r"\bLGG\b", re.IGNORECASE), "gliome de bas grade"),
    (re.compile(r"\bRCP\b", re.IGNORECASE), "reunion de concertation pluridisciplinaire"),
    (re.compile(r"\bdMMR\b"), "deficit MMR"),
    (re.compile(r"\bpMMR\b"), "MMR proficient"),
    (re.compile(r"\bLOH\b", re.IGNORECASE), "perte d'heterozygotie"),
]


def expand_abbreviations(text: str) -> str:
    """Expand common clinical abbreviations in *text*.

    Returns a copy with abbreviations replaced by their full forms.
    The original text should be kept for span referencing.

    Parameters
    ----------
    text : str
        Section or document text.

    Returns
    -------
    str
        Text with abbreviations expanded.
    """
    for pattern, expansion in _ABBREVIATION_PATTERNS:
        text = pattern.sub(expansion, text)
    return text


# ---------------------------------------------------------------------------
# Step 4: Fuzzy matching for short tokens (Phase C1)
# ---------------------------------------------------------------------------

def fuzzy_match(
    token: str,
    vocab: list[str],
    threshold: int = 85,
) -> str | None:
    """Find the best fuzzy match for *token* in *vocab*.

    Uses ``rapidfuzz.fuzz.ratio`` for Levenshtein-based similarity.
    Only considers tokens of length 4–12 characters to avoid false
    positives on very short or very long strings.

    Parameters
    ----------
    token : str
        The input token to match (e.g. a gene name or drug name).
    vocab : list[str]
        The list of canonical vocabulary entries to match against.
    threshold : int
        Minimum similarity score (0–100) to accept a match.

    Returns
    -------
    str | None
        The best matching vocabulary entry, or ``None`` if no match
        meets the threshold or the token length is outside 4–12.
    """
    if not token or not vocab:
        return None
    if not (4 <= len(token) <= 12):
        return None

    from rapidfuzz import fuzz

    best: str | None = None
    best_score: float = 0.0
    for entry in vocab:
        score = fuzz.ratio(token.lower(), entry.lower())
        if score > best_score:
            best_score = score
            best = entry
    return best if best_score >= threshold else None
