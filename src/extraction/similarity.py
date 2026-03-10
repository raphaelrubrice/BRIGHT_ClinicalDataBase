"""Semantic similarity matching for categorical field values.

Provides a four-tier cascade to map extracted text spans to the closest
controlled vocabulary option:

1. Exact match (case-insensitive, accent-normalized)
2. Normalisation map lookup (reuse validation tables)
3. Fuzzy match (rapidfuzz Levenshtein)
4. spaCy vector similarity (fr_core_news_lg)

Falls back to ``"NA"`` when no tier produces a match.

Public API
----------
- ``match_to_vocab()`` — Main entry point for span → vocab matching.
"""

from __future__ import annotations

import logging
from typing import Optional

from .text_normalisation import fuzzy_match, normalise

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded spaCy model
# ---------------------------------------------------------------------------

_spacy_nlp = None


def _ensure_spacy():
    """Load ``fr_core_news_lg`` on first call."""
    global _spacy_nlp
    if _spacy_nlp is not None:
        return
    try:
        import spacy
        _spacy_nlp = spacy.load("fr_core_news_lg")
        logger.info("Loaded spaCy model fr_core_news_lg for similarity matching.")
    except (ImportError, OSError) as exc:
        logger.warning(
            "spaCy model fr_core_news_lg not available (%s). "
            "Tier 4 (vector similarity) will be skipped.",
            exc,
        )
        _spacy_nlp = False  # sentinel: tried and failed


# ---------------------------------------------------------------------------
# Normalisation map import (lazy to avoid circular imports)
# ---------------------------------------------------------------------------

_norm_map: dict[str, str] | None = None


def _get_norm_map() -> dict[str, str]:
    global _norm_map
    if _norm_map is None:
        from .validation import _NORMALISATION_MAP
        _norm_map = _NORMALISATION_MAP
    return _norm_map


# ---------------------------------------------------------------------------
# Main matching function
# ---------------------------------------------------------------------------

def match_to_vocab(
    span_text: str,
    allowed_values: set[str],
    field_name: str = "",
) -> tuple[str, float]:
    """Match *span_text* to the closest option in *allowed_values*.

    Returns ``(matched_value, confidence_score)`` where confidence
    decreases with each successive tier.

    Parameters
    ----------
    span_text : str
        The raw extracted text span.
    allowed_values : set[str]
        The controlled vocabulary options (may include ``"NA"``).
    field_name : str
        Optional field name for logging.

    Returns
    -------
    tuple[str, float]
        ``(matched_value, confidence)`` — ``"NA"`` with ``0.0`` if
        no tier matched.
    """
    if not span_text or not span_text.strip():
        return ("NA", 0.0)

    # Candidates excluding "NA" (we only want to match real options)
    candidates = {v for v in allowed_values if v != "NA"} if allowed_values else set()
    if not candidates:
        return (span_text, 0.5)

    span_stripped = span_text.strip()
    span_lower = span_stripped.lower()
    span_norm = normalise(span_stripped)

    # ------------------------------------------------------------------
    # Tier 1: Exact match (case-insensitive, accent-normalized)
    # ------------------------------------------------------------------
    for candidate in candidates:
        if span_lower == candidate.lower() or span_norm == normalise(candidate):
            logger.debug(
                "Field '%s': Tier 1 exact match '%s' -> '%s'",
                field_name, span_stripped, candidate,
            )
            return (candidate, 1.0)

    # ------------------------------------------------------------------
    # Tier 2: Normalisation map lookup
    # ------------------------------------------------------------------
    norm_map = _get_norm_map()
    normalised = norm_map.get(span_lower)
    if normalised is not None:
        # Check if normalised value is in candidates
        for candidate in candidates:
            if normalised.lower() == candidate.lower():
                logger.debug(
                    "Field '%s': Tier 2 norm map '%s' -> '%s' -> '%s'",
                    field_name, span_stripped, normalised, candidate,
                )
                return (candidate, 0.95)

    # ------------------------------------------------------------------
    # Tier 3: Fuzzy match (rapidfuzz)
    # ------------------------------------------------------------------
    best_fuzzy: Optional[str] = None
    best_fuzzy_score: float = 0.0
    for candidate in candidates:
        result = fuzzy_match(span_stripped, [candidate], threshold=70)
        if result is not None:
            try:
                from rapidfuzz import fuzz
                score = fuzz.ratio(span_lower, candidate.lower()) / 100.0
            except ImportError:
                score = 0.85
            if score > best_fuzzy_score:
                best_fuzzy_score = score
                best_fuzzy = candidate

    if best_fuzzy is not None and best_fuzzy_score >= 0.70:
        logger.debug(
            "Field '%s': Tier 3 fuzzy match '%s' -> '%s' (score=%.2f)",
            field_name, span_stripped, best_fuzzy, best_fuzzy_score,
        )
        return (best_fuzzy, best_fuzzy_score)

    # ------------------------------------------------------------------
    # Tier 4: spaCy vector similarity (no minimum threshold)
    # ------------------------------------------------------------------
    _ensure_spacy()
    if _spacy_nlp and _spacy_nlp is not False:
        span_doc = _spacy_nlp(span_stripped)
        best_sim_candidate: Optional[str] = None
        best_sim_score: float = -1.0
        for candidate in candidates:
            cand_doc = _spacy_nlp(candidate)
            sim = span_doc.similarity(cand_doc)
            if sim > best_sim_score:
                best_sim_score = sim
                best_sim_candidate = candidate

        if best_sim_candidate is not None:
            logger.debug(
                "Field '%s': Tier 4 spaCy similarity '%s' -> '%s' (sim=%.3f)",
                field_name, span_stripped, best_sim_candidate, best_sim_score,
            )
            return (best_sim_candidate, round(best_sim_score, 4))

    # ------------------------------------------------------------------
    # Tier 5: No match — return "NA"
    # ------------------------------------------------------------------
    logger.info(
        "Field '%s': no match for '%s' in %s. Returning 'NA'.",
        field_name, span_stripped, candidates,
    )
    return ("NA", 0.0)
