"""Semantic similarity matching for categorical field values.

Provides a four-tier cascade to map extracted text spans to the closest
controlled vocabulary option:

1. Exact match (case-insensitive, accent-normalized)
2. Normalisation map lookup (reuse validation tables)
3. Fuzzy match (rapidfuzz Levenshtein)
4. spaCy Vector similarity (routed by language detection)

Falls back to ``"NA"`` when no tier produces a match.

Public API
----------
- ``match_to_vocab()`` — Main entry point for span → vocab matching.
"""

from __future__ import annotations

import logging
from typing import Optional

# Assumption: text_normalisation.py exists in the same package
from .text_normalisation import fuzzy_match, normalise
from .schema import vocab_has_autre

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded spaCy models (Large models for better clinical word coverage)
# ---------------------------------------------------------------------------

_models: dict[str, any] = {}


def _get_nlp_for_lang(text: str):
    """Detect language and return the corresponding spaCy model.
    
    Defaults to 'en_core_web_lg' if detection fails or language is unsupported.
    For clinical words, 'lg' models are preferred over 'md' or 'sm' because 
    they contain significantly more unique vectors.
    """
    global _models
    
    lang = "en"  # Default
    try:
        from langdetect import detect
        lang = detect(text)
    except Exception as e:
        logger.warning("Language detection failed, defaulting to English: %s", e)

    # Map detected language to specific spaCy models
    model_map = {
        "fr": "fr_core_news_lg",
        "en": "en_core_web_lg"
    }
    
    target_model = model_map.get(lang, "en_core_web_lg")

    if target_model not in _models:
        try:
            import spacy
            logger.info("Loading spaCy model: %s", target_model)
            _models[target_model] = spacy.load(target_model)
        except (ImportError, OSError) as exc:
            logger.warning(
                "spaCy model %s not available (%s). Tier 4 will be skipped.",
                target_model, exc
            )
            _models[target_model] = None
            
    return _models.get(target_model)


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

    # Candidates excluding "NA" and "autre" (we only want to match real options)
    candidates = {v for v in allowed_values if v not in ("NA", "autre")} if allowed_values else set()

    # Track best score across all tiers for "autre" fallback decision
    _best_tier_score: float = 0.0
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
    _best_tier_score = max(_best_tier_score, best_fuzzy_score)

    # ------------------------------------------------------------------
    # Tier 4: Language-aware spaCy similarity
    # ------------------------------------------------------------------
    nlp = _get_nlp_for_lang(span_stripped)
    
    if nlp:
        span_doc = nlp(span_stripped)
        best_sim_candidate: Optional[str] = None
        best_sim_score: float = -1.0
        
        for candidate in candidates:
            cand_doc = nlp(candidate)
            if span_doc.has_vector and cand_doc.has_vector:
                sim = span_doc.similarity(cand_doc)
            else:
                sim = 0.0 
                
            if sim > best_sim_score:
                best_sim_score = sim
                best_sim_candidate = candidate

        if best_sim_candidate is not None and best_sim_score > 0.45:
            logger.debug(
                "Field '%s': Tier 4 Vector similarity '%s' -> '%s' (sim=%.3f)",
                field_name, span_stripped, best_sim_candidate, best_sim_score,
            )
            return (best_sim_candidate, round(best_sim_score, 4))
        _best_tier_score = max(_best_tier_score, best_sim_score)

    # ------------------------------------------------------------------
    # Tier 5: "autre" fallback — accept raw value as free text
    # ------------------------------------------------------------------
    # If the vocab has an "autre" category and no tier produced a decent
    # match (best < 0.45), the extracted value is genuinely novel —
    # accept it as-is with moderate confidence.
    if vocab_has_autre(allowed_values) and _best_tier_score < 0.45:
        logger.info(
            "Field '%s': no vocab match for '%s' (best_score=%.2f) but "
            "'autre' category present — accepting raw value.",
            field_name, span_stripped, _best_tier_score,
        )
        return (span_stripped, 0.6)

    # ------------------------------------------------------------------
    # Tier 6: No match — return "NA"
    # ------------------------------------------------------------------
    logger.info(
        "Field '%s': no match for '%s' in %s. Returning 'NA'.",
        field_name, span_stripped, candidates,
    )
    return ("NA", 0.0)