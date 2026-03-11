"""Find & Check extractor for controlled-vocabulary fields.

Mirrors the ``DateExtractor`` strategy:

1. **Find** – scan the document for field-specific identification terms
   (marker names, gene names, chromosome arms, …) using exact or fuzzy
   matching.
2. **Check** – in the surrounding context of each hit, fuzzy-match
   against all category surface-form terms to assign a category value.
3. **Assign** – greedy allocation (highest combined score first, one
   value per field).

Fields whose best candidate does not pass the category threshold remain
unset (``"NA"`` by default) and will be routed to the GLiNER tier.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from rapidfuzz import fuzz

from src.extraction.controlled_vocab_data import (
    CONTROLLED_REGISTRY_EN,
    CONTROLLED_REGISTRY_FR,
    FieldVocabConfig,
)
from src.extraction.schema import ExtractionValue
from src.extraction.text_normalisation import normalise

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

# Short terms (≤3 chars) use word-boundary regex; longer terms use fuzzy.
_SHORT_TERM_MAX_LEN = 3


def _build_short_pattern(term: str) -> re.Pattern[str]:
    """Compile a case-insensitive word-boundary regex for *term*."""
    # For terms that contain special regex chars (e.g. "+", "-"), escape.
    escaped = re.escape(term)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE | re.UNICODE)


def _get_context(
    text: str,
    hit_start: int,
    hit_end: int,
    half_window: int,
) -> str:
    """Return normalised context around a hit, masking the hit itself."""
    ctx_start = max(0, hit_start - half_window)
    ctx_end = min(len(text), hit_end + half_window)
    # Concatenate before + after, replacing the hit with a space
    raw = text[ctx_start:hit_start] + " " + text[hit_end:ctx_end]
    return raw


# ───────────────────────────────────────────────────────────────────────
# Main class
# ───────────────────────────────────────────────────────────────────────

class ControlledExtractor:
    """Find & Check extractor for all controlled-vocabulary fields."""

    # Post-processing remaps applied after category assignment.
    _POST_REMAP: dict[str, dict[str, str]] = {
        "ihc_atrx": {"positif": "maintenu"},
    }

    def __init__(self) -> None:
        # Pre-compile short-term patterns for each registry entry.
        # Keyed by (registry_id, field_name, term_normalised).
        self._short_pats: dict[tuple[str, str, str], re.Pattern[str]] = {}
        for reg_id, registry in (("fr", CONTROLLED_REGISTRY_FR),
                                  ("en", CONTROLLED_REGISTRY_EN)):
            for field_name, cfg in registry.items():
                for term in cfg.identification_list:
                    norm_term = normalise(term)
                    if len(norm_term) <= _SHORT_TERM_MAX_LEN:
                        key = (reg_id, field_name, norm_term)
                        self._short_pats[key] = _build_short_pattern(norm_term)

    # ─── public API ───────────────────────────────────────────────────

    def extract(
        self,
        text: str,
        feature_subset: list[str],
        language: str = "fr",
    ) -> dict[str, ExtractionValue]:
        """Run Find & Check on *text* for the requested fields.

        Parameters
        ----------
        text : str
            Full document text (original, NFC-normalised).
        feature_subset : list[str]
            Field names to extract (only those with a registry entry
            will be processed).
        language : str
            ``"fr"`` or ``"en"`` — selects the vocabulary registry.

        Returns
        -------
        dict[str, ExtractionValue]
            Mapping field_name → extraction result for fields that were
            successfully matched.  Fields with no confident match are
            *not* included (they remain for downstream tiers).
        """
        registry = (CONTROLLED_REGISTRY_FR if language.startswith("fr")
                     else CONTROLLED_REGISTRY_EN)
        reg_id = "fr" if language.startswith("fr") else "en"

        text_norm = normalise(text)

        # ------------------------------------------------------------------
        # Step 1: FIND — collect identification hits across all fields
        # ------------------------------------------------------------------
        # Each hit: (field_name, start, end, id_score)
        hits: list[tuple[str, int, int, float]] = []

        for field_name in feature_subset:
            cfg = registry.get(field_name)
            if cfg is None:
                continue
            self._find_hits(
                text_norm, text, cfg, field_name, reg_id, hits,
            )

        if not hits:
            return {}

        # ------------------------------------------------------------------
        # Step 2: CHECK — for each hit, score all categories
        # ------------------------------------------------------------------
        # candidate: (field, category, combined_score, start, end, source_span)
        candidates: list[tuple[str, str, float, int, int, str]] = []

        for field_name, start, end, id_score in hits:
            cfg = registry[field_name]
            ctx_raw = _get_context(text, start, end, cfg.context_half_window)
            ctx_norm = normalise(ctx_raw)

            best_cat: Optional[str] = None
            best_cat_adjusted: float = 0.0   # adjusted = raw + length bonus
            best_raw_score: float = 0.0      # raw partial_ratio (for confidence)

            for category, terms in cfg.category_specific_dict.items():
                if not terms:          # "autre" → empty list
                    continue
                for term in terms:
                    term_norm = normalise(term)
                    if not term_norm:
                        continue
                    raw_score = fuzz.partial_ratio(term_norm, ctx_norm)
                    if raw_score < cfg.cat_fuzzy_threshold:
                        continue
                    # Length bonus: prefer longer (more specific) terms
                    # when fuzzy scores are tied or close.  Up to +15 pts.
                    length_bonus = min(15.0, len(term_norm) / 3.0)
                    adjusted = raw_score + length_bonus
                    if adjusted > best_cat_adjusted:
                        best_cat_adjusted = adjusted
                        best_raw_score = raw_score
                        best_cat = category

            if best_cat is not None:
                combined = id_score * 0.3 + best_raw_score * 0.7
                source_span = text[start:end]
                candidates.append(
                    (field_name, best_cat, combined, start, end, source_span)
                )

        if not candidates:
            return {}

        # ------------------------------------------------------------------
        # Step 3: ASSIGN — greedy, highest combined score first
        # ------------------------------------------------------------------
        candidates.sort(key=lambda c: c[2], reverse=True)

        results: dict[str, ExtractionValue] = {}
        for field_name, category, combined, start, end, source_span in candidates:
            if field_name in results:
                continue  # already assigned

            # Post-processing remap (e.g. ihc_atrx positif → maintenu)
            remap = self._POST_REMAP.get(field_name)
            if remap:
                category = remap.get(category, category)

            results[field_name] = ExtractionValue(
                value=category,
                source_span=source_span,
                source_span_start=start,
                source_span_end=end,
                extraction_tier="rule",
                confidence=round(combined / 100, 2),
                vocab_valid=True,
            )

        return results

    # ─── internals ────────────────────────────────────────────────────

    def _find_hits(
        self,
        text_norm: str,
        text_orig: str,
        cfg: FieldVocabConfig,
        field_name: str,
        reg_id: str,
        out: list[tuple[str, int, int, float]],
    ) -> None:
        """Populate *out* with identification hits for one field."""
        for term in cfg.identification_list:
            term_norm = normalise(term)
            if not term_norm:
                continue

            if len(term_norm) <= _SHORT_TERM_MAX_LEN:
                # Exact word-boundary search on normalised text
                pat = self._short_pats.get((reg_id, field_name, term_norm))
                if pat is None:
                    pat = _build_short_pattern(term_norm)
                for m in pat.finditer(text_norm):
                    out.append((field_name, m.start(), m.end(), 100.0))
            else:
                # Sliding-window fuzzy search for longer terms
                self._fuzzy_scan(
                    text_norm, term_norm, cfg.id_fuzzy_threshold,
                    field_name, out,
                )

    @staticmethod
    def _fuzzy_scan(
        text_norm: str,
        term_norm: str,
        threshold: int,
        field_name: str,
        out: list[tuple[str, int, int, float]],
    ) -> None:
        """Scan *text_norm* for fuzzy occurrences of *term_norm*.

        Uses a sliding window of ``len(term_norm) * 1.5`` characters,
        stepping by ``max(1, len(term_norm) // 2)`` for efficiency.
        """
        term_len = len(term_norm)
        win_size = int(term_len * 1.5) + 1
        step = max(1, term_len // 2)
        text_len = len(text_norm)

        # First try a quick exact substring check
        idx = text_norm.find(term_norm)
        while idx != -1:
            out.append((field_name, idx, idx + term_len, 100.0))
            idx = text_norm.find(term_norm, idx + term_len)

        if any(s >= 100.0 for (f, _, _, s) in out if f == field_name):
            return  # exact hits found, skip fuzzy

        pos = 0
        while pos + win_size <= text_len:
            window = text_norm[pos:pos + win_size]
            score = fuzz.partial_ratio(term_norm, window)
            if score >= threshold:
                # Approximate character span
                out.append((field_name, pos, min(pos + win_size, text_len), score))
                # Jump past this match to avoid duplicates
                pos += win_size
            else:
                pos += step

        # Handle tail
        if text_len > win_size:
            tail = text_norm[text_len - win_size:]
            score = fuzz.partial_ratio(term_norm, tail)
            if score >= threshold:
                out.append((
                    field_name,
                    text_len - win_size,
                    text_len,
                    score,
                ))
