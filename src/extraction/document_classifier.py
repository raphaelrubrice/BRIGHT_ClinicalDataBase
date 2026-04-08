"""Document type detection.

Classifies clinical documents as one of five types:
    ``anapath``, ``molecular_report``, ``consultation``, ``rcp``, ``radiology``.

Strategy
--------
**Keyword scoring**: weighted keyword matching.  ``strong`` keyword → 3 points,
``moderate`` keyword → 1 point.  The document type with the highest score wins.

Public API
----------
- ``DocumentClassifier``          — main classifier object.
- ``ClassificationResult``        — return value with type, scores & flags.
- ``DOCUMENT_TYPE_KEYWORDS``      — editable keyword dictionary.
- ``classify_document(text)``     — convenience function (keyword‑only).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid document types (mirrors schema.DOCUMENT_TYPES)
# ---------------------------------------------------------------------------

VALID_DOCUMENT_TYPES: list[str] = [
    "anapath",
    "molecular_report",
    "consultation",
    "rcp",
    "radiology",
]

# ---------------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------------

DOCUMENT_TYPE_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "anapath": {
        "strong": [
            "anatomopathologique",
            "anatomie pathologique",
            "anapath",
            "examen macroscopique",
            "examen microscopique",
            "immunohistochimie",
            "IHC",
            "pièce opératoire",
            "biopsie stéréotaxique",
            "histologie",
            "compte rendu anatomopathologique",
            "compte-rendu anatomopathologique",
        ],
        "moderate": [
            "fixation formolée",
            "inclusion en paraffine",
            "coloration HES",
            "Ki67",
            "GFAP",
            "Olig2",
            "necrose",
            "nécrose",
            "prolifération endothéliocapillaire",
            "mitoses",
        ],
    },
    "molecular_report": {
        "strong": [
            "biologie moléculaire",
            "analyse moléculaire",
            "CGH-array",
            "CGH array",
            "séquençage",
            "panel NGS",
            "altérations chromosomiques",
            "profil moléculaire",
            "analyse génomique",
            "profil génomique",
        ],
        "moderate": [
            "IDH1",
            "IDH2",
            "TERT",
            "MGMT",
            "1p/19q",
            "CDKN2A",
            "amplification EGFR",
            "codélétion",
            "méthylation",
            "mutation",
        ],
    },
    "consultation": {
        "strong": [
            "compte-rendu de consultation",
            "compte rendu de consultation",
            "consultation du",
            "vu en consultation",
            "vue en consultation",
            "vu(e) en consultation",
            "examen clinique",
            "interrogatoire",
            "karnofsky",
            "Karnofsky",
            "consultation de suivi",
        ],
        "moderate": [
            "traitement en cours",
            "chimio",
            "témozolomide",
            "avastin",
            "irradiation",
            "antécédents",
            "examen neurologique",
            "plainte",
            "autonomie",
        ],
    },
    "rcp": {
        "strong": [
            "réunion de concertation pluridisciplinaire",
            "RCP",
            "staff",
            "réunion pluridisciplinaire",
            "décision thérapeutique collégiale",
            "discussion en RCP",
            "avis RCP",
        ],
        "moderate": [
            "proposition thérapeutique",
            "discussion collégiale",
            "décision collégiale",
            "protocole thérapeutique",
        ],
    },
    "radiology": {
        "strong": [
            "IRM cérébrale",
            "scanner cérébral",
            "imagerie",
            "compte-rendu radiologique",
            "compte rendu radiologique",
            "séquences FLAIR",
            "T1 gadolinium",
            "prise de contraste",
            "IRM de contrôle",
            "IRM encéphalique",
        ],
        "moderate": [
            "lésion",
            "effet de masse",
            "œdème péri-lésionnel",
            "oedème péri-lésionnel",
            "spectroscopie",
            "perfusion",
            "diffusion",
            "rehaussement",
        ],
    },
}

# Scoring weights
_STRONG_WEIGHT: int = 3
_MODERATE_WEIGHT: int = 1

# Ambiguity threshold: if the gap between top two scores is ≤ this,
# the classification is flagged as ambiguous.
_AMBIGUITY_THRESHOLD: int = 2


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Result of document‑type classification.

    Attributes
    ----------
    document_type : str
        Predicted document type (one of ``VALID_DOCUMENT_TYPES``).
    scores : dict[str, int]
        Per‑type keyword score.
    confidence : float
        Normalised confidence in ``[0, 1]``.  Computed as
        ``(top_score - second) / max(top_score, 1)``.
    is_ambiguous : bool
        ``True`` when the gap between the top two scores is
        ≤ ``_AMBIGUITY_THRESHOLD``.
    used_llm_fallback : bool
        ``True`` if the LLM fallback was invoked to resolve ambiguity.
    matched_keywords : dict[str, list[str]]
        For each type, the keywords that were actually found in the text.
    """

    document_type: str
    scores: dict[str, int] = field(default_factory=dict)
    confidence: float = 0.0
    is_ambiguous: bool = False
    used_llm_fallback: bool = False
    matched_keywords: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Keyword scoring engine
# ---------------------------------------------------------------------------

def _score_text(
    text: str,
    keywords: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Score *text* against all keyword groups.

    Returns
    -------
    scores : dict[str, int]
        Per‑type cumulative score.
    matched : dict[str, list[str]]
        Per‑type list of keywords that matched.
    """
    if keywords is None:
        keywords = DOCUMENT_TYPE_KEYWORDS

    # Pre-process: lowercase for case-insensitive matching
    text_lower = text.lower()

    scores: dict[str, int] = {}
    matched: dict[str, list[str]] = {}

    for doc_type, strength_groups in keywords.items():
        total = 0
        type_matched: list[str] = []

        for strength, kw_list in strength_groups.items():
            weight = _STRONG_WEIGHT if strength == "strong" else _MODERATE_WEIGHT
            for kw in kw_list:
                # Case-insensitive search
                if kw.lower() in text_lower:
                    total += weight
                    type_matched.append(kw)

        scores[doc_type] = total
        matched[doc_type] = type_matched

    return scores, matched


def _rank_scores(scores: dict[str, int]) -> list[tuple[str, int]]:
    """Return scores sorted descending by value."""
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _compute_confidence(ranked: list[tuple[str, int]]) -> float:
    """Normalised confidence ∈ [0, 1].

    ``(top - second) / max(top, 1)``
    """
    if not ranked:
        return 0.0
    top = ranked[0][1]
    second = ranked[1][1] if len(ranked) > 1 else 0
    if top == 0:
        return 0.0
    return (top - second) / top


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class DocumentClassifier:
    """Weighted keyword classifier.

    Parameters
    ----------
    keywords : dict, optional
        Override the default ``DOCUMENT_TYPE_KEYWORDS``.
    ambiguity_threshold : int
        Score gap below which the result is flagged as ambiguous.
    """

    def __init__(
        self,
        keywords: dict[str, dict[str, list[str]]] | None = None,
        ambiguity_threshold: int = _AMBIGUITY_THRESHOLD,
    ) -> None:
        self._keywords = keywords or DOCUMENT_TYPE_KEYWORDS
        self._ambiguity_threshold = ambiguity_threshold

    # ── public API ──────────────────────────────────────────────────────

    def classify(self, text: str) -> ClassificationResult:
        """Classify *text* and return a ``ClassificationResult``.

        Steps
        -----
        1. Compute keyword scores for every document type.
        2. Rank types by score.
        3. Return the prediction, scores, confidence, and matched keywords.
        """
        if not text or not text.strip():
            return ClassificationResult(
                document_type="consultation",  # safe default
                scores={dt: 0 for dt in VALID_DOCUMENT_TYPES},
                confidence=0.0,
                is_ambiguous=True,
                used_llm_fallback=False,
                matched_keywords={dt: [] for dt in VALID_DOCUMENT_TYPES},
            )

        scores, matched = _score_text(text, self._keywords)
        ranked = _rank_scores(scores)
        top_type, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0

        is_ambiguous = (top_score - second_score) <= self._ambiguity_threshold
        confidence = _compute_confidence(ranked)

        # All-zero scores → ambiguous
        if top_score == 0:
            is_ambiguous = True

        return ClassificationResult(
            document_type=top_type,
            scores=scores,
            confidence=confidence,
            is_ambiguous=is_ambiguous,
            used_llm_fallback=False,
            matched_keywords=matched,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def classify_document(text: str) -> ClassificationResult:
    """Classify *text* using keyword scoring.

    Convenience shortcut equivalent to ``DocumentClassifier().classify(text)``.
    """
    return DocumentClassifier().classify(text)
