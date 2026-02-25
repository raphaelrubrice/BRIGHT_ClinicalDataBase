"""Section detection and segmentation.

Parses structured clinical documents into named sections for targeted
feature extraction.

Public API
----------
- ``SECTION_PATTERNS``        – regex patterns keyed by canonical section name.
- ``SECTION_TO_FEATURES``     – maps each section to the feature fields likely
                                to appear within it.
- ``SectionDetector``         – stateless detector: call ``detect(text)`` to
                                obtain a ``dict[str, str]`` of section → text.
- ``get_features_for_sections`` – given detected section names, return the
                                  union of relevant feature fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.extraction.schema import (
    ALL_BIO_FIELD_NAMES,
    ALL_CLINIQUE_FIELD_NAMES,
    FEATURE_GROUPS,
)

# ---------------------------------------------------------------------------
# Section header regex patterns (case-insensitive)
# ---------------------------------------------------------------------------
# Each pattern captures a section header that commonly introduces a block of
# text in French clinical documents (anapath reports, consultations, etc.).
# Patterns deliberately avoid matching the keyword when it appears as an
# inline mention (e.g., "IHC : positif") by requiring the header to be near
# the start of a line or preceded by common title-like punctuation.

# Compile flags shared by all patterns.
_RE_FLAGS = re.IGNORECASE | re.MULTILINE

SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "ihc": re.compile(
        r"^[ \t]*(?:immunohistochimie|(?<!\w)IHC(?!\w)|marqueurs?\s+immuno(?:histochim(?:iques?|ie))?)"
        r"[ \t]*[:.\-—–]*[ \t]*$",
        _RE_FLAGS,
    ),
    "molecular": re.compile(
        r"^[ \t]*(?:biologie\s+mol[eé]culaire|analyse\s+mol[eé]culaire"
        r"|panel\s+NGS|s[eé]quen[cç]age|r[eé]sultats?\s+mol[eé]culaire)"
        r"[ \t]*[:.\-—–]*[ \t]*$",
        _RE_FLAGS,
    ),
    "chromosomal": re.compile(
        r"^[ \t]*(?:CGH[\s\-]?array|alt[eé]rations?\s+chromosomiques?"
        r"|profil\s+g[eé]nomique|analyse\s+chromosomique)"
        r"[ \t]*[:.\-—–]*[ \t]*$",
        _RE_FLAGS,
    ),
    "macroscopy": re.compile(
        r"^[ \t]*(?:examen\s+macroscopique|macroscopie|description\s+macroscopique)"
        r"[ \t]*[:.\-—–]*[ \t]*$",
        _RE_FLAGS,
    ),
    "microscopy": re.compile(
        r"^[ \t]*(?:examen\s+microscopique|microscopie|description\s+microscopique"
        r"|histologie)"
        r"[ \t]*[:.\-—–]*[ \t]*$",
        _RE_FLAGS,
    ),
    "conclusion": re.compile(
        r"^[ \t]*(?:conclusion|diagnostic|synth[eè]se\s+diagnostique"
        r"|diagnostic\s+int[eé]gr[eé])"
        r"[ \t]*[:.\-—–]*[ \t]*$",
        _RE_FLAGS,
    ),
    "history": re.compile(
        r"^[ \t]*(?:ant[eé]c[eé]dents?|histoire\s+de\s+la\s+maladie|anamn[eè]se"
        r"|(?:r[eé]sum[eé]\s+de\s+l')?historique)"
        r"[ \t]*[:.\-—–]*[ \t]*$",
        _RE_FLAGS,
    ),
    "treatment": re.compile(
        r"^[ \t]*(?:traitements?|proposition\s+th[eé]rapeutique"
        r"|th[eé]rapeutique|protocole\s+th[eé]rapeutique"
        r"|d[eé]cision\s+th[eé]rapeutique)"
        r"[ \t]*[:.\-—–]*[ \t]*$",
        _RE_FLAGS,
    ),
    "clinical_exam": re.compile(
        r"^[ \t]*(?:examen\s+clinique|examen\s+neurologique"
        r"|interrogatoire|examen\s+physique)"
        r"[ \t]*[:.\-—–]*[ \t]*$",
        _RE_FLAGS,
    ),
    "radiology": re.compile(
        r"^[ \t]*(?:imagerie|IRM(?:\s+c[eé]r[eé]brale)?|scanner(?:\s+c[eé]r[eé]bral)?"
        r"|radiologie|bilan\s+radiologique|compte[\s\-]rendu\s+radiologique)"
        r"[ \t]*[:.\-—–]*[ \t]*$",
        _RE_FLAGS,
    ),
}

# A more lenient set of patterns for inline detection (when strict heading
# patterns find nothing).  These match headers even without full-line anchoring
# but still require the keyword to be at the start of a line.
SECTION_PATTERNS_LENIENT: dict[str, re.Pattern[str]] = {
    "ihc": re.compile(
        r"^[ \t]*(?:immunohistochimie|(?<!\w)IHC(?!\w)|marqueurs?\s+immuno)",
        _RE_FLAGS,
    ),
    "molecular": re.compile(
        r"^[ \t]*(?:biologie\s+mol[eé]culaire|analyse\s+mol[eé]culaire"
        r"|panel\s+NGS|s[eé]quen[cç]age)",
        _RE_FLAGS,
    ),
    "chromosomal": re.compile(
        r"^[ \t]*(?:CGH[\s\-]?array|alt[eé]rations?\s+chromosomiques?"
        r"|profil\s+g[eé]nomique)",
        _RE_FLAGS,
    ),
    "macroscopy": re.compile(
        r"^[ \t]*(?:examen\s+macroscopique|macroscopie|description\s+macroscopique)",
        _RE_FLAGS,
    ),
    "microscopy": re.compile(
        r"^[ \t]*(?:examen\s+microscopique|microscopie|description\s+microscopique"
        r"|histologie)",
        _RE_FLAGS,
    ),
    "conclusion": re.compile(
        r"^[ \t]*(?:conclusion|diagnostic|synth[eè]se\s+diagnostique)",
        _RE_FLAGS,
    ),
    "history": re.compile(
        r"^[ \t]*(?:ant[eé]c[eé]dents?|histoire\s+de\s+la\s+maladie|anamn[eè]se)",
        _RE_FLAGS,
    ),
    "treatment": re.compile(
        r"^[ \t]*(?:traitements?|proposition\s+th[eé]rapeutique"
        r"|th[eé]rapeutique|protocole\s+th[eé]rapeutique)",
        _RE_FLAGS,
    ),
    "clinical_exam": re.compile(
        r"^[ \t]*(?:examen\s+clinique|examen\s+neurologique|interrogatoire)",
        _RE_FLAGS,
    ),
    "radiology": re.compile(
        r"^[ \t]*(?:imagerie|IRM(?:\s+c[eé]r[eé]brale)?|scanner(?:\s+c[eé]r[eé]bral)?"
        r"|radiologie|bilan\s+radiologique)",
        _RE_FLAGS,
    ),
}


# ---------------------------------------------------------------------------
# Section → feature mapping
# ---------------------------------------------------------------------------
# Maps each canonical section name to the list of feature field names that
# are most likely found in that section.  A feature may appear in more than
# one section (e.g., ``grade`` in both ``conclusion`` and ``microscopy``).
#
# ``full_text`` is not listed here because it is a fallback that targets
# *all* fields — handled separately via ``get_features_for_sections()``.

SECTION_TO_FEATURES: dict[str, list[str]] = {
    # --- Biological sections (primarily from anapath / molecular reports) ---
    "ihc": [
        "ihc_idh1", "ihc_p53", "ihc_atrx", "ihc_fgfr3", "ihc_braf",
        "ihc_hist_h3k27m", "ihc_hist_h3k27me3", "ihc_egfr_hirsch",
        "ihc_gfap", "ihc_olig2", "ihc_ki67", "ihc_mmr",
    ],
    "molecular": [
        "mol_idh1", "mol_idh2", "mol_tert", "mol_CDKN2A", "mol_h3f3a",
        "mol_hist1h3b", "mol_braf", "mol_mgmt", "mol_fgfr1", "mol_egfr_mut",
        "mol_prkca", "mol_p53", "mol_pten", "mol_cic", "mol_fubp1", "mol_atrx",
    ],
    "chromosomal": [
        "ch1p", "ch19q", "ch10p", "ch10q", "ch7p", "ch7q", "ch9p", "ch9q",
        "ampli_mdm2", "ampli_cdk4", "ampli_egfr", "ampli_met", "ampli_mdm4",
        "fusion_fgfr", "fusion_ntrk", "fusion_autre",
    ],
    "macroscopy": [
        # Macroscopy sections rarely yield structured fields but may mention
        # specimen identifiers and surgery metadata.
        "num_labo", "date_chir",
    ],
    "microscopy": [
        "diag_histologique", "grade",
        "histo_necrose", "histo_pec", "histo_mitoses",
        "ihc_ki67",  # Ki67 is sometimes described in microscopy text
    ],
    "conclusion": [
        "diag_histologique", "diag_integre", "classification_oms", "grade",
        # Summary sections often restate key molecular/IHC results
        "ihc_idh1", "mol_idh1", "mol_mgmt",
        "ch1p", "ch19q",
    ],

    # --- Clinical sections (primarily from consultation notes, RCP) ---
    "history": [
        "date_1er_symptome", "epilepsie_1er_symptome",
        "ceph_hic_1er_symptome", "deficit_1er_symptome",
        "cognitif_1er_symptome", "autre_trouble_1er_symptome",
        "antecedent_tumoral", "activite_professionnelle",
        # General demographics often stated in anamnesis
        "date_de_naissance", "sexe", "nip",
    ],
    "treatment": [
        "chimios", "chm_date_debut", "chm_date_fin", "chm_cycles",
        "chir_date", "type_chirurgie",
        "rx_date_debut", "rx_date_fin", "rx_dose",
        "anti_epileptiques", "essai_therapeutique",
        "corticoides", "optune",
    ],
    "clinical_exam": [
        "ik_clinique",
        "epilepsie", "ceph_hic", "deficit", "cognitif", "autre_trouble",
        "progress_clinique",
    ],
    "radiology": [
        "exam_radio_date_decouverte",
        "contraste_1er_symptome", "oedeme_1er_symptome", "calcif_1er_symptome",
        "tumeur_lateralite", "tumeur_position",
        "progress_radiologique",
    ],
}

# Features that may appear *outside* any detected section header (e.g.,
# in the document preamble or free-flowing prose).  These are checked
# against the full text whenever section detection is used.
_PREAMBLE_FEATURES: list[str] = [
    "nip", "date_chir", "num_labo",
    "date_de_naissance", "sexe",
    "neuroncologue", "neurochirurgien", "radiotherapeute",
    "localisation_radiotherapie", "localisation_chir",
    "date_deces", "infos_deces",
    "dn_date", "evol_clinique",
    "date_progression",
    "progress_clinique", "progress_radiologique",
]


# ---------------------------------------------------------------------------
# SectionMatch helper
# ---------------------------------------------------------------------------

@dataclass
class SectionMatch:
    """A single matched section header location."""

    section_name: str
    header_text: str  # the text of the matched header line
    start: int  # character offset of the header in the document
    end: int  # character offset of the end of the header line
    body_start: int  # character offset where the section body begins (after header)


# ---------------------------------------------------------------------------
# SectionDetector
# ---------------------------------------------------------------------------

@dataclass
class SectionDetector:
    """Regex-based clinical document section detector.

    Usage
    -----
    >>> detector = SectionDetector()
    >>> sections = detector.detect(document_text)
    >>> sections.keys()  # e.g. dict_keys(['macroscopy', 'microscopy', 'ihc', 'conclusion'])

    If no section headers are found the detector returns
    ``{"full_text": <entire document>}``.
    """

    strict: bool = True
    """If *True* (default), use the strict patterns that require headers to be
    on their own line.  If *False*, use the lenient patterns that match headers
    at the start of a line even if followed by additional content."""

    min_section_length: int = 10
    """Minimum length (in characters) for a section body.  Sections shorter
    than this are merged with the following section or discarded."""

    def _get_patterns(self) -> dict[str, re.Pattern[str]]:
        """Return the pattern set to use based on ``self.strict``."""
        return SECTION_PATTERNS if self.strict else SECTION_PATTERNS_LENIENT

    # -----------------------------------------------------------------
    # Core detection
    # -----------------------------------------------------------------

    def _find_header_matches(self, text: str) -> list[SectionMatch]:
        """Scan *text* for all section header matches.

        Returns a list of ``SectionMatch`` ordered by their position in the
        document.  If duplicate section names are found (e.g., two
        "conclusion" headers), they are deduplicated by keeping only the
        *first* occurrence.
        """
        patterns = self._get_patterns()
        matches: list[SectionMatch] = []
        seen_names: set[str] = set()

        for name, pattern in patterns.items():
            for m in pattern.finditer(text):
                if name in seen_names:
                    continue  # keep first occurrence only
                seen_names.add(name)
                header_text = m.group(0).strip()

                # body_start: skip past the matched header and any trailing
                # whitespace / newline.
                body_start = m.end()
                # Advance past the immediate newline after the header, if any.
                while body_start < len(text) and text[body_start] in ("\r", "\n"):
                    body_start += 1

                matches.append(
                    SectionMatch(
                        section_name=name,
                        header_text=header_text,
                        start=m.start(),
                        end=m.end(),
                        body_start=body_start,
                    )
                )

        # Sort by position in document.
        matches.sort(key=lambda sm: sm.start)
        return matches

    def detect(self, text: str) -> dict[str, str]:
        """Segment *text* into named sections.

        Returns
        -------
        dict[str, str]
            Mapping of canonical section name → section body text.
            If no section headers are detected, returns
            ``{"full_text": text}``.
        """
        if not text or not text.strip():
            return {"full_text": text}

        matches = self._find_header_matches(text)

        # If strict mode found nothing, retry with lenient patterns.
        if not matches and self.strict:
            lenient_detector = SectionDetector(
                strict=False, min_section_length=self.min_section_length
            )
            lenient_matches = lenient_detector._find_header_matches(text)
            if lenient_matches:
                matches = lenient_matches
            else:
                return {"full_text": text}

        if not matches:
            return {"full_text": text}

        sections: dict[str, str] = {}

        # Optional preamble: text before the first section header.
        if matches[0].start > 0:
            preamble = text[: matches[0].start].strip()
            if preamble:
                sections["preamble"] = preamble

        # Split text between consecutive headers.
        for i, sm in enumerate(matches):
            if i + 1 < len(matches):
                body = text[sm.body_start : matches[i + 1].start]
            else:
                body = text[sm.body_start :]

            body = body.strip()

            # Skip empty or negligibly short sections (merge with next).
            if len(body) < self.min_section_length:
                continue

            sections[sm.section_name] = body

        # Edge case: all sections were too short → fallback.
        if not sections or (len(sections) == 1 and "preamble" in sections):
            return {"full_text": text}

        return sections

    def detect_with_metadata(self, text: str) -> DetectionResult:
        """Like ``detect`` but returns richer metadata alongside the sections.

        Returns
        -------
        DetectionResult
            Contains the section dict, matched header info, and a flag
            indicating whether the fallback was used.
        """
        if not text or not text.strip():
            return DetectionResult(
                sections={"full_text": text},
                matches=[],
                used_fallback=True,
            )

        matches = self._find_header_matches(text)

        # Lenient fallback
        used_lenient = False
        if not matches and self.strict:
            lenient_detector = SectionDetector(
                strict=False, min_section_length=self.min_section_length
            )
            matches = lenient_detector._find_header_matches(text)
            used_lenient = True

        if not matches:
            return DetectionResult(
                sections={"full_text": text},
                matches=[],
                used_fallback=True,
            )

        sections: dict[str, str] = {}

        if matches[0].start > 0:
            preamble = text[: matches[0].start].strip()
            if preamble:
                sections["preamble"] = preamble

        for i, sm in enumerate(matches):
            if i + 1 < len(matches):
                body = text[sm.body_start : matches[i + 1].start]
            else:
                body = text[sm.body_start :]
            body = body.strip()
            if len(body) < self.min_section_length:
                continue
            sections[sm.section_name] = body

        if not sections or (len(sections) == 1 and "preamble" in sections):
            return DetectionResult(
                sections={"full_text": text},
                matches=matches,
                used_fallback=True,
            )

        return DetectionResult(
            sections=sections,
            matches=matches,
            used_fallback=used_lenient,
        )


# ---------------------------------------------------------------------------
# DetectionResult — richer return type
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Extended result from section detection."""

    sections: dict[str, str] = field(default_factory=dict)
    matches: list[SectionMatch] = field(default_factory=list)
    used_fallback: bool = False

    @property
    def section_names(self) -> list[str]:
        """Canonical section names detected (excluding ``preamble`` and
        ``full_text``)."""
        return [
            k for k in self.sections
            if k not in ("preamble", "full_text")
        ]


# ---------------------------------------------------------------------------
# Feature mapping helpers
# ---------------------------------------------------------------------------

def get_features_for_sections(
    section_names: list[str],
    *,
    include_preamble: bool = True,
) -> list[str]:
    """Return the union of feature fields associated with *section_names*.

    Parameters
    ----------
    section_names : list[str]
        Canonical section names (keys of ``SECTION_TO_FEATURES``).
        Unknown names are silently ignored.
    include_preamble : bool
        If *True* (default), always include ``_PREAMBLE_FEATURES`` because
        identifier / demographic fields often appear outside formal sections.

    Returns
    -------
    list[str]
        Sorted, deduplicated list of field names.
    """
    if "full_text" in section_names:
        # Full-text fallback → all fields are relevant.
        return sorted(set(ALL_BIO_FIELD_NAMES + ALL_CLINIQUE_FIELD_NAMES))

    result: set[str] = set()
    for name in section_names:
        features = SECTION_TO_FEATURES.get(name, [])
        result.update(features)

    if include_preamble:
        result.update(_PREAMBLE_FEATURES)

    return sorted(result)


def get_section_for_feature(feature_name: str) -> list[str]:
    """Return the section name(s) where *feature_name* is most likely found.

    Parameters
    ----------
    feature_name : str
        A field name from the schema (e.g., ``"ihc_idh1"``).

    Returns
    -------
    list[str]
        List of canonical section names.  May be empty if the feature is
        not mapped to any specific section (will only appear in full_text
        fallback).
    """
    result: list[str] = []
    for section_name, features in SECTION_TO_FEATURES.items():
        if feature_name in features:
            result.append(section_name)
    return result
