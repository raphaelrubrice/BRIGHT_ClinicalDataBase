"""Tier 1 rule-based feature extractors.

Regex and pattern-based extraction for dates, IHC results, molecular
status, chromosomal alterations, binary fields, numerical values,
amplifications, and fusions from French neuro-oncology documents.

Public API
----------
- ``extract_dates``          – Generic date extraction + normalisation.
- ``extract_ihc``            – IHC marker extraction.
- ``extract_molecular``      – Molecular status extraction.
- ``extract_chromosomal``    – Chromosomal alteration extraction.
- ``extract_binary``         – Binary (oui/non) field extraction.
- ``extract_numerical``      – Numerical value extraction (Ki67, IK, …).
- ``extract_amplifications`` – Gene amplification extraction.
- ``extract_fusions``        – Gene fusion extraction.
- ``run_rule_extraction``    – Master function running all extractors.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from src.extraction.schema import (
    ExtractionValue,
    FieldDefinition,
    FieldType,
    ALL_FIELDS_BY_NAME,
    FEATURE_GROUPS,
    get_field,
)
from src.extraction.negation import AssertionAnnotator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RE_FLAGS = re.IGNORECASE | re.UNICODE

# French month names / abbreviations → month number
_FRENCH_MONTHS: dict[str, int] = {
    # Full names
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3,
    "avril": 4, "mai": 5, "juin": 6, "juillet": 7,
    "août": 8, "aout": 8, "septembre": 9, "octobre": 10,
    "novembre": 11, "décembre": 12, "decembre": 12,
    # Abbreviations
    "janv": 1, "jan": 1, "fév": 2, "fev": 2, "févr": 2, "fevr": 2,
    "avr": 4, "juil": 7, "juill": 7,
    "sept": 9, "oct": 10, "nov": 11, "déc": 12, "dec": 12,
}


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.1  Date extractor (generalised)
# ═══════════════════════════════════════════════════════════════════════════

# Pattern 1: DD/MM/YYYY  or DD-MM-YYYY or DD.MM.YYYY
_PAT_DATE_DMY = re.compile(
    r"\b(?P<day>\d{1,2})[/.\-](?P<month>\d{1,2})[/.\-](?P<year>\d{4})\b"
)

# Pattern 2: YYYY/MM/DD or YYYY-MM-DD
_PAT_DATE_YMD = re.compile(
    r"\b(?P<year>\d{4})[/.\-](?P<month>\d{1,2})[/.\-](?P<day>\d{1,2})\b"
)

# Pattern 3: DD Month YYYY  (French month names)
_MONTH_NAMES_RE = "|".join(sorted(_FRENCH_MONTHS.keys(), key=len, reverse=True))
_PAT_DATE_FULL_FR = re.compile(
    r"\b(?P<day>\d{1,2})\s+(?P<month>" + _MONTH_NAMES_RE + r")\s+(?P<year>\d{4})\b",
    _RE_FLAGS,
)

# Pattern 4: Abbreviated month-year  (e.g. "janv-25", "déc-10")
_PAT_DATE_ABBREV = re.compile(
    r"\b(?P<month>" + _MONTH_NAMES_RE + r")[.\-](?P<year>\d{2,4})\b",
    _RE_FLAGS,
)

# Pattern 5: Year only  (e.g. "en 2008", "depuis 2015")
_PAT_DATE_YEAR_ONLY = re.compile(
    r"\b(?:en|depuis|année)\s+(?P<year>(?:19|20)\d{2})\b",
    _RE_FLAGS,
)


def _normalise_year(y: str | int) -> str:
    """Normalise a 2-digit or 4-digit year to 4 digits."""
    y_int = int(y)
    if y_int < 100:
        y_int += 2000 if y_int < 50 else 1900
    return f"{y_int:04d}"


def _normalise_month_name(name: str) -> int:
    """Convert a French month name/abbreviation to a 1-based month number."""
    return _FRENCH_MONTHS.get(name.lower().rstrip("."), 0)


def extract_dates(text: str) -> list[tuple[str, str, int, int]]:
    """Extract and normalise all dates from *text*.

    Returns
    -------
    list[tuple[str, str, int, int]]
        Each element is ``(normalised_date, raw_span, start_offset, end_offset)``.
        ``normalised_date`` is in ``DD/MM/YYYY`` format.  Partial dates
        (year only) are normalised to ``01/01/YYYY``.
    """
    results: list[tuple[str, str, int, int]] = []
    seen_positions: set[tuple[int, int]] = set()

    def _add(day: int, month: int, year: str, raw: str, start: int, end: int) -> None:
        key = (start, end)
        if key in seen_positions:
            return
        seen_positions.add(key)
        y = _normalise_year(year)
        results.append((f"{day:02d}/{month:02d}/{y}", raw, start, end))

    # Pattern 1: DD/MM/YYYY
    for m in _PAT_DATE_DMY.finditer(text):
        _add(int(m.group("day")), int(m.group("month")), m.group("year"),
             m.group(), m.start(), m.end())

    # Pattern 2: YYYY/MM/DD
    for m in _PAT_DATE_YMD.finditer(text):
        _add(int(m.group("day")), int(m.group("month")), m.group("year"),
             m.group(), m.start(), m.end())

    # Pattern 3: DD Month YYYY
    for m in _PAT_DATE_FULL_FR.finditer(text):
        month_num = _normalise_month_name(m.group("month"))
        if month_num:
            _add(int(m.group("day")), month_num, m.group("year"),
                 m.group(), m.start(), m.end())

    # Pattern 4: Abbreviated month-year
    for m in _PAT_DATE_ABBREV.finditer(text):
        month_num = _normalise_month_name(m.group("month"))
        if month_num:
            y = m.group("year")
            _add(1, month_num, y, m.group(), m.start(), m.end())

    # Pattern 5: Year only
    for m in _PAT_DATE_YEAR_ONLY.finditer(text):
        pos = (m.start(), m.end())
        if pos not in seen_positions:
            y = m.group("year")
            seen_positions.add(pos)
            results.append((f"01/01/{y}", m.group(), m.start(), m.end()))

    # Sort by position in text
    results.sort(key=lambda x: x[2])
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.2  IHC result extractor
# ═══════════════════════════════════════════════════════════════════════════

IHC_MARKERS: list[str] = [
    "idh1", "idh-1", "p53", "atrx", "fgfr3", "braf", "h3k27m",
    "h3k27me3", "egfr", "gfap", "olig2", "ki67", "ki-67",
    "mmr", "mlh1", "msh2", "msh6", "pms2",
]

# Canonical marker name mapping (normalise aliases)
_IHC_CANONICAL: dict[str, str] = {
    "idh1": "ihc_idh1",
    "idh-1": "ihc_idh1",
    "p53": "ihc_p53",
    "atrx": "ihc_atrx",
    "fgfr3": "ihc_fgfr3",
    "braf": "ihc_braf",
    "h3k27m": "ihc_hist_h3k27m",
    "h3k27me3": "ihc_hist_h3k27me3",
    "egfr": "ihc_egfr_hirsch",
    "gfap": "ihc_gfap",
    "olig2": "ihc_olig2",
    "ki67": "ihc_ki67",
    "ki-67": "ihc_ki67",
    "mmr": "ihc_mmr",
    "mlh1": "ihc_mmr",
    "msh2": "ihc_mmr",
    "msh6": "ihc_mmr",
    "pms2": "ihc_mmr",
}

# Value normalisation
_IHC_VALUE_NORM: dict[str, str] = {
    "positive": "positif",
    "positif": "positif",
    "positifs": "positif",
    "+": "positif",
    "négative": "negatif",
    "negative": "negatif",
    "négatif": "negatif",
    "negatif": "negatif",
    "-": "negatif",
    "maintenu": "maintenu",
    "maintenue": "maintenu",
    "conservé": "maintenu",
    "conserve": "maintenu",
    "conservée": "maintenu",
    "conservee": "maintenu",
    "perte d'expression": "negatif",
    "perte d'expression": "negatif",
}

_IHC_PATTERN = re.compile(
    r"(?P<marker>"
    + "|".join(re.escape(m) for m in sorted(IHC_MARKERS, key=len, reverse=True))
    + r")"
    r"\s*[:=\-\s]\s*"
    r"(?P<value>"
    r"positif[s]?|n[ée]gatif(?:ve)?|positive?|n[ée]gative?"
    r"|maintenu[e]?|perte\s+d[''']expression"
    r"|conserv[ée]e?"
    r"|\+|\-"
    r"|\d+\s*(?:[àa]\s*\d+\s*)?%"
    r"|<?\.?\s*\d+\s*%"
    r"|score\s+(?:de\s+)?\d+"
    r")",
    _RE_FLAGS,
)


def extract_ihc(text: str) -> dict[str, ExtractionValue]:
    """Extract IHC results from *text*.

    Returns a dict mapping canonical field names (e.g. ``ihc_idh1``)
    to ``ExtractionValue`` objects.
    """
    results: dict[str, ExtractionValue] = {}

    for m in _IHC_PATTERN.finditer(text):
        marker_raw = m.group("marker").lower().strip()
        value_raw = m.group("value").strip().lower()

        field_name = _IHC_CANONICAL.get(marker_raw)
        if field_name is None:
            continue

        # Normalise value
        normalised = _IHC_VALUE_NORM.get(value_raw)
        if normalised is None:
            # Try to extract a percentage / score
            pct_match = re.search(r"(\d+)\s*%", value_raw)
            score_match = re.search(r"score\s+(?:de\s+)?(\d+)", value_raw)
            if pct_match:
                normalised = pct_match.group(1)
            elif score_match:
                normalised = score_match.group(1)
            else:
                normalised = value_raw

        # Only keep first match per field (first occurrence in text)
        if field_name not in results:
            results[field_name] = ExtractionValue(
                value=normalised,
                source_span=m.group(),
                source_span_start=m.start(),
                source_span_end=m.end(),
                extraction_tier="rule",
                confidence=0.9,
                vocab_valid=True,
            )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.3  Molecular status extractor
# ═══════════════════════════════════════════════════════════════════════════

MOL_GENES: list[str] = [
    "idh1", "idh-1", "idh2", "idh-2",
    "tert", "cdkn2a", "h3f3a", "hist1h3b",
    "braf", "mgmt", "fgfr1",
    "egfr", "prkca", "p53", "tp53",
    "pten", "cic", "fubp1", "atrx",
]

_MOL_CANONICAL: dict[str, str] = {
    "idh1": "mol_idh1", "idh-1": "mol_idh1",
    "idh2": "mol_idh2", "idh-2": "mol_idh2",
    "tert": "mol_tert",
    "cdkn2a": "mol_CDKN2A",
    "h3f3a": "mol_h3f3a",
    "hist1h3b": "mol_hist1h3b",
    "braf": "mol_braf",
    "mgmt": "mol_mgmt",
    "fgfr1": "mol_fgfr1",
    "egfr": "mol_egfr_mut",
    "prkca": "mol_prkca",
    "p53": "mol_p53", "tp53": "mol_p53",
    "pten": "mol_pten",
    "cic": "mol_cic",
    "fubp1": "mol_fubp1",
    "atrx": "mol_atrx",
}

# Status normalisation
_MOL_STATUS_NORM: dict[str, str] = {
    "wt": "wt",
    "wild-type": "wt",
    "wild type": "wt",
    "sauvage": "wt",
    "type sauvage": "wt",
    "non muté": "wt",
    "non mutée": "wt",
    "non mute": "wt",
    "non mutee": "wt",
    "absence de mutation": "wt",
    "pas de mutation": "wt",
    "muté": "mute",
    "mutée": "mute",
    "mute": "mute",
    "mutee": "mute",
    "mutation": "mute",
    "présence de mutation": "mute",
    "muté(e)": "mute",
    # MGMT specific
    "méthylé": "methyle",
    "methylé": "methyle",
    "methyle": "methyle",
    "methylation positive": "methyle",
    "méthylation positive": "methyle",
    "non méthylé": "non methyle",
    "non methylé": "non methyle",
    "non methyle": "non methyle",
    "methylation negative": "non methyle",
    "méthylation négative": "non methyle",
    "non methylation": "non methyle",
    "absence de méthylation": "non methyle",
    "absence de methylation": "non methyle",
}

# Known variant pattern (e.g. R132H, C228T, V600E, K27M, G34R, p.R132H)
_VARIANT_PATTERN = re.compile(
    r"(?:p\.)?[A-Z]\d+[A-Z]",
    re.IGNORECASE,
)

# Molecular pattern: <gene> <sep> <status_or_variant>
_MOL_PATTERN = re.compile(
    r"(?P<gene>"
    + "|".join(re.escape(g) for g in sorted(MOL_GENES, key=len, reverse=True))
    + r")"
    r"\s*[:=\-\s]\s*"
    r"(?P<status>"
    r"wt|wild[- ]?type|sauvage|type\s+sauvage"
    r"|non\s+mut[ée]e?"
    r"|mut[ée]e?|mutation"
    r"|pr[ée]sence\s+de\s+mutation"
    r"|absence\s+de\s+mutation"
    r"|pas\s+de\s+mutation"
    r"|m[ée]thyl[ée]|non\s+m[ée]thyl[ée]"
    r"|m[ée]thylation\s+(?:positive|n[ée]gative)"
    r"|absence\s+de\s+m[ée]thylation"
    r"|(?:p\.)?[A-Z]\d+[A-Z]"  # Variant like R132H
    r")",
    _RE_FLAGS,
)

# Special pattern: "pas de mutation <gene>" / "absence de mutation <gene>"
_MOL_NEGATED_PATTERN = re.compile(
    r"(?:pas\s+de\s+mutation|absence\s+de\s+mutation)\s+"
    r"(?:du?\s+gène?\s+)?"
    r"(?P<gene>"
    + "|".join(re.escape(g) for g in sorted(MOL_GENES, key=len, reverse=True))
    + r")",
    _RE_FLAGS,
)

# Special pattern: "mutation <gene> <variant>"
_MOL_MUTATION_GENE_PATTERN = re.compile(
    r"mutation\s+(?:du?\s+(?:gène?\s+)?)?(?:promoteur\s+(?:du?\s+)?)?(?P<gene>"
    + "|".join(re.escape(g) for g in sorted(MOL_GENES, key=len, reverse=True))
    + r")"
    r"(?:\s*[:(\s]\s*(?P<variant>(?:p\.)?[A-Z]\d+[A-Z])\s*[)]?)?",
    _RE_FLAGS,
)

# Codeletion pattern
_CODELETION_PATTERN = re.compile(
    r"(?:cod[ée]l[ée]tion|co-d[ée]l[ée]tion)\s+(?:des?\s+)?(?:bras?\s+)?1p[/\s]+(?:et\s+)?19q",
    _RE_FLAGS,
)


def extract_molecular(text: str) -> dict[str, ExtractionValue]:
    """Extract molecular biology results from *text*.

    Returns a dict mapping canonical field names (e.g. ``mol_idh1``)
    to ``ExtractionValue`` objects.
    """
    results: dict[str, ExtractionValue] = {}

    def _set(field_name: str, value: str, raw: str, start: int, end: int) -> None:
        if field_name not in results:
            results[field_name] = ExtractionValue(
                value=value,
                source_span=raw,
                source_span_start=start,
                source_span_end=end,
                extraction_tier="rule",
                confidence=0.9,
                vocab_valid=True,
            )

    # 1) Regular <gene> <sep> <status> pattern
    for m in _MOL_PATTERN.finditer(text):
        gene_raw = m.group("gene").lower().strip()
        status_raw = m.group("status").strip().lower()
        field_name = _MOL_CANONICAL.get(gene_raw)
        if field_name is None:
            continue

        normalised = _MOL_STATUS_NORM.get(status_raw)
        if normalised is None:
            # Likely a variant string
            if _VARIANT_PATTERN.match(status_raw):
                normalised = "mute"
                _set(field_name, normalised, m.group(), m.start(), m.end())
                # Also store the variant detail
                continue
            else:
                normalised = status_raw

        _set(field_name, normalised, m.group(), m.start(), m.end())

    # 2) "pas de mutation <gene>" pattern
    for m in _MOL_NEGATED_PATTERN.finditer(text):
        gene_raw = m.group("gene").lower().strip()
        field_name = _MOL_CANONICAL.get(gene_raw)
        if field_name:
            _set(field_name, "wt", m.group(), m.start(), m.end())

    # 3) "mutation (du) (promoteur) <gene> (<variant>)" pattern
    for m in _MOL_MUTATION_GENE_PATTERN.finditer(text):
        gene_raw = m.group("gene").lower().strip()
        field_name = _MOL_CANONICAL.get(gene_raw)
        if field_name:
            variant = m.group("variant")
            value = "mute"
            _set(field_name, value, m.group(), m.start(), m.end())

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.4  Chromosomal alteration extractor
# ═══════════════════════════════════════════════════════════════════════════

_CHROMOSOME_ARMS: list[str] = ["1p", "19q", "10p", "10q", "7p", "7q", "9p", "9q"]

_CHR_CANONICAL: dict[str, str] = {arm: f"ch{arm}" for arm in _CHROMOSOME_ARMS}

_CHR_STATUS_NORM: dict[str, str] = {
    "gain": "gain",
    "perte": "perte",
    "perte partielle": "perte partielle",
    "délétion": "perte",
    "deletion": "perte",
    "deleted": "perte",
    "del": "perte",
    "normal": "gain",   # "normal" ≠ gain, but we mark separately below
    "normale": "gain",
    "perte homozygote": "perte",
    "perte hétérozygote": "perte partielle",
}

_CHR_PATTERN = re.compile(
    r"(?P<arm>"
    + "|".join(re.escape(a) for a in sorted(_CHROMOSOME_ARMS, key=len, reverse=True))
    + r")"
    r"\s*[:=\-\s]\s*"
    r"(?P<status>"
    r"gain|perte(?:\s+partielle)?(?:\s+(?:homo|h[ée]t[ée]ro)zygote)?"
    r"|d[ée]l[ée]tion|deleted?|del"
    r"|normal[e]?"
    r")",
    _RE_FLAGS,
)

# Special: "absence de perte de <arm>"
_CHR_ABSENCE_PATTERN = re.compile(
    r"(?:absence\s+de\s+(?:perte|d[ée]l[ée]tion)|pas\s+de\s+(?:perte|d[ée]l[ée]tion))"
    r"\s+(?:du\s+(?:bras\s+)?)?(?P<arm>"
    + "|".join(re.escape(a) for a in sorted(_CHROMOSOME_ARMS, key=len, reverse=True))
    + r")",
    _RE_FLAGS,
)


def extract_chromosomal(text: str) -> dict[str, ExtractionValue]:
    """Extract chromosomal alterations from *text*.

    Returns a dict mapping canonical field names (e.g. ``ch1p``)
    to ``ExtractionValue`` objects.
    """
    results: dict[str, ExtractionValue] = {}

    def _set(field_name: str, value: str, raw: str, start: int, end: int) -> None:
        if field_name not in results:
            results[field_name] = ExtractionValue(
                value=value,
                source_span=raw,
                source_span_start=start,
                source_span_end=end,
                extraction_tier="rule",
                confidence=0.9,
                vocab_valid=True,
            )

    # Regular <arm> : <status>
    for m in _CHR_PATTERN.finditer(text):
        arm = m.group("arm").lower()
        status_raw = m.group("status").strip().lower()
        field_name = _CHR_CANONICAL.get(arm)
        if field_name is None:
            continue
        normalised = _CHR_STATUS_NORM.get(status_raw, status_raw)
        _set(field_name, normalised, m.group(), m.start(), m.end())

    # Codeletion 1p/19q
    for m in _CODELETION_PATTERN.finditer(text):
        _set("ch1p", "perte", m.group(), m.start(), m.end())
        _set("ch19q", "perte", m.group(), m.start(), m.end())

    # "absence de perte de <arm>"
    for m in _CHR_ABSENCE_PATTERN.finditer(text):
        arm = m.group("arm").lower()
        field_name = _CHR_CANONICAL.get(arm)
        if field_name:
            # No deletion → we don't set a value (or could set to "normal"
            # but the schema uses gain/perte/perte partielle only).
            pass

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.5  Binary field extractor
# ═══════════════════════════════════════════════════════════════════════════

# Keywords / synonyms for binary clinical fields
_BINARY_KEYWORDS: dict[str, list[str]] = {
    "epilepsie": [
        "épilepsie", "epilepsie", "crises comitiales", "crises convulsives",
        "crise convulsive", "crise comitiale", "crise épileptique",
        "crises épileptiques", "comitialité",
    ],
    "ceph_hic": [
        "céphalées", "cephalees", "céphalée", "HTIC",
        "hypertension intracrânienne", "hypertension intracranienne",
    ],
    "deficit": [
        "déficit", "deficit", "déficitaire", "hémiplégie",
        "hémiparésie", "hemiparesie", "parésie", "paralysie",
    ],
    "cognitif": [
        "troubles cognitifs", "trouble cognitif", "confusion",
        "troubles mnésiques", "trouble mnésique", "ralentissement",
    ],
    "histo_necrose": [
        "nécrose", "necrose", "nécroses", "plages de nécrose",
        "foyers de nécrose", "nécrose palissadique",
    ],
    "histo_pec": [
        "prolifération endothéliocapillaire", "proliferation endotheliocapillaire",
        "prolifération endothélio-capillaire", "PEC",
        "hyperplasie endothéliocapillaire",
    ],
    "corticoides": [
        "corticoïdes", "corticoides", "corticothérapie", "dexaméthasone",
        "dexamethasone", "solumédrol", "solumedrol", "médrol", "prednisone",
        "cortancyl", "prednisolone",
    ],
    "optune": [
        "optune", "ttfields", "tt-fields", "tumor treating fields",
        "champs électriques",
    ],
    "anti_epileptiques": [
        "anti-épileptique", "antiépileptique", "anti-epileptique",
        "antiepileptique", "keppra", "lévétiracétam", "levetiracetam",
        "valproate", "dépakine", "depakine", "lacosamide", "vimpat",
        "lamotrigine",
    ],
    "essai_therapeutique": [
        "essai thérapeutique", "essai therapeutique", "protocole de recherche",
        "inclusion dans un essai", "étude clinique",
    ],
    "contraste_1er_symptome": [
        "prise de contraste", "rehaussement", "enhancement",
    ],
    "oedeme_1er_symptome": [
        "œdème", "oedème", "oedeme", "œdème péri-lésionnel",
        "oedème péri-lésionnel", "oedeme peri-lesionnel",
    ],
    "calcif_1er_symptome": [
        "calcification", "calcifications", "calcifié",
    ],
    "progress_clinique": [
        "progression clinique", "aggravation clinique",
    ],
    "progress_radiologique": [
        "progression radiologique", "progression à l'imagerie",
        "augmentation de taille", "croissance tumorale",
    ],
    "antecedent_tumoral": [
        "antécédent tumoral", "antecedent tumoral",
        "antécédent de tumeur", "antécédents tumoraux",
    ],
}

# Map binary keywords to their corresponding "first symptom" variants
_BINARY_1ER_SYMPTOME: dict[str, str] = {
    "epilepsie": "epilepsie_1er_symptome",
    "ceph_hic": "ceph_hic_1er_symptome",
    "deficit": "deficit_1er_symptome",
    "cognitif": "cognitif_1er_symptome",
}


def extract_binary(
    text: str,
    annotator: Optional[AssertionAnnotator] = None,
) -> dict[str, ExtractionValue]:
    """Extract binary (oui/non) fields from *text*.

    Uses keyword matching + negation detection to determine if each
    feature is present or negated in the text.

    Parameters
    ----------
    text : str
        The document or section text.
    annotator : AssertionAnnotator, optional
        If provided, used for negation detection.
    """
    results: dict[str, ExtractionValue] = {}

    for field_name, keywords in _BINARY_KEYWORDS.items():
        for kw in keywords:
            pattern = re.compile(re.escape(kw), _RE_FLAGS)
            match = pattern.search(text)
            if match is None:
                continue

            # Determine negation
            is_negated = False
            if annotator is not None:
                spans = [(match.start(), match.end(), field_name)]
                annotated = annotator.annotate(text, spans)
                if annotated and annotated[0].is_negated:
                    is_negated = True
            else:
                # Quick regex negation check
                context_start = max(0, match.start() - 50)
                context = text[context_start:match.start()]
                if re.search(r"(?i)\b(?:pas\s+(?:de|d['']\s*)|absence\s+(?:de|d['']\s*)|sans|aucun[e]?|ni)\s*$", context):
                    is_negated = True

            value = "non" if is_negated else "oui"

            if field_name not in results:
                results[field_name] = ExtractionValue(
                    value=value,
                    source_span=match.group(),
                    source_span_start=match.start(),
                    source_span_end=match.end(),
                    extraction_tier="rule",
                    confidence=0.8,
                    vocab_valid=True,
                )
            break  # First keyword match per field

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.6  Numerical extractor
# ═══════════════════════════════════════════════════════════════════════════

_PAT_KI67 = re.compile(
    r"(?:ki[- ]?67|index\s+de\s+prolif[ée]ration)"
    r"\s*[:=\-\s]\s*"
    r"(?:(?:environ|~)\s*)?"
    r"(?P<value>\d+(?:\s*(?:[àa-]\s*\d+))?)\s*%",
    _RE_FLAGS,
)

_PAT_KARNOFSKY = re.compile(
    r"(?:IK|Karnofsky|KPS|indice\s+de\s+Karnofsky)"
    r"\s*[:=\-àa\s]\s*"
    r"(?P<value>\d{2,3})\s*%?",
    _RE_FLAGS,
)

_PAT_MITOSES = re.compile(
    r"(?P<value>\d+)\s*mitoses?(?:\s*/\s*\d+\s*HPF)?",
    _RE_FLAGS,
)

_PAT_GRADE = re.compile(
    r"grade\s*[:=\-\s]?\s*(?P<value>[1-4]|I{1,3}V?|IV)",
    _RE_FLAGS,
)

_PAT_DOSE_GY = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*Gy",
    _RE_FLAGS,
)

_PAT_CYCLES = re.compile(
    r"(?P<value>\d+)\s*(?:cycles?|cures?)",
    _RE_FLAGS,
)

# Roman numeral to integer
_ROMAN_TO_INT: dict[str, int] = {
    "I": 1, "II": 2, "III": 3, "IV": 4,
}


def extract_numerical(text: str) -> dict[str, ExtractionValue]:
    """Extract numerical clinical values from *text*.

    Extracts: Ki67, Karnofsky index, mitoses count, grade, dose (Gy),
    and chemo cycles.
    """
    results: dict[str, ExtractionValue] = {}

    def _set(field_name: str, value: Any, raw: str, start: int, end: int,
             confidence: float = 0.9) -> None:
        if field_name not in results:
            results[field_name] = ExtractionValue(
                value=value,
                source_span=raw,
                source_span_start=start,
                source_span_end=end,
                extraction_tier="rule",
                confidence=confidence,
                vocab_valid=True,
            )

    # Ki67
    for m in _PAT_KI67.finditer(text):
        val = m.group("value").strip()
        _set("ihc_ki67", val, m.group(), m.start(), m.end())

    # Karnofsky
    for m in _PAT_KARNOFSKY.finditer(text):
        val = int(m.group("value"))
        _set("ik_clinique", val, m.group(), m.start(), m.end())

    # Mitoses
    for m in _PAT_MITOSES.finditer(text):
        val = int(m.group("value"))
        _set("histo_mitoses", val, m.group(), m.start(), m.end())

    # Grade
    for m in _PAT_GRADE.finditer(text):
        val_str = m.group("value").strip()
        if val_str in _ROMAN_TO_INT:
            val = _ROMAN_TO_INT[val_str]
        else:
            try:
                val = int(val_str)
            except ValueError:
                continue
        _set("grade", val, m.group(), m.start(), m.end())

    # Dose Gy
    for m in _PAT_DOSE_GY.finditer(text):
        val_str = m.group("value").replace(",", ".")
        _set("rx_dose", val_str, m.group(), m.start(), m.end(), confidence=0.8)

    # Chemo cycles
    for m in _PAT_CYCLES.finditer(text):
        val = int(m.group("value"))
        _set("chm_cycles", val, m.group(), m.start(), m.end(), confidence=0.8)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.7  Amplification extractor
# ═══════════════════════════════════════════════════════════════════════════

_AMPLI_GENES: list[str] = ["mdm2", "cdk4", "egfr", "met", "mdm4"]

_AMPLI_CANONICAL: dict[str, str] = {
    "mdm2": "ampli_mdm2",
    "cdk4": "ampli_cdk4",
    "egfr": "ampli_egfr",
    "met": "ampli_met",
    "mdm4": "ampli_mdm4",
}

# Pattern: "amplification <gene>" or "<gene> amplifié(e)"
_AMPLI_PATTERN = re.compile(
    r"(?:amplification\s+(?:de\s+|du\s+gène\s+)?(?P<gene1>"
    + "|".join(re.escape(g) for g in sorted(_AMPLI_GENES, key=len, reverse=True))
    + r"))"
    r"|(?:(?P<gene2>"
    + "|".join(re.escape(g) for g in sorted(_AMPLI_GENES, key=len, reverse=True))
    + r")\s+amplifié[e]?)",
    _RE_FLAGS,
)

# Negated amplification: "pas d'amplification <gene>", "absence d'amplification <gene>"
_AMPLI_NEGATED_PATTERN = re.compile(
    r"(?:pas\s+d[''']?\s*amplification|absence\s+d[''']?\s*amplification)"
    r"\s+(?:de\s+|du\s+gène\s+)?(?P<gene>"
    + "|".join(re.escape(g) for g in sorted(_AMPLI_GENES, key=len, reverse=True))
    + r")",
    _RE_FLAGS,
)


def extract_amplifications(text: str) -> dict[str, ExtractionValue]:
    """Extract gene amplification results from *text*.

    Returns a dict mapping canonical field names (e.g. ``ampli_mdm2``)
    to ``ExtractionValue`` objects with value ``"oui"`` or ``"non"``.
    """
    results: dict[str, ExtractionValue] = {}

    def _set(field_name: str, value: str, raw: str, start: int, end: int) -> None:
        if field_name not in results:
            results[field_name] = ExtractionValue(
                value=value,
                source_span=raw,
                source_span_start=start,
                source_span_end=end,
                extraction_tier="rule",
                confidence=0.9,
                vocab_valid=True,
            )

    # Negated amplifications (check FIRST — negated patterns are more
    # specific and must take priority via the "first match wins" `_set`).
    for m in _AMPLI_NEGATED_PATTERN.finditer(text):
        gene = m.group("gene").lower()
        field_name = _AMPLI_CANONICAL.get(gene)
        if field_name:
            _set(field_name, "non", m.group(), m.start(), m.end())

    # Positive amplifications
    for m in _AMPLI_PATTERN.finditer(text):
        gene = (m.group("gene1") or m.group("gene2")).lower()
        field_name = _AMPLI_CANONICAL.get(gene)
        if field_name:
            _set(field_name, "oui", m.group(), m.start(), m.end())

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.8  Fusion extractor
# ═══════════════════════════════════════════════════════════════════════════

_FUSION_GENES: list[str] = ["fgfr", "ntrk", "alk", "ros1", "met", "braf"]

_FUSION_CANONICAL: dict[str, str] = {
    "fgfr": "fusion_fgfr",
    "ntrk": "fusion_ntrk",
    # Others map to fusion_autre
}

# Pattern: "fusion <gene>" or "réarrangement <gene>"
_FUSION_PATTERN = re.compile(
    r"(?:fusion|r[ée]arrangement|translocation)\s+(?:de\s+|du\s+gène\s+)?(?P<gene>"
    + "|".join(re.escape(g) for g in sorted(_FUSION_GENES, key=len, reverse=True))
    + r")",
    _RE_FLAGS,
)

# Negated: "pas de fusion <gene>"
_FUSION_NEGATED_PATTERN = re.compile(
    r"(?:pas\s+de\s+(?:fusion|r[ée]arrangement)|absence\s+de\s+(?:fusion|r[ée]arrangement))"
    r"\s+(?:de\s+|du\s+gène\s+)?(?P<gene>"
    + "|".join(re.escape(g) for g in sorted(_FUSION_GENES, key=len, reverse=True))
    + r")",
    _RE_FLAGS,
)


def extract_fusions(text: str) -> dict[str, ExtractionValue]:
    """Extract gene fusion results from *text*.

    Returns a dict mapping canonical field names (e.g. ``fusion_fgfr``)
    to ``ExtractionValue`` objects with value ``"oui"`` or ``"non"``.
    """
    results: dict[str, ExtractionValue] = {}

    def _set(field_name: str, value: str, raw: str, start: int, end: int) -> None:
        if field_name not in results:
            results[field_name] = ExtractionValue(
                value=value,
                source_span=raw,
                source_span_start=start,
                source_span_end=end,
                extraction_tier="rule",
                confidence=0.9,
                vocab_valid=True,
            )

    # Negated fusions (check FIRST — more specific pattern wins).
    for m in _FUSION_NEGATED_PATTERN.finditer(text):
        gene = m.group("gene").lower()
        field_name = _FUSION_CANONICAL.get(gene, "fusion_autre")
        _set(field_name, "non", m.group(), m.start(), m.end())

    # Positive fusions
    for m in _FUSION_PATTERN.finditer(text):
        gene = m.group("gene").lower()
        field_name = _FUSION_CANONICAL.get(gene, "fusion_autre")
        _set(field_name, "oui", m.group(), m.start(), m.end())

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4.3   Master rule-based extraction function
# ═══════════════════════════════════════════════════════════════════════════

# Section → extractor mapping: which extractors are relevant for each section
_SECTION_EXTRACTORS: dict[str, list[str]] = {
    "ihc": ["ihc", "numerical"],
    "molecular": ["molecular", "amplification", "fusion"],
    "chromosomal": ["chromosomal", "amplification"],
    "macroscopy": ["binary", "numerical"],
    "microscopy": ["binary", "numerical", "ihc"],
    "conclusion": ["ihc", "molecular", "chromosomal", "numerical", "amplification", "fusion"],
    "history": ["date", "binary"],
    "treatment": ["date", "binary", "numerical"],
    "clinical_exam": ["binary", "numerical"],
    "radiology": ["binary", "date"],
    "full_text": ["date", "ihc", "molecular", "chromosomal", "binary", "numerical",
                  "amplification", "fusion"],
}


def run_rule_extraction(
    text: str,
    sections: dict[str, str],
    feature_subset: list[str],
    annotator: Optional[AssertionAnnotator] = None,
) -> dict[str, ExtractionValue]:
    """Run all rule-based extractors on the text/sections.

    Only keeps features present in *feature_subset*.

    Parameters
    ----------
    text : str
        The full document text.
    sections : dict[str, str]
        Section name → section text, as returned by ``SectionDetector``.
    feature_subset : list[str]
        List of field names that should be extracted (from
        ``FEATURE_ROUTING``).
    annotator : AssertionAnnotator, optional
        For negation-aware binary extraction.

    Returns
    -------
    dict[str, ExtractionValue]
        Mapping ``field_name → ExtractionValue`` for successfully
        extracted fields.
    """
    feature_set = set(feature_subset)
    all_results: dict[str, ExtractionValue] = {}

    def _merge(new: dict[str, ExtractionValue], section_name: str) -> None:
        """Merge new extractions, keeping the first (highest-priority) result."""
        for fname, ev in new.items():
            if fname in feature_set and fname not in all_results:
                ev.section = section_name
                all_results[fname] = ev

    # Run extractors on each section
    for section_name, section_text in sections.items():
        if not section_text.strip():
            continue

        extractor_names = _SECTION_EXTRACTORS.get(
            section_name,
            _SECTION_EXTRACTORS["full_text"],
        )

        if "date" in extractor_names:
            date_results = extract_dates(section_text)
            # Dates are generic — we don't auto-assign to fields here.
            # They'll be matched to specific date fields by the pipeline.
            # For now, store the first date per relevant date field.
            date_fields_in_subset = [
                f for f in feature_set
                if f not in all_results and _is_date_field(f)
            ]
            for i, dfname in enumerate(date_fields_in_subset):
                if i < len(date_results):
                    norm, raw, start, end = date_results[i]
                    all_results[dfname] = ExtractionValue(
                        value=norm,
                        source_span=raw,
                        source_span_start=start,
                        source_span_end=end,
                        extraction_tier="rule",
                        confidence=0.7,
                        section=section_name,
                        vocab_valid=True,
                    )

        if "ihc" in extractor_names:
            _merge(extract_ihc(section_text), section_name)

        if "molecular" in extractor_names:
            _merge(extract_molecular(section_text), section_name)

        if "chromosomal" in extractor_names:
            _merge(extract_chromosomal(section_text), section_name)

        if "binary" in extractor_names:
            _merge(extract_binary(section_text, annotator), section_name)

        if "numerical" in extractor_names:
            _merge(extract_numerical(section_text), section_name)

        if "amplification" in extractor_names:
            _merge(extract_amplifications(section_text), section_name)

        if "fusion" in extractor_names:
            _merge(extract_fusions(section_text), section_name)

    # If sections didn't cover everything, run on full text as catch-all
    if "full_text" not in sections:
        remaining = feature_set - set(all_results.keys())
        if remaining:
            if "ihc" in _relevant_groups(remaining):
                _merge(extract_ihc(text), "full_text")
            if "molecular" in _relevant_groups(remaining):
                _merge(extract_molecular(text), "full_text")
            if "chromosomal" in _relevant_groups(remaining):
                _merge(extract_chromosomal(text), "full_text")
            if "binary" in _relevant_groups(remaining):
                _merge(extract_binary(text, annotator), "full_text")
            if "numerical" in _relevant_groups(remaining):
                _merge(extract_numerical(text), "full_text")
            if "amplification" in _relevant_groups(remaining):
                _merge(extract_amplifications(text), "full_text")
            if "fusion" in _relevant_groups(remaining):
                _merge(extract_fusions(text), "full_text")

    return all_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_date_field(field_name: str) -> bool:
    """Return True if the field is a date-type field."""
    fd = ALL_FIELDS_BY_NAME.get(field_name)
    return fd is not None and fd.field_type == FieldType.DATE


def _relevant_groups(field_names: set[str]) -> set[str]:
    """Determine which extractor groups are relevant for the given field names."""
    groups: set[str] = set()
    for fn in field_names:
        if fn.startswith("ihc_"):
            groups.add("ihc")
        elif fn.startswith("mol_"):
            groups.add("molecular")
        elif fn.startswith("ch") and len(fn) <= 5:
            groups.add("chromosomal")
        elif fn.startswith("ampli_"):
            groups.add("amplification")
        elif fn.startswith("fusion_"):
            groups.add("fusion")
        elif fn.startswith("histo_"):
            groups.update({"binary", "numerical"})
        elif fn in ("grade", "ik_clinique", "histo_mitoses", "ihc_ki67",
                     "rx_dose", "chm_cycles"):
            groups.add("numerical")
        else:
            fd = ALL_FIELDS_BY_NAME.get(fn)
            if fd and fd.field_type == FieldType.CATEGORICAL:
                if fd.allowed_values and fd.allowed_values <= {"oui", "non", "Oui", "Non"}:
                    groups.add("binary")
    return groups
