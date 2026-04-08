"""Tier 1 rule-based feature extractors.

Regex and pattern-based extraction for dates, IHC results, molecular
status, chromosomal alterations, binary fields, numerical values,
amplifications, and fusions from French neuro-oncology documents.

Public API
----------
- ``extract_dates``              – Generic date extraction + normalisation.
- ``extract_ihc``                – IHC marker extraction.
- ``extract_molecular``          – Molecular status extraction.
- ``extract_chromosomal``        – Chromosomal alteration extraction.
- ``extract_binary``             – Binary (oui/non) field extraction.
- ``extract_numerical``          – Numerical value extraction (Ki67, IK, …).
- ``extract_amplifications``     – Gene amplification extraction.
- ``extract_fusions``            – Gene fusion extraction.
- ``extract_sexe``               – Patient sex (M/F) extraction.
- ``extract_tumeur_lateralite``  – Tumour laterality extraction.
- ``extract_evol_clinique``      – Clinical evolution label extraction.
- ``extract_type_chirurgie``     – Surgery type extraction.
- ``extract_classification_oms`` – WHO classification year extraction.
- ``extract_chimios``            – Chemotherapy drug name extraction.
- ``extract_tumeur_position``    – Tumour anatomical position extraction.
- ``extract_diag_histologique``  – Histological diagnosis extraction.
- ``run_rule_extraction``        – Master function running all extractors.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .schema import (
    ExtractionValue,
    FieldDefinition,
    FieldType,
    ALL_FIELDS_BY_NAME,
    FEATURE_GROUPS,
    get_field,
)
from .negation import AssertionAnnotator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RE_FLAGS = re.IGNORECASE | re.UNICODE

# Pseudonymised birthdate: "YYYY-??-??"
_PAT_PSEUDO_BIRTHDATE = re.compile(
    r"(\d{4})-\?\?-\?\?",
    _RE_FLAGS,
)

# Context keywords for date-field assignment (replaces positional matching)
_DATE_CONTEXT_KEYWORDS: dict[str, list[str]] = {
    "annee_de_naissance": [
        "né(e) le", "naissance", "DDN", "date de naissance", "année de naissance",
        "né le", "née le", "né en", "née en", "né(e) en"
    ],
    "date_rcp": ["rcp", "réunion de concertation pluridisciplinaire", "concertation", "staff"],
    "date_chir": [
        "chirurgie", "opéré", "intervention", "opération", "exérèse", "biopsie",
        "opéré le", "opérée le", "résection", "craniotomie", "craniectomie", "reprise chir",
    ],
    "chm_date_debut": [
        "début chimio", "chimiothérapie débutée", "TMZ depuis", "témozolomide depuis",
        "début de chimiothérapie", "TMZ le", "bévacizumab le", "avastin le",
        "lomustine le", "premier cycle le", "débuté le",
    ],
    "chm_date_fin": [
        "fin chimio", "arrêt chimio", "dernière cure", "fin de chimiothérapie",
        "fin TMZ", "dernier cycle"
    ],
    "rx_date_debut": [
        "début radiothérapie", "RT débutée", "irradiation depuis", "début RT",
        "début de radiothérapie", "RT le", "radio le", "irrad",
    ],
    "rx_date_fin": [
        "fin radiothérapie", "fin RT", "fin de radiothérapie", "fin d'irradiation",
        "fin radio", "fin irradiation"
    ],
    "date_1er_symptome": [
        "premier symptôme", "1er symptôme", "début des troubles", "apparition",
        "début des symptômes", "symptomatologie initiale", "premiers signes",
        "début de la symptomatologie"
    ],
    "exam_radio_date_decouverte": [
        "découverte", "IRM du", "scanner du", "imagerie du",
        "IRM de découverte", "scanner initial", "imagerie initiale", "TDM du"
    ],
    "date_progression": [
        "progression", "récidive", "rechute", "PD", "rechute le", "récidive le",
        "prise de contraste", "rehaussement", "augmentation de taille"
    ],
    "date_deces": ["décès", "décédé", "décédée"],
    "dn_date": [
        "dernière nouvelle", "dernières nouvelles", "consultation du",
        "vu(e) le", "vu le", "vue le", "vu en consultation", "revu le",
        "dernières nouvelles du", "DN du",
    ],
}

_DATE_CONTEXT_WINDOW = {
    "chm_date_debut": 300, "chm_date_fin": 300,
    "rx_date_debut": 300, "rx_date_fin": 300,
    "date_1er_symptome": 300,
}
_DEFAULT_CONTEXT_HALF = 200


def _assign_dates_by_context(
    date_results: list[tuple[str, str, int, int]],
    date_fields: list[str],
    text: str,
) -> dict[str, ExtractionValue]:
    """Assign extracted dates to the correct date fields using keyword context.

    For each extracted date, looks at 200 chars before and 200 chars after
    the date position in *text* and matches against keyword patterns for each
    date field.  Context is normalised (accent-stripped + lowercase) before
    keyword matching.  When two keywords compete for the same date, the
    nearest one wins.  Multiple dates can be assigned to the same field
    (the nearest keyword match wins).

    Returns
    -------
    dict[str, ExtractionValue]
        Mapping ``field_name → ExtractionValue`` for dates assigned by context.
    """
    from src.extraction.text_normalisation import normalise as _norm

    results: dict[str, ExtractionValue] = {}

    for _i, (norm, raw, start, end) in enumerate(date_results):
        best_field: str | None = None
        best_distance: int = 999999

        for field_name in date_fields:
            # Determine window size for this specific field
            half_win = _DATE_CONTEXT_WINDOW.get(field_name, _DEFAULT_CONTEXT_HALF)
            ctx_start = max(0, start - half_win)
            ctx_end = min(len(text), end + half_win)
            context = _norm(text[ctx_start:ctx_end])

            keywords = _DATE_CONTEXT_KEYWORDS.get(field_name, [])
            for kw in keywords:
                kw_norm = _norm(kw)
                pos = context.find(kw_norm)
                if pos != -1:
                    date_pos_in_ctx = start - ctx_start
                    distance = abs(pos - date_pos_in_ctx)
                    if distance < best_distance:
                        best_distance = distance
                        best_field = field_name

        if best_field is not None:
            # Keep nearest match per field (overwrite if closer)
            existing = results.get(best_field)
            if existing is None or best_distance < getattr(existing, "_ctx_distance", 999999):
                assigned_value = norm
                # For annee_de_naissance, extract year only from DD/MM/YYYY
                if best_field == "annee_de_naissance":
                    parts = norm.split("/")
                    if len(parts) == 3:
                        assigned_value = parts[2]
                ev = ExtractionValue(
                    value=assigned_value,
                    source_span=raw,
                    source_span_start=start,
                    source_span_end=end,
                    extraction_tier="rule",
                    confidence=0.7,
                    vocab_valid=True,
                )
                ev._ctx_distance = best_distance  # type: ignore[attr-defined]
                results[best_field] = ev

    return results


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
    "idh1", "idh-1", "p53", "atrx", "fgfr3", "braf",
    "h3k27m", "h3 k27m", "h3.3 k27m", "h3.3k27m", "histone h3 k27m",
    "h3k27me3", "h3 k27me3",
    "egfr", "egfr hirsch", "score hirsch",
    "gfap", "olig2", "ki67", "ki-67",
    "mmr", "mlh1", "msh2", "msh6", "pms2", "dmmr", "pmmr", "deficit mmr",
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
    "h3 k27m": "ihc_hist_h3k27m",
    "h3.3 k27m": "ihc_hist_h3k27m",
    "h3.3k27m": "ihc_hist_h3k27m",
    "histone h3 k27m": "ihc_hist_h3k27m",
    "h3k27me3": "ihc_hist_h3k27me3",
    "h3 k27me3": "ihc_hist_h3k27me3",
    "egfr": "ihc_egfr_hirsch",
    "egfr hirsch": "ihc_egfr_hirsch",
    "score hirsch": "ihc_egfr_hirsch",
    "gfap": "ihc_gfap",
    "olig2": "ihc_olig2",
    "ki67": "ihc_ki67",
    "ki-67": "ihc_ki67",
    "mmr": "ihc_mmr",
    "mlh1": "ihc_mmr",
    "msh2": "ihc_mmr",
    "msh6": "ihc_mmr",
    "pms2": "ihc_mmr",
    "dmmr": "ihc_mmr",
    "pmmr": "ihc_mmr",
    "deficit mmr": "ihc_mmr",
}

# Value normalisation
_IHC_VALUE_NORM: dict[str, str] = {
    # --- positif ---
    "positive": "positif",
    "positif": "positif",
    "positifs": "positif",
    "+": "positif",
    "surexprimé": "positif",
    "surexprime": "positif",
    "surexpression": "positif",
    "exprimé": "positif",
    "exprime": "positif",
    "présent": "positif",
    "present": "positif",
    # --- negatif ---
    "négative": "negatif",
    "negative": "negatif",
    "négatif": "negatif",
    "negatif": "negatif",
    "-": "negatif",
    "perte d'expression": "negatif",
    "perte d'expression": "negatif",
    "absent": "negatif",
    "absence d'expression": "negatif",
    "perte": "negatif",
    "non exprimé": "negatif",
    "non exprime": "negatif",
    "non détecté": "negatif",
    "non detecte": "negatif",
    "non retrouvé": "negatif",
    "non retrouve": "negatif",
    "perdu": "negatif",
    # --- maintenu ---
    "maintenu": "maintenu",
    "maintenue": "maintenu",
    "conservé": "maintenu",
    "conserve": "maintenu",
    "conservée": "maintenu",
    "conservee": "maintenu",
    "expression conservée": "maintenu",
    "expression conservee": "maintenu",
    "expression maintenue": "maintenu",
    "préservé": "maintenu",
    "preserve": "maintenu",
    "normal": "maintenu",
}

_IHC_PATTERN = re.compile(
    r"(?P<marker>"
    + "|".join(re.escape(m) for m in sorted(IHC_MARKERS, key=len, reverse=True))
    + r")"
    r"\s*[:=\-\s]\s*"
    r"(?P<value>"
    r"positif[s]?|n[ée]gatif(?:ve)?|positive?|n[ée]gative?"
    r"|maintenu[e]?|perte\s+d[''']expression|absence\s+d[''']expression"
    r"|conserv[ée]e?|expression\s+(?:conserv[ée]e?|maintenue)"
    r"|surexprim[ée]|surexpression|exprim[ée]|pr[ée]sent"
    r"|absent|perte|perdu"
    r"|non\s+(?:exprim[ée]|d[ée]tect[ée]|retrouv[ée])"
    r"|pr[ée]serv[ée]|normal"
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
            # Try to extract a percentage range (e.g. "15 à 20%") or single %
            range_match = re.search(r"(\d+)\s*[àa]\s*(\d+)\s*%", value_raw)
            lt_match = re.search(r"<\s*(\d+)\s*%", value_raw)
            pct_match = re.search(r"(\d+)\s*%", value_raw)
            score_match = re.search(r"score\s+(?:de\s+)?(\d+)", value_raw)
            if range_match:
                normalised = f"{range_match.group(1)}-{range_match.group(2)}"
            elif lt_match:
                normalised = f"<{lt_match.group(1)}"
            elif pct_match:
                normalised = pct_match.group(1)
            elif score_match:
                normalised = score_match.group(1)
            else:
                normalised = value_raw

        # Map ATRX positif to maintenu
        if field_name == "ihc_atrx" and normalised == "positif":
            normalised = "maintenu"

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
    # --- wt ---
    "wt": "wt",
    "wild-type": "wt",
    "wild type": "wt",
    "sauvage": "wt",
    "type sauvage": "wt",
    "séquence sauvage": "wt",
    "sequence sauvage": "wt",
    "non muté": "wt",
    "non mutée": "wt",
    "non mute": "wt",
    "non mutee": "wt",
    "non muté(e)": "wt",
    "absence de mutation": "wt",
    "pas de mutation": "wt",
    "pas de mutation détectée": "wt",
    "pas de mutation detectee": "wt",
    "absence de mutation détectée": "wt",
    "absence de mutation detectee": "wt",
    "statut wt": "wt",
    "status wt": "wt",
    # --- mute ---
    "muté": "mute",
    "mutée": "mute",
    "mute": "mute",
    "mutee": "mute",
    "mutation": "mute",
    "présence de mutation": "mute",
    "muté(e)": "mute",
    "mutation détectée": "mute",
    "mutation detectee": "mute",
    "mutation identifiée": "mute",
    "mutation identifiee": "mute",
    "variant pathogène": "mute",
    "variant pathogene": "mute",
    "altéré": "mute",
    "altere": "mute",
    "altérée": "mute",
    "alteree": "mute",
    "altéré(e)": "mute",
    # --- methyle (MGMT) ---
    "méthylé": "methyle",
    "methylé": "methyle",
    "methyle": "methyle",
    "methylation positive": "methyle",
    "méthylation positive": "methyle",
    "promoteur méthylé": "methyle",
    "promoteur methyle": "methyle",
    "méthylation du promoteur": "methyle",
    "methylation du promoteur": "methyle",
    "hyperméthylé": "methyle",
    "hypermethyle": "methyle",
    # --- non methyle (MGMT) ---
    "non méthylé": "non methyle",
    "non methylé": "non methyle",
    "non methyle": "non methyle",
    "methylation negative": "non methyle",
    "méthylation négative": "non methyle",
    "non methylation": "non methyle",
    "absence de méthylation": "non methyle",
    "absence de methylation": "non methyle",
    "méthylation absente": "non methyle",
    "methylation absente": "non methyle",
    "non hyperméthylé": "non methyle",
    "non hypermethyle": "non methyle",
    "promoteur non méthylé": "non methyle",
    "promoteur non methyle": "non methyle",
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
    r"wt|wild[- ]?type|sauvage|type\s+sauvage|s[ée]quence\s+sauvage"
    r"|statut?\s+wt|status\s+wt"
    r"|non\s+mut[ée]e?(?:\(e\))?"
    r"|mut[ée]e?(?:\(e\))?|mutation"
    r"|mutation\s+(?:d[ée]tect[ée]e?|identifi[ée]e?)"
    r"|variant\s+pathog[eè]ne"
    r"|alt[ée]r[ée]e?(?:\(e\))?"
    r"|pr[ée]sence\s+de\s+mutation"
    r"|absence\s+de\s+mutation(?:\s+d[ée]tect[ée]e?)?"
    r"|pas\s+de\s+mutation(?:\s+d[ée]tect[ée]e?)?"
    r"|promoteur\s+(?:non\s+)?m[ée]thyl[ée]"
    r"|m[ée]thylation\s+(?:du\s+promoteur|positive|n[ée]gative|absente)"
    r"|hyper?m[ée]thyl[ée]|non\s+hyper?m[ée]thyl[ée]"
    r"|m[ée]thyl[ée]|non\s+m[ée]thyl[ée]"
    r"|absence\s+de\s+m[ée]thylation"
    r"|(?:p\.)?[A-Z]\d+[A-Z]"  # Variant like R132H
    r")",
    _RE_FLAGS,
)

# Special pattern: "pas de mutation <gene>" / "absence de mutation <gene>"
_MOL_NEGATED_PATTERN = re.compile(
    r"(?:pas\s+de\s+mutation(?:\s+d[ée]tect[ée]e?)?|absence\s+de\s+mutation(?:\s+d[ée]tect[ée]e?)?)\s+"
    r"(?:du?\s+g[eè]ne?\s+)?"
    r"(?P<gene>"
    + "|".join(re.escape(g) for g in sorted(MOL_GENES, key=len, reverse=True))
    + r")",
    _RE_FLAGS,
)

# Special pattern: "mutation <gene> <variant>"
_MOL_MUTATION_GENE_PATTERN = re.compile(
    r"(?:mutation|mutation\s+(?:d[ée]tect[ée]e?|identifi[ée]e?))\s+"
    r"(?:du?\s+(?:g[eè]ne?\s+)?)?(?:promoteur\s+(?:du?\s+)?)?(?P<gene>"
    + "|".join(re.escape(g) for g in sorted(MOL_GENES, key=len, reverse=True))
    + r")"
    r"(?:\s*[:(\s]\s*(?P<variant>(?:p\.)?[A-Z]\d+[A-Z])\s*[)]?)?",
    _RE_FLAGS,
)

# MGMT special pattern (dedicated, covers compact notations)
_MGMT_SPECIAL_PATTERN = re.compile(
    r"MGMT\s*[:\-]?\s*(?P<status>"
    r"promoteur\s+(?:non\s+)?m[ée]thyl[ée]"
    r"|m[ée]thylation\s+(?:du\s+promoteur|positive|n[ée]gative|absente)"
    r"|m[ée]thyl[ée]|non\s+m[ée]thyl[ée]"
    r"|hyper?m[ée]thyl[ée]|non\s+hyper?m[ée]thyl[ée]"
    r"|wt|mut[ée]e?"
    r")",
    _RE_FLAGS,
)

# Additional negation patterns: "statut WT <gene>", "gène <gene> sauvage"
_MOL_WT_STATUS_PATTERN = re.compile(
    r"(?:statut?\s+WT|status\s+WT)\s+"
    r"(?:du?\s+g[eè]ne?\s+)?"
    r"(?P<gene>"
    + "|".join(re.escape(g) for g in sorted(MOL_GENES, key=len, reverse=True))
    + r")",
    _RE_FLAGS,
)

_MOL_GENE_SAUVAGE_PATTERN = re.compile(
    r"g[eè]ne\s+(?P<gene>"
    + "|".join(re.escape(g) for g in sorted(MOL_GENES, key=len, reverse=True))
    + r")\s+sauvage",
    _RE_FLAGS,
)

# Codeletion pattern (standard order: codélétion 1p/19q)
_CODELETION_PATTERN = re.compile(
    r"(?:cod[ée]l[ée]tion|co-d[ée]l[ée]tion)\s+(?:des?\s+)?(?:bras?\s+)?1p[/\s]+(?:et\s+)?19q",
    _RE_FLAGS,
)

# Reversed codeletion: "1p19q codélétion"
_CODELETION_REVERSED_PATTERN = re.compile(
    r"1p[/\s]*19q\s+(?:cod[ée]l[ée]t|co-d[ée]l[ée]t)",
    _RE_FLAGS,
)

# CDKN2A homozygous deletion → sets mol_CDKN2A and ch9p
_CDKN2A_HOMODEL_PATTERN = re.compile(
    r"d[ée]l[ée]tion\s+(?:homo(?:zygote)?|bi[- ]?all[ée]lique)\s+(?:de\s+)?CDKN2A",
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
            if value == "mute":
                value = f"mute ({raw.strip()})"
            
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
            # Check for variant pattern
            variant_match = _VARIANT_PATTERN.search(status_raw)
            if variant_match:
                normalised = "mute"
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

    # 4) MGMT special pattern (compact notations)
    for m in _MGMT_SPECIAL_PATTERN.finditer(text):
        status_raw = m.group("status").strip().lower()
        normalised = _MOL_STATUS_NORM.get(status_raw, status_raw)
        _set("mol_mgmt", normalised, m.group(), m.start(), m.end())

    # 5) "statut WT <gene>" / "status WT <gene>"
    for m in _MOL_WT_STATUS_PATTERN.finditer(text):
        gene_raw = m.group("gene").lower().strip()
        field_name = _MOL_CANONICAL.get(gene_raw)
        if field_name:
            _set(field_name, "wt", m.group(), m.start(), m.end())

    # 6) "gène <gene> sauvage"
    for m in _MOL_GENE_SAUVAGE_PATTERN.finditer(text):
        gene_raw = m.group("gene").lower().strip()
        field_name = _MOL_CANONICAL.get(gene_raw)
        if field_name:
            _set(field_name, "wt", m.group(), m.start(), m.end())

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.4  Chromosomal alteration extractor
# ═══════════════════════════════════════════════════════════════════════════

_CHROMOSOME_ARMS: list[str] = ["1p", "19q", "10p", "10q", "7p", "7q", "9p", "9q"]

_CHR_CANONICAL: dict[str, str] = {arm: f"ch{arm}" for arm in _CHROMOSOME_ARMS}

_CHR_STATUS_NORM: dict[str, str] = {
    # --- gain ---
    "gain": "gain",
    "gain de signal": "gain",
    "polysomie": "gain",
    "trisomie": "gain",
    "normal": "gain",
    "normale": "gain",
    # --- perte ---
    "perte": "perte",
    "délétion": "perte",
    "deletion": "perte",
    "deleted": "perte",
    "del": "perte",
    "monosomie": "perte",
    "perte de signal": "perte",
    "perte d'hétérozygotie": "perte",
    "perte d'heterozygotie": "perte",
    "loh": "perte",
    "perte allélique": "perte",
    "perte allelique": "perte",
    "perte homozygote": "perte",
    # --- perte partielle ---
    "perte partielle": "perte partielle",
    "perte hétérozygote": "perte partielle",
    "perte heterozygote": "perte partielle",
    "délétion focale": "perte partielle",
    "deletion focale": "perte partielle",
    "délétion partielle": "perte partielle",
    "deletion partielle": "perte partielle",
    "perte focale": "perte partielle",
    "perte hémizygote": "perte partielle",
    "perte hemizygote": "perte partielle",
}

_CHR_PATTERN = re.compile(
    r"(?P<arm>"
    + "|".join(re.escape(a) for a in sorted(_CHROMOSOME_ARMS, key=len, reverse=True))
    + r")"
    r"\s*[:=\-\s]\s*"
    r"(?P<status>"
    r"gain(?:\s+de\s+signal)?|perte(?:\s+(?:partielle|focale|all[ée]lique|de\s+signal|d[''']h[ée]t[ée]rozygotie|h[ée]mi?zygote|homo(?:zygote)?|h[ée]t[ée]ro(?:zygote)?))?"
    r"|d[ée]l[ée]tion(?:\s+(?:focale|partielle))?"
    r"|deleted?|del"
    r"|monosomie|polysomie|trisomie|loh"
    r"|normal[e]?"
    r")",
    _RE_FLAGS,
)

# CGH array short notation: "1p -", "7q +", "19q -"
_CHR_CGH_SHORT_PATTERN = re.compile(
    r"\b(?P<arm>" + "|".join(re.escape(a) for a in sorted(_CHROMOSOME_ARMS, key=len, reverse=True))
    + r")\s*(?P<sign>[+\-])\s",
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

    # CGH array short notation: "1p -", "7q +"
    for m in _CHR_CGH_SHORT_PATTERN.finditer(text):
        arm = m.group("arm").lower()
        sign = m.group("sign")
        field_name = _CHR_CANONICAL.get(arm)
        if field_name:
            value = "gain" if sign == "+" else "perte"
            _set(field_name, value, m.group(), m.start(), m.end())

    # Codeletion 1p/19q (standard order)
    for m in _CODELETION_PATTERN.finditer(text):
        _set("ch1p", "perte", m.group(), m.start(), m.end())
        _set("ch19q", "perte", m.group(), m.start(), m.end())

    # Codeletion 1p19q (reversed order)
    for m in _CODELETION_REVERSED_PATTERN.finditer(text):
        _set("ch1p", "perte", m.group(), m.start(), m.end())
        _set("ch19q", "perte", m.group(), m.start(), m.end())

    # CDKN2A homozygous deletion → sets mol_CDKN2A and ch9p
    for m in _CDKN2A_HOMODEL_PATTERN.finditer(text):
        _set("mol_CDKN2A", "mute", m.group(), m.start(), m.end())
        _set("ch9p", "perte", m.group(), m.start(), m.end())

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
        "foyers de nécrose", "nécrose palissadique", "nécroses en palissade", "nécrose extensive",
    ],
    "histo_pec": [
        "prolifération endothéliocapillaire", "proliferation endotheliocapillaire",
        "prolifération endothélio-capillaire", "PEC",
        "hyperplasie endothéliocapillaire", "prolifération vasculaire", "hyperplasie vasculaire", "néovascularisation",
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
        "tumeur antérieure", "néoplasie antérieure", "cancer antérieur", "tumeur préexistante",
    ],
    "epilepsie_1er_symptome": [
        "crise inaugurale", "crise révélatrice", "épilepsie inaugurale",
    ],
}

# Map binary keywords to their corresponding "first symptom" variants
_BINARY_1ER_SYMPTOME: dict[str, str] = {
    "epilepsie": "epilepsie_1er_symptome",
    "ceph_hic": "ceph_hic_1er_symptome",
    "deficit": "deficit_1er_symptome",
    "cognitif": "cognitif_1er_symptome",
}

_FIRST_SYMPTOM_MARKERS = [
    "premier", "initial", "inaugural", "révélateur", "découverte",
    "motif de consultation", "1er symptôme", "symptôme révélateur",
]

def _is_first_symptom_context(text: str, match_start: int, window: int = 200) -> bool:
    context = text[max(0, match_start - window):match_start].lower()
    return any(marker in context for marker in _FIRST_SYMPTOM_MARKERS)


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
            pattern = re.compile(r"\b" + re.escape(kw) + r"\b", _RE_FLAGS)
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
            
            # Disambiguate first-symptom
            target_field = field_name
            if field_name in _BINARY_1ER_SYMPTOME:
                if _is_first_symptom_context(text, match.start()):
                    target_field = _BINARY_1ER_SYMPTOME[field_name]

            if target_field not in results:
                results[target_field] = ExtractionValue(
                    value=value,
                    source_span=match.group(),
                    source_span_start=match.start(),
                    source_span_end=match.end(),
                    extraction_tier="rule",
                    confidence=0.8,
                    vocab_valid=True,
                )
            # Cannot break early if disambiguating, as a later match in the text 
            # might be the other variant (first symptom vs regular).
            # We break after satisfying both if it's a dual field, or just 1 if not.
            if field_name not in _BINARY_1ER_SYMPTOME:
                break
            elif target_field == field_name and _BINARY_1ER_SYMPTOME[field_name] in results:
                break
            elif target_field == _BINARY_1ER_SYMPTOME[field_name] and field_name in results:
                break

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
        val = str(int(m.group("value")))
        _set("ik_clinique", val, m.group(), m.start(), m.end())

    # Mitoses
    for m in _PAT_MITOSES.finditer(text):
        val = str(int(m.group("value")))
        _set("histo_mitoses", val, m.group(), m.start(), m.end())

    # Grade
    for m in _PAT_GRADE.finditer(text):
        val_str = m.group("value").strip()
        if val_str in _ROMAN_TO_INT:
            val = str(_ROMAN_TO_INT[val_str])
        else:
            try:
                val = str(int(val_str))
            except ValueError:
                continue
        _set("grade", val, m.group(), m.start(), m.end())

    # Dose Gy
    for m in _PAT_DOSE_GY.finditer(text):
        val_str = m.group("value").replace(",", ".")
        _set("rx_dose", val_str, m.group(), m.start(), m.end(), confidence=0.8)

    # Chemo cycles
    for m in _PAT_CYCLES.finditer(text):
        val = str(int(m.group("value")))
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
# 4.2.9  Sex extractor (Phase A1)
# ═══════════════════════════════════════════════════════════════════════════

_PAT_SEX_HEADER = re.compile(r"\|\s*(?P<sex>[MF])\s*\|")

_PAT_SEX_SALUTATION_F = re.compile(
    r"\b(?:Madame|Mme|née\s+le)\b",
    _RE_FLAGS,
)
_PAT_SEX_SALUTATION_M = re.compile(
    r"\b(?:Monsieur|né\s+le)\b",
    _RE_FLAGS,
)

_PAT_SEX_AGREEMENT_F = re.compile(r"\bvue\s+en\b", _RE_FLAGS)
_PAT_SEX_AGREEMENT_M = re.compile(r"\bvu\s+en\b", _RE_FLAGS)

_SEX_TOKEN_MAP: dict[str, str] = {
    "homme": "M", "femme": "F",
    "féminin": "F", "feminin": "F",
    "masculin": "M",
}

_PAT_SEX_TOKENS = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in _SEX_TOKEN_MAP) + r")\b",
    _RE_FLAGS,
)


def extract_sexe(text: str) -> dict[str, ExtractionValue]:
    """Extract patient sex (M/F) from *text*.

    Priority: header line > salutation > past participle agreement > tokens.
    """
    # 1. Header: "| F |" or "| M |"
    m = _PAT_SEX_HEADER.search(text)
    if m:
        return {"sexe": ExtractionValue(
            value=m.group("sex").upper(),
            source_span=m.group(), source_span_start=m.start(),
            source_span_end=m.end(), extraction_tier="rule",
            confidence=0.95, vocab_valid=True,
        )}

    # 2. Salutation
    m = _PAT_SEX_SALUTATION_F.search(text)
    if m:
        return {"sexe": ExtractionValue(
            value="F",
            source_span=m.group(), source_span_start=m.start(),
            source_span_end=m.end(), extraction_tier="rule",
            confidence=0.80, vocab_valid=True,
        )}
    m = _PAT_SEX_SALUTATION_M.search(text)
    if m:
        return {"sexe": ExtractionValue(
            value="M",
            source_span=m.group(), source_span_start=m.start(),
            source_span_end=m.end(), extraction_tier="rule",
            confidence=0.80, vocab_valid=True,
        )}

    # 3. Past participle agreement: "vue en" → F, "vu en" → M
    m = _PAT_SEX_AGREEMENT_F.search(text)
    if m:
        return {"sexe": ExtractionValue(
            value="F",
            source_span=m.group(), source_span_start=m.start(),
            source_span_end=m.end(), extraction_tier="rule",
            confidence=0.65, vocab_valid=True,
        )}
    m = _PAT_SEX_AGREEMENT_M.search(text)
    if m:
        return {"sexe": ExtractionValue(
            value="M",
            source_span=m.group(), source_span_start=m.start(),
            source_span_end=m.end(), extraction_tier="rule",
            confidence=0.65, vocab_valid=True,
        )}

    # 4. Explicit tokens: "homme", "femme", "masculin", "féminin"
    m = _PAT_SEX_TOKENS.search(text)
    if m:
        token = m.group().lower()
        sex = _SEX_TOKEN_MAP.get(token)
        if sex:
            return {"sexe": ExtractionValue(
                value=sex,
                source_span=m.group(), source_span_start=m.start(),
                source_span_end=m.end(), extraction_tier="rule",
                confidence=0.70, vocab_valid=True,
            )}

    return {}


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.10  Tumour laterality extractor (Phase A2)
# ═══════════════════════════════════════════════════════════════════════════

_LATERALITY_NORM: dict[str, str] = {
    "gauche": "gauche", "droit": "droite", "droite": "droite",
    "bilateral": "bilateral", "bilaterale": "bilateral",
    "bilatéral": "bilateral", "bilatérale": "bilateral",
    "median": "median", "mediane": "median",
    "médian": "median", "médiane": "median",
}

_PAT_LAT_EXPLICIT = re.compile(
    r"(?:lat[eé]ralit[eé]|c[oô]t[eé])\s*[:\-]?\s*"
    r"(?P<side>gauche|droit(?:e)?|bilat[eé]ral(?:e)?|m[eé]dian(?:e)?)",
    _RE_FLAGS,
)

_PAT_LAT_ANATOMICAL = re.compile(
    r"(?:h[eé]misph[eè]re|lobe|frontal|temporal|pari[eé]tal|occipital"
    r"|thalamus|insula|cervelet)\s+"
    r"(?P<side>gauche|droit(?:e)?|bilat[eé]ral(?:e)?)",
    _RE_FLAGS,
)

_PAT_LAT_PROXIMITY_SIDE = re.compile(
    r"\b(?P<side>gauche|droit(?:e)?|bilat[eé]ral(?:e)?|m[eé]dian(?:e)?)\b",
    _RE_FLAGS,
)

_PAT_LAT_TUMOUR_ANCHOR = re.compile(
    r"\b(?:tumeur|l[eé]sion|processus|masse)\b",
    _RE_FLAGS,
)


def extract_tumeur_lateralite(text: str) -> dict[str, ExtractionValue]:
    """Extract tumour laterality (gauche/droite/bilateral/median).

    Rules:
    1. Explicit label: "latéralité : gauche"
    2. Anatomical adjective: "lobe frontal droit"
    3. Side keyword within 200 chars of tumour anchor word
    """
    # 1. Explicit label
    m = _PAT_LAT_EXPLICIT.search(text)
    if m:
        raw_side = m.group("side").lower()
        norm = _LATERALITY_NORM.get(raw_side, raw_side)
        return {"tumeur_lateralite": ExtractionValue(
            value=norm,
            source_span=m.group(), source_span_start=m.start(),
            source_span_end=m.end(), extraction_tier="rule",
            confidence=0.95, vocab_valid=True,
        )}

    # 2. Anatomical adjective
    m = _PAT_LAT_ANATOMICAL.search(text)
    if m:
        raw_side = m.group("side").lower()
        norm = _LATERALITY_NORM.get(raw_side, raw_side)
        return {"tumeur_lateralite": ExtractionValue(
            value=norm,
            source_span=m.group(), source_span_start=m.start(),
            source_span_end=m.end(), extraction_tier="rule",
            confidence=0.90, vocab_valid=True,
        )}

    # 3. Side keyword near tumour anchor (within 200 chars)
    for anchor in _PAT_LAT_TUMOUR_ANCHOR.finditer(text):
        search_start = max(0, anchor.start() - 200)
        search_end = min(len(text), anchor.end() + 200)
        window = text[search_start:search_end]
        m = _PAT_LAT_PROXIMITY_SIDE.search(window)
        if m:
            raw_side = m.group("side").lower()
            norm = _LATERALITY_NORM.get(raw_side, raw_side)
            abs_start = search_start + m.start()
            abs_end = search_start + m.end()
            return {"tumeur_lateralite": ExtractionValue(
                value=norm,
                source_span=m.group(), source_span_start=abs_start,
                source_span_end=abs_end, extraction_tier="rule",
                confidence=0.75, vocab_valid=True,
            )}

    return {}


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.11  Clinical evolution extractor (Phase A3)
# ═══════════════════════════════════════════════════════════════════════════

_PAT_EVOL_EXPLICIT = re.compile(
    r"(?:[eé]volution|[eé]tape|temps|timepoint|dn\s+n[°º]?\s*|point)"
    r"\s*[:\-]?\s*(?P<label>initial|P\d+|terminal)",
    _RE_FLAGS,
)

_PAT_EVOL_HEADER = re.compile(
    r"(?:^|[\-:–—|/]\s*)(?P<label>initial|terminal)\b"
    r"|(?:^|[\-:–—|/\s]\s*)(?P<label2>P\d+)\b",
    _RE_FLAGS | re.MULTILINE,
)

_ORDINAL_PROGRESSION: dict[str, str] = {
    "premiere": "P1", "première": "P1", "1ere": "P1",
    "1ère": "P1", "1re": "P1",
    "deuxieme": "P2", "deuxième": "P2", "2eme": "P2",
    "2ème": "P2", "2e": "P2",
    "troisieme": "P3", "troisième": "P3", "3eme": "P3",
    "3ème": "P3", "3e": "P3",
    "quatrieme": "P4", "quatrième": "P4", "4eme": "P4",
    "4ème": "P4", "4e": "P4",
    "cinquieme": "P5", "cinquième": "P5", "5eme": "P5",
    "5ème": "P5", "5e": "P5",
}

_PAT_EVOL_ORDINAL = re.compile(
    r"(?P<ord>"
    + "|".join(re.escape(k) for k in sorted(_ORDINAL_PROGRESSION, key=len, reverse=True))
    + r")\s+progression",
    _RE_FLAGS,
)

_PAT_EVOL_INITIAL_CONTEXT = re.compile(
    r"\b(?:premi[eè]re\s+consultation|bilan\s+initial|diagnostique?\s+initial)\b",
    _RE_FLAGS,
)

_PROGRESSION_PATTERNS = [
    (re.compile(r"(?:1[èe]re|premi[èe]re)\s+(?:récidive|progression)", _RE_FLAGS), "P1"),
    (re.compile(r"(?:2[èe]me|deuxi[èe]me)\s+(?:récidive|progression)", _RE_FLAGS), "P2"),
    (re.compile(r"(?:3[èe]me|troisi[èe]me)\s+(?:récidive|progression)", _RE_FLAGS), "P3"),
    (re.compile(r"\brécidive\b", _RE_FLAGS), "P1"),  # bare "récidive" defaults to P1
]


def extract_evol_clinique(text: str) -> dict[str, ExtractionValue]:
    """Extract clinical evolution label (initial/P1/P2/…/terminal).

    Priority: explicit label > document header > ordinal progression > context.
    Prefers NA over a wrong label.
    """
    # 1. Explicit label
    m = _PAT_EVOL_EXPLICIT.search(text)
    if m:
        label = m.group("label")
        label = label.lower() if label.lower() == "initial" or label.lower() == "terminal" else label.upper()
        return {"evol_clinique": ExtractionValue(
            value=label,
            source_span=m.group(), source_span_start=m.start(),
            source_span_end=m.end(), extraction_tier="rule",
            confidence=0.95, vocab_valid=True,
        )}

    # 2. Document header (first 200 chars)
    header = text[:200]
    m = _PAT_EVOL_HEADER.search(header)
    if m:
        label = m.group("label") or m.group("label2")
        label = label.lower() if label.lower() in ("initial", "terminal") else label.upper()
        return {"evol_clinique": ExtractionValue(
            value=label,
            source_span=m.group(), source_span_start=m.start(),
            source_span_end=m.end(), extraction_tier="rule",
            confidence=0.85, vocab_valid=True,
        )}

    # 3. Ordinal progression: "deuxième progression" → P2
    m = _PAT_EVOL_ORDINAL.search(text)
    if m:
        ord_key = m.group("ord").lower()
        label = _ORDINAL_PROGRESSION.get(ord_key)
        if label:
            return {"evol_clinique": ExtractionValue(
                value=label,
                source_span=m.group(), source_span_start=m.start(),
                source_span_end=m.end(), extraction_tier="rule",
                confidence=0.85, vocab_valid=True,
            )}

    # 3.5 Progression markers
    for pat, label in _PROGRESSION_PATTERNS:
        m = pat.search(text)
        if m:
            return {"evol_clinique": ExtractionValue(
                value=label,
                source_span=m.group(), source_span_start=m.start(),
                source_span_end=m.end(), extraction_tier="rule",
                confidence=0.85, vocab_valid=True,
            )}

    # 4. Context inference (flagged — lower confidence)
    m = _PAT_EVOL_INITIAL_CONTEXT.search(text)
    if m:
        # If we are in a history context, this is likely a past "initial" mention, not the current one.
        # But we don't have the section easily here, so we just set confidence lower.
        return {"evol_clinique": ExtractionValue(
            value="initial",
            source_span=m.group(), source_span_start=m.start(),
            source_span_end=m.end(), extraction_tier="rule",
            confidence=0.60, vocab_valid=True,
            flagged=True,
        )}

    return {}


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.12  Surgery type extractor (Phase A4)
# ═══════════════════════════════════════════════════════════════════════════

_PAT_CHIR_COMPLETE = re.compile(
    r"(?:ex[eé]r[eè]se|r[eé]section)\s+(?:macroscopiquement\s+)?(?:compl[eè]te?|totale?|en\s+totalit[eé])\b",
    _RE_FLAGS,
)
_PAT_CHIR_GTR = re.compile(r"\bGTR\b")

_PAT_CHIR_PARTIELLE = re.compile(
    r"(?:ex[eé]r[eè]se|r[eé]section)\s+(?:partielle?|sub-?totale?|incompl[eè]te?)\b",
    _RE_FLAGS,
)
_PAT_CHIR_STR = re.compile(r"\bSTR\b")

_PAT_CHIR_BIOPSIE = re.compile(r"\bbiopsie\b", _RE_FLAGS)
_PAT_CHIR_BST = re.compile(r"\bBST\b")

_PAT_CHIR_EXERESE_BARE = re.compile(
    r"\b(?:ex[eé]r[eè]se|r[eé]section)\b",
    _RE_FLAGS,
)

_PAT_CHIR_ATTENTE = re.compile(
    r"(?:en\s+attente\s+de\s+chirurgie|chirurgie\s+(?:à\s+planifier|pr[eé]vue|non\s+r[eé]alis[eé]e))",
    _RE_FLAGS,
)


def extract_type_chirurgie(text: str) -> dict[str, ExtractionValue]:
    """Extract surgery type from *text*.

    Priority (most specific first):
    exerese complete > exerese partielle > biopsie > exerese (bare) > en attente
    """
    def _ev(value: str, m: re.Match, confidence: float = 0.90) -> dict[str, ExtractionValue]:
        return {"type_chirurgie": ExtractionValue(
            value=value,
            source_span=m.group(), source_span_start=m.start(),
            source_span_end=m.end(), extraction_tier="rule",
            confidence=confidence, vocab_valid=True,
        )}

    # Check most specific patterns first
    m = _PAT_CHIR_COMPLETE.search(text)
    if m:
        return _ev("exerese complete", m)
    m = _PAT_CHIR_GTR.search(text)
    if m:
        return _ev("exerese complete", m)

    m = _PAT_CHIR_PARTIELLE.search(text)
    if m:
        return _ev("exerese partielle", m)
    m = _PAT_CHIR_STR.search(text)
    if m:
        return _ev("exerese partielle", m)

    m = _PAT_CHIR_BIOPSIE.search(text)
    if m:
        # Check context: if preceded by "pièce de", this is a specimen description, 
        # not the surgery type itself (unless it's the only thing we have)
        context_start = max(0, m.start() - 20)
        context = text[context_start:m.start()].lower()
        if "pièce de" in context or "piece de" in context:
            # We skip this match and search for another one later in the text if possible
            # But the regex search only finds the first one. Let's use finditer.
            pass # We'll handle this in a rewrite below
        else:
            return _ev("biopsie", m, 0.85)
            
    # Rewrite biopsie search to handle "pièce de biopsie"
    for m in _PAT_CHIR_BIOPSIE.finditer(text):
        context_start = max(0, m.start() - 20)
        context = text[context_start:m.start()].lower()
        if "pièce de" not in context and "piece de" not in context:
            return _ev("biopsie", m, 0.85)

    m = _PAT_CHIR_BST.search(text)
    if m:
        return _ev("biopsie", m, 0.85)

    m = _PAT_CHIR_EXERESE_BARE.search(text)
    if m:
        # Same check for bare exerese
        return _ev("exerese", m, 0.70)

    # Re-evaluate bare exerese with the same logic
    for m in _PAT_CHIR_EXERESE_BARE.finditer(text):
        context_start = max(0, m.start() - 20)
        context = text[context_start:m.start()].lower()
        if "pièce d" not in context and "piece d" not in context:
            return _ev("exerese", m, 0.70)

    m = _PAT_CHIR_ATTENTE.search(text)
    if m:
        return _ev("en attente", m, 0.80)

    return {}


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.13  WHO classification year extractor (Phase A5)
# ═══════════════════════════════════════════════════════════════════════════

_VALID_OMS_YEARS = {"2007", "2016", "2021"}

_PAT_OMS_CLASSIFICATION = re.compile(
    r"(?:classification|class\.?)\s+(?:OMS|WHO)\s+(?:de\s+)?(?P<year>20\d{2})",
    _RE_FLAGS,
)

_PAT_OMS_WHO_YEAR = re.compile(
    r"(?:OMS|WHO)\s+(?P<year>20\d{2})",
    _RE_FLAGS,
)

_PAT_OMS_YEAR_ONLY = re.compile(
    r"classification\s+(?P<year>20\d{2})",
    _RE_FLAGS,
)


def extract_classification_oms(text: str) -> dict[str, ExtractionValue]:
    """Extract WHO classification year (2007/2016/2021) from *text*.
    
    Two-pass approach:
    1. First pass: search only in conclusion/diagnostic context
    2. Second pass: search full text, prefer the most recent valid year
    """
    all_matches = []
    
    # Collect all matches across all patterns
    for pat in (_PAT_OMS_CLASSIFICATION, _PAT_OMS_WHO_YEAR, _PAT_OMS_YEAR_ONLY):
        for m in pat.finditer(text):
            all_matches.append(m)
            
    if not all_matches:
        return {}
        
    # Pass 1: look near conclusion/diagnostic keywords
    for m in all_matches:
        year = m.group("year")
        if year not in _VALID_OMS_YEARS:
            continue
            
        context_start = max(0, m.start() - 200)
        context_end = min(len(text), m.end() + 200)
        context = text[context_start:context_end].lower()
        if any(kw in context for kw in ("conclusion", "diagnostic", "synthèse", "retenu")):
            return {"classification_oms": ExtractionValue(
                value=year,
                source_span=m.group(), source_span_start=m.start(),
                source_span_end=m.end(), extraction_tier="rule",
                confidence=0.95, vocab_valid=True,
            )}

    # Pass 2: fallback — pick the highest (most recent) valid year
    valid_matches = [(int(m.group("year")), m) for m in all_matches if m.group("year") in _VALID_OMS_YEARS]
    
    if valid_matches:
        best_year, best_m = max(valid_matches, key=lambda x: x[0])
        return {"classification_oms": ExtractionValue(
            value=str(best_year),
            source_span=best_m.group(), source_span_start=best_m.start(),
            source_span_end=best_m.end(), extraction_tier="rule",
            confidence=0.95, vocab_valid=True,
        )}
        
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.14  Drug dictionary extractor (Phase B1)
# ═══════════════════════════════════════════════════════════════════════════

_DRUG_SYNONYMS: dict[str, list[str]] = {
    "temozolomide": ["temozolomide", "témozolomide", "TMZ", "temodar", "temodal"],
    "carmustine": ["carmustine", "BCNU", "gliadel"],
    "lomustine": ["lomustine", "CCNU", "bélustine", "belustine"],
    "bevacizumab": ["bevacizumab", "bévacizumab", "avastin"],
    "vincristine": ["vincristine", "oncovin"],
    "carboplatin": ["carboplatine", "carboplatin"],
    "cisplatin": ["cisplatine", "cisplatin"],
    "etoposide": ["étoposide", "etoposide", "vepesid"],
    "irinotecan": ["irinotécan", "irinotecan", "campto"],
    "nilotinib": ["nilotinib", "tasigna"],
    "regorafenib": ["régorafénib", "regorafenib", "stivarga"],
    "paxalisib": ["paxalisib"],
    "marizomib": ["marizomib"],
}

# Build case-insensitive patterns per canonical drug name
_DRUG_PATTERNS: list[tuple[str, re.Pattern[str]]] = []
for _canonical, _synonyms in _DRUG_SYNONYMS.items():
    _alt = "|".join(re.escape(s) for s in sorted(_synonyms, key=len, reverse=True))
    _DRUG_PATTERNS.append(
        (_canonical, re.compile(r"\b(?:" + _alt + r")\b", _RE_FLAGS))
    )

# Context keywords for chemotherapy
_PAT_CHIMIO_CONTEXT = re.compile(
    r"\b(?:chimioth[eé]rapie|protocole|traitement|cure|ligne)\b",
    _RE_FLAGS,
)


def extract_chimios(text: str) -> dict[str, ExtractionValue]:
    """Extract chemotherapy drug names from *text* using a drug dictionary.

    Returns a dict with key ``chimios``, value being the canonical drug
    name(s) joined with `` + `` when multiple are found.
    Unrecognised drug-like patterns are flagged for review.
    """
    found_drugs: list[tuple[str, re.Match]] = []

    for canonical, pattern in _DRUG_PATTERNS:
        m = pattern.search(text)
        if m:
            found_drugs.append((canonical, m))

    if not found_drugs:
        return {}

    # Sort alphabetically and dedupe
    drug_names = sorted(list(dict.fromkeys(d[0] for d in found_drugs)))
    
    # We still need the min and max positions across all matched drugs to capture the whole span
    first_match = min(found_drugs, key=lambda x: x[1].start())[1]
    last_match = max(found_drugs, key=lambda x: x[1].end())[1]

    value = " + ".join(drug_names)
    span_text = text[first_match.start():last_match.end()]

    return {"chimios": ExtractionValue(
        value=value,
        source_span=span_text,
        source_span_start=first_match.start(),
        source_span_end=last_match.end(),
        extraction_tier="rule",
        confidence=0.85,
        vocab_valid=True,
    )}


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.15  Tumour position extractor (Phase B2)
# ═══════════════════════════════════════════════════════════════════════════

_ALL_LOCATIONS: list[str] = [
    # Composite lobes
    "parieto-occipital", "pariéto-occipital",
    "fronto-temporal", "fronto-pariétal", "fronto-parietal",
    "temporo-pariétal", "temporo-parietal",
    "temporo-occipital", "pariéto-temporal",
    "fronto-temporo-pariétal",
    # Lobes
    "frontal", "temporal", "pariétal", "parietal", "occipital",
    "insulaire", "cingulaire",
    # Deep structures
    "thalamus", "thalamique",
    "ganglions de la base", "noyaux gris centraux",
    "putamen", "noyau caudé", "noyau caude",
    "capsule interne",
    "corps calleux",
    "cervelet", "cérébelleux", "cerebelleux",
    "tronc cérébral", "tronc cerebral",
    "bulbe", "protubérance", "protuberance", "mésencéphale", "mesencephale",
    "hypothalamus", "hypophyse", "selle turcique",
    "angle ponto-cérébelleux", "angle ponto-cerebelleux",
    "ventricule", "ventriculaire",
    "épendyme", "ependyme", "épendymaire", "ependymaire",
    "méninges", "meninges", "méningé", "meninge",
    "leptoméningé", "leptomeninge",
]

_LOC_ALTERNATION = "|".join(
    re.escape(loc) for loc in sorted(_ALL_LOCATIONS, key=len, reverse=True)
)

_PAT_POSITION_CONTEXT = re.compile(
    r"(?:situ[eé]e?\s+(?:en\s+|au\s+niveau\s+(?:du?\s+)?)?|localis[eé]e?\s+(?:au?\s+)?"
    r"|tumeur\s+(?:du?\s+)?|l[eé]sion\s+(?:du?\s+)?)"
    r"(?P<loc>" + _LOC_ALTERNATION + r")",
    _RE_FLAGS,
)

_PAT_POSITION_BARE = re.compile(
    r"\b(?P<loc>" + _LOC_ALTERNATION + r")\b",
    _RE_FLAGS,
)

# Normalise location names to canonical form
_LOC_NORM: dict[str, str] = {
    "parietal": "pariétal", "cerebelleux": "cérébelleux",
    "cérébelleux": "cérébelleux",
    "tronc cerebral": "tronc cérébral",
    "noyau caude": "noyau caudé",
    "protuberance": "protubérance", "mesencephale": "mésencéphale",
    "ependyme": "épendyme", "ependymaire": "épendymaire",
    "meninges": "méninges", "meninge": "méningé",
    "leptomeninge": "leptoméningé",
    "angle ponto-cerebelleux": "angle ponto-cérébelleux",
    "thalamique": "thalamus",
    "noyaux gris centraux": "ganglions de la base",
    "ventriculaire": "ventricule",
    "épendymaire": "épendyme",
    "pariéto-occipital": "parieto-occipital",
    "fronto-pariétal": "fronto-parietal",
    "temporo-pariétal": "temporo-parietal",
}


def extract_tumeur_position(text: str) -> dict[str, ExtractionValue]:
    """Extract tumour anatomical position from *text*.

    Uses a closed neuro-oncology anatomical vocabulary.
    Multiple structures are joined with ' + '.
    Returns None (empty dict) for unmatched — no LLM fallback.
    """
    found: list[tuple[str, int, int]] = []

    # 1. Context-prefixed matches (higher confidence)
    for m in _PAT_POSITION_CONTEXT.finditer(text):
        loc = m.group("loc").lower()
        norm = _LOC_NORM.get(loc, loc)
        found.append((norm, m.start(), m.end()))

    # 2. If none found via context, try bare location near tumour keywords
    if not found:
        for m in _PAT_POSITION_BARE.finditer(text):
            loc = m.group("loc").lower()
            norm = _LOC_NORM.get(loc, loc)
            found.append((norm, m.start(), m.end()))

    if not found:
        return {}

    # Deduplicate locations while preserving order
    seen: set[str] = set()
    unique_locs: list[str] = []
    for loc, _, _ in found:
        if loc not in seen:
            seen.add(loc)
            unique_locs.append(loc)

    value = " + ".join(unique_locs)
    first_start = found[0][1]
    last_end = found[-1][2]

    return {"tumeur_position": ExtractionValue(
        value=value,
        source_span=text[first_start:last_end],
        source_span_start=first_start,
        source_span_end=last_end,
        extraction_tier="rule",
        confidence=0.80,
        vocab_valid=True,
    )}


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.16  Histological diagnosis extractor (Phase C5)
# ═══════════════════════════════════════════════════════════════════════════

_DIAGNOSIS_VOCAB: dict[str, list[str]] = {
    "glioblastome": [
        "glioblastome", "GBM", "glioblastoma", "gliolastome",
        "glialastome", "glioblasatome",
    ],
    "astrocytome": ["astrocytome", "astrocytoma"],
    "oligodendrogliome": ["oligodendrogliome", "oligodendroglioma", "oligo"],
    "oligoastrocytome": ["oligoastrocytome", "oligoastrocytoma"],
    "gliome": ["gliome", "glioma", "gliomes de bas grade"],
    "medulloblastome": ["médulloblastome", "medulloblastome", "PNET"],
    "ependymome": ["épendymome", "ependymome", "ependymoma"],
    "meningiome": ["méningiome", "meningiome", "meningioma"],
    "schwannome": ["schwannome", "schwannoma", "neurinome"],
    "lymphome": ["lymphome", "DLBCL", "PCNSL"],
    "metastase": ["métastase", "metastase", "localisation secondaire", "carcinose"],
}

# Build combined pattern sorted longest-first to avoid substring collisions
_ALL_DIAG_SYNONYMS: list[tuple[str, str]] = []
for _canon, _syns in _DIAGNOSIS_VOCAB.items():
    for _syn in _syns:
        _ALL_DIAG_SYNONYMS.append((_syn, _canon))
_ALL_DIAG_SYNONYMS.sort(key=lambda x: len(x[0]), reverse=True)

_PAT_DIAG_VOCAB = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s, _ in _ALL_DIAG_SYNONYMS) + r")\b",
    _RE_FLAGS,
)

_PAT_DIAG_RAW_CONTEXT = re.compile(
    r"diagnostic\s*[:\-]?\s*(?P<raw>[^\n\r.]{5,80})",
    _RE_FLAGS,
)

# Reverse lookup: synonym → canonical
_DIAG_REVERSE: dict[str, str] = {
    s.lower(): c for s, c in _ALL_DIAG_SYNONYMS
}


def extract_diag_histologique(text: str) -> dict[str, ExtractionValue]:
    """Extract histological diagnosis from *text* using vocabulary lookup.

    Searches in conclusion/diagnostic sections first (if text contains them),
    falls back to full text. Returns canonical form.
    """
    m = _PAT_DIAG_VOCAB.search(text)
    if m:
        canon = _DIAG_REVERSE.get(m.group().lower(), m.group().lower())
        return {"diag_histologique": ExtractionValue(
            value=canon,
            source_span=m.group(), source_span_start=m.start(),
            source_span_end=m.end(), extraction_tier="rule",
            confidence=0.85, vocab_valid=True,
        )}

    # Fallback: raw text near "diagnostic :"
    m = _PAT_DIAG_RAW_CONTEXT.search(text)
    if m:
        raw = m.group("raw").strip().rstrip(".,;:")
        return {"diag_histologique": ExtractionValue(
            value=raw,
            source_span=m.group(), source_span_start=m.start(),
            source_span_end=m.end(), extraction_tier="rule",
            confidence=0.50, vocab_valid=False,
            flagged=True,
        )}

    return {}


# ═══════════════════════════════════════════════════════════════════════════
# 4.3   Master rule-based extraction function
# ═══════════════════════════════════════════════════════════════════════════

# Section → extractor mapping: which extractors are relevant for each section
_SECTION_EXTRACTORS: dict[str, list[str]] = {
    "ihc": ["ihc", "numerical"],
    "molecular": ["molecular", "amplification", "fusion"],
    "chromosomal": ["chromosomal", "amplification"],
    "macroscopy": ["binary", "numerical"],
    "microscopy": ["binary", "numerical", "ihc", "diag_histologique"],
    "conclusion": ["ihc", "molecular", "chromosomal", "numerical", "amplification",
                    "fusion", "evol_clinique", "classification_oms", "diag_histologique"],
    "history": ["date", "binary", "type_chirurgie", "evol_clinique"],
    "treatment": ["date", "binary", "numerical", "type_chirurgie", "chimios"],
    "clinical_exam": ["binary", "numerical"],
    "radiology": ["binary", "date"],
    "equipe_soignante": ["date"],
    "demographics": ["date", "sexe"],
    "summary": ["date", "binary", "numerical", "evol_clinique", "type_chirurgie",
                "chimios", "diag_histologique"],
    "rcp_decision": ["date", "binary", "numerical", "type_chirurgie", "chimios"],
    "full_text": ["date", "ihc", "molecular", "chromosomal", "binary", "numerical",
                  "amplification", "fusion", "sexe", "tumeur_lateralite",
                  "evol_clinique", "type_chirurgie", "classification_oms",
                  "chimios", "tumeur_position", "diag_histologique"],
}


def run_rule_extraction(
    text: str,
    sections: dict[str, str],
    feature_subset: list[str],
    annotator: Optional[AssertionAnnotator] = None,
    enabled_extractors: Optional[set[str]] = None,
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
    enabled_extractors : set[str] or None
        Names of sub-extractors to activate.  ``None`` means all are active.
        Valid names: ``"date"``, ``"ihc"``, ``"molecular"``, ``"chromosomal"``,
        ``"binary"``, ``"numerical"``, ``"amplification"``, ``"fusion"``,
        ``"sexe"``, ``"tumeur_lateralite"``, ``"evol_clinique"``,
        ``"type_chirurgie"``, ``"classification_oms"``, ``"chimios"``,
        ``"tumeur_position"``, ``"diag_histologique"``.

    Returns
    -------
    dict[str, ExtractionValue]
        Mapping ``field_name → ExtractionValue`` for successfully
        extracted fields.
    """
    def _on(name: str) -> bool:
        return enabled_extractors is None or name in enabled_extractors

    feature_set = set(feature_subset)
    all_results: dict[str, ExtractionValue] = {}

    # --- Phase 0.3: Pseudonymised birthdate extraction ---
    if _on("date") and "annee_de_naissance" in feature_set:
        m = _PAT_PSEUDO_BIRTHDATE.search(text)
        if m:
            year = m.group(1)
            all_results["annee_de_naissance"] = ExtractionValue(
                value=year,
                source_span=m.group(0),
                source_span_start=m.start(),
                source_span_end=m.end(),
                extraction_tier="rule",
                confidence=0.5,
                section="full_text",
                vocab_valid=True,
            )

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

        if _on("date") and "date" in extractor_names:
            date_results = extract_dates(section_text)
            # Context-aware date assignment (Phase 0.1)
            date_fields_in_subset = [
                f for f in feature_set
                if f not in all_results and _is_date_field(f)
            ]
            context_assigned = _assign_dates_by_context(
                date_results, date_fields_in_subset, section_text
            )
            for fname, ev in context_assigned.items():
                if fname == "annee_de_naissance" and isinstance(ev.value, str):
                    ev.value = ev.value[-4:]  # Only keep YYYY
                ev.section = section_name
                all_results[fname] = ev

        if _on("ihc") and "ihc" in extractor_names:
            _merge(extract_ihc(section_text), section_name)

        if _on("molecular") and "molecular" in extractor_names:
            _merge(extract_molecular(section_text), section_name)

        if _on("chromosomal") and "chromosomal" in extractor_names:
            _merge(extract_chromosomal(section_text), section_name)

        if _on("binary") and "binary" in extractor_names:
            _merge(extract_binary(section_text, annotator), section_name)

        if _on("numerical") and "numerical" in extractor_names:
            _merge(extract_numerical(section_text), section_name)

        if _on("amplification") and "amplification" in extractor_names:
            _merge(extract_amplifications(section_text), section_name)

        if _on("fusion") and "fusion" in extractor_names:
            _merge(extract_fusions(section_text), section_name)

        # --- Phase A/B/C new extractors ---
        if _on("sexe") and "sexe" in extractor_names:
            _merge(extract_sexe(section_text), section_name)

        if _on("tumeur_lateralite") and "tumeur_lateralite" in extractor_names:
            _merge(extract_tumeur_lateralite(section_text), section_name)

        if _on("evol_clinique") and "evol_clinique" in extractor_names:
            _merge(extract_evol_clinique(section_text), section_name)

        if _on("type_chirurgie") and "type_chirurgie" in extractor_names:
            _merge(extract_type_chirurgie(section_text), section_name)

        if _on("classification_oms") and "classification_oms" in extractor_names:
            _merge(extract_classification_oms(section_text), section_name)

        if _on("chimios") and "chimios" in extractor_names:
            _merge(extract_chimios(section_text), section_name)

        if _on("tumeur_position") and "tumeur_position" in extractor_names:
            _merge(extract_tumeur_position(section_text), section_name)

        if _on("diag_histologique") and "diag_histologique" in extractor_names:
            _merge(extract_diag_histologique(section_text), section_name)

    # If sections didn't cover everything, run on full text as catch-all
    if "full_text" not in sections:
        remaining = feature_set - set(all_results.keys())
        if remaining:
            if _on("ihc") and "ihc" in _relevant_groups(remaining):
                _merge(extract_ihc(text), "full_text")
            if _on("molecular") and "molecular" in _relevant_groups(remaining):
                _merge(extract_molecular(text), "full_text")
            if _on("chromosomal") and "chromosomal" in _relevant_groups(remaining):
                _merge(extract_chromosomal(text), "full_text")
            if _on("binary") and "binary" in _relevant_groups(remaining):
                _merge(extract_binary(text, annotator), "full_text")
            if _on("numerical") and "numerical" in _relevant_groups(remaining):
                _merge(extract_numerical(text), "full_text")
            if _on("amplification") and "amplification" in _relevant_groups(remaining):
                _merge(extract_amplifications(text), "full_text")
            if _on("fusion") and "fusion" in _relevant_groups(remaining):
                _merge(extract_fusions(text), "full_text")

            # Phase A/B/C fallback extractors
            if _on("sexe") and "sexe" in remaining:
                _merge(extract_sexe(text), "full_text")
            if _on("tumeur_lateralite") and "tumeur_lateralite" in remaining:
                _merge(extract_tumeur_lateralite(text), "full_text")
            if _on("evol_clinique") and "evol_clinique" in remaining:
                _merge(extract_evol_clinique(text), "full_text")
            if _on("type_chirurgie") and "type_chirurgie" in remaining:
                _merge(extract_type_chirurgie(text), "full_text")
            if _on("classification_oms") and "classification_oms" in remaining:
                _merge(extract_classification_oms(text), "full_text")
            if _on("chimios") and "chimios" in remaining:
                _merge(extract_chimios(text), "full_text")
            if _on("tumeur_position") and "tumeur_position" in remaining:
                _merge(extract_tumeur_position(text), "full_text")
            if _on("diag_histologique") and "diag_histologique" in remaining:
                _merge(extract_diag_histologique(text), "full_text")

    return all_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_date_field(field_name: str) -> bool:
    """Return True if the field is a date-type field."""
    if field_name == "annee_de_naissance":
        return True
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
