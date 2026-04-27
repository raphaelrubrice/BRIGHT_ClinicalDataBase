"""Dedicated date-field extractor using eds.dates + fuzzy context matching.

Strategy
--------
1. Run ``eds.dates`` on the full document to detect all date spans.
2. Exclude the consultation / document date (passed in or auto-detected).
3. For each remaining date, extract a ±50-char context window (masking
   neighbouring date spans) and fuzzy-match against bilingual (FR / EN)
   keyword repertoires to decide which clinical date field it belongs to.
4. Greedily assign dates to fields (highest confidence first).  Unmatched
   dates are left unassigned, better NA than a wrong date.

Public API
----------
- ``DateExtractor`` , instantiate once, call ``.extract()`` per document.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import edsnlp
from rapidfuzz import fuzz

from .schema import ExtractionValue
from .text_normalisation import normalise as _norm

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Bilingual keyword repertoires
# ═══════════════════════════════════════════════════════════════════════════

_DATE_REPERTOIRE_FR: dict[str, list[str]] = {
    "date_deces": [
        "décès", "décédé", "décédée", "mort le", "date de décès",
        "date du décès", "décès le", "décès survenu",
        "est décédé(e)", "mort survenue", "fin de vie le", 
        "constat de décès", "jour du décès", "date du deces", "exitus le",
    ],
    "date_chir": [
        "chirurgie", "opéré", "opérée", "intervention", "opération",
        "exérèse", "biopsie", "résection", "craniotomie", "craniectomie",
        "opéré le", "opérée le", "reprise chir", "intervention le",
        "acte chirurgical", "intervention chirurgicale", "résection tumorale", 
        "exérèse tumorale", "macroscopiquement complète le", "resection le", 
        "exerese le", "biopsie stéréotaxique le", "biopsie chirurgicale le",
        "geste chirurgical", "intervention neurochirurgicale",
    ],
    "date_rcp": [
        "rcp", "réunion de concertation", "concertation pluridisciplinaire",
        "staff", "réunion pluridisciplinaire", "réunion de concertation pluridisciplinaire",
        "réunion neuro-oncologique", "staff de neuro-oncologie", "dossier présenté en rcp", 
        "dossier discuté en rcp", "RCP du", "staff du", "réunion de concertation du",
        "décision de rcp", "proposition rcp", "avis rcp",
    ],
    "chm_date_debut": [
        "début chimio", "chimiothérapie débutée", "TMZ depuis",
        "témozolomide depuis", "début de chimiothérapie", "premier cycle",
        "débuté le", "TMZ le", "bévacizumab le", "avastin le",
        "lomustine le", "chimio débutée",
        "initiation chimiothérapie", "chimiothérapie initiée", "début cure", 
        "1ère cure", "première cure", "C1", "début traitement médical", 
        "mise sous TMZ", "mise sous témozolomide", "chimiothérapie commencée",
        "protocole stupp débuté", "début stupp",
    ],
    "chm_date_fin": [
        "fin chimio", "arrêt chimio", "dernière cure",
        "fin de chimiothérapie", "dernier cycle", "fin TMZ",
        "arrêt de chimiothérapie",
        "fin cure", "dernière chimiothérapie", "arrêt de la chimiothérapie", 
        "chimiothérapie arrêtée", "fin du traitement", "fin du TMZ", 
        "fin du témozolomide", "suspension chimiothérapie", "fin de l'entretien",
        "fin du protocole", "traitement suspendu le",
    ],
    "rx_date_debut": [
        "début radiothérapie", "RT débutée", "irradiation depuis",
        "début RT", "début de radiothérapie", "RT le", "radio le",
        "irradiation le", "radiothérapie débutée",
        "radiothérapie initiée", "initiation radiothérapie", "début de la RT", 
        "première séance de radiothérapie", "début rayons", "mise en route radiothérapie",
        "début radiothérapie focale", "début de l'irradiation",
    ],
    "rx_date_fin": [
        "fin radiothérapie", "fin RT", "fin de radiothérapie",
        "fin d'irradiation", "fin radio", "fin irradiation",
        "arrêt radiothérapie",
        "radiothérapie terminée", "fin des rayons", "dernière séance de radiothérapie", 
        "clôture radiothérapie", "irradiation terminée", "fin de la RT",
        "fin du stupp radiothérapie",
    ],
    "date_1er_symptome": [
        "premier symptôme", "1er symptôme", "début des troubles",
        "apparition", "premiers signes", "début des symptômes",
        "symptomatologie initiale", "début de la symptomatologie",
        "premières manifestations", "début de l'histoire", "premiers épisodes", 
        "crise inaugurale", "symptômes initiaux", "début de la maladie", 
        "première crise comitiale", "tableau clinique initial", 
        "première crise", "début clinique",
    ],
    "exam_radio_date_decouverte": [
        "découverte", "IRM du", "scanner du", "imagerie du",
        "IRM de découverte", "scanner initial", "imagerie initiale",
        "TDM du", "IRM initiale",
        "IRM diagnostique", "imagerie diagnostique", "scanner diagnostique", 
        "IRM mettant en évidence", "IRM initiale du", "scanner initial du", 
        "TDM diagnostique", "IRM cérébrale du", "examen de découverte",
        "IRM cérébrale de découverte", "bilan initial",
    ],
    "date_progression": [
        "progression", "récidive", "rechute", "récidive le",
        "prise de contraste", "rehaussement", "augmentation de taille",
        "PD", "progression le",
        "reprise évolutive", "progression tumorale", "aggravation radiologique", 
        "récidive tumorale", "échappement", "évolution péjorative", 
        "majoration de la prise de contraste", "progression radiologique", 
        "récidive clinique", "reprise de la maladie", "augmentation du volume",
    ],
    "dn_date": [
        "dernière nouvelle", "dernières nouvelles", "consultation du",
        "vu le", "vue le", "revu le", "revue le", "DN du",
        "vu en consultation", "dernières nouvelles du",
        "dernières nouvelles au", "point clinique du", "vu(e) ce jour", 
        "dernière consultation", "dernier contact", "nouvelles du", "suivi du",
        "vu(e) en HDJ le", "vu(e) en hôpital de jour", "évaluation ce jour",
    ],
}

_DATE_REPERTOIRE_EN: dict[str, list[str]] = {
    "date_deces": [
        "death", "died", "deceased", "date of death", "passed away",
        "death on", "died on", "patient died",
        "passed away on", "date of passing", "exitus", "pronounced dead", 
        "expired on", "time of death",
    ],
    "date_chir": [
        "surgery", "operated", "operation", "resection", "biopsy",
        "craniotomy", "excision", "surgical", "operated on",
        "surgical intervention", "tumor resection", "tumour resection", 
        "surgical resection", "biopsied on", "resected on", "craniotomy on", 
        "excision on", "neurosurgery on",
    ],
    "date_rcp": [
        "tumor board", "MDT meeting", "multidisciplinary",
        "board meeting", "tumour board", "MDT",
        "neuro-oncology board", "multidisciplinary team", "presented at tumor board", 
        "discussed at MDT", "MDT on", "tumor board on", "consensus meeting",
    ],
    "chm_date_debut": [
        "chemo started", "chemotherapy started", "TMZ since",
        "first cycle", "started on", "chemotherapy initiated",
        "began chemotherapy",
        "chemotherapy initiated", "started chemo", "first cycle of", "cycle 1", 
        "commenced chemotherapy", "initiation of chemotherapy", "placed on TMZ",
        "started TMZ",
    ],
    "chm_date_fin": [
        "chemo ended", "last cycle", "end of chemotherapy",
        "chemotherapy completed", "chemotherapy stopped",
        "finished chemotherapy",
        "chemotherapy finished", "last chemo", "chemotherapy discontinued", 
        "chemotherapy terminated", "completion of chemotherapy", "end of chemo",
        "TMZ stopped",
    ],
    "rx_date_debut": [
        "radiotherapy started", "radiation started", "RT started",
        "irradiation since", "radiation initiated", "began radiation",
        "radiation initiated", "radiotherapy initiated", "started RT", 
        "started radiation", "first fraction", "commenced radiotherapy",
    ],
    "rx_date_fin": [
        "radiotherapy ended", "radiation ended", "end of radiotherapy",
        "RT completed", "radiation completed", "finished radiation",
        "radiation finished", "radiotherapy completed", "RT ended", 
        "last fraction", "completion of radiotherapy",
    ],
    "date_1er_symptome": [
        "first symptom", "onset", "initial symptoms", "first signs",
        "symptom onset", "onset of symptoms", "first presentation",
        "initial presentation", "first manifestation", "onset of disease", 
        "symptom beginning", "first seizure", "presenting symptom",
        "clinical onset",
    ],
    "exam_radio_date_decouverte": [
        "discovery", "MRI on", "CT on", "imaging on", "initial MRI",
        "initial scan", "CT scan on", "discovered on",
        "diagnostic MRI", "diagnostic CT", "baseline MRI", "baseline imaging", 
        "initial imaging", "scan showing", "MRI demonstrating",
        "baseline scan", "discovery scan",
    ],
    "date_progression": [
        "progression", "recurrence", "relapse", "disease progression",
        "tumor progression", "tumour progression",
        "disease recurrence", "tumor recurrence", "tumour recurrence", 
        "radiological progression", "clinical progression", "worsening on", "growth on",
        "tumor growth",
    ],
    "dn_date": [
        "last follow-up", "last seen", "follow-up on", "last news",
        "seen on", "last visit", "most recent visit",
        "last contact", "most recent follow-up", "last evaluation", 
        "seen in clinic on", "date of last contact", "latest news",
        "recent visit",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Consultation-date fallback regex  (mirrors ops._extract_consult_date_regex)
# ═══════════════════════════════════════════════════════════════════════════

_CONSULT_DATE_RE = re.compile(
    r"(?i:(?:consultation.+du *|Paris(?:,| +,|) +le +|"
    r"Date *de *r[ée]ception *:? *))"
    r"((?:(?:\d{4}|\d{2})/\d{2}/(?:\d{4}|\d{2}))"
    r"|(?:\d{2} +\D+ +\d{4}))",
    re.MULTILINE,
)

_FRENCH_MONTHS: dict[str, int] = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3,
    "avril": 4, "mai": 5, "juin": 6, "juillet": 7,
    "août": 8, "aout": 8, "septembre": 9, "octobre": 10,
    "novembre": 11, "décembre": 12, "decembre": 12,
    "janv": 1, "jan": 1, "fév": 2, "fev": 2, "févr": 2, "fevr": 2,
    "avr": 4, "juil": 7, "juill": 7,
    "sept": 9, "oct": 10, "nov": 11, "déc": 12, "dec": 12,
}


def _parse_raw_consult_date(raw: str) -> str | None:
    """Parse a raw date string to DD/MM/YYYY.  Returns None on failure."""
    if "/" in raw:
        parts = [int(p) for p in raw.split("/")]
        if len(parts) != 3:
            return None
        # Ensure year first if max component is in position 0
        if parts[0] > 31:
            year, b, c = parts
            month, day = (b, c) if b <= 12 else (c, b)
        else:
            a, b, year = parts
            if year < 100:
                year += 2000 if year < 50 else 1900
            day, month = (a, b) if b <= 12 else (b, a)
        return f"{day:02d}/{month:02d}/{year:04d}"
    # "DD Month YYYY"
    tokens = raw.split()
    if len(tokens) != 3:
        return None
    try:
        day = int(tokens[0])
        month = _FRENCH_MONTHS.get(tokens[1].lower().rstrip("."))
        year = int(tokens[2])
    except (ValueError, TypeError):
        return None
    if not month:
        return None
    return f"{day:02d}/{month:02d}/{year:04d}"


def _extract_consult_date_from_text(text: str) -> str | None:
    """Try to extract consultation date from text via regex (no LLM)."""
    m = _CONSULT_DATE_RE.search(text)
    if m is None:
        return None
    return _parse_raw_consult_date(m.group(1).strip())


# ═══════════════════════════════════════════════════════════════════════════
# Context helper
# ═══════════════════════════════════════════════════════════════════════════

_HALF_WINDOW = 50
_FUZZY_THRESHOLD = 60


def _get_context(
    text: str,
    date_start: int,
    date_end: int,
    all_spans: list[tuple[int, int]],
    half_window: int = _HALF_WINDOW,
) -> str:
    """Return textual context around a date span, masking other date spans."""
    ctx_start = max(0, date_start - half_window)
    ctx_end = min(len(text), date_end + half_window)
    raw = text[ctx_start:date_start] + " " + text[date_end:ctx_end]

    # Mask other date spans that overlap with the context window
    for s, e in all_spans:
        if (s, e) == (date_start, date_end):
            continue
        # Compute overlap with context window
        os_ = max(s, ctx_start)
        oe_ = min(e, ctx_end)
        if os_ < oe_:
            # Replace overlapping region with spaces in raw
            # Adjust to raw-local coordinates
            # raw = text[ctx_start:date_start] + " " + text[date_end:ctx_end]
            # Before-part length = date_start - ctx_start
            # After-part starts at position (date_start - ctx_start + 1)
            before_len = date_start - ctx_start
            if oe_ <= date_start:
                # Overlap is in the "before" part
                local_s = os_ - ctx_start
                local_e = oe_ - ctx_start
                raw = raw[:local_s] + " " * (local_e - local_s) + raw[local_e:]
            elif os_ >= date_end:
                # Overlap is in the "after" part
                # In raw, the after part starts at index (before_len + 1)
                local_s = before_len + 1 + (os_ - date_end)
                local_e = before_len + 1 + (oe_ - date_end)
                if local_e <= len(raw):
                    raw = raw[:local_s] + " " * (local_e - local_s) + raw[local_e:]

    return raw


# ═══════════════════════════════════════════════════════════════════════════
# DateExtractor
# ═══════════════════════════════════════════════════════════════════════════

class DateExtractor:
    """Extract clinical date fields using eds.dates + fuzzy context matching."""

    def __init__(self) -> None:
        self._nlp = edsnlp.blank("eds")
        self._nlp.add_pipe("eds.dates")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        feature_subset: list[str],
        language: str = "fr",
        consultation_date: str | None = None,
    ) -> dict[str, ExtractionValue]:
        """Extract date fields from *text*.

        Parameters
        ----------
        text : str
            Full document text.
        feature_subset : list[str]
            Date field names to extract (only DATE-type fields expected).
        language : str
            ``"fr"`` or ``"en"``, selects keyword repertoire.
        consultation_date : str | None
            Consultation date in DD/MM/YYYY to exclude.  If ``None``, an
            automatic regex-based extraction is attempted.

        Returns
        -------
        dict[str, ExtractionValue]
        """
        if not feature_subset:
            return {}

        # -- 1. Detect all dates via eds.dates --------------------------------
        doc = self._nlp(text)
        date_spans = doc.spans.get("dates", [])
        if not date_spans:
            return {}

        # Build structured list: (normalised_date, raw_span, start, end)
        detected: list[tuple[str, str, int, int]] = []
        for span in date_spans:
            dt = getattr(span._, "date", None)
            if dt is None or not dt.year:
                continue
            if dt.month and dt.day:
                norm = f"{dt.day:02d}/{dt.month:02d}/{dt.year:04d}"
            elif dt.month:
                norm = f"01/{dt.month:02d}/{dt.year:04d}"
            else:
                norm = f"01/01/{dt.year:04d}"
            detected.append((norm, span.text, span.start_char, span.end_char))

        if not detected:
            return {}

        # -- 2. Resolve consultation date & exclude ---------------------------
        if consultation_date is None:
            consultation_date = _extract_consult_date_from_text(text)

        if consultation_date:
            consult_norm = consultation_date.strip()
            detected = [
                d for d in detected
                if d[0] != consult_norm
            ]

        if not detected:
            return {}

        # -- 3. Build (date_idx, field, score) candidates ---------------------
        repertoire = (
            _DATE_REPERTOIRE_FR if language.startswith("fr")
            else _DATE_REPERTOIRE_EN
        )

        all_spans_pos = [(d[2], d[3]) for d in detected]
        candidates: list[tuple[int, str, float, str, str, int, int]] = []
        # Each element: (date_idx, field, score, norm_date, raw_span, start, end)

        for idx, (norm_date, raw_span, start, end) in enumerate(detected):
            ctx = _norm(_get_context(text, start, end, all_spans_pos))

            for field_name in feature_subset:
                keywords = repertoire.get(field_name, [])
                if not keywords:
                    continue

                best_score = 0.0
                for kw in keywords:
                    kw_norm = _norm(kw)
                    score = fuzz.partial_ratio(kw_norm, ctx)
                    if score > best_score:
                        best_score = score

                if best_score >= _FUZZY_THRESHOLD:
                    candidates.append(
                        (idx, field_name, best_score, norm_date,
                         raw_span, start, end)
                    )

        # -- 4. Greedy assignment (highest score first) -----------------------
        candidates.sort(key=lambda c: c[2], reverse=True)
        assigned_fields: set[str] = set()
        assigned_dates: set[int] = set()
        results: dict[str, ExtractionValue] = {}

        for (date_idx, field_name, score, norm_date,
             raw_span, start, end) in candidates:
            if field_name in assigned_fields:
                continue
            if date_idx in assigned_dates:
                continue

            results[field_name] = ExtractionValue(
                value=norm_date,
                source_span=raw_span,
                source_span_start=start,
                source_span_end=end,
                extraction_tier="rule",
                confidence=round(score / 100, 2),
                vocab_valid=True,
            )
            assigned_fields.add(field_name)
            assigned_dates.add(date_idx)

        return results