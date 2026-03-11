import re
from typing import Any
import edsnlp
import edsnlp.pipes as eds

from src.extraction.schema import ExtractionValue, get_field, ControlledVocab
from src.extraction.rule_extraction import (
    _IHC_VALUE_NORM,
    _MOL_STATUS_NORM,
    _CHR_STATUS_NORM,
    _VARIANT_PATTERN,
    _ROMAN_TO_INT,
    _LATERALITY_NORM,
    extract_dates,
    _assign_dates_by_context,
    _DIAGNOSIS_VOCAB,
    _DRUG_SYNONYMS,
    _DATE_CONTEXT_KEYWORDS,
)
from src.extraction.text_normalisation import normalise as _norm
from collections import defaultdict
from src.extraction.section_detector import get_section_for_feature

# ---------------------------------------------------------------------------
# Matcher Dictionaries
# ---------------------------------------------------------------------------

TERM_DICT: dict[str, list[str]] = {
    "optune": ["optune", "ttfields", "tt-fields", "tumor treating fields", "champs électriques"],
    "corticoides": ["corticoïdes", "corticoides", "corticothérapie", "dexaméthasone", "dexamethasone", "solumédrol", "solumedrol", "médrol", "prednisone", "cortancyl", "prednisolone"],
    "anti_epileptiques": ["anti-épileptique", "antiépileptique", "anti-epileptique", "antiepileptique", "keppra", "lévétiracétam", "levetiracetam", "valproate", "dépakine", "depakine", "lacosamide", "vimpat", "lamotrigine"],
    "type_chirurgie": ["exérèse complète", "exérèse partielle", "biopsie", "gtr", "str", "exérèse", "resection"],
    "epilepsie_1er_symptome": ["épilepsie", "epilepsie", "crises comitiales", "crises convulsives", "crise convulsive", "crise comitiale", "crise épileptique", "crises épileptiques", "comitialité"],
    "ceph_hic_1er_symptome": ["céphalées", "cephalees", "céphalée", "htic", "hypertension intracrânienne", "hypertension intracranienne"],
    "deficit_1er_symptome": ["déficit", "deficit", "déficitaire", "hémiplégie", "hémiparésie", "hemiparesie", "parésie", "paralysie"],
    "cognitif_1er_symptome": ["troubles cognitifs", "trouble cognitif", "confusion", "troubles mnésiques", "trouble mnésique", "ralentissement"],
    "histo_necrose": ["nécrose", "necrose", "nécroses", "plages de nécrose", "foyers de nécrose", "nécrose palissadique"],
    "histo_pec": ["prolifération endothéliocapillaire", "proliferation endotheliocapillaire", "prolifération endothélio-capillaire", "pec", "hyperplasie endothéliocapillaire"],
    "prise_de_contraste": ["prise de contraste", "rehaussement", "injection"],
    "dominance_cerebrale": ["droitier", "gaucher", "ambidextre", "latéralité manuelle", "lateralite manuelle"],
    "sexe": ["homme", "femme", "masculin", "féminin", "feminin", "monsieur", "madame", "mme"],
    "tumeur_lateralite": ["gauche", "droit", "droite", "bilateral", "bilaterale", "bilatéral", "bilatérale", "median", "mediane", "médian", "médiane"],
    "evol_clinique": ["initial", "p1", "p2", "p3", "p4", "p5", "terminal"],
}

REGEX_DICT: dict[str, list[str]] = {
    "sexe": [r"(?i)\|\s*[MF]\s*\|"],
    "classification_oms": [
        r"(?i)(?:classification|class\.?)\s+(?:OMS|WHO)\s+(?:de\s+)?20\d{2}",
        r"(?i)(?:OMS|WHO)\s+20\d{2}",
        r"(?i)classification\s+20\d{2}"
    ],
    "grade": [r"(?i)\bgrade\s*[:=\-\s]?\s*(?:[1-4]|I{1,3}V?|IV)\b"],
    "ik_clinique": [
        r"(?i)(?:indice\s+de\s+)?karnofsky\s*(?:[:=\-]|(?:est\s+)?(?:estim[eé]e?|[eé]valu[eé]e?)\s+[àa]\s+|de\s+)?\s*\d{2,3}", 
        r"(?i)\bik\s*(?:[:=\-]|(?:est\s+)?(?:estim[eé]e?|[eé]valu[eé]e?)\s+[àa]\s+|de\s+)?\s*\d{2,3}"
    ],
    "rx_dose": [r"(?i)dose\s+(?:de\s+)?(?:radio|rt)[^\d]*?\d+(?:[.,]\d+)?\s*gy", r"(?i)\b\d+(?:[.,]\d+)?\s*gy\b"],
    "chm_cycles": [r"(?i)\b\d+\s+cycles?\b"],
    "histo_mitoses": [r"(?i)\b\d+\s*mitoses?\b"],
}

_OPT_SEP = r"\s*(?:[:=\-]|est\s+(?:estim[eé]e?\s+[àa]\s+|[eé]valu[eé]e?\s+[àa]\s+|de\s+|un\s+score\s+de\s+)?)?\s*"
IHC_REGEX = _OPT_SEP + r"(" + r"\d+\s*(?:[àa\-]\s*\d+\s*)?%|<?\.?\s*\d+\s*%|score\s+(?:de\s+)?\d+|positif[s]?|n[ée]gatif(?:ve)?|positive?|n[ée]gative?|maintenu[e]?|perte\s+d['’]?\s*expression|absence\s+d['’]?\s*expression|conserv[ée]e?|expression\s+(?:conserv[ée]e?|maintenue)|surexprim[ée]|surexpression|exprim[ée]|pr[ée]sent|absent|perte|perdu|non\s+(?:exprim[ée]|d[ée]tect[ée]|retrouv[ée])|pr[ée]serv[ée]|normal|\+(?!\d)|\-(?!\d)" + r")"
MOL_REGEX = _OPT_SEP + r"(" + r"wt|wild[- ]?type|sauvage|type\s+sauvage|s[ée]quence\s+sauvage|statut?\s+wt|status\s+wt|non\s+mut[ée]e?(?:\(e\))?|mut[ée]e?(?:\(e\))?|mutation|mutation\s+(?:d[ée]tect[ée]e?|identifi[ée]e?)|variant\s+pathog[eè]ne|alt[ée]r[ée]e?(?:\(e\))?|pr[ée]sence\s+de\s+mutation|absence\s+de\s+mutation(?:\s+d[ée]tect[ée]e?)?|pas\s+de\s+mutation(?:\s+d[ée]tect[ée]e?)?|promoteur\s+(?:non\s+)?m[ée]thyl[ée]|m[ée]thylation\s+(?:du\s+promoteur|positive|n[ée]gative|absente)|hyper?m[ée]thyl[ée]|non\s+hyper?m[ée]thyl[ée]|m[ée]thyl[ée]|non\s+m[ée]thyl[ée]|absence\s+de\s+m[ée]thylation|(?:p\.)?[A-Z]\d+[A-Z]" + r")"
CHR_REGEX = _OPT_SEP + r"(" + r"gain(?:\s+de\s+signal)?|perte(?:\s+(?:partielle|focale|all[ée]lique|de\s+signal|d['']h[ée]t[ée]rozygotie|h[ée]mi?zygote|homo(?:zygote)?|h[ée]t[ée]ro(?:zygote)?))?|d[ée]l[ée]tion(?:\s+(?:focale|partielle))?|deleted?|del|monosomie|polysomie|trisomie|loh|normal[e]?" + r")"

def _build_ihc_pattern(source, terms):
    return dict(
        source=source,
        terms=terms,
        regex=[r"(?i)hirsch.*?est.*?(?:score\s+)?(\d+)"] if source == "ihc_egfr_hirsch" else [],
        assign=[
            dict(
                name="value",
                regex=IHC_REGEX,
                window=10,
                replace_entity=True,
                reduce_mode="keep_first"
            )
        ]
    )

def _build_mol_pattern(source, terms):
    return dict(
        source=source,
        terms=terms,
        regex=[r"(?i)(?:statut?\s+WT|status\s+WT)\s+(?:du?\s+g[eè]ne?\s+)?(?:" + "|".join(terms) + r")",
               r"(?i)g[eè]ne\s+(?:" + "|".join(terms) + r")\s+sauvage",
               r"(?i)(?:pas\s+de\s+mutation(?:\s+d[ée]tect[ée]e?)?|absence\s+de\s+mutation(?:\s+d[ée]tect[ée]e?)?)\s+(?:du?\s+g[eè]ne?\s+)?(?:" + "|".join(terms) + r")"],
        assign=[
            dict(
                name="value",
                regex=MOL_REGEX,
                window=10,
                replace_entity=True,
                reduce_mode="keep_first"
            )
        ]
    )

def _build_chr_pattern(source, terms, additional_regex=None):
    res = dict(
        source=source,
        terms=terms,
        assign=[
            dict(
                name="value",
                regex=CHR_REGEX,
                window=8,
                replace_entity=True,
                reduce_mode="keep_first"
            )
        ]
    )
    if additional_regex:
        res["regex"] = additional_regex
    return res

CM_PATTERNS = [
    _build_ihc_pattern("ihc_idh1", ["idh-1", "idh1"]),
    _build_ihc_pattern("ihc_p53", ["p53"]),
    _build_ihc_pattern("ihc_atrx", ["atrx"]),
    _build_ihc_pattern("ihc_fgfr3", ["fgfr3"]),
    _build_ihc_pattern("ihc_braf", ["braf"]),
    _build_ihc_pattern("ihc_hist_h3k27m", ["h3 k27m", "h3.3 k27m", "h3k27m", "histone h3 k27m"]),
    _build_ihc_pattern("ihc_hist_h3k27me3", ["h3 k27me3", "h3k27me3"]),
    _build_ihc_pattern("ihc_egfr_hirsch", ["egfr hirsch", "score hirsch", "egfr"]),
    _build_ihc_pattern("ihc_gfap", ["gfap"]),
    _build_ihc_pattern("ihc_olig2", ["olig2"]),
    _build_ihc_pattern("ihc_ki67", ["ki-67", "ki67", "index de prolifération"]),
    _build_ihc_pattern("ihc_mmr", ["mmr", "mlh1", "msh2", "msh6", "pms2", "dmmr", "pmmr", "deficit mmr"]),
    
    _build_mol_pattern("mol_idh1", ["idh-1", "idh1"]),
    _build_mol_pattern("mol_idh2", ["idh-2", "idh2"]),
    _build_mol_pattern("mol_tert", ["tert"]),
    _build_mol_pattern("mol_CDKN2A", ["cdkn2a"]),
    _build_mol_pattern("mol_h3f3a", ["h3f3a"]),
    _build_mol_pattern("mol_hist1h3b", ["hist1h3b"]),
    _build_mol_pattern("mol_braf", ["braf"]),
    _build_mol_pattern("mol_mgmt", ["mgmt"]),
    _build_mol_pattern("mol_fgfr1", ["fgfr1"]),
    _build_mol_pattern("mol_egfr_mut", ["egfr"]),
    _build_mol_pattern("mol_prkca", ["prkca"]),
    _build_mol_pattern("mol_p53", ["p53", "tp53"]),
    _build_mol_pattern("mol_pten", ["pten"]),
    _build_mol_pattern("mol_cic", ["cic"]),
    _build_mol_pattern("mol_fubp1", ["fubp1"]),
    _build_mol_pattern("mol_atrx", ["atrx"]),
    
    _build_chr_pattern("ch1p", ["1p"], [r"(?i)\b1p\s*([+\-])\s", r"(?i)(?:absence\s+de\s+(?:perte|d[ée]l[ée]tion)|pas\s+de\s+(?:perte|d[ée]l[ée]tion))\s+(?:du\s+(?:bras\s+)?)?1p"]),
    _build_chr_pattern("ch19q", ["19q"], [r"(?i)\b19q\s*([+\-])\s", r"(?i)(?:absence\s+de\s+(?:perte|d[ée]l[ée]tion)|pas\s+de\s+(?:perte|d[ée]l[ée]tion))\s+(?:du\s+(?:bras\s+)?)?19q"]),
    _build_chr_pattern("ch10p", ["10p"], [r"(?i)\b10p\s*([+\-])\s"]),
    _build_chr_pattern("ch10q", ["10q"], [r"(?i)\b10q\s*([+\-])\s"]),
    _build_chr_pattern("ch7p", ["7p"], [r"(?i)\b7p\s*([+\-])\s"]),
    _build_chr_pattern("ch7q", ["7q"], [r"(?i)\b7q\s*([+\-])\s"]),
    _build_chr_pattern("ch9p", ["9p"], [r"(?i)\b9p\s*([+\-])\s"]),
    _build_chr_pattern("ch9q", ["9q"], [r"(?i)\b9q\s*([+\-])\s"]),
    
    dict(
        source="codeletion_1p_19q",
        regex=[r"(?i)(?:cod[ée]l[ée]tion|co-d[ée]l[ée]tion)\s+(?:des?\s+)?(?:bras?\s+)?1p[/\s]+(?:et\s+)?19q", r"(?i)1p[/\s]*19q\s+(?:cod[ée]l[ée]t|co-d[ée]l[ée]t)"]
    ),
    dict(
        source="del_cdkn2a",
        regex=[r"(?i)d[ée]l[ée]tion\s+(?:homo(?:zygote)?|bi[- ]?all[ée]lique)\s+(?:de\s+)?cdkn2a"]
    ),
]


class EDSExtractor:
    def __init__(self):
        self._nlp = edsnlp.blank("eds")
        self._nlp.add_pipe("eds.sentences")
        self._nlp.add_pipe("eds.normalizer")
        self._nlp.add_pipe("eds.sections")
        
        self._nlp.add_pipe(
            "eds.matcher", 
            config=dict(terms=TERM_DICT, regex=REGEX_DICT, attr="LOWER")
        )
        self._nlp.add_pipe("eds.dates")
        self._nlp.add_pipe("eds.quantities")
        
        self._nlp.add_pipe("eds.terminology", name="terminology_diag", config=dict(
            label="diag_histologique",
            terms=_DIAGNOSIS_VOCAB,
            attr="LOWER"
        ))
        
        self._nlp.add_pipe("eds.terminology", name="terminology_drugs", config=dict(
            label="chimios",
            terms=_DRUG_SYNONYMS,
            attr="LOWER"
        ))
        
        self._nlp.add_pipe("eds.hemiplegia", config=dict(span_setter={"ents": True, "deficit_1er_symptome": True}))
        self._nlp.add_pipe("eds.dementia", config=dict(span_setter={"ents": True, "cognitif_1er_symptome": True}))
        
        self._nlp.add_pipe("eds.contextual_matcher", config=dict(
            attr="LOWER",
            patterns=CM_PATTERNS,
            label="contextual_feature"
        ))

        self._nlp.add_pipe("eds.negation")
        self._nlp.add_pipe("eds.family")
        self._nlp.add_pipe("eds.hypothesis")
        self._nlp.add_pipe("eds.history")

    def _parse_ihc_assigned(self, val: str) -> str:
        val = val.lower()
        range_match = re.search(r"(\d+)\s*[àa\-]\s*(\d+)\s*%", val)
        lt_match = re.search(r"<\s*(\d+)\s*%", val)
        pct_match = re.search(r"(\d+)\s*%", val)
        score_match = re.search(r"score\s+(?:de\s+)?(\d+)", val)

        if range_match: return f"{range_match.group(1)}-{range_match.group(2)}"
        if lt_match: return f"<{lt_match.group(1)}"
        if pct_match: return pct_match.group(1)
        if score_match: return score_match.group(1)
        
        for raw in sorted(_IHC_VALUE_NORM.keys(), key=len, reverse=True):
            if re.search(r"\b" + re.escape(raw) + r"\b", val) or raw in ["+", "-"] and raw in val:
                return _IHC_VALUE_NORM[raw]
        return val

    def _parse_mol_assigned(self, val: str) -> str:
        val = val.lower()
        if re.search(r"pas de mutation|absence de mutation|sauvage|wt|non mut", val):
            return "wt"
        if re.search(_VARIANT_PATTERN, val):
            return "mute"
        for raw in sorted(_MOL_STATUS_NORM.keys(), key=len, reverse=True):
            if re.search(r"\b" + re.escape(raw) + r"\b", val):
                return _MOL_STATUS_NORM[raw]
        return "mute"

    def _parse_chr_assigned(self, val: str) -> str | None:
        val = val.lower()
        if re.search(r"absence de (?:perte|d[ée]l[ée]tion)|pas de (?:perte|d[ée]l[ée]tion)", val):
            return None
        cgh = re.search(r"\b(?:1p|19q|10p|10q|7p|7q|9p|9q)\s*([+\-])\s", val)
        if cgh:
            return "gain" if cgh.group(1) == "+" else "perte"
        
        for raw in sorted(_CHR_STATUS_NORM.keys(), key=len, reverse=True):
            if re.search(r"\b" + re.escape(raw) + r"\b", val):
                return _CHR_STATUS_NORM[raw]
        return "perte"


    def extract(self, text: str, sections: dict[str, str], feature_subset: list[str]) -> dict[str, ExtractionValue]:
        doc = self._nlp(text)
        has_sections = len(sections) > 1 or (len(sections) == 1 and "full_text" not in sections)
        
        candidates: list[dict] = []
        chimios_found = []
        chimio_span_start = -1
        chimio_span_end = -1

        all_ents = list(doc.ents)
        for span_group in ["dates", "quantities"]:
            if span_group in doc.spans:
                all_ents.extend(doc.spans[span_group])
                
        seen_spans = set()
        unique_ents = []
        for ent in all_ents:
            span_id = (ent.start, ent.end, ent.label_)
            if span_id not in seen_spans:
                seen_spans.add(span_id)
                unique_ents.append(ent)

        for ent in unique_ents:
            field_name = ent.label_
            if field_name == "contextual_feature" and hasattr(ent._, "source"):
                field_name = ent._.source
            
            if ent._.family:
                continue
            
            if getattr(ent._, 'hypothesis', False) and field_name not in ["diag_histologique", "tumeur_lateralite", "classification_oms", "grade"]:
                continue
            
            if getattr(ent._, 'history', False) and field_name not in ["chimios", "type_chirurgie", "date_1er_symptome", "date", "dates", "classification_oms", "grade"]:
                continue

            emits = []
            ent_sec = "full_text"
            if hasattr(ent._, "section") and getattr(ent._, "section") is not None:
                ent_sec = ent._.section.label_
                
            if field_name == "codeletion_1p_19q":
                emits = [("ch1p", "perte"), ("ch19q", "perte"), ("ch1p19q_codel", "oui")]
            elif field_name == "del_cdkn2a":
                emits = [("mol_CDKN2A", "mute"), ("ch9p", "perte")]
            elif field_name == "sexe":
                t = ent.text.lower()
                if "femme" in t or "féminin" in t or "feminin" in t or "madame" in t or "mme" in t or "f" in t.replace("|", "").strip():
                    sex_val = "F"
                else:
                    sex_val = "M"
                emits = [("sexe", sex_val)]
            elif field_name == "tumeur_lateralite":
                norm = _LATERALITY_NORM.get(ent.text.lower(), ent.text.lower())
                emits = [("tumeur_lateralite", norm)]
            elif field_name == "dominance_cerebrale":
                emits = [("dominance_cerebrale", ent.text.lower())]
            elif field_name == "classification_oms":
                year_match = re.search(r"20\d{2}", ent.text)
                if year_match and year_match.group(0) in ["2007", "2016", "2021"]:
                    emits = [("classification_oms", year_match.group(0))]
            elif field_name == "grade":
                grade_match = re.search(r"(?:[1-4]|I{1,3}V?|IV)", ent.text, re.IGNORECASE)
                if grade_match:
                    val_str = grade_match.group(0).upper()
                    grade_val = _ROMAN_TO_INT.get(val_str) or (int(val_str) if val_str.isdigit() else None)
                    if grade_val is not None:
                        emits = [("grade", grade_val)]
            elif field_name == "type_chirurgie":
                n = _norm(ent.text.lower())
                if "complet" in n or "gtr" in n or "total" in n: type_val = "exerese complete"
                elif "partiel" in n or "incomplet" in n or "subtotal" in n or "str" in n: type_val = "exerese partielle"
                elif "biopsie" in n or "bst" in n: type_val = "biopsie"
                elif "attente" in n: type_val = "en attente"
                else: type_val = "exerese"
                emits = [("type_chirurgie", type_val)]
            elif field_name in ["optune", "corticoides", "anti_epileptiques", "epilepsie_1er_symptome", "ceph_hic_1er_symptome", "deficit_1er_symptome", "cognitif_1er_symptome", "histo_necrose", "histo_pec", "prise_de_contraste"]:
                is_negated = getattr(ent._, "negation", False)
                emits = [(field_name, "non" if is_negated else "oui")]
            elif field_name == "evol_clinique":
                emits = [("evol_clinique", ent.text.lower() if ent.text.lower() in ["initial", "terminal"] else ent.text.upper())]
            elif field_name == "diag_histologique":
                canon = ent.kb_id_ if getattr(ent, "kb_id_", "") else ent.text.lower()
                emits = [("diag_histologique", canon)]
            elif field_name == "chimios":
                canon = ent.kb_id_ if getattr(ent, "kb_id_", "") else None
                if not canon:
                    for k, syns in _DRUG_SYNONYMS.items():
                        if ent.text.lower() in [s.lower() for s in syns]:
                            canon = k
                            break
                    if not canon: 
                        canon = ent.text.lower()
                if canon not in chimios_found:
                    chimios_found.append(canon)
                    if chimio_span_start == -1 or ent.start_char < chimio_span_start:
                        chimio_span_start = ent.start_char
                    if ent.end_char > chimio_span_end:
                        chimio_span_end = ent.end_char
                continue
            elif field_name in ["date", "dates"]:
                dt = getattr(ent._, "date", None)
                if dt and dt.year:
                    if getattr(ent, "sent", None):
                        surrounding_text = ent.sent.text.lower()
                    else:
                        surrounding_text = text[max(0, ent.start_char - 40):min(len(text), ent.end_char + 40)].lower()
                        
                    for t_field, keywords in _DATE_CONTEXT_KEYWORDS.items():
                        for kw in keywords:
                            if kw in surrounding_text:
                                if t_field == "annee_de_naissance":
                                    fmt_date = str(dt.year)
                                else:
                                    fmt_date = f"{dt.day:02d}/{dt.month:02d}/{dt.year}" if dt.month and dt.day else str(dt.year)
                                emits.append((t_field, fmt_date))
                                break
            elif field_name in ["rx_dose", "histo_mitoses", "chm_cycles", "ik_clinique"]:
                if field_name == "ik_clinique":
                    num_match = re.search(r"(\d{2,3})", ent.text)
                else:
                    num_match = re.search(r"(\d+(?:[.,]\d+)?)", ent.text)
                if num_match:
                    val_str = num_match.group(1).replace(",", ".")
                    val_num = int(float(val_str)) if field_name in ["histo_mitoses", "chm_cycles", "ik_clinique"] else val_str
                    emits = [(field_name, val_num)]
            
            if not emits:
                if field_name not in feature_subset:
                    continue
                value = None
                span_to_parse = ent.text
                if hasattr(ent._, "assigned"):
                    assign_dict = ent._.assigned
                    if assign_dict and "value" in assign_dict:
                        assigned_val = assign_dict["value"]
                        if isinstance(assigned_val, list) and len(assigned_val) > 0:
                            span_to_parse = assigned_val[0].text if hasattr(assigned_val[0], 'text') else str(assigned_val[0])
                        elif hasattr(assigned_val, 'text'):
                            span_to_parse = assigned_val.text
                        else:
                            span_to_parse = str(assigned_val)
                            
                if field_name.startswith("ihc_"):
                    value = self._parse_ihc_assigned(span_to_parse)
                    if field_name == "ihc_atrx" and value == "negatif" and "perte" in ent.text.lower():
                        value = "perte"
                elif field_name.startswith("mol_"):
                    value = self._parse_mol_assigned(span_to_parse)
                    if field_name == "mol_mgmt":
                        # FIXED: Removed the "mgmt" check and check both the assigned span and original text
                        check_text = (span_to_parse + " " + ent.text).lower()
                        if "promoteur" in check_text or "méthyl" in check_text or "methyl" in check_text:
                            if re.search(r"non |im|hypo", check_text): value = "non methyle"
                            elif "hyper" in check_text: value = "hypermethyle"
                            else: value = "methyle"
                elif field_name.startswith("ch"):
                    value = self._parse_chr_assigned(span_to_parse)
                
                if value is not None:
                    emits = [(field_name, value)]
            
            if emits:
                for subfield, subval in emits:
                    if subfield in feature_subset:
                        candidates.append({
                            "field": subfield,
                            "value": subval,
                            "span": ent.text,
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "section": ent_sec
                        })

        grouped = defaultdict(list)
        for c in candidates:
            grouped[c["field"]].append(c)

        results = {}
        for field, c_list in grouped.items():
            if not has_sections:
                best = c_list[0]
            else:
                preferred = get_section_for_feature(field)
                c_list.sort(key=lambda x: (
                    not (x["section"] in preferred),
                    not (x["section"] == "preamble"),
                    x["start"]
                ))
                best = c_list[0]
            
            results[field] = ExtractionValue(
                value=best["value"],
                source_span=best["span"],
                source_span_start=best["start"],
                source_span_end=best["end"],
                extraction_tier="rule",
                confidence=0.9,
                vocab_valid=True,
            )

        if "chimios" in feature_subset and chimios_found:
            results["chimios"] = ExtractionValue(
                value=" + ".join(chimios_found),
                source_span=text[chimio_span_start:chimio_span_end],
                source_span_start=chimio_span_start,
                source_span_end=chimio_span_end,
                extraction_tier="rule",
                confidence=0.85,
                vocab_valid=True,
            )

        return results