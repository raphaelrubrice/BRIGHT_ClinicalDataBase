"""Disambiguator step prior to GLiNER extraction.

This module provides the `Disambiguator` class, which scans document text for
known, highly ambiguous clinical terms (e.g. trial names, specific chemo agents)
and injects specifying context right after them (e.g., `" (essai thérapeutique)"`).

This helps the downstream GLiNER model by providing explicit local context.

Because this modifies the text, it returns the modified string alongside an
`offset_mapper` function to perfectly align extracted spans back to the original
unmodified document text.
"""

from __future__ import annotations

import logging
import re
from typing import Callable
from rapidfuzz import fuzz

from src.extraction.text_normalisation import normalise

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────
# Clinical Entities Repertoires & Contexts
# ───────────────────────────────────────────────────────────────────────

_DISAMBIGUATION_FR: dict[str, dict[str, list[str] | str]] = {
    "essai_therapeutique": {
        "context": " (essai thérapeutique)",
        "terms": [
            "essai", "étude", "protocole", "ivy", "biomede", "biomede-2", "fore", "gliofocus", "gliomrs", "eortc", 
            "gliostar", "incyte", "fight 209", "fight-209", "indigo", "keynote", "legato", "navig", "navig-1", 
            "olatmz", "optimum", "polo", "polca", "rosalie", "resilience", "rc-102", "sonobird", "sonofirst", 
            "sonocloud", "strateglio", "temotrad", "trident", "ucpvax", "ucp-vax", "tiger", "checkmate", 
            "marizornib", "olagli", "revolumab", "target", "stellar", "onconeurotek", "onconeurotek 2", "rcp", 
            "phrc", "eortc 26981", "avaglio", "centric", "core", "ef-14", "rtog", "geino", "cohorte", "registre", 
            "essai clinique", "phase i", "phase ii", "phase iii"
        ]
    },
    "chimio_protocole": {
        "context": " (protocole chimiothérapie)",
        "terms": [
            "agile", "stupp", "stupp accéléré", "pcv", "carbo-vp16", "memmat", "ccnu-avastin", "témo-avastin", 
            "tmz-avastin", "carbo-avastin", "bevacizu", "bev", "rt-tmz", "chimioradiothérapie", "crt", "bev-iri", 
            "folfiri", "folfox", "r-chop", "protocole de stupp"
        ]
    },
    "optune": {
        "context": " (dispositif optune)",
        "terms": [
            "optune", "ttf", "novocure", "tumor treating fields", "optunegitmz", "ttfields", "dispositif optune", 
            "électrodes optune", "traitement par champs électriques"
        ]
    },
    "chimios": {
        "context": " (chimiothérapie)",
        "terms": [
            "chimio", "chimiothérapie", "temodal", "témodal", "tmz", "temozok", "avastin", "bevacizumab", 
            "lomustine", "belustine", "natulan", "procarbazine", "vincristine", "vcr", "carboplatine", "carbo", 
            "vp 16", "etoposide", "pemigatinib", "pemi", "carmustine", "gliadel", "bcnu", "fotémustine", "muphoran", 
            "irinotécan", "campto", "méthotrexate", "mtx", "oncovin", "cyclophosphamide", "endoxan", "doxorubicine", 
            "nivolumab", "pembrolizumab", "dacarbazine"
        ]
    },
    "anti_epileptiques": {
        "context": " (traitement antiépileptique)",
        "terms": [
            "anti-épileptique", "anti-comitiaux", "keppra", "levetiracetam", "lev", "vimpat", "lacosamide", 
            "urbanyl", "clobazam", "lamotrigine", "fycompa", "perampanel", "briviact", "valproate", "dépakine", 
            "micropakine", "carbamazépine", "tégrétol", "phénytoïne", "di-hydan", "oxcarbazépine", "trileptal", 
            "zonisamide", "zonegran", "prégabaline", "lyrica", "gabapentine", "neurontin", "topiramate", "epitomax", 
            "clonazépam", "rivotril", "aed"
        ]
    },
    "corticoides": {
        "context": " (corticoïde)",
        "terms": [
            "corticoides", "ctc", "prednisone", "prednisolone", "medrol", "solumedrol", "dexamethasone", "dexa", 
            "hmn", "mondor", "cortancyl", "hydrocortisone", "bétaméthasone", "célestène", "méthylprednisolone", 
            "dectancyl", "cortisone", "corticothérapie"
        ]
    },
    "hopitaux": {
        "context": " (hôpital)",
        "terms": [
            "pitié", "pitié-salpêtrière", "psl", "mondor", "saint louis", "st louis", "foch", "sainte anne", 
            "st anne", "st joseph", "lariboisière", "larib", "lrb", "beaujon", "bjn", "bicêtre", "kb", 
            "charles foix", "champigny", "jeanne garnier", "usp", "ssr", "bottard", "gustave roussy", "igr", 
            "institut curie", "curie", "hegp", "georges pompidou", "necker", "tenon", "cochin", "hôtel-dieu", 
            "avicenne", "val de grâce", "percy", "chu", "chr", "clcc", "chru", "fondation rothschild"
        ]
    },
    "professions": {
        "context": " (professionnel de santé)",
        "terms": [
            "kiné", "kinésithérapeute", "orthophoniste", "ergo", "ergothérapeute", "neurochirurgien", "nch", 
            "neurochir", "radiologue", "méd nucl", "médecin nucléaire", "neuropath", "neuropathologiste", 
            "rééducateur", "umasp", "équipe soins palliatifs", "infirmière référente", "infirmière", "ide", "ipa", 
            "psychologue", "psychiatre", "radiothérapeute", "oncologue", "onco", "neuro-oncologue", "neurologue", 
            "anesthésiste", "mar", "idec", "as", "diététicien", "diététicienne", "assistante sociale", "arc", "tec", 
            "pharmacien", "anapath"
        ]
    }
}

_DISAMBIGUATION_EN: dict[str, dict[str, list[str] | str]] = {
    "essai_therapeutique": {
        "context": " (clinical trial)",
        "terms": [
            "trial", "study", "protocol", "ivy", "biomede", "biomede-2", "fore", "gliofocus", "gliomrs", "eortc", 
            "gliostar", "incyte", "fight 209", "fight-209", "indigo", "keynote", "legato", "navig", "navig-1", 
            "olatmz", "optimum", "polo", "polca", "rosalie", "resilience", "rc-102", "sonobird", "sonofirst", 
            "sonocloud", "strateglio", "temotrad", "trident", "ucpvax", "ucp-vax", "tiger", "checkmate", 
            "marizornib", "olagli", "revolumab", "target", "stellar", "onconeurotek", "onconeurotek 2", "mdt", 
            "eortc 26981", "avaglio", "centric", "core", "ef-14", "rtog", "geino", "cohort", "registry", "phase i", 
            "phase ii", "phase iii"
        ]
    },
    "chimio_protocole": {
        "context": " (chemotherapy protocol)",
        "terms": [
            "agile", "stupp", "accelerated stupp", "pcv", "carbo-vp16", "memmat", "ccnu-avastin", "tmz-avastin", 
            "carbo-avastin", "bev", "rt-tmz", "chemoradiotherapy", "crt", "bev-iri", "folfiri", "folfox", "r-chop", 
            "stupp protocol"
        ]
    },
    "optune": {
        "context": " (tumor treating fields)",
        "terms": [
            "optune", "ttf", "novocure", "tumor treating fields", "optunegitmz", "ttfields", "optune device", 
            "optune arrays"
        ]
    },
    "chimios": {
        "context": " (chemotherapy agent)",
        "terms": [
            "chemo", "chemotherapy", "temodar", "temozolomide", "tmz", "temozok", "avastin", "bevacizumab", 
            "lomustine", "gleostine", "ceenu", "natulan", "matulane", "procarbazine", "vincristine", "vcr", 
            "carboplatin", "carbo", "vp-16", "vp 16", "etoposide", "pemigatinib", "pemi", "carmustine", "gliadel", 
            "bcnu", "fotemustine", "irinotecan", "camptosar", "methotrexate", "mtx", "oncovin", "cyclophosphamide", 
            "cytoxan", "doxorubicin", "nivolumab", "pembrolizumab", "dacarbazine"
        ]
    },
    "anti_epileptiques": {
        "context": " (anticonvulsant treatment)",
        "terms": [
            "aed", "anti-epileptic drug", "anticonvulsant", "keppra", "levetiracetam", "lev", "vimpat", "lacosamide", 
            "onfi", "clobazam", "lamotrigine", "lamictal", "fycompa", "perampanel", "briviact", "brivaracetam", 
            "valproate", "depakote", "carbamazepine", "tegretol", "phenytoin", "dilantin", "oxcarbazepine", 
            "trileptal", "zonisamide", "zonegran", "pregabalin", "lyrica", "gabapentin", "neurontin", "topiramate", 
            "topamax", "clonazepam", "klonopin"
        ]
    },
    "corticoides": {
        "context": " (corticosteroid)",
        "terms": [
            "corticosteroids", "steroids", "ctc", "prednisone", "prednisolone", "medrol", "solumedrol", 
            "dexamethasone", "dexa", "decadron", "hydrocortisone", "betamethasone", "cortisone", "steroid therapy"
        ]
    },
    "hopitaux": {
        "context": " (hospital)",
        "terms": [
            "pitié", "pitié-salpêtrière", "psl", "mondor", "saint louis", "st louis", "foch", "sainte anne", 
            "st anne", "st joseph", "lariboisière", "larib", "lrb", "beaujon", "bjn", "bicêtre", "kb", 
            "charles foix", "champigny", "jeanne garnier", "usp", "ssr", "bottard", "gustave roussy", "igr", 
            "institut curie", "curie", "hegp", "georges pompidou", "necker", "tenon", "cochin", "hôtel-dieu", 
            "avicenne", "val de grâce", "percy", "chu", "chr", "chru", "fondation rothschild", "hospital", 
            "medical center", "clinic", "institute"
        ]
    },
    "professions": {
        "context": " (healthcare professional)",
        "terms": [
            "pt", "physical therapist", "speech therapist", "ot", "occupational therapist", "neurosurgeon", 
            "radiologist", "nuclear medicine physician", "neuropathologist", "pathologist", "palliative care team", 
            "nurse", "rn", "aprn", "psychologist", "psychiatrist", "radiation oncologist", "oncologist", 
            "neuro-oncologist", "neurologist", "anesthesiologist", "social worker", "clinical research coordinator", 
            "crc", "pharmacist"
        ]
    }
}


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

_SHORT_TERM_MAX_LEN = 3


def _build_short_pattern(term: str) -> re.Pattern[str]:
    """Compile a case-insensitive word-boundary regex for *term*."""
    escaped = re.escape(term)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE | re.UNICODE)


# ───────────────────────────────────────────────────────────────────────
# Main class
# ───────────────────────────────────────────────────────────────────────

class Disambiguator:
    """Find known clinical entities and inject context to help GLiNER."""

    # We only inject context if the term is found without another overlapping context nearby
    _FUZZY_THRESHOLD = 90

    def __init__(self) -> None:
        # Pre-compile short-term patterns.
        # Keyed by (language, category, term_normalised).
        self._short_pats: dict[tuple[str, str, str], re.Pattern[str]] = {}
        for reg_id, registry in (("fr", _DISAMBIGUATION_FR),
                                 ("en", _DISAMBIGUATION_EN)):
            for category, data in registry.items():
                for term in data["terms"]:
                    norm_term = normalise(term)
                    if len(norm_term) <= _SHORT_TERM_MAX_LEN:
                        key = (reg_id, category, norm_term)
                        self._short_pats[key] = _build_short_pattern(norm_term)

    # ─── public API ───────────────────────────────────────────────────

    def apply(self, text: str, language: str = "fr") -> tuple[str, Callable[[int], int]]:
        """Run Disambiguator on *text*.

        Parameters
        ----------
        text : str
            Full document text.
        language : str
            ``"fr"`` or ``"en"`` — selects the contexts registry.

        Returns
        -------
        tuple[str, Callable[[int], int]]
            - The modified text string (with injected contexts).
            - An `offset_mapper` function `f(new_index)` -> `original_index` 
              to map slices from the modified text back to the original text.
        """
        registry = _DISAMBIGUATION_FR if language.startswith("fr") else _DISAMBIGUATION_EN
        reg_id = "fr" if language.startswith("fr") else "en"

        text_norm = normalise(text)

        # 1. Collect all valid hits across all categories
        # Each hit: (start_idx, end_idx, context_to_inject)
        hits: list[tuple[int, int, str]] = []

        for category, data in registry.items():
            context_to_add = str(data["context"])
            self._find_hits(
                text_norm, text, data["terms"], reg_id, category, context_to_add, hits
            )

        if not hits:
            # Short-circuit if nothing found. Mapper is identity.
            return text, lambda i: i

        # 2. Sort hits by start position. 
        # Handle overlaps by greedily picking the first longest one.
        hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
        
        filtered_hits: list[tuple[int, int, str]] = []
        last_end = -1
        for start, end, ctx in hits:
            # Skip if hit overlaps with a previously accepted hit
            if start < last_end:
                continue
            
            # Additional check: Don't inject if the context is already present right after
            # Look ahead up to 5 characters (spaces, punctuation)
            lookahead = text_norm[end:min(end + 5 + len(ctx), len(text_norm))]
            if normalise(ctx) in lookahead:
                continue
                
            filtered_hits.append((start, end, ctx))
            last_end = end

        if not filtered_hits:
            return text, lambda i: i

        # 3. Apply changes and build the offset map
        # We process hits from left to right.
        modified_text = []
        
        # A list of (new_idx, original_idx) correspondences where divergence happens
        mappings: list[tuple[int, int]] = [(0, 0)]
        
        curr_orig = 0
        curr_new = 0
        
        for start, end, ctx in filtered_hits:
            # Add unmodified text up to the end of the term
            chunk = text[curr_orig:end]
            modified_text.append(chunk)
            
            curr_new += len(chunk)
            curr_orig = end
            
            # Inject context
            modified_text.append(ctx)
            
            # The context was added, so new index advances but original doesn't
            mappings.append((curr_new, curr_orig)) # Just before context
            
            curr_new += len(ctx)
            
            # After context
            mappings.append((curr_new, curr_orig))

        # Add remaining text
        modified_text.append(text[curr_orig:])
        final_text = "".join(modified_text)

        # Build mapping function for binary search
        from bisect import bisect_right
        
        def offset_mapper(idx: int) -> int:
            """Map an index from `final_text` back to `text`."""
            if idx <= 0:
                return 0
            if idx >= len(final_text):
                return len(text)
                
            # Find the segment
            i = bisect_right(mappings, (idx, float('inf'))) - 1
            if i < 0:
                return idx # Fallback
                
            new_idx_anchor, orig_idx_anchor = mappings[i]
            
            # If we fall inside an injected context (the region where curr_orig didn't advance)
            # Both new_idx_before_ctx and new_idx_after_ctx map to the exact same orig_idx_anchor.
            # If `idx` is between them, we just return the orig_idx_anchor so the span stops exactly there.
            
            # Actually, `mappings[i]` tells us what the offset was at `new_idx_anchor`.
            # Let's see if we are in an injected block.
            if i > 0 and mappings[i][1] == mappings[i-1][1]:
                # We are in an injected block. Any index inside this block maps to the anchor
                return orig_idx_anchor

            # We are in normal text
            return orig_idx_anchor + (idx - new_idx_anchor)

        return final_text, offset_mapper


    # ─── internals ────────────────────────────────────────────────────

    def _find_hits(
        self,
        text_norm: str,
        text_orig: str,
        terms_list: list[str],
        reg_id: str,
        category: str,
        context_to_add: str,
        out: list[tuple[int, int, str]],
    ) -> None:
        """Populate *out* with hits for one category."""
        for term in terms_list:
            term_norm = normalise(term)
            if not term_norm:
                continue

            if len(term_norm) <= _SHORT_TERM_MAX_LEN:
                # Exact word-boundary search on normalised text
                pat = self._short_pats.get((reg_id, category, term_norm))
                if pat is None:
                    pat = _build_short_pattern(term_norm)
                for m in pat.finditer(text_norm):
                    out.append((m.start(), m.end(), context_to_add))
            else:
                # Sliding-window fuzzy search for longer terms
                self._fuzzy_scan(
                    text_norm, term_norm, self._FUZZY_THRESHOLD,
                    context_to_add, out,
                )

    @staticmethod
    def _fuzzy_scan(
        text_norm: str,
        term_norm: str,
        threshold: int,
        context_to_add: str,
        out: list[tuple[int, int, str]],
    ) -> None:
        """Scan *text_norm* for fuzzy occurrences of *term_norm*."""
        term_len = len(term_norm)
        win_size = int(term_len * 1.5) + 1
        step = max(1, term_len // 2)
        text_len = len(text_norm)

        # First try exact substring check
        idx = text_norm.find(term_norm)
        while idx != -1:
            out.append((idx, idx + term_len, context_to_add))
            idx = text_norm.find(term_norm, idx + term_len)

        # Optimisation: if we found the term exactly, don't bother with fuzzy
        # This prevents picking up weird partial matches when a perfect match exists.
        # But here we search globally for the whole document, so let's continue fuzzy 
        # but skip areas close to the exact match. For simplicity, we just do fuzzy.
        
        pos = 0
        while pos + win_size <= text_len:
            window = text_norm[pos:pos + win_size]
            score = fuzz.partial_ratio(term_norm, window)
            if score >= threshold:
                out.append((pos, min(pos + win_size, text_len), context_to_add))
                pos += win_size
            else:
                pos += step

        # Handle tail
        if text_len > win_size:
            tail = text_norm[text_len - win_size:]
            score = fuzz.partial_ratio(term_norm, tail)
            if score >= threshold:
                out.append((
                    text_len - win_size,
                    text_len,
                    context_to_add,
                ))
