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
from typing import Callable, Optional
from rapidfuzz import fuzz

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
# Lazy-loaded spaCy models (Aligned with similarity.py pattern)
# ───────────────────────────────────────────────────────────────────────

_models: dict[str, any] = {}

def _get_nlp(lang: str):
    """Lazy load the spaCy model appropriate for the given language."""
    global _models
    model_name = "fr_core_news_lg" if lang.startswith("fr") else "en_core_web_lg"
    
    if model_name not in _models:
        try:
            import spacy
            logger.info("Loading spaCy model: %s for Disambiguator", model_name)
            _models[model_name] = spacy.load(model_name)
        except (ImportError, OSError) as exc:
            logger.error("spaCy model %s not available. Disambiguator requires it. (%s)", model_name, exc)
            raise
            
    return _models[model_name]


# ───────────────────────────────────────────────────────────────────────
# Main class
# ───────────────────────────────────────────────────────────────────────

class Disambiguator:
    """Find known clinical entities using a spaCy-tokenized sliding window and rapidfuzz."""

    def __init__(self, similarity_threshold: float = 0.85, max_context_proximity: int = 5) -> None:
        """
        Parameters
        ----------
        similarity_threshold : float
            Threshold for rapidfuzz matching. 
            0.85 gracefully handles typos (e.g. "temodla" instead of "temodal").
        max_context_proximity : int
            Maximum token distance between two identical context injections 
            to trigger deduplication. Keeps only the latest one.
            (e.g. 5 prevents stuttering like "essai clinique (trial) onconeurotek 2 (trial)")
        """
        self.similarity_threshold = similarity_threshold
        self.max_context_proximity = max_context_proximity
        
        # Stores pre-tokenized terminologies by language to avoid repeated processing
        self._processed_terms: dict[str, dict] = {"fr": None, "en": None}

    def _init_language(self, language: str) -> None:
        """Pre-tokenize all vocabulary terms using spaCy for strict alignment."""
        reg_id = "fr" if language.startswith("fr") else "en"
        
        # Already initialized
        if self._processed_terms[reg_id] is not None:
            return
            
        nlp = _get_nlp(reg_id)
        registry = _DISAMBIGUATION_FR if reg_id == "fr" else _DISAMBIGUATION_EN
        
        terms_list = []
        max_tokens = 0
        
        for category, data in registry.items():
            context = str(data["context"])
            for term in data["terms"]:
                term_norm = term.lower().strip()
                if not term_norm:
                    continue
                
                # Tokenize term using spaCy (ensures same hyphen/punctuation logic as documents)
                doc = nlp(term_norm)
                tokens = [t.text for t in doc if not t.is_space]
                num_tokens = len(tokens)
                
                if num_tokens == 0:
                    continue
                    
                max_tokens = max(max_tokens, num_tokens)
                term_joined = " ".join(tokens)
                terms_list.append((term_joined, num_tokens, context))
        
        # Sort descending by character length to prioritize compound terms
        terms_list.sort(key=lambda x: len(x[0]), reverse=True)
        
        self._processed_terms[reg_id] = {
            "terms": terms_list,
            "max_tokens": max_tokens
        }

    # ─── public API ───────────────────────────────────────────────────

    def apply(self, text: str, language: str = "fr") -> tuple[str, Callable[[int], int]]:
        """Run Disambiguator on *text*.

        Parameters
        ----------
        text : str
            Full document text.
        language : str
            ``"fr"`` or ``"en"`` — selects the contexts registry and spaCy model.

        Returns
        -------
        tuple[str, Callable[[int], int]]
            - The modified text string (with injected contexts).
            - An `offset_mapper` function `f(new_index)` -> `original_index` 
              to map slices from the modified text back to the original text.
        """
        reg_id = "fr" if language.startswith("fr") else "en"
        
        # Ensure vocab is parsed with spaCy for the requested language
        self._init_language(reg_id)
        registry_data = self._processed_terms.get(reg_id)
        
        if not registry_data or not registry_data["terms"]:
            return text, lambda i: i
            
        terms_list = registry_data["terms"]
        
        # Allow max window to exceed the longest term token count by 2
        # just in case the document separates hyphens heavily.
        max_window = registry_data["max_tokens"] + 2
        
        # 1. Tokenize document using spaCy to match terminology splits
        nlp = _get_nlp(reg_id)
        doc = nlp(text)
        
        # Keep track of text tokens and exact character boundaries, skipping pure spaces
        doc_tokens = [(t.text, t.idx, t.idx + len(t.text)) for t in doc if not t.is_space]
        
        if not doc_tokens:
            return text, lambda i: i

        candidate_hits: list[dict] = []
        
        # 2. Sliding window over the text: Collect ALL possible matches exceeding threshold
        for i in range(len(doc_tokens)):
            for window_size in range(1, max_window + 1):
                if i + window_size > len(doc_tokens):
                    break
                    
                window_slice = doc_tokens[i : i + window_size]
                
                # Join exact tokens from document
                window_text_joined = " ".join([t[0].lower() for t in window_slice])
                
                # Compare against all terms using RapidFuzz
                for term_joined, num_tokens, context in terms_list:
                    
                    # Optimization: skip if lengths are wildly different
                    # (helps speed up and prevents impossible matches from evaluating)
                    len_diff = abs(len(window_text_joined) - len(term_joined))
                    if len_diff > max(5, len(term_joined) * 0.3):
                        continue
                        
                    # Calculate Levenshtein similarity via rapidfuzz
                    score = fuzz.ratio(window_text_joined, term_joined) / 100.0
                    
                    if score >= self.similarity_threshold:
                        candidate_hits.append({
                            "start_char": window_slice[0][1],
                            "end_char": window_slice[-1][2],
                            "start_tok": i,
                            "end_tok": i + window_size,
                            "score": score,
                            "term_len": len(term_joined),
                            "context": context
                        })

        if not candidate_hits:
            return text, lambda i: i

        # 3. Overlap Resolution
        # Sort candidates by best score first, then by the length of the matched term 
        # (this perfectly resolves the "onconeurotek 2" vs "onconeurotek 2 le" issue)
        candidate_hits.sort(key=lambda h: (h["score"], h["term_len"]), reverse=True)

        final_hits = []
        consumed_toks = set()

        for h in candidate_hits:
            # Skip if any token in this match was already consumed by a better/longer match
            if any(t in consumed_toks for t in range(h["start_tok"], h["end_tok"])):
                continue
                
            # Anti-double injection safeguard: don't inject if context is naturally present right after
            ctx_lower = h["context"].strip().lower()
            lookahead_slice = text[h["end_char"] : h["end_char"] + len(h["context"]) + 5].lower()
            if ctx_lower in lookahead_slice:
                # Mark tokens as consumed so we don't accidentally tag a sub-part of them, 
                # but don't add to final_hits
                for t in range(h["start_tok"], h["end_tok"]):
                    consumed_toks.add(t)
                continue
                
            final_hits.append(h)
            for t in range(h["start_tok"], h["end_tok"]):
                consumed_toks.add(t)

        # 4. Sort final accepted hits chronologically by start position
        final_hits.sort(key=lambda h: h["start_char"])

        # 5. Proximity Deduplication Heuristic
        # Remove identical context injections that are too close to each other,
        # keeping only the latest one to lighten the text.
        deduplicated_hits = []
        for i in range(len(final_hits)):
            keep = True
            curr_hit = final_hits[i]
            
            # Look ahead to see if the same context repeats shortly
            for j in range(i + 1, len(final_hits)):
                next_hit = final_hits[j]
                
                # Distance in tokens between end of current hit and start of next hit
                dist = next_hit["start_tok"] - curr_hit["end_tok"]
                
                if dist > self.max_context_proximity:
                    break  # Hits are too far apart, stop looking
                    
                if curr_hit["context"] == next_hit["context"]:
                    # An identical context will be injected very soon.
                    # We drop the current injection.
                    keep = False
                    break
                    
            if keep:
                deduplicated_hits.append(curr_hit)

        # 6. Apply changes and build the offset mapping array
        modified_text = []
        mappings: list[tuple[int, int]] = [(0, 0)]
        
        curr_orig = 0
        curr_new = 0
        
        for h in deduplicated_hits:
            start = h["start_char"]
            end = h["end_char"]
            ctx = h["context"]
            
            chunk = text[curr_orig:end]
            modified_text.append(chunk)
            
            curr_new += len(chunk)
            curr_orig = end
            
            modified_text.append(ctx)
            
            # Anchor just before context
            mappings.append((curr_new, curr_orig)) 
            
            curr_new += len(ctx)
            
            # Anchor right after context
            mappings.append((curr_new, curr_orig))

        modified_text.append(text[curr_orig:])
        final_text = "".join(modified_text)

        # 7. Build mapper function
        from bisect import bisect_right
        
        def offset_mapper(idx: int) -> int:
            """Map an index from `final_text` back to `text`."""
            if idx <= 0:
                return 0
            if idx >= len(final_text):
                return len(text)
                
            i = bisect_right(mappings, (idx, float('inf'))) - 1
            if i < 0:
                return idx
                
            new_idx_anchor, orig_idx_anchor = mappings[i]
            
            # Detect injected block
            if i + 1 < len(mappings) and mappings[i][1] == mappings[i+1][1]:
                return orig_idx_anchor

            # Normal text space
            return orig_idx_anchor + (idx - new_idx_anchor)

        return final_text, offset_mapper