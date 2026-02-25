"""EDS-NLP negation, hypothesis, and history detection wrapper.

Wraps EDS-NLP assertion annotation components to determine whether
extracted spans are negated, hypothetical, or historical.

Public API
----------
- ``AnnotatedSpan``       – Result dataclass for a single span annotation.
- ``AssertionAnnotator``  – Pre-annotate text spans with assertion status.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedSpan:
    """A single annotated span with assertion status.

    Attributes
    ----------
    text : str
        The text of the span.
    start : int
        Character offset (start) in the original text.
    end : int
        Character offset (end) in the original text.
    label : str
        Semantic label (e.g. ``"epilepsie"``, ``"IDH1"``).
    is_negated : bool
        Whether the span is negated (e.g. "pas d'épilepsie").
    is_hypothesis : bool
        Whether the span is hypothetical (e.g. "possiblement…").
    is_history : bool
        Whether the span refers to a historical event.
    """

    text: str
    start: int
    end: int
    label: str = ""
    is_negated: bool = False
    is_hypothesis: bool = False
    is_history: bool = False


# ---------------------------------------------------------------------------
# Regex-based negation detection (fallback)
# ---------------------------------------------------------------------------

# Common French negation cues that precede or surround a medical term.
_NEGATION_PATTERNS: list[re.Pattern[str]] = [
    # "pas de X", "pas d'X"
    re.compile(r"(?i)\bpas\s+(?:de|d['']\s*)", re.UNICODE),
    # "absence de X", "absence d'X"
    re.compile(r"(?i)\babsence\s+(?:de|d['']\s*)", re.UNICODE),
    # "sans X"
    re.compile(r"(?i)\bsans\s+", re.UNICODE),
    # "aucun(e) X"
    re.compile(r"(?i)\baucun(?:e)?\s+", re.UNICODE),
    # "ni X" (e.g. "ni gain ni perte")
    re.compile(r"(?i)\bni\s+", re.UNICODE),
    # "non X" (free-standing: "non méthylé", "non muté")
    re.compile(r"(?i)\bnon\s+", re.UNICODE),
    # "ne … pas" (catch partial patterns near spans)
    re.compile(r"(?i)\bn['']?\s*(?:est|a|montre|révèle|retrouve|objective)\s+pas\b", re.UNICODE),
    # "négatif / negative" immediately before/after
    re.compile(r"(?i)\bn[ée]gatif(?:ve)?\b", re.UNICODE),
]

# Context window (characters) around a span to search for negation cues.
_CONTEXT_WINDOW: int = 60

_HYPOTHESIS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\bpossible(?:ment)?\b", re.UNICODE),
    re.compile(r"(?i)\bprobable(?:ment)?\b", re.UNICODE),
    re.compile(r"(?i)\bsuspecté[e]?\b", re.UNICODE),
    re.compile(r"(?i)\bsuspecion\b", re.UNICODE),
    re.compile(r"(?i)\bsuspicion\b", re.UNICODE),
    re.compile(r"(?i)\bà\s+confirmer\b", re.UNICODE),
    re.compile(r"(?i)\bà\s+(?:confronter|corréler)\b", re.UNICODE),
    re.compile(r"(?i)\béventuel(?:le(?:ment)?)?\b", re.UNICODE),
    re.compile(r"(?i)\bhypothèse\b", re.UNICODE),
]

_HISTORY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\bantécédent(?:s)?\b", re.UNICODE),
    re.compile(r"(?i)\bhistoire\s+de\b", re.UNICODE),
    re.compile(r"(?i)\bhistorique(?:ment)?\b", re.UNICODE),
    re.compile(r"(?i)\bancien(?:ne)?(?:ment)?\b", re.UNICODE),
    re.compile(r"(?i)\bpr[ée]c[ée]demment\b", re.UNICODE),
    re.compile(r"(?i)\baut[ée]rieurement\b", re.UNICODE),
    re.compile(r"(?i)\ben\s+\d{4}\b", re.UNICODE),
]


# Sentence-ending punctuation used to prevent negation cues from bleeding
# across sentence boundaries.
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?;]\s", re.UNICODE)


def _has_pattern_near_span(
    text: str,
    span_start: int,
    span_end: int,
    patterns: list[re.Pattern[str]],
    window: int = _CONTEXT_WINDOW,
    look_after: bool = False,
) -> bool:
    """Return True if any pattern matches near the span.

    By default searches the *window* chars **before** the span.  When
    *look_after* is True, also searches the *window* chars **after** the
    span.  Sentence boundaries (e.g. ``. ``) are respected so that cues
    from a different sentence are not counted.
    """
    # --- Look BEFORE the span ---
    ctx_start = max(0, span_start - window)
    before_context = text[ctx_start:span_end]
    # Trim to the last sentence boundary before the span so we don't
    # accidentally pick up cues from a preceding sentence.
    boundary_matches = list(_SENTENCE_BOUNDARY_RE.finditer(before_context))
    if boundary_matches:
        # Keep only context from the last sentence boundary onwards
        last_boundary_end = boundary_matches[-1].end()
        # Only trim if the boundary is before the span start (relative)
        rel_span_start = span_start - ctx_start
        if last_boundary_end <= rel_span_start:
            before_context = before_context[last_boundary_end:]
            ctx_start = ctx_start + last_boundary_end

    for pat in patterns:
        m = pat.search(before_context)
        if m is not None:
            cue_end_abs = ctx_start + m.end()
            if cue_end_abs <= span_end:
                return True

    # --- Look AFTER the span (optional) ---
    if look_after:
        ctx_end = min(len(text), span_end + window)
        after_context = text[span_start:ctx_end]
        # Trim at the first sentence boundary after the span
        rel_span_end = span_end - span_start
        boundary_after = _SENTENCE_BOUNDARY_RE.search(after_context, rel_span_end)
        if boundary_after:
            after_context = after_context[:boundary_after.start()]

        for pat in patterns:
            m = pat.search(after_context)
            if m is not None:
                return True

    return False


# ---------------------------------------------------------------------------
# AssertionAnnotator
# ---------------------------------------------------------------------------

class AssertionAnnotator:
    """Pre-annotate text spans with negation/hypothesis/history status.

    The annotator attempts to use **edsnlp** (spaCy-based NLP pipeline)
    for high-quality assertion detection.  If edsnlp is not available,
    it falls back to a regex-based approach.

    Parameters
    ----------
    use_edsnlp : bool, optional
        Whether to try loading edsnlp (default ``True``).  Set to
        ``False`` for testing without the dependency.
    """

    def __init__(self, use_edsnlp: bool = True) -> None:
        self._nlp = None
        self._backend: str = "regex"  # "edsnlp" or "regex"

        if use_edsnlp:
            try:
                self._init_edsnlp()
                self._backend = "edsnlp"
            except Exception:
                # edsnlp not installed or failed → fall back to regex
                pass

    # -- edsnlp initialisation -----------------------------------------------

    def _init_edsnlp(self) -> None:
        """Initialise an edsnlp pipeline with assertion components."""
        import edsnlp

        nlp = edsnlp.blank("eds")
        nlp.add_pipe("eds.sentences")
        nlp.add_pipe("eds.negation")
        nlp.add_pipe("eds.hypothesis")
        nlp.add_pipe("eds.history")
        self._nlp = nlp

    # -- Public API -----------------------------------------------------------

    @property
    def backend(self) -> str:
        """Return ``"edsnlp"`` or ``"regex"`` depending on the active backend."""
        return self._backend

    def annotate(
        self,
        text: str,
        spans: list[tuple[int, int, str]],
    ) -> list[AnnotatedSpan]:
        """Annotate a list of spans within *text*.

        Parameters
        ----------
        text : str
            The full document (or section) text.
        spans : list[tuple[int, int, str]]
            Each element is ``(start_offset, end_offset, label)``.

        Returns
        -------
        list[AnnotatedSpan]
            One ``AnnotatedSpan`` per input span, enriched with
            ``is_negated``, ``is_hypothesis``, and ``is_history`` flags.
        """
        if self._backend == "edsnlp" and self._nlp is not None:
            return self._annotate_edsnlp(text, spans)
        return self._annotate_regex(text, spans)

    # -- Backends -------------------------------------------------------------

    def _annotate_edsnlp(
        self,
        text: str,
        spans: list[tuple[int, int, str]],
    ) -> list[AnnotatedSpan]:
        """Use edsnlp pipeline for assertion annotation."""
        import spacy
        from spacy.tokens import Span as SpacySpan

        doc = self._nlp.make_doc(text)  # type: ignore[union-attr]

        # We need to create spaCy Span objects for each input span.
        # edsnlp works on doc.ents or doc.spans — we use doc.spans["ruler"].
        spacy_spans: list[SpacySpan] = []
        for start_char, end_char, label in spans:
            span = doc.char_span(start_char, end_char, label=label or "ENTITY")
            if span is not None:
                spacy_spans.append(span)
            else:
                # char_span can fail if offsets don't align to tokens.
                # Use alignment_mode="expand" as fallback.
                span = doc.char_span(
                    start_char, end_char,
                    label=label or "ENTITY",
                    alignment_mode="expand",
                )
                if span is not None:
                    spacy_spans.append(span)

        doc.spans["ruler"] = spacy_spans

        # Run the assertion pipeline on the doc
        doc = self._nlp(doc)  # type: ignore[union-attr]

        # Build results — map back using character offsets
        annotated_spans_by_offset: dict[tuple[int, int], SpacySpan] = {}
        for sp in doc.spans.get("ruler", []):
            annotated_spans_by_offset[(sp.start_char, sp.end_char)] = sp

        results: list[AnnotatedSpan] = []
        for start_char, end_char, label in spans:
            sp = annotated_spans_by_offset.get((start_char, end_char))
            if sp is not None:
                results.append(AnnotatedSpan(
                    text=sp.text,
                    start=sp.start_char,
                    end=sp.end_char,
                    label=label,
                    is_negated=getattr(sp._, "negation", False),
                    is_hypothesis=getattr(sp._, "hypothesis", False),
                    is_history=getattr(sp._, "history", False),
                ))
            else:
                # Span was not aligned — fall back to regex for this span
                regex_result = self._annotate_regex(text, [(start_char, end_char, label)])
                results.extend(regex_result)

        return results

    def _annotate_regex(
        self,
        text: str,
        spans: list[tuple[int, int, str]],
    ) -> list[AnnotatedSpan]:
        """Regex-based fallback for assertion annotation."""
        results: list[AnnotatedSpan] = []
        for start, end, label in spans:
            span_text = text[start:end]
            is_neg = _has_pattern_near_span(text, start, end, _NEGATION_PATTERNS)
            is_hyp = _has_pattern_near_span(text, start, end, _HYPOTHESIS_PATTERNS, look_after=True)
            is_hist = _has_pattern_near_span(text, start, end, _HISTORY_PATTERNS, look_after=True)
            results.append(AnnotatedSpan(
                text=span_text,
                start=start,
                end=end,
                label=label,
                is_negated=is_neg,
                is_hypothesis=is_hyp,
                is_history=is_hist,
            ))
        return results

    def detect_negation(self, text: str, target: str) -> bool:
        """Convenience: check if *target* is negated anywhere in *text*.

        Searches for *target* in *text* (case-insensitive) and checks
        negation on the first match.

        Returns ``True`` if the target is negated, ``False`` otherwise
        (including when the target is not found).
        """
        match = re.search(re.escape(target), text, re.IGNORECASE)
        if match is None:
            return False
        results = self.annotate(text, [(match.start(), match.end(), target)])
        return results[0].is_negated if results else False
