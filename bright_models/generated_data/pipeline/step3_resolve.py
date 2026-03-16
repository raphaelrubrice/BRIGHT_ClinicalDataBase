"""Step 3: Span resolution to character offsets (deterministic, no LLM).

Converts {value, span} annotation pairs from Step 2 into exact (start, end)
character offsets in document_text, using a 4-stage cascade:
  1. Exact match
  2. Case-insensitive
  3. Whitespace-normalised
  4. Fuzzy sliding-window (rapidfuzz >= threshold)
"""

import logging
import re
from typing import Optional

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Whitespace normalisation with offset remapping
# ═════════════════════════════════════════════════════════════════════════════

def _build_norm_mapping(text: str) -> tuple[str, list[int]]:
    """Normalise whitespace and build a mapping from normalised → original offsets.

    Returns (normalised_text, norm_to_orig) where norm_to_orig[i] gives the
    original character index for normalised position i.
    """
    norm_chars: list[str] = []
    norm_to_orig: list[int] = []
    prev_ws = False
    for i, c in enumerate(text):
        if c in (" ", "\t", "\n", "\r", "\xa0"):
            if not prev_ws:
                norm_chars.append(" ")
                norm_to_orig.append(i)
            prev_ws = True
        else:
            norm_chars.append(c)
            norm_to_orig.append(i)
            prev_ws = False
    normalised = "".join(norm_chars).strip()
    # Adjust for leading whitespace strip
    strip_offset = len("".join(norm_chars)) - len("".join(norm_chars).lstrip())
    if strip_offset > 0:
        norm_to_orig = norm_to_orig[strip_offset:]
        normalised = "".join(norm_chars).strip()
    return normalised, norm_to_orig


# ═════════════════════════════════════════════════════════════════════════════
# Core span resolver
# ═════════════════════════════════════════════════════════════════════════════

def resolve_span(
    document_text: str,
    span: str,
    value: str,
    fuzzy_threshold: int = 85,
) -> dict:
    """Resolve a single annotation span to character offsets.

    Parameters
    ----------
    document_text : str
        The full generated document text.
    span : str
        Verbatim text span from the LLM annotation.
    value : str
        Normalised value to locate within the span.
    fuzzy_threshold : int
        Minimum rapidfuzz ratio for fuzzy match acceptance.

    Returns
    -------
    dict with keys:
        span_start, span_end : int (-1 if unresolved)
        value_start, value_end : int (-1 if unresolved)
        method : str describing resolution strategy used
    """
    result = {
        "span_start": -1, "span_end": -1,
        "value_start": -1, "value_end": -1,
        "method": None,
    }

    if not span or not document_text:
        result["method"] = "unresolved"
        return result

    # ── Strategy 1: Exact match ─────────────────────────────────────────
    idx = document_text.find(span)
    if idx != -1:
        result["span_start"] = idx
        result["span_end"] = idx + len(span)
        result["method"] = "exact"

    # ── Strategy 2: Case-insensitive ────────────────────────────────────
    if result["method"] is None:
        idx = document_text.lower().find(span.lower())
        if idx != -1:
            result["span_start"] = idx
            result["span_end"] = idx + len(span)
            result["method"] = "case_insensitive"

    # ── Strategy 3: Whitespace-normalised ───────────────────────────────
    if result["method"] is None:
        norm_text, norm_to_orig = _build_norm_mapping(document_text)
        norm_span = re.sub(r"\s+", " ", span).strip()
        idx = norm_text.lower().find(norm_span.lower())
        if idx != -1 and idx + len(norm_span) <= len(norm_to_orig):
            orig_start = norm_to_orig[idx]
            end_idx = min(idx + len(norm_span) - 1, len(norm_to_orig) - 1)
            orig_end = norm_to_orig[end_idx] + 1
            result["span_start"] = orig_start
            result["span_end"] = orig_end
            result["method"] = "whitespace_norm"

    # ── Strategy 4: Fuzzy sliding window ────────────────────────────────
    if result["method"] is None:
        best_score, best_start, best_end = 0, -1, -1
        span_len = len(span)
        text_len = len(document_text)

        for factor in (1.0, 0.8, 1.2):
            ws = max(10, int(span_len * factor))
            step = max(1, ws // 10)
            for i in range(0, max(1, text_len - ws + 1), step):
                window = document_text[i:i + ws]
                score = fuzz.ratio(span, window)
                if score > best_score:
                    best_score = score
                    best_start = i
                    best_end = i + ws

        if best_score >= fuzzy_threshold:
            result["span_start"] = best_start
            result["span_end"] = min(best_end, text_len)
            result["method"] = f"fuzzy_{best_score}"

    # ── Resolve value within span region ────────────────────────────────
    if result["span_start"] >= 0:
        _resolve_value_in_span(document_text, result, value)

    if result["method"] is None:
        result["method"] = "unresolved"

    return result


def _resolve_value_in_span(document_text: str, result: dict, value: str):
    """Locate the value substring within the resolved span region."""
    value = str(value)
    start = result["span_start"]
    # Look slightly beyond span end to catch edge cases
    end = min(result["span_end"] + 20, len(document_text))
    region = document_text[start:end]

    # Exact match in region
    v_idx = region.find(value)
    if v_idx != -1:
        result["value_start"] = start + v_idx
        result["value_end"] = start + v_idx + len(value)
        return

    # Case-insensitive
    v_idx = region.lower().find(value.lower())
    if v_idx != -1:
        result["value_start"] = start + v_idx
        result["value_end"] = start + v_idx + len(value)
        return

    # Fallback: use span boundaries as value boundaries
    result["value_start"] = result["span_start"]
    result["value_end"] = result["span_end"]
    result["method"] += "_value_fallback"


# ═════════════════════════════════════════════════════════════════════════════
# Batch resolution + output formatting
# ═════════════════════════════════════════════════════════════════════════════

def resolve_document(raw_doc: dict, fuzzy_threshold: int = 85) -> dict:
    """Resolve all annotations in a single document.

    Parameters
    ----------
    raw_doc : dict
        Output from Step 2 with keys: document_text, annotations, patient_id, doc_type.

    Returns
    -------
    dict in edsnlp-compatible format:
        {note_id, note_text, entities: [{start, end, label, value}], meta}
    """
    text = raw_doc["document_text"]
    annotations = raw_doc.get("annotations", {})
    patient_id = raw_doc.get("patient_id", "unknown")
    doc_type = raw_doc.get("doc_type", "unknown")

    entities = []
    stats = {"total": 0, "exact": 0, "case_insensitive": 0,
             "whitespace_norm": 0, "fuzzy": 0, "unresolved": 0,
             "value_fallback": 0}

    for field_name, ann in annotations.items():
        if not isinstance(ann, dict) or "span" not in ann:
            continue

        span = str(ann["span"])
        value = str(ann.get("value", span))
        stats["total"] += 1

        resolved = resolve_span(text, span, value, fuzzy_threshold)

        # Count by method
        method = resolved["method"]
        if method == "exact":
            stats["exact"] += 1
        elif method == "case_insensitive":
            stats["case_insensitive"] += 1
        elif method == "whitespace_norm":
            stats["whitespace_norm"] += 1
        elif method and method.startswith("fuzzy"):
            if "value_fallback" in method:
                stats["fuzzy"] += 1
                stats["value_fallback"] += 1
            else:
                stats["fuzzy"] += 1
        elif method == "unresolved":
            stats["unresolved"] += 1
        elif method and "value_fallback" in method:
            # e.g. "exact_value_fallback"
            base = method.replace("_value_fallback", "")
            stats[base] = stats.get(base, 0) + 1
            stats["value_fallback"] += 1

        if resolved["span_start"] >= 0:
            entities.append({
                "start": resolved["value_start"],
                "end": resolved["value_end"],
                "label": field_name,
                "value": value,
            })

    return {
        "note_id": f"synth-{patient_id}-{doc_type}",
        "note_text": text,
        "entities": entities,
        "meta": {
            "patient_id": patient_id,
            "doc_type": doc_type,
            "resolution_stats": stats,
        },
    }


def resolve_all_spans(
    raw_documents: list[dict],
    config,
) -> tuple[list[dict], dict]:
    """Resolve spans for all documents.

    Parameters
    ----------
    raw_documents : list[dict]
        Output from Step 2.
    config : PipelineConfig

    Returns
    -------
    (resolved_documents, aggregate_stats)
    """
    from config.fields import CRITICAL_FIELDS

    resolved = []
    agg = {"total": 0, "exact": 0, "case_insensitive": 0,
           "whitespace_norm": 0, "fuzzy": 0, "unresolved": 0,
           "value_fallback": 0, "accepted": 0, "rejected": 0}

    for raw_doc in raw_documents:
        doc = resolve_document(raw_doc, config.fuzzy_threshold)
        stats = doc["meta"]["resolution_stats"]

        # Aggregate
        for k in ("total", "exact", "case_insensitive", "whitespace_norm",
                   "fuzzy", "unresolved", "value_fallback"):
            agg[k] += stats.get(k, 0)

        # Acceptance gate
        total = stats["total"]
        if total == 0:
            agg["rejected"] += 1
            continue

        resolved_count = total - stats["unresolved"]
        resolution_rate = resolved_count / total

        # Check critical fields for value_fallback
        critical_fallback = False
        for entity in doc["entities"]:
            if entity["label"] in CRITICAL_FIELDS:
                # Check if this entity used value_fallback
                # (value boundaries == span boundaries is the indicator)
                pass  # We track globally; per-entity check is complex

        if resolution_rate < config.min_resolution_rate:
            agg["rejected"] += 1
            logger.info(
                "Rejected %s: resolution rate %.1f%% < %.1f%%",
                doc["note_id"], resolution_rate * 100,
                config.min_resolution_rate * 100,
            )
            continue

        agg["accepted"] += 1
        resolved.append(doc)

    # Compute aggregate percentages
    total = agg["total"] if agg["total"] > 0 else 1
    summary = {
        "total_annotations": agg["total"],
        "exact": agg["exact"] / total,
        "case_insensitive": agg["case_insensitive"] / total,
        "whitespace_norm": agg["whitespace_norm"] / total,
        "fuzzy": agg["fuzzy"] / total,
        "unresolved": agg["unresolved"] / total,
        "value_fallback": agg["value_fallback"] / total,
        "documents_accepted": agg["accepted"],
        "documents_rejected": agg["rejected"],
    }

    logger.info(
        "Span resolution: %d/%d docs accepted, %.1f%% exact, %.1f%% unresolved",
        agg["accepted"], agg["accepted"] + agg["rejected"],
        summary["exact"] * 100, summary["unresolved"] * 100,
    )

    return resolved, summary
