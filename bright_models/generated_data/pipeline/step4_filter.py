"""Step 4: Quality filtering cascade.

Applies 6 filters sequentially to resolved documents:
  1. Resolution rate
  2. Clinical coherence (WHO 2021)
  3. Document length
  4. Profile-to-annotation match
  5. Near-duplicate detection (MinHash LSH)
  6. Language quality (French, no prompt leakage)
"""

import logging
import re
from collections import Counter, defaultdict

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Individual filters
# ═════════════════════════════════════════════════════════════════════════════

def filter_resolution_rate(doc: dict, config) -> str:
    """Check annotation resolution rate. Returns 'pass', 'review', or 'reject'."""
    stats = doc.get("meta", {}).get("resolution_stats", {})
    total = stats.get("total", 0)
    if total == 0:
        return "reject"
    unresolved = stats.get("unresolved", 0)
    rate = (total - unresolved) / total
    if rate < config.reject_resolution_below:
        return "reject"
    if rate < config.review_resolution_below:
        return "review"
    return "pass"


def filter_clinical_coherence(doc: dict) -> str:
    """Check WHO 2021 coherence on extracted annotations."""
    from pipeline.who_validation import validate_document_coherence

    # Build annotations dict from entities
    annotations = {}
    for ent in doc.get("entities", []):
        annotations[ent["label"]] = ent.get("value", "")

    errors = validate_document_coherence(annotations)
    if errors:
        logger.debug("WHO violations in %s: %s", doc.get("note_id"), errors)
        return "reject"
    return "pass"


def filter_length(doc: dict, config) -> str:
    """Check document length is within realistic bounds."""
    doc_type = doc.get("meta", {}).get("doc_type", "consultation")
    limits = config.length_limits.get(doc_type, {"min": 200, "max": 6000})
    text_len = len(doc.get("note_text", ""))
    if text_len < limits["min"] or text_len > limits["max"]:
        return "review"
    return "pass"


def filter_profile_match(
    doc: dict,
    profile: dict,
    config,
) -> str:
    """Compare annotation values against original profile fields."""
    entities_by_label = {}
    for ent in doc.get("entities", []):
        entities_by_label[ent["label"]] = str(ent.get("value", ""))

    # Only check fields present in both profile and annotations
    checkable = []
    for k, v in profile.items():
        if v is None or k in ("patient_id", "document_types"):
            continue
        if k in entities_by_label:
            checkable.append(k)

    if not checkable:
        return "pass"

    matches = 0
    for field in checkable:
        profile_val = str(profile[field]).lower().strip()
        annot_val = entities_by_label[field].lower().strip()
        if fuzz.ratio(profile_val, annot_val) >= 80:
            matches += 1

    rate = matches / len(checkable)
    if rate < config.profile_match_reject:
        return "reject"
    if rate < config.profile_match_review:
        return "review"
    return "pass"


def filter_deduplication(
    documents: list[dict],
    similarity_threshold: float = 0.85,
) -> list[dict]:
    """Remove near-duplicates using MinHash LSH.

    Returns the deduplicated list (keeps the document with higher resolution rate
    from each duplicate cluster).
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        logger.warning("datasketch not installed, skipping deduplication")
        return documents

    # Build MinHash for each document
    def _text_shingles(text: str, k: int = 5) -> set:
        words = text.lower().split()
        return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}

    lsh = MinHashLSH(threshold=similarity_threshold, num_perm=128)
    minhashes = {}

    for i, doc in enumerate(documents):
        m = MinHash(num_perm=128)
        for shingle in _text_shingles(doc.get("note_text", "")):
            m.update(shingle.encode("utf-8"))
        minhashes[i] = m
        try:
            lsh.insert(str(i), m)
        except ValueError:
            pass  # duplicate key, already inserted

    # Find clusters and keep best per cluster
    seen = set()
    keep = []
    for i, doc in enumerate(documents):
        if i in seen:
            continue
        # Query for similar documents
        similar = lsh.query(minhashes[i])
        cluster_indices = [int(s) for s in similar if int(s) not in seen]

        if not cluster_indices:
            keep.append(doc)
            seen.add(i)
            continue

        # Keep the one with highest resolution rate
        best_idx = max(
            cluster_indices,
            key=lambda idx: _resolution_rate(documents[idx]),
        )
        keep.append(documents[best_idx])
        seen.update(cluster_indices)

    removed = len(documents) - len(keep)
    if removed > 0:
        logger.info("Deduplication removed %d documents", removed)
    return keep


def _resolution_rate(doc: dict) -> float:
    stats = doc.get("meta", {}).get("resolution_stats", {})
    total = stats.get("total", 1)
    unresolved = stats.get("unresolved", 0)
    return (total - unresolved) / max(total, 1)


# French medical vocabulary for language quality check
_FRENCH_MEDICAL_TERMS = {
    "patient", "patiente", "tumeur", "examen", "traitement", "diagnostic",
    "histologique", "chirurgie", "biopsie", "imagerie", "irm", "scanner",
    "consultation", "service", "hôpital", "clinique", "médecin",
    "neurologie", "oncologie", "radiothérapie", "chimiothérapie",
    "glioblastome", "astrocytome", "oligodendrogliome", "gliome",
    "immunohistochimie", "moléculaire", "anatomopathologie",
}

_ENGLISH_FUNCTION_WORDS = {
    "the", "is", "are", "was", "were", "have", "has", "been", "being",
    "will", "would", "could", "should", "this", "that", "these", "those",
    "with", "from", "they", "their", "which", "there", "about", "into",
}


def filter_language_quality(doc: dict, config) -> str:
    """Check language quality: French, medical terms, no prompt leakage."""
    text = doc.get("note_text", "")
    text_lower = text.lower()

    # Check for prompt leakage
    for term in config.prompt_leakage_terms:
        if term.lower() in text_lower:
            return "reject"

    # Check for French medical terms (at least 5)
    words = set(re.findall(r"\b\w+\b", text_lower))
    medical_count = len(words & _FRENCH_MEDICAL_TERMS)
    if medical_count < 5:
        return "reject"

    # Check for excessive English (> 10% English function words)
    all_words = re.findall(r"\b\w+\b", text_lower)
    if all_words:
        english_count = sum(1 for w in all_words if w in _ENGLISH_FUNCTION_WORDS)
        if english_count / len(all_words) > 0.10:
            return "reject"

    return "pass"


# ═════════════════════════════════════════════════════════════════════════════
# Main filter pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_filter_pipeline(
    resolved_documents: list[dict],
    config,
    profiles_lookup: dict[str, dict],
) -> tuple[list[dict], dict]:
    """Apply all quality filters.

    Parameters
    ----------
    resolved_documents : list[dict]
        Output from Step 3 in edsnlp format.
    config : PipelineConfig
    profiles_lookup : dict
        patient_id -> profile dict for profile-match checking.

    Returns
    -------
    (passing_documents, report)
    """
    rejection_reasons: Counter = Counter()
    review_reasons: Counter = Counter()

    # Filters 1-4, 6: per-document filters
    candidates = []
    for doc in resolved_documents:
        rejected = False
        pid = doc.get("meta", {}).get("patient_id", "")

        # Filter 1: Resolution rate
        result = filter_resolution_rate(doc, config)
        if result == "reject":
            rejection_reasons["resolution_rate"] += 1
            rejected = True
        elif result == "review":
            review_reasons["resolution_rate"] += 1

        # Filter 2: Clinical coherence
        if not rejected:
            result = filter_clinical_coherence(doc)
            if result == "reject":
                rejection_reasons["clinical_coherence"] += 1
                rejected = True

        # Filter 3: Length
        if not rejected:
            result = filter_length(doc, config)
            if result == "reject":
                rejection_reasons["length"] += 1
                rejected = True
            elif result == "review":
                review_reasons["length"] += 1

        # Filter 4: Profile match
        if not rejected and pid in profiles_lookup:
            result = filter_profile_match(doc, profiles_lookup[pid], config)
            if result == "reject":
                rejection_reasons["profile_match"] += 1
                rejected = True
            elif result == "review":
                review_reasons["profile_match"] += 1

        # Filter 6: Language quality
        if not rejected:
            result = filter_language_quality(doc, config)
            if result == "reject":
                rejection_reasons["language_quality"] += 1
                rejected = True

        if not rejected:
            candidates.append(doc)

    # Filter 5: Deduplication (operates on the full candidate list)
    final = filter_deduplication(candidates, config.dedup_similarity)
    dedup_removed = len(candidates) - len(final)
    if dedup_removed > 0:
        rejection_reasons["deduplication"] = dedup_removed

    report = {
        "input_count": len(resolved_documents),
        "output_count": len(final),
        "rejection_reasons": dict(rejection_reasons),
        "review_flags": dict(review_reasons),
        "pass_rate": len(final) / max(len(resolved_documents), 1),
    }

    logger.info(
        "Filter pipeline: %d → %d documents (%.1f%% pass rate)",
        report["input_count"], report["output_count"],
        report["pass_rate"] * 100,
    )
    logger.info("Rejection breakdown: %s", dict(rejection_reasons))

    return final, report
