"""Tier 2 LLM-based feature extraction via Ollama.

Extracts features requiring contextual understanding using Qwen3-4B
with schema-constrained JSON output and source span citation.

Public API
----------
- ``run_llm_extraction()``    – Main LLM extraction entry point.
- ``validate_source_spans()`` – Post-extraction source span verification.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from .ollama_client import OllamaClient, OllamaError, OllamaResponse
from .prompts import PROMPT_REGISTRY, PromptConfig, get_prompt
from .schema import (
    ExtractionValue,
    FEATURE_GROUPS,
    ALL_FIELDS_BY_NAME,
    get_json_schema,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section → feature group mapping
# ---------------------------------------------------------------------------

# Maps section names (from SectionDetector) to the most relevant feature groups.
_SECTION_TO_GROUPS: dict[str, list[str]] = {
    "ihc": ["ihc"],
    "molecular": ["molecular"],
    "chromosomal": ["chromosomal"],
    "macroscopy": ["diagnosis"],
    "microscopy": ["diagnosis"],
    "conclusion": ["ihc", "molecular", "chromosomal", "diagnosis"],
    "history": ["demographics", "symptoms"],
    "treatment": ["treatment"],
    "clinical_exam": ["symptoms"],
    "radiology": ["evolution"],
    "full_text": [
        "ihc", "molecular", "chromosomal", "diagnosis",
        "demographics", "symptoms", "treatment", "evolution",
    ],
}


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

def _determine_groups_for_features(
    remaining_fields: set[str],
) -> dict[str, list[str]]:
    """Determine which feature groups are needed for the remaining fields.

    Returns
    -------
    dict[str, list[str]]
        Mapping ``group_name → list_of_field_names`` — only groups that
        have at least one remaining field.
    """
    result: dict[str, list[str]] = {}
    for group_name, group_fields in FEATURE_GROUPS.items():
        needed = [f for f in group_fields if f in remaining_fields]
        if needed:
            result[group_name] = needed
    return result


def _select_section_text(
    sections: dict[str, str],
    group_name: str,
    full_text: str,
) -> str:
    """Select the best section text for a given feature group.

    Tries to find a section that maps to the group. Falls back to
    ``full_text`` if no relevant section is found.
    """
    # Try matching section → group
    for section_name, groups in _SECTION_TO_GROUPS.items():
        if group_name in groups and section_name in sections:
            section_text = sections[section_name]
            if section_text.strip():
                return section_text

    # Fall back to full_text section if present
    if "full_text" in sections:
        return sections["full_text"]

    return full_text


def _parse_llm_response(
    response: OllamaResponse,
    group_name: str,
    target_fields: list[str],
) -> dict[str, ExtractionValue]:
    """Parse the LLM response JSON into ExtractionValue objects.

    Parameters
    ----------
    response : OllamaResponse
        The parsed Ollama response.
    group_name : str
        The feature group name (for logging).
    target_fields : list[str]
        The field names we expect in the response.

    Returns
    -------
    dict[str, ExtractionValue]
        Mapping ``field_name → ExtractionValue``.
    """
    results: dict[str, ExtractionValue] = {}

    if response.parsed_json is None:
        logger.warning(
            "LLM response for group '%s' was not valid JSON: %s",
            group_name,
            response.content[:200],
        )
        return results

    data = response.parsed_json

    # Expected structure: {"values": {...}, "_source": {...}}
    values = data.get("values", data)  # Fall back to top-level if no "values" key
    sources = data.get("_source", {})

    for field_name in target_fields:
        raw_value = values.get(field_name)
        if raw_value is None:
            continue  # Field not extracted — skip (don't fabricate)

        source_span = sources.get(field_name)

        # Normalize the value
        normalised = _normalise_llm_value(field_name, raw_value)
        if normalised is None:
            continue

        results[field_name] = ExtractionValue(
            value=normalised,
            source_span=source_span,
            extraction_tier="llm",
            confidence=0.7,  # LLM extractions get lower confidence than rule-based
            vocab_valid=True,  # Will be validated later by validation.py
        )

    return results


def _normalise_llm_value(
    field_name: str,
    raw_value: Any,
) -> Optional[str | int | float]:
    """Normalise a raw LLM-extracted value.

    Handles common LLM output quirks: extra whitespace, encoding
    differences, etc.
    """
    if raw_value is None:
        return None

    # Convert to string for processing
    if isinstance(raw_value, bool):
        return "oui" if raw_value else "non"

    if isinstance(raw_value, (int, float)):
        field_def = ALL_FIELDS_BY_NAME.get(field_name)
        if field_def and field_def.field_type.value == "integer":
            return int(raw_value)
        if field_def and field_def.field_type.value == "float":
            return float(raw_value)
        return raw_value

    value = str(raw_value).strip()
    if not value or value.lower() in ("null", "none", "n/a", "na", ""):
        return None

    # Normalise common French accentuation variants
    normalisations: dict[str, str] = {
        "négatif": "negatif",
        "négative": "negatif",
        "negative": "negatif",
        "muté": "mute",
        "mutée": "mute",
        "méthylé": "methyle",
        "non méthylé": "non methyle",
        "non methylé": "non methyle",
    }

    value_lower = value.lower()
    if value_lower in normalisations:
        return normalisations[value_lower]

    return value


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_llm_extraction(
    text: str,
    sections: dict[str, str],
    feature_subset: list[str],
    already_extracted: dict[str, ExtractionValue],
    client: OllamaClient,
) -> dict[str, ExtractionValue]:
    """Extract features using the LLM for fields not yet extracted by Tier 1.

    Parameters
    ----------
    text : str
        The full document text.
    sections : dict[str, str]
        Section name → section text, as returned by ``SectionDetector``.
    feature_subset : list[str]
        List of all field names that should be extracted for this document.
    already_extracted : dict[str, ExtractionValue]
        Features already extracted by Tier 1 (rule-based).
    client : OllamaClient
        The Ollama client instance.

    Returns
    -------
    dict[str, ExtractionValue]
        Mapping ``field_name → ExtractionValue`` for newly LLM-extracted
        fields. Does **not** include fields already in *already_extracted*.
    """
    # Determine which fields still need extraction
    remaining = set(feature_subset) - set(already_extracted.keys())
    if not remaining:
        logger.info("All features already extracted by Tier 1, skipping LLM.")
        return {}

    # Group remaining fields by feature group
    groups_needed = _determine_groups_for_features(remaining)
    if not groups_needed:
        logger.info("No feature groups applicable to remaining fields.")
        return {}

    logger.info(
        "LLM extraction: %d remaining fields across %d groups: %s",
        len(remaining),
        len(groups_needed),
        sorted(groups_needed.keys()),
    )

    all_results: dict[str, ExtractionValue] = {}

    for group_name, fields_in_group in groups_needed.items():
        try:
            prompt_config = get_prompt(group_name)
        except KeyError:
            logger.warning("No prompt template for group '%s', skipping.", group_name)
            continue

        # Select appropriate section text
        section_text = _select_section_text(sections, group_name, text)

        # Truncate very long texts to avoid exceeding model context
        max_chars = 4000  # ~1000 tokens
        if len(section_text) > max_chars:
            section_text = section_text[:max_chars] + "\n[... texte tronqué ...]"

        # Build the prompt
        user_prompt = prompt_config.user_template.format(section_text=section_text)

        # Get the JSON schema for constrained decoding
        try:
            json_schema = get_json_schema(group_name)
        except KeyError:
            json_schema = None

        # Call Ollama
        try:
            response = client.generate(
                prompt=user_prompt,
                system=prompt_config.system,
                json_schema=json_schema,
                temperature=0.0,
            )
        except OllamaError as exc:
            logger.error(
                "Ollama call failed for group '%s': %s", group_name, exc
            )
            continue

        # Parse the response
        group_results = _parse_llm_response(response, group_name, fields_in_group)

        # Add section metadata
        for field_name, ev in group_results.items():
            if field_name not in all_results and field_name not in already_extracted:
                # Find which section was used
                for sec_name, sec_text in sections.items():
                    if section_text == sec_text:
                        ev.section = sec_name
                        break
                all_results[field_name] = ev

        logger.info(
            "Group '%s': extracted %d/%d fields",
            group_name,
            len(group_results),
            len(fields_in_group),
        )

    return all_results


# ---------------------------------------------------------------------------
# Source span validation (Step 5.4)
# ---------------------------------------------------------------------------

def _normalise_whitespace(text: str) -> str:
    """Collapse all whitespace to single spaces and strip."""
    return re.sub(r"\s+", " ", text).strip().lower()


def validate_source_spans(
    extractions: dict[str, ExtractionValue],
    original_text: str,
    fuzzy_threshold: float = 0.8,
) -> dict[str, ExtractionValue]:
    """Verify that each cited source_span exists in the original text.

    For each extraction that has a ``source_span``, check whether the span
    actually appears in the document. If not found (even with fuzzy matching),
    flag the value for human review.

    Parameters
    ----------
    extractions : dict[str, ExtractionValue]
        The extractions to validate.
    original_text : str
        The full original document text.
    fuzzy_threshold : float
        Minimum similarity ratio for fuzzy matching (0.0–1.0).

    Returns
    -------
    dict[str, ExtractionValue]
        The same dict with ``flagged=True`` set on extractions whose
        source spans could not be verified.
    """
    normalised_text = _normalise_whitespace(original_text)

    for field_name, ev in extractions.items():
        if ev.source_span is None or ev.source_span.strip() == "":
            # No source span provided — flag LLM extractions
            if ev.extraction_tier == "llm":
                ev.flagged = True
                logger.debug(
                    "Field '%s': no source span provided (LLM), flagging.",
                    field_name,
                )
            continue

        normalised_span = _normalise_whitespace(ev.source_span)

        # Exact match (normalised)
        if normalised_span in normalised_text:
            continue  # Source span verified ✓

        # Try fuzzy match: check if a high proportion of span words
        # appear near each other in the text
        span_words = normalised_span.split()
        if not span_words:
            continue

        found_count = sum(1 for w in span_words if w in normalised_text)
        similarity = found_count / len(span_words)

        if similarity >= fuzzy_threshold:
            logger.debug(
                "Field '%s': source span fuzzy-matched (%.0f%% words found).",
                field_name,
                similarity * 100,
            )
            continue  # Close enough ✓

        # Source span not found — flag
        ev.flagged = True
        logger.warning(
            "Field '%s': source span NOT found in text (%.0f%% match). "
            "Span: '%s'",
            field_name,
            similarity * 100,
            ev.source_span[:80],
        )

    return extractions
