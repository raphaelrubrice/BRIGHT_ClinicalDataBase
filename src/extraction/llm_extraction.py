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
    FieldType,
    FEATURE_GROUPS,
    ALL_FIELDS_BY_NAME,
    get_json_schema,
)

logger = logging.getLogger(__name__)

# Reject pseudo-token values leaked from pseudonymisation
_PSEUDO_TOKEN_RE = re.compile(
    r'\[?(NOM|PRENOM|TEL|MAIL|ADDRESS|HOPITAL|VILLE|IPP|DATE|ZIP|SSID|NDA)_[A-Fa-f0-9]+\]?'
)


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
    "equipe_soignante": ["demographics"],
    "demographics": ["demographics"],
    "summary": ["demographics", "evolution", "treatment"],
    "rcp_decision": ["treatment", "evolution"],
    "full_text": [
        "ihc", "molecular", "chromosomal", "diagnosis",
        "demographics", "symptoms", "treatment", "evolution",
    ],
}


# Fields that are fully rule-based and should NEVER be sent to the LLM.
# This provides an explicit safety net beyond the "already_extracted" filter,
# ensuring these fields are never included in an LLM prompt even if the rule
# extractor returned None (we prefer None over a hallucinated value).

_LLM_FIELDS: set[str] = {
    "diag_integre",
    "localisation_chir",
    "localisation_radiotherapie",
    "infos_deces",
}

_RULE_ONLY_FIELDS: set[str] = set(ALL_FIELDS_BY_NAME.keys()) - _LLM_FIELDS


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


_GROUP_KEYWORDS: dict[str, list[str]] = {
    "ihc": ["ihc", "immunohistochimie", "idh1", "p53", "atrx", "ki67", "gfap", "olig2", "h3k27"],
    "molecular": ["moléculaire", "moleculaire", "mutation", "idh", "tert", "mgmt", "braf", "cdkn2a"],
    "chromosomal": ["cgh", "chromosom", "1p", "19q", "amplification", "fusion", "délétion"],
    "diagnosis": ["diagnostic", "histologique", "glioblastome", "astrocytome", "grade", "oms", "nécrose", "pec", "mitose"],
    "demographics": ["référent", "referent", "équipe", "neurochirur", "neuro-onco", "radiothérap", "sexe", "profession", "né le", "née le", "naissance"],
    "symptoms": ["symptôme", "symptome", "épilepsie", "epilepsie", "céphalée", "cephalee", "déficit", "deficit", "karnofsky", "ik "],
    "treatment": ["chimiothérapie", "chimiotherapie", "temozolomide", "tmz", "radiothérapie", "radiotherapie", "chirurgie", "optune", "corticoïde", "gy"],
    "evolution": ["évolution", "evolution", "progression", "récidive", "recidive", "décès", "deces", "dernière nouvelle", "latéralité"],
}


def _select_section_text(
    sections: dict[str, str],
    group_name: str,
    full_text: str,
) -> str:
    """Select the best section text for a given feature group.

    Tries to find a section that maps to the group. Falls back to
    ``full_text`` with smart paragraph selection if no relevant section
    is found.
    """
    # Try matching section → group
    for section_name, groups in _SECTION_TO_GROUPS.items():
        if group_name in groups and section_name in sections:
            section_text = sections[section_name]
            if section_text.strip():
                return section_text

    # Fall back to full_text with smart paragraph selection
    raw = sections.get("full_text", full_text)
    return _select_relevant_paragraphs(raw, group_name)


def _select_relevant_paragraphs(text: str, group_name: str, max_chars: int = 8000) -> str:
    """Select the most relevant paragraphs for a feature group from full text.

    Splits text into paragraphs, scores each by keyword relevance to the
    target group, and returns the top paragraphs (in document order) up
    to *max_chars*.
    """
    if len(text) <= max_chars:
        return text

    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    if len(paragraphs) <= 1:
        paragraphs = text.split('\n')

    keywords = _GROUP_KEYWORDS.get(group_name, [])
    if not keywords:
        return text[:max_chars]

    # Score each paragraph
    scored: list[tuple[int, int, str]] = []  # (score, original_index, text)
    for i, para in enumerate(paragraphs):
        para_lower = para.lower()
        score = sum(1 for kw in keywords if kw in para_lower)
        scored.append((score, i, para))

    # Sort by score descending, then pick top ones respecting max_chars
    scored.sort(key=lambda x: -x[0])

    selected_indices: list[int] = []
    total_chars = 0
    for score, idx, para in scored:
        if score == 0 and selected_indices:
            break  # Don't add irrelevant paragraphs if we have some relevant ones
        if total_chars + len(para) > max_chars:
            if not selected_indices:
                # Must include at least one paragraph
                selected_indices.append(idx)
            break
        selected_indices.append(idx)
        total_chars += len(para)

    # Return in document order
    selected_indices.sort()
    return "\n\n".join(paragraphs[i] for i in selected_indices)


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

        # Reject pseudo-token values
        if isinstance(normalised, str) and _PSEUDO_TOKEN_RE.search(normalised):
            logger.debug("Field '%s': rejected pseudo-token value '%s'", field_name, normalised)
            continue

        # Reject unreasonable dates
        field_def = ALL_FIELDS_BY_NAME.get(field_name)
        if field_def and field_def.field_type == FieldType.DATE and isinstance(normalised, str):
            if not _is_reasonable_date(normalised):
                logger.debug("Field '%s': rejected unreasonable date '%s'", field_name, normalised)
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


def _is_reasonable_date(date_str: str) -> bool:
    """Check if a date string (DD/MM/YYYY) is within reasonable bounds."""
    import datetime
    parts = date_str.split("/")
    if len(parts) != 3:
        return True  # Not a parseable date, let it through
    try:
        year = int(parts[2])
        current_year = datetime.date.today().year
        return 1900 <= year <= current_year + 1
    except (ValueError, IndexError):
        return True


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
    remaining = set(feature_subset) - set(already_extracted.keys()) - _RULE_ONLY_FIELDS
    if not remaining:
        logger.info("All features already extracted by Tier 1 or rule-only, skipping LLM.")
        return {}

    all_results: dict[str, ExtractionValue] = {}

    if "diag_integre" in remaining:
        logger.info("Extracting diag_integre via constrained LLM call.")
        diag_res = extract_diag_integre(text, sections, already_extracted, client)
        all_results.update(diag_res)
        remaining.remove("diag_integre")

    if not remaining:
        return all_results

    # Group remaining fields by feature group
    groups_needed = _determine_groups_for_features(remaining)
    if not groups_needed:
        logger.info("No feature groups applicable to remaining fields.")
        return all_results

    logger.info(
        "LLM extraction: %d remaining fields across %d groups: %s",
        len(remaining),
        len(groups_needed),
        sorted(groups_needed.keys()),
    )

    for group_name, fields_in_group in groups_needed.items():
        try:
            prompt_config = get_prompt(group_name)
        except KeyError:
            logger.warning("No prompt template for group '%s', skipping.", group_name)
            continue

        # Select appropriate section text
        section_text = _select_section_text(sections, group_name, text)

        # Truncate very long texts to avoid exceeding model context
        max_chars = 8000  # ~2000 tokens
        if len(section_text) > max_chars:
            section_text = section_text[:max_chars] + "\n[... texte tronqué ...]"

        # Build the prompt
        user_prompt = prompt_config.user_template.replace("{section_text}", section_text)

        # Get the JSON schema for constrained decoding
        try:
            json_schema = get_json_schema(group_name, subset=fields_in_group)
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
# Phase D2: Constrained diag_integre extraction
# ---------------------------------------------------------------------------

_DIAG_INTEGRE_SYSTEM = """\
Tu es un neuropathologiste. À partir des résultats IHC et moléculaires \
fournis, formule le diagnostic intégré selon la classification OMS 2021 \
des tumeurs du SNC. Réponds UNIQUEMENT avec le diagnostic intégré. \
Si les données sont insuffisantes, retourne null. /no_think\
"""

_DIAG_INTEGRE_PROMPT = """\
Voici les résultats déjà extraits pour ce patient :

{extracted_context}

En te basant sur ces résultats et le texte source ci-dessous, \
formule le diagnostic intégré selon la classification OMS 2021.

Format de réponse attendu :
{{"values": {{"diag_integre": "<diagnostic intégré>"}}, \
"_source": {{"diag_integre": "<passage du texte source>"}}}}

Si les données sont insuffisantes pour formuler un diagnostic intégré, \
retourne : {{"values": {{"diag_integre": null}}, "_source": {{}}}}

### Texte source :
{section_text}
"""

# Fields whose pre-extracted values provide context for diag_integre
_DIAG_INTEGRE_CONTEXT_FIELDS = [
    "diag_histologique", "grade", "classification_oms",
    "mol_idh1", "mol_idh2", "mol_tert", "mol_CDKN2A",
    "mol_h3f3a", "mol_hist1h3b", "mol_braf", "mol_mgmt",
    "mol_atrx", "mol_egfr_mut", "mol_pten",
    "ch1p", "ch19q", "ch10q", "ch7p",
    "ihc_idh1", "ihc_p53", "ihc_atrx",
    "ihc_hist_h3k27m", "ihc_hist_h3k27me3",
    "ampli_egfr", "ampli_cdk4", "ampli_mdm2",
]


def _build_diag_integre_context(
    already_extracted: dict[str, ExtractionValue],
) -> str:
    """Build a structured context string from pre-extracted values."""
    lines = []
    for field in _DIAG_INTEGRE_CONTEXT_FIELDS:
        ev = already_extracted.get(field)
        if ev is not None and ev.value is not None:
            lines.append(f"- {field}: {ev.value}")
    return "\n".join(lines) if lines else "(aucun résultat pré-extrait)"


def extract_diag_integre(
    text: str,
    sections: dict[str, str],
    already_extracted: dict[str, ExtractionValue],
    client: "OllamaClient",
) -> dict[str, ExtractionValue]:
    """Extract diag_integre using a constrained LLM call.

    Uses pre-extracted IHC/molecular values as structured context so the
    LLM only needs to assemble the WHO 2021 integrated diagnosis label.
    Returns at most one field: ``diag_integre``.
    """
    context = _build_diag_integre_context(already_extracted)

    # Select best section text
    section_text = _select_section_text(sections, "diagnosis", text)
    max_chars = 4000
    if len(section_text) > max_chars:
        section_text = section_text[:max_chars] + "\n[... tronqué ...]"

    user_prompt = (
        _DIAG_INTEGRE_PROMPT
        .replace("{extracted_context}", context)
        .replace("{section_text}", section_text)
    )

    try:
        response = client.generate(
            prompt=user_prompt,
            system=_DIAG_INTEGRE_SYSTEM,
            json_schema=None,
            temperature=0.0,
        )
    except Exception as exc:
        logger.error("diag_integre LLM call failed: %s", exc)
        return {
            "diag_integre": ExtractionValue(
                value=None,
                source_span=None,
                extraction_tier="llm",
                confidence=0.0,
                flagged=True,
            )
        }

    # Parse response
    parsed = response.parsed_json
    if not parsed or "values" not in parsed:
        logger.warning("diag_integre: malformed LLM response")
        return {
            "diag_integre": ExtractionValue(
                value=None,
                source_span=None,
                extraction_tier="llm",
                confidence=0.0,
                flagged=True,
            )
        }

    raw_value = parsed["values"].get("diag_integre")
    if raw_value is None:
        return {}

    # Validate: should be a non-empty string ≤ 200 chars
    if not isinstance(raw_value, str) or not raw_value.strip() or len(raw_value) > 200:
        return {
            "diag_integre": ExtractionValue(
                value=str(raw_value)[:200] if raw_value else None,
                source_span=parsed.get("_source", {}).get("diag_integre"),
                extraction_tier="llm",
                confidence=0.3,
                flagged=True,
            )
        }

    source_span = None
    if "_source" in parsed and isinstance(parsed["_source"], dict):
        source_span = parsed["_source"].get("diag_integre")

    return {
        "diag_integre": ExtractionValue(
            value=raw_value.strip(),
            source_span=source_span,
            extraction_tier="llm",
            confidence=0.8,
            vocab_valid=True,
        )
    }


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
