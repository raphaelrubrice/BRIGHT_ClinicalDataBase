"""Controlled vocabulary enforcement and value validation.

Checks extracted values against controlled vocabularies, normalizes
common variants, and flags out-of-vocabulary values for human review.

Public API
----------
- ``validate_extraction()`` – Main validation entry point.
- ``normalise_value()``     – Normalise a single value for a given field.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from .schema import (
    ALL_FIELDS_BY_NAME,
    ControlledVocab,
    ExtractionValue,
    FieldDefinition,
    FieldType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalisation tables
# ---------------------------------------------------------------------------

# Common French accent / synonym variants → canonical form.
# Applied *before* controlled-vocabulary checking so that
# e.g. "négatif" is accepted for a field expecting "negatif".
_NORMALISATION_MAP: dict[str, str] = {
    # IHC / binary status
    "négatif": "negatif",
    "négative": "negatif",
    "negative": "negatif",
    "neg": "negatif",
    "positif": "positif",
    "positive": "positif",
    "pos": "positif",
    "+": "positif",
    "-": "negatif",
    "conservé": "maintenu",
    "conservée": "maintenu",
    "conserve": "maintenu",
    "perte d'expression": "negatif",
    "perte d expression": "negatif",
    # Molecular status
    "muté": "mute",
    "mutée": "mute",
    "mutee": "mute",
    "wild-type": "wt",
    "wild type": "wt",
    "sauvage": "wt",
    "type sauvage": "wt",
    "non muté": "wt",
    "non mutée": "wt",
    "non mute": "wt",
    "non mutee": "wt",
    "absence de mutation": "wt",
    "pas de mutation": "wt",
    # Methylation
    "méthylé": "methyle",
    "methylé": "methyle",
    "methylation positive": "methyle",
    "méthylation positive": "methyle",
    "non méthylé": "non methyle",
    "non methylé": "non methyle",
    "methylation negative": "non methyle",
    "méthylation négative": "non methyle",
    "absence de méthylation": "non methyle",
    "absence de methylation": "non methyle",
    # Chromosomal
    "délétion": "perte",
    "deletion": "perte",
    "deleted": "perte",
    "del": "perte",
    "perte homozygote": "perte",
    "perte hétérozygote": "perte partielle",
    # Binary
    "oui": "oui",
    "non": "non",
    "yes": "oui",
    "no": "non",
    "vrai": "oui",
    "faux": "non",
    "true": "oui",
    "false": "non",
    "present": "oui",
    "absent": "non",
    "présent": "oui",
    "présente": "oui",
    # Sex
    "homme": "M",
    "femme": "F",
    "masculin": "M",
    "féminin": "F",
    "feminin": "F",
    "h": "M",
    "f": "F",
    "m": "M",
    # Laterality
    "gauche": "gauche",
    "droite": "droite",
    "bilatéral": "bilateral",
    "bilatérale": "bilateral",
    "médian": "median",
    "médiane": "median",
    "ligne médiane": "median",
    # Surgery type
    "exérèse complète": "exerese complete",
    "exerese complète": "exerese complete",
    "exérèse partielle": "exerese partielle",
    "exerese partielle": "exerese partielle",
    "exérèse": "exerese",
    "biopsie stéréotaxique": "biopsie",
    "biopsie chirurgicale": "biopsie",
    # WHO classification
    "oms 2007": "2007",
    "oms 2016": "2016",
    "oms 2021": "2021",
    "who 2007": "2007",
    "who 2016": "2016",
    "who 2021": "2021",
}


# ---------------------------------------------------------------------------
# Normalisation logic
# ---------------------------------------------------------------------------

def normalise_value(
    field_name: str,
    value: Any,
) -> Optional[str | int | float]:
    """Normalise *value* for *field_name* using the normalisation table.

    Returns the normalised value, or the original value if no normalisation
    applies.  Returns ``None`` if value is empty/null-like.
    """
    if value is None:
        return None

    # Handle booleans
    # Note: Pydantic coerces bool → int (True→1, False→0) in
    # ExtractionValue.value, so isinstance(value, bool) may not trigger.
    if isinstance(value, bool):
        return "oui" if value else "non"

    # Handle numeric types (leave as-is for integer/float fields)
    field_def = ALL_FIELDS_BY_NAME.get(field_name)
    if isinstance(value, (int, float)):
        # For binary fields, 1/0 should map to oui/non
        # (covers the case where Pydantic coerced bool → int)
        if field_def and field_def.allowed_values is not None:
            if set(field_def.allowed_values) == {"oui", "non"}:
                if value == 1:
                    return "oui"
                if value == 0:
                    return "non"
        if field_def and field_def.field_type == FieldType.INTEGER:
            return int(value)
        if field_def and field_def.field_type == FieldType.FLOAT:
            return float(value)
        return value

    # String normalisation
    val_str = str(value).strip()
    if not val_str or val_str.lower() in ("null", "none", "n/a", "na", ""):
        return None

    val_lower = val_str.lower()

    # Look up in normalisation map
    if val_lower in _NORMALISATION_MAP:
        return _NORMALISATION_MAP[val_lower]

    # For integer fields, try parsing
    if field_def and field_def.field_type == FieldType.INTEGER:
        try:
            return int(val_str)
        except ValueError:
            pass

    # For float fields, try parsing
    if field_def and field_def.field_type == FieldType.FLOAT:
        try:
            return float(val_str.replace(",", "."))
        except ValueError:
            pass

    return val_str


# ---------------------------------------------------------------------------
# Vocabulary validation
# ---------------------------------------------------------------------------

def _is_value_valid(
    field_def: FieldDefinition,
    normalised_value: Any,
) -> bool:
    """Check if *normalised_value* passes the field's vocabulary constraint.

    Returns ``True`` if the value is acceptable, ``False`` if it should be
    flagged for review.
    """
    if normalised_value is None:
        return True  # Null is always acceptable for nullable fields

    # Fields with no vocabulary constraint → always valid
    if field_def.allowed_values is None:
        return True

    # Check evolution fields specially
    if field_def.name == "evol_clinique":
        return ControlledVocab.is_valid_evolution(str(normalised_value))

    # Check molecular fields specially (accept variant strings)
    if field_def.group == "molecular":
        return ControlledVocab.is_valid_molecular(str(normalised_value))

    # Standard controlled vocabulary check
    if normalised_value in field_def.allowed_values:
        return True

    # Try case-insensitive match for string values
    if isinstance(normalised_value, str):
        normalised_lower = normalised_value.lower()
        for allowed in field_def.allowed_values:
            if isinstance(allowed, str) and allowed.lower() == normalised_lower:
                return True

    return False


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_extraction(
    extractions: dict[str, ExtractionValue],
    feature_definitions: Optional[dict[str, FieldDefinition]] = None,
) -> dict[str, ExtractionValue]:
    """Validate and normalise all extractions against controlled vocabularies.

    For each extraction:
    1. Normalise the value using the normalisation table.
    2. Check the value against the field's controlled vocabulary.
    3. If the value is outside the allowed set, set ``flagged=True``
       and ``vocab_valid=False``.
    4. Return the updated extractions dict (modified in-place).

    Parameters
    ----------
    extractions : dict[str, ExtractionValue]
        Mapping ``field_name → ExtractionValue``.
    feature_definitions : dict[str, FieldDefinition], optional
        Field metadata. Defaults to ``ALL_FIELDS_BY_NAME`` from schema.

    Returns
    -------
    dict[str, ExtractionValue]
        The same dict, with values normalised and out-of-vocabulary values
        flagged.
    """
    if feature_definitions is None:
        feature_definitions = ALL_FIELDS_BY_NAME

    for field_name, ev in extractions.items():
        field_def = feature_definitions.get(field_name)
        if field_def is None:
            # Unknown field — flag it
            ev.flagged = True
            ev.vocab_valid = False
            logger.warning("Unknown field '%s' encountered during validation.", field_name)
            continue

        # Skip if value is None
        if ev.value is None:
            continue

        # 1. Normalise
        normalised = normalise_value(field_name, ev.value)
        ev.value = normalised

        if normalised is None:
            continue

        # 2. Validate against controlled vocabulary
        is_valid = _is_value_valid(field_def, normalised)
        ev.vocab_valid = is_valid

        if not is_valid:
            ev.flagged = True
            logger.info(
                "Field '%s': value %r is outside controlled vocabulary "
                "(allowed: %s). Flagged for review.",
                field_name,
                normalised,
                field_def.allowed_values,
            )

    return extractions
