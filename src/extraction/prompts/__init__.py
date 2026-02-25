"""Prompt templates for Tier 2 LLM extraction.

Each module contains prompt templates for a specific feature group,
designed for use with Qwen3-4B via Ollama.

Public API
----------
- ``PROMPT_REGISTRY``  – mapping from feature group name to prompt config.
- ``get_prompt(group)`` – return the (system, user_template, fields) for a group.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from .bio_ihc import IHC_FIELDS, IHC_PROMPT, IHC_SYSTEM
from .bio_molecular import MOLECULAR_FIELDS, MOLECULAR_PROMPT, MOLECULAR_SYSTEM
from .bio_chromosomal import CHROMOSOMAL_FIELDS, CHROMOSOMAL_PROMPT, CHROMOSOMAL_SYSTEM
from .bio_diagnosis import DIAGNOSIS_FIELDS, DIAGNOSIS_PROMPT, DIAGNOSIS_SYSTEM
from .clinique_demographics import DEMOGRAPHICS_FIELDS, DEMOGRAPHICS_PROMPT, DEMOGRAPHICS_SYSTEM
from .clinique_symptoms import SYMPTOMS_FIELDS, SYMPTOMS_PROMPT, SYMPTOMS_SYSTEM
from .clinique_treatment import TREATMENT_FIELDS, TREATMENT_PROMPT, TREATMENT_SYSTEM
from .clinique_evolution import EVOLUTION_FIELDS, EVOLUTION_PROMPT, EVOLUTION_SYSTEM


class PromptConfig(NamedTuple):
    """Configuration for a feature group's LLM prompt."""
    system: str
    user_template: str  # Contains ``{section_text}`` placeholder
    fields: list[str]


PROMPT_REGISTRY: dict[str, PromptConfig] = {
    "ihc": PromptConfig(
        system=IHC_SYSTEM,
        user_template=IHC_PROMPT,
        fields=IHC_FIELDS,
    ),
    "molecular": PromptConfig(
        system=MOLECULAR_SYSTEM,
        user_template=MOLECULAR_PROMPT,
        fields=MOLECULAR_FIELDS,
    ),
    "chromosomal": PromptConfig(
        system=CHROMOSOMAL_SYSTEM,
        user_template=CHROMOSOMAL_PROMPT,
        fields=CHROMOSOMAL_FIELDS,
    ),
    "diagnosis": PromptConfig(
        system=DIAGNOSIS_SYSTEM,
        user_template=DIAGNOSIS_PROMPT,
        fields=DIAGNOSIS_FIELDS,
    ),
    "demographics": PromptConfig(
        system=DEMOGRAPHICS_SYSTEM,
        user_template=DEMOGRAPHICS_PROMPT,
        fields=DEMOGRAPHICS_FIELDS,
    ),
    "symptoms": PromptConfig(
        system=SYMPTOMS_SYSTEM,
        user_template=SYMPTOMS_PROMPT,
        fields=SYMPTOMS_FIELDS,
    ),
    "treatment": PromptConfig(
        system=TREATMENT_SYSTEM,
        user_template=TREATMENT_PROMPT,
        fields=TREATMENT_FIELDS,
    ),
    "evolution": PromptConfig(
        system=EVOLUTION_SYSTEM,
        user_template=EVOLUTION_PROMPT,
        fields=EVOLUTION_FIELDS,
    ),
}


def get_prompt(group: str) -> PromptConfig:
    """Return the prompt configuration for a feature group.

    Parameters
    ----------
    group : str
        One of: ``ihc``, ``molecular``, ``chromosomal``, ``diagnosis``,
        ``demographics``, ``symptoms``, ``treatment``, ``evolution``.

    Raises
    ------
    KeyError
        If *group* is not a recognised feature group.
    """
    if group not in PROMPT_REGISTRY:
        raise KeyError(
            f"Unknown prompt group: {group!r}. "
            f"Available: {sorted(PROMPT_REGISTRY)}"
        )
    return PROMPT_REGISTRY[group]
