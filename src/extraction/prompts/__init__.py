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

from .clinique_demographics import DEMOGRAPHICS_FIELDS, DEMOGRAPHICS_PROMPT, DEMOGRAPHICS_SYSTEM
from .clinique_evolution import EVOLUTION_FIELDS, EVOLUTION_PROMPT, EVOLUTION_SYSTEM
from .clinique_symptoms import SYMPTOMS_FIELDS, SYMPTOMS_PROMPT, SYMPTOMS_SYSTEM


class PromptConfig(NamedTuple):
    """Configuration for a feature group's LLM prompt."""
    system: str
    user_template: str  # Contains ``{section_text}`` placeholder
    fields: list[str]


PROMPT_REGISTRY: dict[str, PromptConfig] = {
    "demographics": PromptConfig(
        system=DEMOGRAPHICS_SYSTEM,
        user_template=DEMOGRAPHICS_PROMPT,
        fields=DEMOGRAPHICS_FIELDS,
    ),
    "evolution": PromptConfig(
        system=EVOLUTION_SYSTEM,
        user_template=EVOLUTION_PROMPT,
        fields=EVOLUTION_FIELDS,
    ),
    "symptoms": PromptConfig(
        system=SYMPTOMS_SYSTEM,
        user_template=SYMPTOMS_PROMPT,
        fields=SYMPTOMS_FIELDS,
    ),
}


def get_prompt(group: str) -> PromptConfig:
    """Return the prompt configuration for a feature group.

    Parameters
    ----------
    group : str
        One of: ``demographics``, ``evolution``.

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
