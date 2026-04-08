"""HFExtractor — inference over the 10 fine-tuned raphael-r/bright-eds-* models.

Each model is an edsnlp pipeline saved to the HuggingFace Hub and loadable
via ``edsnlp.load("raphael-r/bright-eds-{group}")``.  Models are loaded
ONE AT A TIME for the whole batch, then released, to minimise GPU memory
usage.

Architecture slot: ML branch (``use_eds=True`` in ExtractionPipeline).
Controlled-field normalization reuses the lookup tables from rule_extraction.py.
Free-text fields (diagnostics, drug names, proper nouns, dates, etc.) are
returned as-is from the entity text.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import torch
import edsnlp

from .schema import ExtractionValue

# ---------------------------------------------------------------------------
# bright_models lives at BRIGHT_ClinicalDataBase/bright_models/ — 2 levels up
# from src/extraction/.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bright_models"))
from utils import GROUPS  # dict[str, list[str]]  # noqa: E402

# Reuse normalisation tables already defined in rule_extraction.py
from .rule_extraction import (  # noqa: E402
    _IHC_VALUE_NORM,
    _MOL_STATUS_NORM,
    _CHR_STATUS_NORM,
    _ROMAN_TO_INT,
    _LATERALITY_NORM,
    _PAT_SEX_HEADER,
    _PAT_SEX_SALUTATION_F,
    _PAT_SEX_SALUTATION_M,
    _PAT_SEX_AGREEMENT_F,
    _PAT_SEX_AGREEMENT_M,
    _SEX_TOKEN_MAP,
    _PAT_SEX_TOKENS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HUB_PREFIX = "raphael-r/bright-eds-"

# Only known label mismatch between model output and canonical field name.
_LABEL_REMAP: dict[str, str] = {"chir_date": "date_chir"}

# IHC field names (value normalised via _IHC_VALUE_NORM)
_IHC_FIELDS: frozenset[str] = frozenset({
    "ihc_idh1", "ihc_atrx", "ihc_p53", "ihc_fgfr3", "ihc_braf",
    "ihc_gfap", "ihc_olig2", "ihc_ki67", "ihc_hist_h3k27m",
    "ihc_hist_h3k27me3", "ihc_egfr_hirsch", "ihc_mmr",
})

# Molecular field names (value normalised via _MOL_STATUS_NORM)
_MOL_FIELDS: frozenset[str] = frozenset({
    "mol_idh1", "mol_idh2", "mol_mgmt", "mol_h3f3a", "mol_hist1h3b",
    "mol_tert", "mol_CDKN2A", "mol_atrx", "mol_cic", "mol_fubp1",
    "mol_fgfr1", "mol_egfr_mut", "mol_prkca", "mol_pten", "mol_p53",
    "mol_braf",
})

# Chromosomal / ampli / fusion field names (value normalised via _CHR_STATUS_NORM)
_CHR_FIELDS: frozenset[str] = frozenset({
    "ch1p", "ch19q", "ch1p19q_codel", "ch7p", "ch7q",
    "ch10p", "ch10q", "ch9p", "ch9q",
    "ampli_egfr", "ampli_cdk4", "ampli_mdm2", "ampli_mdm4", "ampli_met",
    "fusion_fgfr", "fusion_ntrk", "fusion_autre",
})

# Binary fields: entity presence → "oui"; negation handled downstream by
# AssertionAnnotator.
_BINARY_FIELDS: frozenset[str] = frozenset({
    "optune", "corticoides", "anti_epileptiques",
    "histo_necrose", "histo_pec",
    "prise_de_contraste", "epilepsie_1er_symptome",
    "essai_therapeutique",
    "epilepsie", "ceph_hic", "deficit", "cognitif",
    "contraste_1er_symptome", "oedeme_1er_symptome", "calcif_1er_symptome",
    "ceph_hic_1er_symptome", "deficit_1er_symptome", "cognitif_1er_symptome",
    "progress_clinique", "progress_radiologique",
    "reponse_radiologique", "autre_trouble",
    "antecedent_tumoral", "infos_deces",
})

# Pattern to validate classification_oms values (20XX years)
_PAT_OMS_YEAR = re.compile(r"20\d{2}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_label(label: str) -> str:
    """Map model output label to the canonical field name."""
    return _LABEL_REMAP.get(label, label)


def _normalize_sexe(span_text: str) -> Optional[str]:
    """Normalise a raw span to 'M' or 'F' using the same rules as rule_extraction."""
    t = span_text.strip()
    # 1. Single-char direct token
    if t.upper() in ("M", "F"):
        return t.upper()
    # 2. Header pattern: | M | or | F |
    m = _PAT_SEX_HEADER.search(t)
    if m:
        return m.group("sex").upper()
    # 3. Salutation patterns
    if _PAT_SEX_SALUTATION_F.search(t):
        return "F"
    if _PAT_SEX_SALUTATION_M.search(t):
        return "M"
    # 4. Past-participle agreement
    if _PAT_SEX_AGREEMENT_F.search(t):
        return "F"
    if _PAT_SEX_AGREEMENT_M.search(t):
        return "M"
    # 5. Token map (homme/femme/masculin/féminin)
    m = _PAT_SEX_TOKENS.search(t)
    if m:
        return _SEX_TOKEN_MAP.get(m.group().lower())
    return None


def _normalize_span(fname: str, span_text: str) -> Optional[str]:
    """Return a normalised value for *span_text* according to *fname*'s vocabulary.

    Returns ``None`` if the span cannot be mapped to a valid value (the entity
    should then be discarded).  Free-text fields are returned as-is.
    """
    t = span_text.strip()
    t_lower = t.lower()

    if fname in _IHC_FIELDS:
        return _IHC_VALUE_NORM.get(t_lower) or _IHC_VALUE_NORM.get(t)

    if fname in _MOL_FIELDS:
        return _MOL_STATUS_NORM.get(t_lower) or _MOL_STATUS_NORM.get(t)

    if fname in _CHR_FIELDS:
        return _CHR_STATUS_NORM.get(t_lower) or _CHR_STATUS_NORM.get(t)

    if fname == "grade":
        # Accept Roman or Arabic numerals 1-4
        roman = _ROMAN_TO_INT.get(t.upper())
        if roman is not None:
            return str(roman)
        if t in ("1", "2", "3", "4"):
            return t
        return None

    if fname == "tumeur_lateralite":
        return _LATERALITY_NORM.get(t_lower)

    if fname == "classification_oms":
        m = _PAT_OMS_YEAR.search(t)
        return m.group() if m else None

    if fname == "sexe":
        return _normalize_sexe(t)

    if fname in _BINARY_FIELDS:
        # Presence of a span signals "oui"; negation is applied downstream.
        return "oui"

    # Free-text field: return raw span text unchanged.
    return t if t else None


def _span_to_ev(fname: str, ent) -> Optional[ExtractionValue]:
    """Build an ExtractionValue from a spaCy-like entity span."""
    value = _normalize_span(fname, ent.text)
    if value is None:
        return None
    return ExtractionValue(
        value=value,
        source_span=ent.text,
        source_span_start=ent.start_char,
        source_span_end=ent.end_char,
        confidence=None,        # CRF models do not expose calibrated scores
        extraction_tier="hf",
        section=None,
        vocab_valid=None,       # validated later by pipeline validation step
    )


# ---------------------------------------------------------------------------
# HFExtractor
# ---------------------------------------------------------------------------

class HFExtractor:
    """Run inference using the 10 fine-tuned raphael-r/bright-eds-* models.

    Parameters
    ----------
    enabled_groups:
        Subset of GROUPS keys to enable.  ``None`` activates all 10 groups.
    local_model_dir:
        If provided, look for ``{local_model_dir}/{group}`` before falling
        back to the HuggingFace Hub.  Useful for offline Colab runs with
        Drive-cached checkpoints.
    """

    def __init__(
        self,
        enabled_groups: list[str] | None = None,
        local_model_dir: Path | None = None,
    ) -> None:
        self.enabled_groups: list[str] = enabled_groups or list(GROUPS.keys())
        self.local_model_dir: Optional[Path] = (
            Path(local_model_dir) if local_model_dir is not None else None
        )

    def _resolve_model(self, group: str) -> str:
        """Return a local path (str) if a cached model exists, else a Hub repo ID."""
        if self.local_model_dir is not None:
            local = self.local_model_dir / group
            if local.exists():
                return str(local)
        return f"{HUB_PREFIX}{group}"

    def extract_batch(self, texts: list[str]) -> list[dict[str, ExtractionValue]]:
        """Infer on *texts* with each enabled group model loaded exactly once.

        For every group:
        1. Load the model (from disk or HuggingFace Hub).
        2. Run ``nlp.pipe(texts)`` — returns one Doc per text.
        3. Iterate entities; normalise and store in per-document result dicts.
        4. Delete the model and flush the GPU cache before loading the next.

        Returns
        -------
        list[dict[str, ExtractionValue]]
            One dict per input text.  A field is present only when the model
            extracted a non-None normalised value.
        """
        results: list[dict[str, ExtractionValue]] = [{} for _ in texts]

        for group in self.enabled_groups:
            model_id = self._resolve_model(group)
            nlp = edsnlp.load(model_id)

            for i, doc in enumerate(nlp.pipe(texts)):
                for ent in doc.ents:
                    fname = _normalize_label(ent.label_)
                    ev = _span_to_ev(fname, ent)
                    if ev is not None:
                        # Last entity wins per field (models are deterministic,
                        # so only one entity per field is expected).
                        results[i][fname] = ev

            del nlp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def extract(self, text: str) -> dict[str, ExtractionValue]:
        """Convenience wrapper for single-document inference."""
        return self.extract_batch([text])[0]
