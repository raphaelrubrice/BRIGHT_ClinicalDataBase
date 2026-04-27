"""Main extraction pipeline orchestrator.

Wires together document classification, section detection, rule-based
extraction (DateExtractor, ControlledExtractor, regex rules, EDSExtractor),
HuggingFace fine-tuned model extraction (HFExtractor), validation, and
provenance tracking into a single ``extract_document()`` function.

Supports three ablatable modes (see ``ExtractionPipeline``):
- Rules-only  (``use_eds=False``), DateExtractor + ControlledExtractor +
  regex rules + EDSExtractor (rule-based edsnlp patterns).  No HF models.
- ML-only     (``use_rules=False``), HFExtractor (10 fine-tuned models)
  on all fields.  No rule-based extractors.
- Rules + ML  (``use_eds=True, use_rules=True``), both branches run;
  HF wins on ``_HF_PASSING_FIELDS``, rules win on others.

Public API
----------
- ``ExtractionPipeline`` , Main pipeline class.
"""

from __future__ import annotations

import logging
import os
import time
import io
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from .controlled_extractor import ControlledExtractor
from .controlled_vocab_data import CONTROLLED_REGISTRY_FR
from .date_extractor import DateExtractor
from .document_classifier import DocumentClassifier
from .provenance import ExtractionResult
from .eds_extractor import EDSExtractor
from .hf_extractor import HFExtractor
from .negation import AssertionAnnotator
from .rule_extraction import run_rule_extraction
from .schema import (
    ExtractionValue,
    FieldType,
    FEATURE_ROUTING,
    ALL_FIELDS_BY_NAME,
    MappingType,
    get_extractable_fields,
)
from .section_detector import SectionDetector
from .validation import validate_extraction, validate_source_spans

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flip tables for SIMILARITY negation
# ---------------------------------------------------------------------------

_SIMILARITY_FLIP: dict[str, str] = {
    # IHC / presence
    "positif": "negatif",
    "negatif": "positif",
    "positive": "negatif",
    "negative": "positif",
    "maintenu": "negatif",
    # Molecular status: mute ↔ wt
    "mute": "wt",
    "muté": "wt",
    "mutée": "wt",
    "wt": "muté",
    # Methylation: methyle ↔ non methyle
    "methyle": "non methyle",
    "méthylé": "non methyle",
    "non methyle": "méthylé",
    "non méthylé": "méthylé",
    # Lack of gain is not synonymous with loss and vice versa
    # # Chromosomal: gain ↔ perte
    # "gain": "perte",
    # "perte": "gain",
    # "perte partielle": "gain",
}


# ---------------------------------------------------------------------------
# Negation: mapping_type-aware inversion
# ---------------------------------------------------------------------------

def _apply_negation(field_name: str, value, source_span: str | None = None) -> str:
    """Invert an extracted value when negation is detected.

    Dispatch based on the field's ``mapping_type``:

    - **PRESENCE** → return ``"non"``
    - **SIMILARITY** → flip value via ``_SIMILARITY_FLIP``
    - **DIRECT** (free text) → prepend ``"non "`` before the span text

    Special case: ``"en attente"`` is *never* negated.
    """
    val_str = str(value).strip().lower() if value is not None else ""

    # "en attente" is never negated
    if val_str == "en attente":
        return val_str

    field_def = ALL_FIELDS_BY_NAME.get(field_name)
    mapping = field_def.mapping_type if field_def else MappingType.DIRECT

    # PRESENCE fields → always "non"
    if mapping == MappingType.PRESENCE:
        return "non"

    # SIMILARITY fields → flip via table
    if mapping == MappingType.SIMILARITY:
        flipped = _SIMILARITY_FLIP.get(val_str)
        if flipped:
            return flipped
        # If no flip rule matches, return original
        return val_str

    # DIRECT fields → prepend "non " if free text, else use flip table
    flipped = _SIMILARITY_FLIP.get(val_str)
    if flipped:
        return flipped
    # Free text: prepend "non "
    if val_str:
        return f"non {val_str}"
    return val_str


# ---------------------------------------------------------------------------
# EDS passing fields (F1 >= 0.6 on held-out benchmark)
# Used in Rules + ML mode: HF wins on these fields, rules win on the others.
# In ML-only mode HF is used on ALL fields (no filter applied).
# Calibrated from HuggingFace model-card metrics, refine after full evaluation.
# ---------------------------------------------------------------------------

_HF_PASSING_FIELDS: frozenset[str] = frozenset({
    "fusion_autre", "chm_date_debut", "dominance_cerebrale", "ik_clinique",
    "diag_histologique", "rx_fractionnement", "type_chirurgie", "rx_dose",
    "mol_idh1", "tumeur_lateralite", "classification_oms", "tumeur_position",
    "ch10q", "sexe", "grade", "neuroncologue", "ch7p", "anti_epileptiques",
    "histo_mitoses", "date_rcp", "corticoides", "mol_fubp1", "mol_h3f3a",
    "mol_braf", "evol_clinique", "chimio_protocole", "ihc_idh1", "mol_mgmt",
    "annee_de_naissance", "localisation_chir", "mol_atrx", "ihc_gfap", "ch9p",
    "mol_tert", "ihc_ki67", "ihc_atrx", "ihc_olig2", "autre_trouble_1er_symptome",
    "chimios", "ihc_p53", "neurochirurgien", "diag_integre",
    "activite_professionnelle", "infos_deces", "mol_idh2", "mol_cic",
    "mol_CDKN2A", "ihc_hist_h3k27me3", "survie_globale", "ihc_braf", "ch7q",
    "ch1p", "aspect_cellulaire", "chm_cycles", "essai_therapeutique",
    "ch1p19q_codel", "prise_de_contraste", "radiotherapeute", "anatomo_pathologiste",
})


# ---------------------------------------------------------------------------
# Merge helpers for the three-mode pipeline
# ---------------------------------------------------------------------------

def _merge_rules(
    rule_results: dict[str, ExtractionValue],
    controlled_results: dict[str, ExtractionValue],
    date_results: dict[str, ExtractionValue],
    eds_results: dict[str, ExtractionValue],
) -> dict[str, ExtractionValue]:
    """Merge all rule-based result dicts into a single ``rules_merged`` dict.

    Priority: ``date > controlled > eds_results (edsnlp patterns) > rule_results (regex)``.

    EDSExtractor (rule-based edsnlp patterns) sits between controlled-vocab
    extraction and pure-regex rules because its pattern matching is typically
    more specialised.
    """
    merged: dict[str, ExtractionValue] = {}
    merged.update(rule_results)
    merged.update(eds_results)          # edsnlp rule patterns win over pure regex
    merged.update(controlled_results)   # controlled vocab wins over edsnlp patterns
    merged.update(date_results)         # date extractor always wins
    return merged


def _decide_extraction(
    hf_results: dict[str, ExtractionValue],
    rules_merged: dict[str, ExtractionValue],
    use_hf: bool,
    use_rules: bool,
) -> dict[str, ExtractionValue]:
    """Select the best extraction per field based on the active mode.

    Modes
    -----
    - **ML-only** (``use_rules=False``)  → return *hf_results* as-is.
    - **Rules-only** (``use_hf=False``)  → return *rules_merged* as-is.
    - **Rules + ML** (both True)         → HF wins on ``_HF_PASSING_FIELDS``,
      rules win on all others.
    """
    if not use_rules:
        return dict(hf_results)
    if not use_hf:
        return dict(rules_merged)
    # Rules + ML: per-field decision
    merged = dict(rules_merged)
    for fname, hf_val in hf_results.items():
        if fname in _HF_PASSING_FIELDS:
            merged[fname] = hf_val
    return merged


# ---------------------------------------------------------------------------
# Multiprocessing Globals & Worker Functions
# ---------------------------------------------------------------------------

_WORKER_PIPELINE: ExtractionPipeline | None = None

def _init_worker(pipeline_kwargs: dict):
    """Initialise a global pipeline instance for each worker process.
    This avoids pickling heavy ML models (GLiNER, spaCy) during cross-process calls.
    It also enforces single-threading at the C++/backend level to prevent CPU thrashing.
    """
    # 1. Force C++ backend math libraries to use a single thread per process
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # Prevent HuggingFace tokenizers from deadlocking in child processes
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 2. Force PyTorch to use a single thread per process
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass

    global _WORKER_PIPELINE
    _WORKER_PIPELINE = ExtractionPipeline(**pipeline_kwargs)

def _process_single_doc(idx: int, doc: dict):
    """Base function wrapped for multiprocessing map."""
    global _WORKER_PIPELINE
    text = doc.get("text", "")
    doc_id = doc.get("document_id", f"doc_{idx}")
    patient_id = doc.get("patient_id", "")
    consultation_date = doc.get("consultation_date")

    # Capture standard prints separately so they do not interleave
    # with prints from other parallel workers on the console.
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            result = _WORKER_PIPELINE.extract_document(
                text=text,
                document_id=doc_id,
                patient_id=patient_id,
                consultation_date=consultation_date,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to process document '%s': %s", doc_id, exc)
            result = ExtractionResult(
                document_id=doc_id,
                patient_id=patient_id,
            )
            result.add_log(f"Pipeline failed with error: {exc}")

    # Return the captured stdout so the main thread can print it cleanly
    return idx, result, f.getvalue()


# ---------------------------------------------------------------------------
# Main Pipeline Class
# ---------------------------------------------------------------------------

class ExtractionPipeline:
    """End-to-end clinical feature extraction pipeline.

    Supports three modes selectable via ``use_eds`` / ``use_rules``:

    - **Rules-only** (``use_eds=False, use_rules=True``): DateExtractor +
      ControlledExtractor + regex rules + EDSExtractor (rule-based edsnlp
      patterns).  Fully deterministic, no HF models.
    - **ML-only** (``use_eds=True, use_rules=False``): HFExtractor
      (10 fine-tuned ``raphael-r/bright-eds-*`` models) on all fields.
      No rule-based extractors.
    - **Rules + ML** (``use_eds=True, use_rules=True``): both branches run;
      HF wins on ``_HF_PASSING_FIELDS``, rules win on others.

    Pipeline steps (Rules + ML mode):
    1.  Document type classification
    2.  Section detection
    3.  Language detection
    4.  Feature routing (document type → extractable fields)
    5.  DateExtractor              → date_results         (RULES branch)
    6.  ControlledExtractor        → controlled_results   (RULES branch)
    7.  run_rule_extraction()      → rule_results         (RULES branch)
    7.5 EDSExtractor               → eds_results          (RULES branch)
    8.  HFExtractor                → hf_results           (ML branch)
    9.  Merge rules branch         → rules_merged
    10. Decide HF vs Rules         → features
    11. Negation                   → features updated
    12. Controlled vocabulary validation
    13. Source span validation
    14. Provenance record + flagging

    Parameters
    ----------
    use_negation : bool
        Whether to apply negation annotation after extraction.
    use_eds : bool
        Whether to run HuggingFace fine-tuned model extraction (ML branch).
    use_rules : bool
        Whether to run rule-based extraction (DateExtractor,
        ControlledExtractor, regex rules, EDSExtractor).
    enabled_rule_extractors : set[str] or None
        Fine-grained control over which rule sub-extractors to activate.
        ``None`` means all are active.  See ``run_rule_extraction`` for
        valid names.
    enabled_groups : list[str] or None
        HuggingFace model groups to enable.  ``None`` activates all 10.
        Useful for ablation studies (e.g. ``["diagnosis", "ihc"]``).
    local_model_dir : Path or None
        Directory containing locally cached ``bright-eds-{group}`` model
        subdirectories.  Checked before downloading from the Hub.
    verbose : bool
        Whether to emit step-by-step debug logs (default ``False``).
    n_jobs : int
        Number of workers for parallel processing (-1 = max CPUs - 2).
        Ignored when ``use_eds=True`` (HF batch pre-computation forces
        sequential execution).
    """

    def __init__(
        self,
        use_negation: bool = True,
        use_eds: bool = True,
        use_rules: bool = True,
        enabled_rule_extractors: set[str] | None = None,
        enabled_groups: list[str] | None = None,
        local_model_dir: Optional[str] = None,
        verbose: bool = False,
        transparent: bool = False,
        n_jobs: int = -1,
    ):
        self.use_negation = use_negation
        self.use_eds = use_eds
        self.use_rules = use_rules
        self.enabled_rule_extractors = enabled_rule_extractors
        self.enabled_groups = enabled_groups
        self.local_model_dir = local_model_dir
        self.verbose = verbose
        self.transparent = transparent
        self.n_jobs = n_jobs

        # Sub-components
        self.classifier = DocumentClassifier()
        self.section_detector = SectionDetector()

        # RULES branch: EDSExtractor (rule-based edsnlp patterns)
        self._eds_extractor = EDSExtractor() if use_rules else None

        # ML branch: HFExtractor (10 fine-tuned HuggingFace models)
        self._hf_extractor = (
            HFExtractor(enabled_groups=enabled_groups, local_model_dir=local_model_dir)
            if use_eds else None
        )

        self._assertion_annotator = AssertionAnnotator() if use_negation else None
        self._date_extractor = DateExtractor()
        self._controlled_extractor = ControlledExtractor()

    # -- Transparent logging helper -------------------------------------------

    @staticmethod
    def _log_extraction_dict(
        label: str,
        d: dict[str, "ExtractionValue"],
        max_span: int = 60,
    ) -> None:
        """Dump the full contents of an extraction result dict via logger.debug.

        ``transparent=True`` calls this after every intermediate step so that
        each field's value, source span, and confidence are visible in the log.
        """
        if not d:
            logger.debug("  [%s] (empty)", label)
            return
        logger.debug("  [%s], %d field(s):", label, len(d))
        logger.debug(
            "    %-32s %-22s %-20s %s",
            "field", "value", "source_span[:60]", "conf",
        )
        logger.debug("    %s", "-" * 90)
        for fname, ev in sorted(d.items()):
            val = repr(ev.value) if ev.value is not None else "None"
            span = (ev.source_span or "")[:max_span].replace("\n", "↵")
            conf = f"{ev.confidence:.2f}" if ev.confidence is not None else "N/A"
            logger.debug("    %-32s %-22s %-20s %s", fname, val[:22], span[:20], conf)

    # -- Language detection ---------------------------------------------------

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect the language of *text*, defaulting to ``"fr"``."""
        try:
            from langdetect import detect
            lang = detect(text[:2000])  # Use first 2000 chars for speed
            return lang
        except Exception:
            return "fr"

    # -- Main entry point ----------------------------------------------------

    def extract_document(
        self,
        text: str,
        document_id: str = "",
        patient_id: str = "",
        consultation_date: str | None = None,
        _precomputed_hf: dict[str, ExtractionValue] | None = None,
    ) -> ExtractionResult:
        """Run the full extraction pipeline on a single document.

        Parameters
        ----------
        text : str
            The full document text (already pseudonymised if needed).
        document_id : str
            Unique document identifier.
        patient_id : str
            Patient identifier (pseudonymised).
        consultation_date : str | None
            Consultation / document date in DD/MM/YYYY format.  Used by
            the ``DateExtractor`` to exclude it from clinical date
            assignment.  If ``None``, auto-detected via regex.

        Returns
        -------
        ExtractionResult
            Complete extraction result with features, provenance, and
            audit trail.
        """
        t_start = time.perf_counter()
        _v = self.verbose  # local alias for brevity

        result = ExtractionResult(
            document_id=document_id,
            patient_id=patient_id,
        )
        result.add_log(f"Pipeline started for document '{document_id}'.")
        if _v:
            logger.debug("=" * 60)
            logger.debug("[PIPELINE] Starting extraction for document '%s'", document_id)
            logger.debug("=" * 60)

        # -----------------------------------------------------------------
        # Step 1: Classify document type
        # -----------------------------------------------------------------
        if _v:
            logger.debug("[Step 1/14] Classifying document type...")
        classification = self.classifier.classify(text)
        result.document_type = classification.document_type
        result.classification_confidence = classification.confidence
        result.classification_is_ambiguous = classification.is_ambiguous
        result.add_log(
            f"Document classified as '{classification.document_type}' "
            f"(confidence={classification.confidence:.2f}, "
            f"ambiguous={classification.is_ambiguous})."
        )
        if _v:
            logger.debug(
                "           → type='%s', confidence=%.2f, ambiguous=%s",
                classification.document_type,
                classification.confidence,
                classification.is_ambiguous,
            )

        if classification.is_ambiguous:
            result.add_log(
                "Classification is ambiguous, extracted features may be "
                "incomplete. Consider manual review."
            )

        # -----------------------------------------------------------------
        # Step 2: Detect sections
        # -----------------------------------------------------------------
        if _v:
            logger.debug("[Step 2/14] Detecting sections...")
        sections = self.section_detector.detect(text)
        result.sections_detected = list(sections.keys())
        result.add_log(
            f"Sections detected: {result.sections_detected}"
        )
        if _v:
            logger.debug(
                "           → %d sections: %s",
                len(result.sections_detected), result.sections_detected,
            )

        # -----------------------------------------------------------------
        # Step 3: Language detection
        # -----------------------------------------------------------------
        if _v:
            logger.debug("[Step 3/14] Detecting language...")
        language = self._detect_language(text)
        result.add_log(f"Detected language: '{language}'.")
        if _v:
            logger.debug("           → language='%s'", language)

        # -----------------------------------------------------------------
        # Step 4: Determine extractable feature subset
        # -----------------------------------------------------------------
        if _v:
            logger.debug("[Step 4/14] Routing features for document type...")
        try:
            feature_subset = get_extractable_fields(result.document_type)
        except ValueError as exc:
            result.add_log(f"Feature routing error: {exc}. Using full feature set.")
            from .schema import ALL_BIO_FIELD_NAMES, ALL_CLINIQUE_FIELD_NAMES
            feature_subset = sorted(
                set(ALL_BIO_FIELD_NAMES + ALL_CLINIQUE_FIELD_NAMES)
            )

        # Split date fields from non-date fields
        date_fields = [
            f for f in feature_subset
            if f in ALL_FIELDS_BY_NAME
            and ALL_FIELDS_BY_NAME[f].field_type == FieldType.DATE
        ]
        non_date_fields = [f for f in feature_subset if f not in date_fields]

        result.add_log(
            f"Feature subset: {len(feature_subset)} fields to extract "
            f"({len(date_fields)} date, {len(non_date_fields)} non-date)."
        )
        if _v:
            logger.debug(
                "           → %d fields targeted (%d date, %d non-date)",
                len(feature_subset), len(date_fields), len(non_date_fields),
            )

        # =================================================================
        # RULES BRANCH  (Steps 5–7), active when use_rules=True
        # =================================================================

        # -----------------------------------------------------------------
        # Step 5: DateExtractor
        # -----------------------------------------------------------------
        date_results: dict[str, ExtractionValue] = {}
        if self.use_rules and date_fields:
            if _v:
                logger.debug("[Step 5/14] Running DateExtractor (rules branch)...")
            t_date_start = time.perf_counter()
            try:
                date_results = self._date_extractor.extract(
                    text=text,
                    feature_subset=date_fields,
                    language=language,
                    consultation_date=consultation_date,
                )
            except Exception as exc:
                result.add_log(f"DateExtractor failed: {exc}")
                logger.error("DateExtractor failed: %s", exc)
            t_date_elapsed = (time.perf_counter() - t_date_start) * 1000
            result.add_log(
                f"DateExtractor: {len(date_results)} date fields in {t_date_elapsed:.0f}ms."
            )
            if _v:
                logger.debug("           → %d date fields in %.0fms", len(date_results), t_date_elapsed)
            if self.transparent:
                self._log_extraction_dict("Step5 date_results", date_results)

        # -----------------------------------------------------------------
        # Step 6: ControlledExtractor
        # -----------------------------------------------------------------
        controlled_results: dict[str, ExtractionValue] = {}
        if self.use_rules:
            ctrl_registry = CONTROLLED_REGISTRY_FR
            controlled_fields = [f for f in non_date_fields if f in ctrl_registry]
            if controlled_fields:
                if _v:
                    logger.debug("[Step 6/14] Running ControlledExtractor (rules branch)...")
                t_ctrl_start = time.perf_counter()
                try:
                    controlled_results = self._controlled_extractor.extract(
                        text=text,
                        feature_subset=controlled_fields,
                        language=language,
                    )
                except Exception as exc:
                    result.add_log(f"ControlledExtractor failed: {exc}")
                    logger.error("ControlledExtractor failed: %s", exc)
                t_ctrl_elapsed = (time.perf_counter() - t_ctrl_start) * 1000
                result.add_log(
                    f"ControlledExtractor: {len(controlled_results)} fields "
                    f"in {t_ctrl_elapsed:.0f}ms."
                )
                if _v:
                    logger.debug("           → %d fields in %.0fms", len(controlled_results), t_ctrl_elapsed)
                if self.transparent:
                    self._log_extraction_dict("Step6 controlled_results", controlled_results)

        # -----------------------------------------------------------------
        # Step 7: run_rule_extraction (regex / heuristics)
        # -----------------------------------------------------------------
        rule_results: dict[str, ExtractionValue] = {}
        if self.use_rules:
            if _v:
                logger.debug("[Step 7/14] Running rule extraction (regex/heuristics)...")
            t_rules_start = time.perf_counter()
            try:
                rule_results = run_rule_extraction(
                    text=text,
                    sections=sections,
                    feature_subset=non_date_fields,
                    enabled_extractors=self.enabled_rule_extractors,
                )
            except Exception as exc:
                result.add_log(f"run_rule_extraction failed: {exc}")
                logger.error("run_rule_extraction failed: %s", exc)
            t_rules_elapsed = (time.perf_counter() - t_rules_start) * 1000
            result.add_log(
                f"RuleExtraction: {len(rule_results)} fields in {t_rules_elapsed:.0f}ms."
            )
            if _v:
                logger.debug("           → %d fields in %.0fms", len(rule_results), t_rules_elapsed)
            if self.transparent:
                self._log_extraction_dict("Step7 rule_results", rule_results)

        # =================================================================
        # RULES BRANCH, Step 7.5: EDSExtractor (rule-based edsnlp patterns)
        # =================================================================

        # -----------------------------------------------------------------
        # Step 7.5: EDSExtractor (rule-based edsnlp patterns, RULES branch)
        # -----------------------------------------------------------------
        eds_results: dict[str, ExtractionValue] = {}
        if self.use_rules and self._eds_extractor:
            if _v:
                logger.debug("[Step 7.5/14] Running EDSExtractor (RULES branch)...")
            t_eds_start = time.perf_counter()
            try:
                eds_results = self._eds_extractor.extract(
                    text=text,
                    sections=sections,
                    feature_subset=feature_subset,
                )
            except Exception as exc:
                result.add_log(f"EDSExtractor failed: {exc}")
                logger.error("EDSExtractor failed: %s", exc)
            t_eds_elapsed = (time.perf_counter() - t_eds_start) * 1000
            result.add_log(
                f"EDSExtractor: {len(eds_results)} fields in {t_eds_elapsed:.0f}ms."
            )
            if _v:
                logger.debug("           → %d fields in %.0fms", len(eds_results), t_eds_elapsed)
            if self.transparent:
                self._log_extraction_dict("Step7.5 eds_results", eds_results)

        # =================================================================
        # ML BRANCH  (Step 8), active when use_eds=True
        # =================================================================

        # -----------------------------------------------------------------
        # Step 8: HFExtractor (fine-tuned HuggingFace models, ML branch)
        # -----------------------------------------------------------------
        hf_results: dict[str, ExtractionValue] = {}
        if self.use_eds and self._hf_extractor:
            if _v:
                logger.debug("[Step 8/14] Running HFExtractor (ML branch)...")
            t_hf_start = time.perf_counter()
            try:
                if _precomputed_hf is not None:
                    # Provided by extract_batch(), avoids reloading models per doc.
                    hf_results = _precomputed_hf
                else:
                    hf_results = self._hf_extractor.extract(text)
            except Exception as exc:
                result.add_log(f"HFExtractor failed: {exc}")
                logger.error("HFExtractor failed: %s", exc)
            t_hf_elapsed = (time.perf_counter() - t_hf_start) * 1000
            result.add_log(
                f"HFExtractor: {len(hf_results)} fields in {t_hf_elapsed:.0f}ms."
            )
            if _v:
                logger.debug("           → %d fields in %.0fms", len(hf_results), t_hf_elapsed)
            if self.transparent:
                self._log_extraction_dict("Step8 hf_results", hf_results)

        # =================================================================
        # MERGE & DECISION  (Steps 9–10)
        # =================================================================

        # -----------------------------------------------------------------
        # Step 9: Merge rules branch → rules_merged
        # Priority: date > controlled > eds_results > rule_results
        # -----------------------------------------------------------------
        if _v:
            logger.debug("[Step 9/14] Merging rules branch...")
        rules_merged: dict[str, ExtractionValue] = _merge_rules(
            rule_results, controlled_results, date_results, eds_results
        )
        result.rule_results = rule_results
        result.rules_merged = rules_merged
        result.date_results = date_results
        result.controlled_results = controlled_results
        result.eds_results = eds_results
        result.hf_results = hf_results
        result.add_log(
            f"Rules merged: {len(rules_merged)} fields "
            f"(date={len(date_results)}, controlled={len(controlled_results)}, "
            f"eds={len(eds_results)}, rules={len(rule_results)})."
        )
        if _v:
            logger.debug("           → %d fields in rules_merged", len(rules_merged))
        if self.transparent:
            self._log_extraction_dict("Step9 rules_merged", rules_merged)

        # -----------------------------------------------------------------
        # Step 10: Decide HF vs Rules → final merged features
        # -----------------------------------------------------------------
        if _v:
            logger.debug("[Step 10/14] Deciding HF vs Rules per field...")
        merged: dict[str, ExtractionValue] = _decide_extraction(
            hf_results, rules_merged, self.use_eds, self.use_rules
        )
        result.features = merged
        result.tier1_count = len(merged)
        result.add_log(
            f"Decision: {len(merged)} total features selected "
            f"(mode: rules={self.use_rules}, hf={self.use_eds})."
        )
        if self.transparent:
            self._log_extraction_dict("Step10 decided_features", merged)
        if _v:
            logger.debug("           → %d total features after decision", len(merged))
            if merged:
                logger.debug("           ▼ Extracted Values Breakdown:")
                header = f"{'Field Name':<28} | {'Value':<30} | {'Source':<11} | {'Conf'}"
                sep = f"{'-'*28}-+-{'-'*30}-+-{'-'*11}-+-{'-'*4}"
                logger.debug(header)
                logger.debug(sep)
                for fname, ev in sorted(merged.items()):
                    if fname in date_results:
                        src = "Date"
                    elif fname in controlled_results:
                        src = "Controlled"
                    elif fname in eds_results:
                        src = "EDS"
                    elif fname in rule_results:
                        src = "Rules"
                    elif fname in hf_results:
                        src = "HF"
                    else:
                        src = "Unknown"
                    val_repr = repr(ev.value)
                    if len(val_repr) > 30:
                        val_repr = val_repr[:27] + "..."
                    conf_str = f"{ev.confidence:.2f}" if ev.confidence is not None else "N/A"
                    logger.debug(
                        "%s | %s | %s | %s",
                        f"{fname:<28}", f"{val_repr:<30}", f"{src:<11}", conf_str,
                    )

        # -----------------------------------------------------------------
        # Step 11: Negation detection
        # -----------------------------------------------------------------
        if self.use_negation and self._assertion_annotator:
            if _v:
                logger.debug("[Step 11/14] Applying negation detection...")
            # Collect (start, end, field_name) for all spans with a value
            spans_to_check: list[tuple[int, int, str]] = []
            span_fields: list[str] = []
            for fname, ev in merged.items():
                if ev.source_span and ev.value is not None:
                    span_start = text.find(ev.source_span)
                    if span_start == -1:
                        continue
                    spans_to_check.append(
                        (span_start, span_start + len(ev.source_span), fname)
                    )
                    span_fields.append(fname)

            if spans_to_check:
                try:
                    annotations = self._assertion_annotator.annotate(
                        text, spans_to_check
                    )
                    negated_count = 0
                    for ann, fname in zip(annotations, span_fields):
                        if ann.is_negated:
                            ev = merged[fname]
                            original_val = ev.value
                            ev.value = _apply_negation(fname, ev.value, ev.source_span)
                            result.add_log(
                                f"Negation: '{fname}' '{original_val}' → '{ev.value}'"
                            )
                            if self.transparent:
                                logger.debug(
                                    "  [Step11 negation] %-32s %r → %r  (span: %r)",
                                    fname, original_val, ev.value,
                                    (ev.source_span or "")[:60],
                                )
                            negated_count += 1
                        elif self.transparent and ann.is_hypothesis:
                            logger.debug(
                                "  [Step11 hypothesis] %-32s %r  (span: %r)",
                                fname, merged[fname].value,
                                (merged[fname].source_span or "")[:60],
                            )
                    if negated_count:
                        result.add_log(
                            f"Negation: {negated_count} field(s) inverted."
                        )
                    elif _v:
                        logger.debug("           → no negated spans found")
                except Exception as exc:
                    result.add_log(f"Negation step failed: {exc}")
                    logger.error("Negation step failed: %s", exc)

        # -----------------------------------------------------------------
        # Step 12: Validate against controlled vocabularies
        # -----------------------------------------------------------------
        if _v:
            logger.debug("[Step 12/14] Validating against controlled vocabularies...")
        validate_extraction(merged)
        vocab_flagged = [
            fname for fname, ev in merged.items()
            if not ev.vocab_valid
        ]
        if vocab_flagged:
            result.add_log(
                f"Vocabulary validation: {len(vocab_flagged)} fields flagged: "
                f"{vocab_flagged}"
            )
        else:
            result.add_log("Vocabulary validation: all values valid.")
        if _v:
            logger.debug("           → %d fields flagged out of vocabulary", len(vocab_flagged))
        if self.transparent and vocab_flagged:
            for fname in vocab_flagged:
                ev = merged[fname]
                logger.debug(
                    "  [Step12 vocab_invalid] %-32s value=%r", fname, ev.value
                )

        # -----------------------------------------------------------------
        # Step 13: Validate source spans
        # -----------------------------------------------------------------
        if _v:
            logger.debug("[Step 13/14] Validating source spans...")
        validate_source_spans(merged, text)
        span_flagged = [
            fname for fname, ev in merged.items()
            if ev.flagged and fname not in vocab_flagged
        ]
        if span_flagged:
            result.add_log(
                f"Source span validation: {len(span_flagged)} additional "
                f"fields flagged: {span_flagged}"
            )
        else:
            result.add_log("Source span validation: all spans verified.")
        if _v:
            logger.debug("           → %d additional fields flagged", len(span_flagged))

        # -----------------------------------------------------------------
        # Step 14: Build flagged-for-review list
        # -----------------------------------------------------------------
        if _v:
            logger.debug("[Step 14/14] Building flagged-for-review list...")
        result.update_flagged_from_features()
        result.add_log(
            f"Total flagged for review: {len(result.flagged_for_review)} fields."
        )

        # Finalise timing
        result.total_extraction_time_ms = (
            time.perf_counter() - t_start
        ) * 1000
        result.add_log(
            f"Pipeline completed in {result.total_extraction_time_ms:.0f}ms."
        )
        if _v:
            logger.debug(
                "           → %d fields flagged for review", len(result.flagged_for_review)
            )
            logger.debug(
                "[PIPELINE] ✓ Completed in %.0fms, %d features extracted",
                result.total_extraction_time_ms, len(merged),
            )
            logger.debug("=" * 60)

        return result

    # -- Batch processing ----------------------------------------------------

    def extract_batch(
        self,
        documents: list[dict],
        n_jobs: int | None = None,
    ) -> list[ExtractionResult]:
        """Process a list of documents, with HF-aware batching.

        When ``self._hf_extractor`` is active, all HuggingFace models are
        loaded exactly **once** for the whole batch (one per group), results
        are pre-computed, then the per-document pipeline runs sequentially
        with the pre-computed HF results injected (bypassing Step 8).

        When only the RULES branch is active, the existing
        ``ProcessPoolExecutor`` multiprocessing path is used.

        Parameters
        ----------
        documents : list[dict]
            Each dict must have ``'text'``, and optionally
            ``'document_id'``, ``'patient_id'``, and ``'consultation_date'``.
        n_jobs : int, optional
            Number of worker processes for the rules-only path.
            Ignored when HFExtractor is active.

        Returns
        -------
        list[ExtractionResult]
            One ``ExtractionResult`` per input document, preserving the input order.
        """
        n = len(documents)
        if n == 0:
            return []

        # ---------------------------------------------------------------
        # HF-aware path: pre-compute all HF results in one pass, then
        # iterate sequentially (GPU batching >>> CPU parallelism here).
        # ---------------------------------------------------------------
        if self._hf_extractor is not None:
            logger.info(
                "Starting HF-aware sequential batch extraction of %d documents", n
            )
            texts = [d.get("text", "") for d in documents]
            try:
                all_hf: list[dict[str, ExtractionValue]] = (
                    self._hf_extractor.extract_batch(texts)
                )
            except Exception as exc:
                logger.error("HFExtractor.extract_batch failed: %s, falling back to empty", exc)
                all_hf = [{} for _ in documents]

            results: list[ExtractionResult] = []
            for i, doc in enumerate(documents):
                doc_id = doc.get("document_id", f"doc_{i}")
                logger.info("Processing document %d/%d: %s", i + 1, n, doc_id)
                try:
                    result = self.extract_document(
                        text=doc.get("text", ""),
                        document_id=doc_id,
                        patient_id=doc.get("patient_id", ""),
                        consultation_date=doc.get("consultation_date"),
                        _precomputed_hf=all_hf[i],
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to process document '%s': %s", doc_id, exc)
                    result = ExtractionResult(document_id=doc_id, patient_id=doc.get("patient_id", ""))
                    result.add_log(f"Pipeline failed with error: {exc}")
                results.append(result)
            return results

        # ---------------------------------------------------------------
        # Rules-only path: existing sequential / multiprocessing logic.
        # ---------------------------------------------------------------
        n_jobs_to_use = n_jobs if n_jobs is not None else self.n_jobs

        def run_sequential() -> list[ExtractionResult]:
            """Helper to run the extraction sequentially as a fallback or baseline."""
            logger.info("Starting sequential batch extraction of %d documents", n)
            results: list[ExtractionResult] = []
            for i, doc in enumerate(documents):
                text = doc.get("text", "")
                doc_id = doc.get("document_id", f"doc_{i}")
                patient_id = doc.get("patient_id", "")
                consultation_date = doc.get("consultation_date")

                logger.info("Processing document %d/%d: %s", i + 1, n, doc_id)

                try:
                    result = self.extract_document(
                        text=text,
                        document_id=doc_id,
                        patient_id=patient_id,
                        consultation_date=consultation_date,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to process document '%s': %s", doc_id, exc)
                    result = ExtractionResult(document_id=doc_id, patient_id=patient_id)
                    result.add_log(f"Pipeline failed with error: {exc}")

                results.append(result)
            return results

        if n_jobs_to_use == -1:
            n_jobs_to_use = max(1, (os.cpu_count() or 1) - 2)

        # Si 0, 1 seul worker demandé, ou un seul doc fourni, on exécute séquentiellement
        if n_jobs_to_use <= 1 or n == 1:
            return run_sequential()

        # --- Parallélisation via Multiprocessing ---
        logger.info("Starting parallel batch extraction of %d documents with %d workers", n, n_jobs_to_use)

        # On extrait la configuration pour l'injecter dans chaque worker process
        pipeline_kwargs = {
            "use_negation": self.use_negation,
            "use_eds": self.use_eds,
            "use_rules": self.use_rules,
            "enabled_rule_extractors": self.enabled_rule_extractors,
            "enabled_groups": self.enabled_groups,
            "local_model_dir": self.local_model_dir,
            "verbose": self.verbose,
            "transparent": self.transparent,
            "n_jobs": 1  # Inside the worker, we don't spawn more parallel jobs
        }

        try:
            # L'index permet de restituer les résultats exactement dans l'ordre de la liste `documents`
            with ProcessPoolExecutor(
                max_workers=n_jobs_to_use,
                initializer=_init_worker,
                initargs=(pipeline_kwargs,)
            ) as executor:
                
                # Map exécute la fonction avec les tuples (index, dict_document)
                mapped_results = list(executor.map(
                    _process_single_doc, 
                    list(range(n)), 
                    documents
                ))

            # Re-ordonnancement explicite à partir des index retournés
            final_results = [None] * n
            for idx, res, captured_out in mapped_results:
                final_results[idx] = res
                # Si des 'print' (verbose=True) ont eu lieu dans le processus de travail,
                # on les imprime d'un coup ici pour que le DualLogger (ou sys.stdout) le capture
                if captured_out:
                    print(captured_out, end="")

            return final_results

        except Exception as parallel_exc:
            # Fallback en cas d'échec du multiprocessing (ex: erreur de pickling, manque de mémoire, etc.)
            error_msg = f"Parallelization failed ({type(parallel_exc).__name__}: {parallel_exc}). Falling back to sequential execution."
            logger.warning(error_msg)
            
            return run_sequential()

if __name__ == "__main__":
    # Toy example to demonstrate pipeline execution
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("--- Initializing Pipeline ---")
    pipeline = ExtractionPipeline(verbose=True, n_jobs=1)
    
    sample_text = (
        """
[PHI_VILLE_00C6D50AEF]
[PHI_DATE_NAISSANCE_0468266DDC]
Références : MTO/MRO
Compte-Rendu de consultation du [DATE_ABA2C99D41]
Monsieur [PERSON_FBBA500A85] [PERSON_97E34FEF2C], né le [PHI_DATE_NAISSANCE_9688CD0E2B], âgé de 48 ans, a été vu en consultation.
Motif
Prise en charge d'un glioblastome pariétal gauche IDH 1/2 non muté
Antécédents et allergies
Allergies :
- Pas d'allergie connue
Antécédents familiaux :
- Grand-mère paternelle : glioblastome
Père : cancer prostate
Mère : [PERSON_A2BE91B1B5]
2 frères : [PERSON_9BF0B566F1], cancer testicule
2 enfants : [PERSON_A2BE91B1B5]
- Biologie ([DATE_2887B29EC2]) : Hb 15.9 g/dl, PNN 2830/mm 3, lymphocytes 2500/mm 3, plaquettes
171763.000/mm 3, créatininémie 87 µmol/l, iono, bilan hépatique sans anomalie notable.
- IRM cérébrale ([DATE_EE0A5608C5]) : poursuite d'une lente majoration de la lésion rehaussée pariétale
gauche, dont les caractéristiques perfusionnelles sont en faveur d'une pseudo-progression (sous
réserve d'une perfusion ASL non interprétable) : rupture de barrière sans néoangiogenèse en
perfusion DSC, ni signe d'hypercellularité.
Imprimé le [DATE_EB2281C628] 16:59
CR CONSULTATION PSL CONSULT NEURO-ONCO
Pat.: [PERSON_97E34FEF2C] [PERSON_FBBA500A85] | M | [PHI_DATE_NAISSANCE_9688CD0E2B] | INS/NIR : [ID_2C2E3C6F13] | [PHI_NDA_F33890BBBC] | [PHI_NDA_F4416A9037]
Courrier non validé
* Prochaine cs avec IRM cérébrale dans 2 mois.
Consultation du [DATE_ABA2C99D41] :
- Vu avec son épouse.
- Stabilité des troubles phasiques résiduels (paraphasies, manque du mot, compréhension
préservée). Pas de signe d'HTIC. Pas de crise comitiale récente.
- Pas de fièvre ni évènement intercurrent récent.
- Cliniquement, KPS 90%, 95 kg, 193 cm. Examen neurologique : orientation temporo-spatiale
normale, discrètes paraphasies, marche possible sans aide, pas de déficit moteur ni sensitif,
quadranopsie latérale homonyme inférieure droite partielle, pas de syndrome cérébelleux ni
atteinte des autres paires crâniennes. Examen général sans particularité.
- Biologie ([DATE_4F3012EC77]) : NFS, iono, bilan hépatique sans anomalie notable.
- IRM cérébrale ([DATE_0F861C83AC]) : très minime majoration de la plage hypersignal FLAIR péri-
cavitaire le long du corps calleux à gauche. Stabilité du petit rehaussement punctiforme
précédemment apparu.
- Conclusion :
* Stabilité clinique et radiologique.
* Prochaine cs avec IRM cérébrale dans 2 mois.
Examen clinique
Poids : 93 kg ([DATE_135A327D58]), Taille : 193 cm ([DATE_135A327D58]), IMC : 25 kg/m 2, SC 2.2 m 2
Rendez-vous pris
Planification des soins / Suites à donner
- [DATE_62751A3490] à 14:30 : ([PHI_HOPITAL_A994C995ED])
"""
    )
    
    print("\n--- Running Extraction ---")
    result = pipeline.extract_document(
        text=sample_text,
        document_id="toy_doc_001",
        patient_id="toy_pat_123",
        consultation_date="12/05/2023"
    )
    
    print("\n--- Final Output Summary ---")
    print(f"Document ID: {result.document_id}")
    print(f"Patient ID: {result.patient_id}")
    print(f"Document Type: {result.document_type}")
    print(f"Total Features Extracted: {len(result.features)}")
    print(f"Flagged for Review: {len(result.flagged_for_review)} fields")