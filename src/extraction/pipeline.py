"""Main extraction pipeline orchestrator — GLiNER-first architecture.

Wires together document classification, section detection, GLiNER primary
extraction, EDS-NLP qualifier identification and standalone extraction,
validation, and provenance tracking into a single ``extract_document()``
function.

Public API
----------
- ``ExtractionPipeline``  – Main pipeline class.
"""

from __future__ import annotations

import logging
import os
import time
import io
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from .date_extractor import DateExtractor
from .document_classifier import DocumentClassifier
from .provenance import ExtractionResult
from .eds_extractor import EDSExtractor
from .gliner_extractor import GlinerExtractor
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
    "positif": "negatif",
    "negatif": "positif",
    "positive": "negatif",
    "negative": "positif",
    "maintenu": "negatif",
    "mute": "wt",
    "muté": "wt",
    "mutée": "wt",
    "wt": "mute",
    "methyle": "non methyle",
    "méthylé": "non methyle",
    "non methyle": "methyle",
    "non méthylé": "methyle",
    "gain": "perte",
    "perte": "gain",
    "perte partielle": "gain",
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
# Multiprocessing Globals & Worker Functions
# ---------------------------------------------------------------------------

_WORKER_PIPELINE: ExtractionPipeline | None = None

def _init_worker(pipeline_kwargs: dict):
    """Initialise a global pipeline instance for each worker process.
    This avoids pickling heavy ML models (GLiNER, spaCy) during cross-process calls.
    """
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
    """End-to-end clinical feature extraction pipeline (GLiNER-first).

    Orchestrates:
    1. Document type classification
    2. Section detection
    3. Language detection
    4. Feature routing (document type → extractable fields)
    5. Primary extraction via GLiNER (all fields)
    6. EDS-NLP Level 1: qualifier identification for GLiNER entities
    7. EDS-NLP Level 2 / Rules: standalone extraction
    8. Merge (GLiNER precedence, synergy boost, EDS fallback)
    9. Controlled vocabulary validation
    10. Source span validation
    11. Provenance record + flagging

    Parameters
    ----------
    use_negation : bool
        Whether to enable negation / hypothesis / history annotation.
    use_eds : bool
        Whether to enable EDS-NLP standalone extraction.
    use_gliner : bool
        Whether to enable GLiNER primary extraction.
    gliner_model : str
        GLiNER model name (default ``"urchade/gliner_multi-v2.1"``).
    batching_strategy : str
        Batching strategy to use in GLiNER (default ``"heterogeneous"``).
    verbose : bool
        Whether to print step-by-step progress to stdout (default ``False``).
    n_jobs : int
        Number of workers for parallel processing (-1 means max CPUs - 2). Default is -1.
    """

    def __init__(
        self,
        use_negation: bool = True,
        use_eds: bool = True,
        use_gliner: bool = True,
        gliner_model: str = "urchade/gliner_multi-v2.1",
        batching_strategy: str = "heterogeneous",
        verbose: bool = False,
        n_jobs: int = -1,
    ):
        self.use_negation = use_negation
        self.use_eds = use_eds
        self.use_gliner = use_gliner
        self.gliner_model = gliner_model
        self.batching_strategy = batching_strategy
        self.verbose = verbose
        self.n_jobs = n_jobs

        # Sub-components
        self.classifier = DocumentClassifier()
        self.section_detector = SectionDetector()

        self._eds_extractor = EDSExtractor() if use_eds else None

        self._gliner_extractor = None
        if use_gliner:
            self._gliner_extractor = GlinerExtractor(
                model_name=gliner_model,
                batching_strategy=batching_strategy,
            )

        self._assertion_annotator = AssertionAnnotator() if use_negation else None
        self._date_extractor = DateExtractor()

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
            print(f"\n{'='*60}")
            print(f"[PIPELINE] Starting extraction for document '{document_id}'")
            print(f"{'='*60}")

        # -----------------------------------------------------------------
        # Step 1: Classify document type
        # -----------------------------------------------------------------
        if _v:
            print("[Step 1/11] Classifying document type...")
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
            print(f"           → type='{classification.document_type}', "
                  f"confidence={classification.confidence:.2f}, "
                  f"ambiguous={classification.is_ambiguous}")

        if classification.is_ambiguous:
            result.add_log(
                "Classification is ambiguous — extracted features may be "
                "incomplete. Consider manual review."
            )

        # -----------------------------------------------------------------
        # Step 2: Detect sections
        # -----------------------------------------------------------------
        if _v:
            print("[Step 2/11] Detecting sections...")
        sections = self.section_detector.detect(text)
        result.sections_detected = list(sections.keys())
        result.add_log(
            f"Sections detected: {result.sections_detected}"
        )
        if _v:
            print(f"           → {len(result.sections_detected)} sections: {result.sections_detected}")

        # -----------------------------------------------------------------
        # Step 3: Language detection
        # -----------------------------------------------------------------
        if _v:
            print("[Step 3/11] Detecting language...")
        language = self._detect_language(text)
        result.add_log(f"Detected language: '{language}'.")
        if _v:
            print(f"           → language='{language}'")

        # -----------------------------------------------------------------
        # Step 4: Determine extractable feature subset
        # -----------------------------------------------------------------
        if _v:
            print("[Step 4/11] Routing features for document type...")
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
            print(f"           → {len(feature_subset)} fields targeted "
                  f"({len(date_fields)} date, {len(non_date_fields)} non-date)")

        # -----------------------------------------------------------------
        # Step 5: Date extraction (dedicated DateExtractor)
        # -----------------------------------------------------------------
        if _v:
            print("[Step 5/12] Running dedicated date extraction...")
        date_results: dict[str, ExtractionValue] = {}
        if date_fields and self._date_extractor:
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
                f"DateExtractor: extracted {len(date_results)} date fields "
                f"in {t_date_elapsed:.0f}ms."
            )
            if _v:
                print(f"           → {len(date_results)} date fields in {t_date_elapsed:.0f}ms")

        # -----------------------------------------------------------------
        # Step 6: PRIMARY — GLiNER extraction (non-date fields)
        # -----------------------------------------------------------------
        if _v:
            print("[Step 6/12] Running GLiNER primary extraction...")
        gliner_results: dict[str, ExtractionValue] = {}
        if self._gliner_extractor:
            t_gliner_start = time.perf_counter()
            try:
                gliner_results = self._gliner_extractor.extract(
                    text=text,
                    feature_subset=non_date_fields,
                    language=language,
                    verbose=_v,
                )
            except Exception as exc:
                result.add_log(f"GLiNER extraction failed: {exc}")
                logger.error("GLiNER extraction failed: %s", exc)

            t_gliner_elapsed = (time.perf_counter() - t_gliner_start) * 1000
            result.gliner_count = len(gliner_results)
            result.add_log(
                f"GLiNER (primary): extracted {len(gliner_results)} fields "
                f"in {t_gliner_elapsed:.0f}ms."
            )
            if _v:
                print(f"           → {len(gliner_results)} fields extracted in {t_gliner_elapsed:.0f}ms")

        # -----------------------------------------------------------------
        # Step 7: EDS-NLP Level 1 — Qualifier identification for GLiNER
        # -----------------------------------------------------------------
        if _v:
            print("[Step 7/12] Running EDS-NLP qualifier annotation...")
        if self._assertion_annotator and gliner_results:
            qualified_count = 0
            for field_name, ev in list(gliner_results.items()):
                if ev.source_span_start is None or ev.source_span_end is None:
                    continue

                annotations = self._assertion_annotator.annotate(
                    text,
                    [(ev.source_span_start, ev.source_span_end, field_name)],
                )
                if not annotations:
                    continue

                ann = annotations[0]
                if ann.is_negated:
                    ev.value = _apply_negation(field_name, ev.value, ev.source_span)
                    qualified_count += 1
                if ann.is_hypothesis:
                    ev.confidence = round((ev.confidence or 0.5) * 0.7, 4)
                    qualified_count += 1
                if ann.is_history:
                    ev.flagged = True
                    qualified_count += 1

            result.add_log(
                f"EDS-NLP Level 1 (qualifiers): {qualified_count} GLiNER "
                f"entities qualified (negation/hypothesis/history)."
            )
            if _v:
                print(f"           → {qualified_count} entities qualified (negation/hypothesis/history)")

        # -----------------------------------------------------------------
        # Step 8: EDS-NLP Level 2 / Rules — Standalone extraction (non-date)
        # -----------------------------------------------------------------
        if _v:
            print("[Step 8/12] Running EDS-NLP / Rules standalone extraction...")
        t_eds_start = time.perf_counter()
        if self.use_eds and self._eds_extractor:
            eds_results = self._eds_extractor.extract(
                text=text,
                sections=sections,
                feature_subset=non_date_fields,
            )
            result.add_log("EDS-NLP Level 2: standalone extraction.")
        else:
            eds_results = run_rule_extraction(
                text=text,
                sections=sections,
                feature_subset=non_date_fields,
            )
            result.add_log("EDS-NLP Level 2: Regex fallback (legacy logic).")
        t_eds_elapsed = (time.perf_counter() - t_eds_start) * 1000

        result.tier1_count = len(eds_results)
        result.add_log(
            f"EDS/Rules: extracted {len(eds_results)} fields "
            f"in {t_eds_elapsed:.0f}ms."
        )
        if _v:
            print(f"           → {len(eds_results)} fields extracted in {t_eds_elapsed:.0f}ms")

        # -----------------------------------------------------------------
        # Step 9: Merge GLiNER + EDS + Dates (GLiNER precedence for non-date)
        # -----------------------------------------------------------------
        if _v:
            print("[Step 9/12] Merging GLiNER + EDS + Date results...")
        merged: dict[str, ExtractionValue] = {}
        synergy_count = 0

        all_fields = set(list(gliner_results.keys()) + list(eds_results.keys()))
        for fname in all_fields:
            g = gliner_results.get(fname)
            e = eds_results.get(fname)

            if g and e:
                g_val = str(g.value).strip().lower() if g.value is not None else ""
                e_val = str(e.value).strip().lower() if e.value is not None else ""

                if g_val == e_val and g_val != "":
                    # Synergy boost: both agree → boost confidence
                    base_conf = max(g.confidence or 0.0, e.confidence or 0.0)
                    g.confidence = min(1.0, round(base_conf + 0.1, 4))
                    merged[fname] = g
                    synergy_count += 1
                else:
                    # GLiNER takes precedence
                    merged[fname] = g
            elif g:
                merged[fname] = g
            else:
                # EDS fallback for fields GLiNER missed
                merged[fname] = e

        # Inject date results (DateExtractor has sole authority over date fields)
        merged.update(date_results)

        result.features = merged
        result.add_log(
            f"Merged: {len(merged)} total features "
            f"({result.gliner_count} GLiNER + {result.tier1_count} EDS/Rules "
            f"+ {len(date_results)} dates)."
        )
        if _v:
            eds_only = len(merged) - result.gliner_count - len(date_results)
            print(f"           → {len(merged)} total features "
                  f"({result.gliner_count} GLiNER, {max(0, eds_only)} EDS-only fallback, "
                  f"{len(date_results)} dates, {synergy_count} synergy-boosted)")

        # -----------------------------------------------------------------
        # Step 10: Validate against controlled vocabularies
        # -----------------------------------------------------------------
        if _v:
            print("[Step 10/12] Validating against controlled vocabularies...")
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
            print(f"           → {len(vocab_flagged)} fields flagged out of vocabulary")

        # -----------------------------------------------------------------
        # Step 11: Validate source spans
        # -----------------------------------------------------------------
        if _v:
            print("[Step 11/12] Validating source spans...")
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
            print(f"           → {len(span_flagged)} additional fields flagged")

        # -----------------------------------------------------------------
        # Step 12: Build flagged-for-review list
        # -----------------------------------------------------------------
        if _v:
            print("[Step 12/12] Building flagged-for-review list...")
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
            print(f"           → {len(result.flagged_for_review)} fields flagged for review")
            print(f"[PIPELINE] ✓ Completed in {result.total_extraction_time_ms:.0f}ms "
                  f"— {len(merged)} features extracted")
            print(f"{'='*60}\n")

        return result

    # -- Batch processing ----------------------------------------------------

    def extract_batch(
        self,
        documents: list[dict],
        n_jobs: int | None = None,
    ) -> list[ExtractionResult]:
        """Process a list of documents in parallel.

        Parameters
        ----------
        documents : list[dict]
            Each dict must have ``'text'``, and optionally
            ``'document_id'``, ``'patient_id'``, and ``'consultation_date'``.
        n_jobs : int, optional
            Number of worker processes. If None, defaults to `self.n_jobs`.
            If -1, max(1, N_CPU - 2).

        Returns
        -------
        list[ExtractionResult]
            One ``ExtractionResult`` per input document, preserving the input order.
        """
        n = len(documents)
        if n == 0:
            return []

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
            "use_gliner": self.use_gliner,
            "gliner_model": self.gliner_model,
            "batching_strategy": self.batching_strategy,
            "verbose": self.verbose,
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
            error_msg = f"⚠️ Parallelization failed ({type(parallel_exc).__name__}: {parallel_exc}). Falling back to sequential execution..."
            print(error_msg)
            logger.warning(error_msg)
            
            return run_sequential()