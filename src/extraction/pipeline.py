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
import time
from typing import Optional

from .document_classifier import DocumentClassifier
from .provenance import ExtractionResult
from .eds_extractor import EDSExtractor
from .gliner_extractor import GlinerExtractor
from .negation import AssertionAnnotator
from .rule_extraction import run_rule_extraction
from .schema import (
    ExtractionValue,
    FEATURE_ROUTING,
    ALL_FIELDS_BY_NAME,
    get_extractable_fields,
)
from .section_detector import SectionDetector
from .validation import validate_extraction, validate_source_spans

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Negation value flipping
# ---------------------------------------------------------------------------

def _flip_negated_value(field_name: str, value) -> str:
    """Flip an extracted value when negation is detected.

    Instead of discarding the entity, we invert it:
    - Binary (oui/non) → flip
    - IHC (positif/negatif/maintenu) → flip
    - Molecular (mute/wt) → flip
    - Other → prefix with "non"
    """
    val_str = str(value).strip().lower() if value is not None else ""

    # Binary fields
    if val_str == "oui":
        return "non"
    if val_str == "non":
        return "oui"

    # IHC fields
    if field_name.startswith("ihc_"):
        if val_str in ("positif", "positive", "maintenu"):
            return "negatif"
        if val_str in ("negatif", "negative"):
            return "positif"

    # Molecular fields
    if field_name.startswith("mol_"):
        if val_str in ("mute", "muté", "mutée"):
            return "wt"
        if val_str == "wt":
            return "mute"
        if val_str in ("methyle", "méthylé"):
            return "non methyle"
        if val_str in ("non methyle", "non méthylé"):
            return "methyle"

    # Chromosomal fields
    if field_name.startswith("ch") and field_name not in (
        "chir_date", "chimios", "chm_date_debut", "chm_date_fin", "chm_cycles",
    ):
        if val_str == "gain":
            return "perte"
        if val_str in ("perte", "perte partielle"):
            return "gain"

    # Amplification / fusion fields
    if field_name.startswith("ampli_") or field_name.startswith("fusion_"):
        if val_str == "oui":
            return "non"
        if val_str == "non":
            return "oui"

    return val_str


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
    """

    def __init__(
        self,
        use_negation: bool = True,
        use_eds: bool = True,
        use_gliner: bool = True,
        gliner_model: str = "urchade/gliner_multi-v2.1",
    ):
        self.use_negation = use_negation
        self.use_eds = use_eds
        self.use_gliner = use_gliner

        # Sub-components
        self.classifier = DocumentClassifier()
        self.section_detector = SectionDetector()

        self._eds_extractor = EDSExtractor() if use_eds else None

        self._gliner_extractor = None
        if use_gliner:
            self._gliner_extractor = GlinerExtractor(model_name=gliner_model)

        self._assertion_annotator = AssertionAnnotator() if use_negation else None

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

        Returns
        -------
        ExtractionResult
            Complete extraction result with features, provenance, and
            audit trail.
        """
        t_start = time.perf_counter()

        result = ExtractionResult(
            document_id=document_id,
            patient_id=patient_id,
        )
        result.add_log(f"Pipeline started for document '{document_id}'.")

        # -----------------------------------------------------------------
        # Step 1: Classify document type
        # -----------------------------------------------------------------
        classification = self.classifier.classify(text)
        result.document_type = classification.document_type
        result.classification_confidence = classification.confidence
        result.classification_is_ambiguous = classification.is_ambiguous
        result.add_log(
            f"Document classified as '{classification.document_type}' "
            f"(confidence={classification.confidence:.2f}, "
            f"ambiguous={classification.is_ambiguous})."
        )

        if classification.is_ambiguous:
            result.add_log(
                "Classification is ambiguous — extracted features may be "
                "incomplete. Consider manual review."
            )

        # -----------------------------------------------------------------
        # Step 2: Detect sections
        # -----------------------------------------------------------------
        sections = self.section_detector.detect(text)
        result.sections_detected = list(sections.keys())
        result.add_log(
            f"Sections detected: {result.sections_detected}"
        )

        # -----------------------------------------------------------------
        # Step 3: Language detection
        # -----------------------------------------------------------------
        language = self._detect_language(text)
        result.add_log(f"Detected language: '{language}'.")

        # -----------------------------------------------------------------
        # Step 4: Determine extractable feature subset
        # -----------------------------------------------------------------
        try:
            feature_subset = get_extractable_fields(result.document_type)
        except ValueError as exc:
            result.add_log(f"Feature routing error: {exc}. Using full feature set.")
            from .schema import ALL_BIO_FIELD_NAMES, ALL_CLINIQUE_FIELD_NAMES
            feature_subset = sorted(
                set(ALL_BIO_FIELD_NAMES + ALL_CLINIQUE_FIELD_NAMES)
            )

        result.add_log(
            f"Feature subset: {len(feature_subset)} fields to extract."
        )

        # -----------------------------------------------------------------
        # Step 5: PRIMARY — GLiNER extraction (runs FIRST)
        # -----------------------------------------------------------------
        gliner_results: dict[str, ExtractionValue] = {}
        if self._gliner_extractor:
            t_gliner_start = time.perf_counter()
            try:
                gliner_results = self._gliner_extractor.extract(
                    text=text,
                    feature_subset=feature_subset,
                    language=language,
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

        # -----------------------------------------------------------------
        # Step 6: EDS-NLP Level 1 — Qualifier identification for GLiNER
        # -----------------------------------------------------------------
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
                    ev.value = _flip_negated_value(field_name, ev.value)
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

        # -----------------------------------------------------------------
        # Step 7: EDS-NLP Level 2 / Rules — Standalone extraction
        # -----------------------------------------------------------------
        t_eds_start = time.perf_counter()
        if self.use_eds and self._eds_extractor:
            eds_results = self._eds_extractor.extract(
                text=text,
                sections=sections,
                feature_subset=feature_subset,
            )
            result.add_log("EDS-NLP Level 2: standalone extraction.")
        else:
            eds_results = run_rule_extraction(
                text=text,
                sections=sections,
                feature_subset=feature_subset,
            )
            result.add_log("EDS-NLP Level 2: Regex fallback (legacy logic).")
        t_eds_elapsed = (time.perf_counter() - t_eds_start) * 1000

        result.tier1_count = len(eds_results)
        result.add_log(
            f"EDS/Rules: extracted {len(eds_results)} fields "
            f"in {t_eds_elapsed:.0f}ms."
        )

        # -----------------------------------------------------------------
        # Step 8: Merge GLiNER + EDS (GLiNER precedence)
        # -----------------------------------------------------------------
        merged: dict[str, ExtractionValue] = {}

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
                else:
                    # GLiNER takes precedence
                    merged[fname] = g
            elif g:
                merged[fname] = g
            else:
                # EDS fallback for fields GLiNER missed
                merged[fname] = e

        result.features = merged
        result.add_log(
            f"Merged: {len(merged)} total features "
            f"({result.gliner_count} GLiNER + {result.tier1_count} EDS/Rules)."
        )

        # -----------------------------------------------------------------
        # Step 9: Validate against controlled vocabularies
        # -----------------------------------------------------------------
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

        # -----------------------------------------------------------------
        # Step 10: Validate source spans
        # -----------------------------------------------------------------
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

        # -----------------------------------------------------------------
        # Step 11: Build flagged-for-review list
        # -----------------------------------------------------------------
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

        return result

    # -- Batch processing ----------------------------------------------------

    def extract_batch(
        self,
        documents: list[dict],
    ) -> list[ExtractionResult]:
        """Process a list of documents.

        Parameters
        ----------
        documents : list[dict]
            Each dict must have ``'text'``, and optionally
            ``'document_id'`` and ``'patient_id'``.

        Returns
        -------
        list[ExtractionResult]
            One ``ExtractionResult`` per input document.
        """
        results: list[ExtractionResult] = []

        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            doc_id = doc.get("document_id", f"doc_{i}")
            patient_id = doc.get("patient_id", "")

            logger.info("Processing document %d/%d: %s", i + 1, len(documents), doc_id)

            try:
                result = self.extract_document(
                    text=text,
                    document_id=doc_id,
                    patient_id=patient_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to process document '%s': %s", doc_id, exc)
                result = ExtractionResult(
                    document_id=doc_id,
                    patient_id=patient_id,
                )
                result.add_log(f"Pipeline failed with error: {exc}")

            results.append(result)

        return results
