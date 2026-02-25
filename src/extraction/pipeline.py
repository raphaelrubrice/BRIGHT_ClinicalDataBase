"""Main extraction pipeline orchestrator.

Wires together document classification, section detection, Tier 1
rule-based extraction, Tier 2 LLM extraction, validation, and
provenance tracking into a single ``extract_document()`` function.

Public API
----------
- ``ExtractionPipeline``  – Main pipeline class.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from .document_classifier import DocumentClassifier
from .llm_extraction import run_llm_extraction, validate_source_spans
from .negation import AssertionAnnotator
from .ollama_client import OllamaClient, OllamaError
from .provenance import ExtractionResult
from .rule_extraction import run_rule_extraction
from .schema import (
    ExtractionValue,
    FEATURE_ROUTING,
    get_extractable_fields,
)
from .section_detector import SectionDetector
from .validation import validate_extraction

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """End-to-end clinical feature extraction pipeline.

    Orchestrates:
    1. Document type classification
    2. Section detection
    3. Feature routing (document type → extractable fields)
    4. Tier 1 rule-based extraction
    5. Assertion annotation (negation / hypothesis / history)
    6. Tier 2 LLM extraction (for remaining fields)
    7. Merge (Tier 1 takes precedence)
    8. Controlled vocabulary validation
    9. Source span validation
    10. Provenance record + flagging

    Parameters
    ----------
    ollama_model : str
        Ollama model name (default ``"qwen3:4b-instruct"``).
    ollama_base_url : str
        Ollama server URL.
    use_llm : bool
        Whether to enable Tier 2 LLM extraction. When ``False``,
        only Tier 1 (rule-based) extraction is performed.
    use_negation : bool
        Whether to enable negation / hypothesis / history annotation.
    """

    def __init__(
        self,
        ollama_model: str = "qwen3:4b-instruct",
        ollama_base_url: str = "http://localhost:11434",
        use_llm: bool = True,
        use_negation: bool = True,
    ):
        self.use_llm = use_llm
        self.use_negation = use_negation

        # Sub-components
        self.classifier = DocumentClassifier()
        self.section_detector = SectionDetector()

        # Assertion annotator (negation / hypothesis / history)
        self._annotator: Optional[AssertionAnnotator] = None
        if use_negation:
            try:
                self._annotator = AssertionAnnotator()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to initialise AssertionAnnotator: %s. "
                    "Negation detection will be disabled.",
                    exc,
                )

        # Ollama client (initialised lazily to avoid startup overhead)
        self._ollama_client: Optional[OllamaClient] = None
        self._ollama_model = ollama_model
        self._ollama_base_url = ollama_base_url

    # -- Lazy Ollama client --------------------------------------------------

    def _get_ollama_client(self) -> Optional[OllamaClient]:
        """Return the OllamaClient, creating it on first use."""
        if not self.use_llm:
            return None

        if self._ollama_client is None:
            try:
                self._ollama_client = OllamaClient(
                    model=self._ollama_model,
                    base_url=self._ollama_base_url,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to create OllamaClient: %s", exc)
                return None

        return self._ollama_client

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
        # Step 3: Determine extractable feature subset
        # -----------------------------------------------------------------
        try:
            feature_subset = get_extractable_fields(result.document_type)
        except ValueError as exc:
            result.add_log(f"Feature routing error: {exc}. Using full feature set.")
            # Fall back to all fields
            from .schema import ALL_BIO_FIELD_NAMES, ALL_CLINIQUE_FIELD_NAMES
            feature_subset = sorted(
                set(ALL_BIO_FIELD_NAMES + ALL_CLINIQUE_FIELD_NAMES)
            )

        result.add_log(
            f"Feature subset: {len(feature_subset)} fields to extract."
        )

        # -----------------------------------------------------------------
        # Step 4: Tier 1 — Rule-based extraction
        # -----------------------------------------------------------------
        t_tier1_start = time.perf_counter()
        tier1_results = run_rule_extraction(
            text=text,
            sections=sections,
            feature_subset=feature_subset,
            annotator=self._annotator,
        )
        t_tier1_elapsed = (time.perf_counter() - t_tier1_start) * 1000

        result.tier1_count = len(tier1_results)
        result.add_log(
            f"Tier 1 (rule-based): extracted {len(tier1_results)} fields "
            f"in {t_tier1_elapsed:.0f}ms."
        )

        # -----------------------------------------------------------------
        # Step 5: Determine remaining unextracted features
        # -----------------------------------------------------------------
        remaining = set(feature_subset) - set(tier1_results.keys())
        result.add_log(
            f"Remaining after Tier 1: {len(remaining)} fields."
        )

        # -----------------------------------------------------------------
        # Step 6: Tier 2 — LLM extraction (for remaining features)
        # -----------------------------------------------------------------
        tier2_results: dict[str, ExtractionValue] = {}

        if remaining and self.use_llm:
            client = self._get_ollama_client()
            if client is not None:
                t_tier2_start = time.perf_counter()
                try:
                    tier2_results = run_llm_extraction(
                        text=text,
                        sections=sections,
                        feature_subset=list(remaining),
                        already_extracted=tier1_results,
                        client=client,
                    )
                except Exception as exc:  # noqa: BLE001
                    result.add_log(
                        f"Tier 2 LLM extraction failed: {exc}"
                    )
                    logger.error("Tier 2 LLM extraction failed: %s", exc)

                t_tier2_elapsed = (time.perf_counter() - t_tier2_start) * 1000
                result.tier2_count = len(tier2_results)
                result.add_log(
                    f"Tier 2 (LLM): extracted {len(tier2_results)} fields "
                    f"in {t_tier2_elapsed:.0f}ms."
                )
            else:
                result.add_log(
                    "Tier 2 (LLM) skipped: Ollama client unavailable."
                )
        elif not remaining:
            result.add_log("Tier 2 (LLM) skipped: all features extracted by Tier 1.")
        else:
            result.add_log("Tier 2 (LLM) skipped: LLM extraction disabled.")

        # -----------------------------------------------------------------
        # Step 7: Merge Tier 1 + Tier 2 (Tier 1 takes precedence)
        # -----------------------------------------------------------------
        merged: dict[str, ExtractionValue] = {}

        # Tier 1 always wins on conflicts
        merged.update(tier1_results)

        # Add Tier 2 results only for fields not in Tier 1
        for fname, ev in tier2_results.items():
            if fname not in merged:
                merged[fname] = ev

        result.features = merged
        result.add_log(
            f"Merged: {len(merged)} total features "
            f"({result.tier1_count} Tier 1 + {result.tier2_count} Tier 2)."
        )

        # -----------------------------------------------------------------
        # Step 8: Validate against controlled vocabularies
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
        # Step 9: Validate source spans
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
        # Step 10: Build flagged-for-review list
        # -----------------------------------------------------------------
        result.update_flagged_from_features()
        result.add_log(
            f"Total flagged for review: {len(result.flagged_for_review)} fields."
        )

        # -----------------------------------------------------------------
        # Step 11: Extract document date (if available from Tier 1)
        # -----------------------------------------------------------------
        # Look for date fields in extracted features
        for date_field in ("date_chir", "dn_date", "chir_date"):
            ev = merged.get(date_field)
            if ev and ev.value:
                result.document_date = str(ev.value)
                break

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
