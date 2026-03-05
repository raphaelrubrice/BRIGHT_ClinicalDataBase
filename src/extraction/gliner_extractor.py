"""GLiNER-based extractor for unstructured narrative clinical fields.

Extracts complex entities that are difficult to capture with pure rules,
using a fine-tuned GLiNER model (e.g. urchade/gliner_multi-v2.1).
"""

from __future__ import annotations

import logging
from typing import Any

from src.extraction.schema import ExtractionValue

logger = logging.getLogger(__name__)


class GlinerExtractor:
    """Zero-shot/Few-shot entity extractor using GLiNER."""

    GLINER_FIELDS: set[str] = {
        "epilepsie_1er_symptome",
        "deficit_1er_symptome",
        "tumeur_position",
        "activite_professionnelle",
        "antecedent_tumoral",
        "evol_clinique",
        "progress_clinique",
        "progress_radiologique",
        "localisation_chir",
        "localisation_radiotherapie",
        "diag_histologique",
        "histo_necrose",
        "histo_pec",
    }

    _LABEL_MAP: dict[str, str] = {
        "epilepsie_1er_symptome": "crise d'epilepsie ou convulsion inaugurale",
        "deficit_1er_symptome": "deficit neurologique initial",
        "tumeur_position": "localisation anatomique de la tumeur",
        "activite_professionnelle": "profession ou activite professionnelle",
        "antecedent_tumoral": "antecedent de tumeur cerebrale",
        "evol_clinique": "evolution clinique ou progression",
        "progress_clinique": "progression clinique",
        "progress_radiologique": "progression radiologique ou imagerie",
        "localisation_chir": "localisation anatomique de la chirurgie",
        "localisation_radiotherapie": "zone irradiee ou localisation de radiotherapie",
        "diag_histologique": "diagnostic histologique ou type tumoral",
        "histo_necrose": "necrose tumorale",
        "histo_pec": "proliferation endotheliocapillaire",
    }
    
    # Reverse map for quick lookup
    _REVERSE_LABEL_MAP: dict[str, str] = {v: k for k, v in _LABEL_MAP.items()}

    def __init__(
        self,
        model_name: str = "urchade/gliner_multi-v2.1",
        chunk_size: int = 384,
        chunk_overlap: int = 50,
        confidence_threshold: float = 0.5,
    ):
        self._model_name = model_name
        self._model = None  # Lazy loading
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._confidence_threshold = confidence_threshold

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        
        logger.info(f"Loading GLiNER model: {self._model_name}")
        try:
            from gliner import GLiNER
            # Load the model with ONNX optimization if available
            self._model = GLiNER.from_pretrained(
                self._model_name, load_onnx_model=True
            )
            logger.info("GLiNER model loaded successfully.")
        except ImportError:
            logger.error("Failed to import gliner. Make sure it is installed.")
            raise

    def _chunk_text(self, text: str, sections: dict[str, str]) -> list[str]:
        """Split text into manageable chunks.
        
        Uses sections first. If a section is too long, splits it further.
        """
        chunks = []
        
        # Simple fallback if no sections provided
        if not sections:
            # Naive word-based chunking
            words = text.split()
            for i in range(0, len(words), self._chunk_size - self._chunk_overlap):
                chunks.append(" ".join(words[i:i + self._chunk_size]))
            return chunks

        # Section-based chunking
        for sec_name, sec_text in sections.items():
            if not sec_text.strip():
                continue
                
            words = sec_text.split()
            if len(words) <= self._chunk_size:
                chunks.append(sec_text)
            else:
                for i in range(0, len(words), self._chunk_size - self._chunk_overlap):
                    chunks.append(" ".join(words[i:i + self._chunk_size]))

        return chunks

    def extract(self, text: str, sections: dict[str, str], feature_subset: list[str]) -> dict[str, ExtractionValue]:
        """Extract targeting fields using GLiNER.
        
        Parameters
        ----------
        text : str
            The full document text.
        sections : dict[str, str]
            Parsed sections of the document.
        feature_subset : list[str]
            Fields to extract for this document type.
            
        Returns
        -------
        dict[str, ExtractionValue]
            The extracted values mapped by field name.
        """
        target_fields = set(feature_subset) & self.GLINER_FIELDS
        if not target_fields:
            return {}

        self._ensure_model()
        
        labels_to_extract = [self._LABEL_MAP[f] for f in target_fields]
        chunks = self._chunk_text(text, sections)
        
        results: dict[str, ExtractionValue] = {}
        
        for chunk in chunks:
            # entities is a list of dicts: {'text': ..., 'label': ..., 'score': ..., 'start': ..., 'end': ...}
            entities = self._model.predict_entities(
                chunk, labels_to_extract, threshold=self._confidence_threshold
            )
            
            for ent in entities:
                label = ent["label"]
                score = ent["score"]
                span_text = ent["text"]
                
                field_name = self._REVERSE_LABEL_MAP.get(label)
                if not field_name:
                    continue
                
                # We simply keep the first match with best confidence for now
                if field_name not in results or score > results[field_name].confidence:
                    # Find offset in the original text (naive approach via find, but works for exact chunks)
                    start_char = text.find(span_text)
                    end_char = start_char + len(span_text) if start_char != -1 else None
                    if start_char == -1:
                        start_char = None
                    
                    results[field_name] = ExtractionValue(
                        value=span_text,
                        source_span=span_text,
                        source_span_start=start_char,
                        source_span_end=end_char,
                        extraction_tier="gliner",
                        confidence=round(float(score), 4),
                        vocab_valid=True, # Needs specific post-processing validation if categorical
                    )
                    
        return results
