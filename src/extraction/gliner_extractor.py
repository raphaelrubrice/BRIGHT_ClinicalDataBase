"""GLiNER-based extractor for unstructured narrative clinical fields.

Extracts complex entities that are difficult to capture with pure rules,
using a fine-tuned GLiNER model (e.g. urchade/gliner_multi-v2.1).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.extraction.schema import ExtractionValue, ALL_FIELDS_BY_NAME
from src.extraction.rule_extraction import (
    _IHC_VALUE_NORM,
    _MOL_STATUS_NORM,
    _CHR_STATUS_NORM,
    _LATERALITY_NORM,
)

logger = logging.getLogger(__name__)


class GlinerExtractor:
    """Zero-shot/Few-shot entity extractor using GLiNER."""

    # Expanded list of complex fields to extract via GLiNER
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
        "ihc_idh1",
        "ihc_p53",
        "ihc_atrx",
        "ihc_fgfr3",
        "ihc_braf",
        "ihc_hist_h3k27m",
        "ihc_hist_h3k27me3",
        "ihc_egfr_hirsch",
        "ihc_gfap",
        "ihc_olig2",
        "ihc_ki67",
        "ihc_mmr",
        "mol_idh1",
        "mol_idh2",
        "mol_tert",
        "mol_CDKN2A",
        "mol_h3f3a",
        "mol_hist1h3b",
        "mol_braf",
        "mol_mgmt",
        "mol_fgfr1",
        "mol_egfr_mut",
        "mol_prkca",
        "mol_p53",
        "mol_pten",
        "mol_cic",
        "mol_fubp1",
        "mol_atrx",
        "ch1p",
        "ch19q",
        "ch10p",
        "ch10q",
        "ch7p",
        "ch7q",
        "ch9p",
        "ch9q",
        "ampli_mdm2",
        "ampli_cdk4",
        "ampli_egfr",
        "ampli_met",
        "ampli_mdm4",
        "fusion_fgfr",
        "fusion_ntrk",
        "fusion_autre",
        "ik_clinique",
    }

    _LABEL_MAP: dict[str, str] = {
        # Clinical
        "epilepsie_1er_symptome": "epilepsy at onset",
        "deficit_1er_symptome": "initial neurological deficit",
        "tumeur_position": "tumor anatomical location",
        "activite_professionnelle": "patient profession",
        "antecedent_tumoral": "history of brain tumor",
        "evol_clinique": "clinical evolution status",
        "progress_clinique": "clinical progression",
        "progress_radiologique": "radiological progression",
        "localisation_chir": "surgery location",
        "localisation_radiotherapie": "radiotherapy location",
        # Histology
        "diag_histologique": "histological diagnosis",
        "histo_necrose": "tumor necrosis",
        "histo_pec": "endothelial proliferation",
        # IHC
        "ihc_idh1": "IDH1 immunohistochemistry status",
        "ihc_p53": "p53 immunohistochemistry status",
        "ihc_atrx": "ATRX immunohistochemistry status",
        "ihc_fgfr3": "FGFR3 immunohistochemistry status",
        "ihc_braf": "BRAF immunohistochemistry status",
        "ihc_hist_h3k27m": "H3K27M immunohistochemistry status",
        "ihc_hist_h3k27me3": "H3K27me3 immunohistochemistry status",
        "ihc_egfr_hirsch": "EGFR immunohistochemistry score",
        "ihc_gfap": "GFAP immunohistochemistry status",
        "ihc_olig2": "Olig2 immunohistochemistry status",
        "ihc_ki67": "Ki67 proliferation index",
        "ihc_mmr": "MMR immunohistochemistry status",
        # Molecular
        "mol_idh1": "IDH1 molecular mutation",
        "mol_idh2": "IDH2 molecular mutation",
        "mol_tert": "TERT promoter mutation",
        "mol_CDKN2A": "CDKN2A molecular alteration",
        "mol_h3f3a": "H3F3A molecular mutation",
        "mol_hist1h3b": "HIST1H3B molecular mutation",
        "mol_braf": "BRAF molecular mutation",
        "mol_mgmt": "MGMT promoter methylation",
        "mol_fgfr1": "FGFR1 molecular mutation",
        "mol_egfr_mut": "EGFR molecular mutation",
        "mol_prkca": "PRKCA molecular mutation",
        "mol_p53": "TP53 molecular mutation",
        "mol_pten": "PTEN molecular alteration",
        "mol_cic": "CIC molecular mutation",
        "mol_fubp1": "FUBP1 molecular mutation",
        "mol_atrx": "ATRX molecular mutation",
        # Chromosomal
        "ch1p": "chromosome 1p alteration",
        "ch19q": "chromosome 19q alteration",
        "ch10p": "chromosome 10p alteration",
        "ch10q": "chromosome 10q alteration",
        "ch7p": "chromosome 7p alteration",
        "ch7q": "chromosome 7q alteration",
        "ch9p": "chromosome 9p alteration",
        "ch9q": "chromosome 9q alteration",
        "ampli_mdm2": "MDM2 amplification",
        "ampli_cdk4": "CDK4 amplification",
        "ampli_egfr": "EGFR amplification",
        "ampli_met": "MET amplification",
        "ampli_mdm4": "MDM4 amplification",
        "fusion_fgfr": "FGFR fusion",
        "fusion_ntrk": "NTRK fusion",
        "fusion_autre": "other gene fusion",
        "ik_clinique": "Karnofsky performance score",
    }
    
    # Reverse map for quick lookup
    _REVERSE_LABEL_MAP: dict[str, str] = {v: k for k, v in _LABEL_MAP.items()}

    # Field-specific confidence thresholds to balance precision/recall
    _CONFIDENCE_THRESHOLDS: dict[str, float] = {
        # High confidence required for critical clinical fields
        "epilepsie_1er_symptome": 0.5,
        "deficit_1er_symptome": 0.5,
        "evol_clinique": 0.5,
        "progress_clinique": 0.5,
        
        # Lower thresholds for text-heavy or highly variable fields
        "tumeur_position": 0.3,
        "activite_professionnelle": 0.3,
        "localisation_chir": 0.3,
        "localisation_radiotherapie": 0.3,
        "diag_histologique": 0.3,
        
        # Default fallback is 0.4
    }

    def __init__(
        self,
        model_name: str = "urchade/gliner_multi-v2.1",
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        confidence_threshold: float = 0.4,
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
            # Load the model with PyTorch backend explicitly, ONNX disabled for stability
            self._model = GLiNER.from_pretrained(
                self._model_name, load_onnx_model=False
            )
            logger.info("GLiNER model loaded successfully with PyTorch backend.")
        except ImportError:
            logger.error("Failed to import gliner. Make sure it is installed.")
            raise

    def _chunk_text(self, text: str, sections: dict[str, str]) -> list[str]:
        """Split text into manageable chunks safely below the token limit.
        
        Splits by sentences first, grouping them up to `_chunk_size` words.
        If a single sentence is extremely long, falls back to word chunking.
        """
        def split_into_chunks(text_to_split: str) -> list[str]:
            # Load basic eds-nlp pipeline for robust sentence splitting
            try:
                import edsnlp
                if not hasattr(self, '_nlp_splitter'):
                    self._nlp_splitter = edsnlp.blank("eds")
                    self._nlp_splitter.add_pipe("eds.sentences")
                
                doc = self._nlp_splitter(text_to_split)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except ImportError:
                # Fallback to naive splitting if eds-nlp is unavailable
                import re
                sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', text_to_split) if s.strip()]
            
            local_chunks = []
            current_chunk = []
            current_word_count = 0
            
            for sentence in sentences:
                words_in_sentence = len(sentence.split())
                
                if current_word_count + words_in_sentence > self._chunk_size and current_chunk:
                    local_chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
                
                # If a single sentence is larger than chunk size, split it by words
                if words_in_sentence > self._chunk_size:
                    words = sentence.split()
                    for i in range(0, len(words), self._chunk_size - self._chunk_overlap):
                        local_chunks.append(" ".join(words[i:i + self._chunk_size]))
                else:
                    current_chunk.append(sentence)
                    current_word_count += words_in_sentence
                    
            if current_chunk:
                local_chunks.append(" ".join(current_chunk))
                
            return local_chunks

        chunks = []
        
        # Simple fallback if no sections provided
        if not sections:
            return split_into_chunks(text)

        # Section-based chunking
        for sec_name, sec_text in sections.items():
            if not sec_text.strip():
                continue
            chunks.extend(split_into_chunks(sec_text))

        return chunks

    def _postprocess_span(self, field_name: str, span_text: str) -> str | int | float | None:
        """Convert GLiNER extracted text to normalised value based on field schema."""
        field_def = ALL_FIELDS_BY_NAME.get(field_name)
        if not field_def:
            return span_text
            
        text_lower = span_text.lower().strip()
        
        # 1. Binary/Boolean (check for negation or implicit presence)
        if field_def.allowed_values == {"oui", "non"} or field_def.name.startswith("ampli_") or field_def.name.startswith("fusion_"):
            if re.search(r"(?i)\b(?:pas\s+(?:de|d['']\s*)|absence\s+(?:de|d['']\s*)|sans|aucun[e]?|ni)\b", text_lower):
                return "non"
            return "oui"
            
        # 2. IHC fields
        if field_name.startswith("ihc_"):
            if field_name == "ihc_ki67":
                m = re.search(r"(\d+)", text_lower)
                return m.group(1) if m else span_text
            for key, val in _IHC_VALUE_NORM.items():
                if key in text_lower:
                    if field_name == "ihc_atrx" and val == "positif":
                        return "maintenu"
                    return val
            return _IHC_VALUE_NORM.get(text_lower, span_text)
            
        # 3. Molecular fields
        if field_name.startswith("mol_"):
            for key, val in _MOL_STATUS_NORM.items():
                if key in text_lower:
                    return val
            if re.search(r"(?:p\.)?[a-z]\d+[a-z]", text_lower):
                return "mute"
            return _MOL_STATUS_NORM.get(text_lower, span_text)
            
        # 4. Chromosomal fields
        if field_name.startswith("ch") and field_name not in ("chir_date", "chimios", "chm_date_debut", "chm_date_fin", "chm_cycles"):
            for key, val in _CHR_STATUS_NORM.items():
                if key in text_lower:
                    return val
            return _CHR_STATUS_NORM.get(text_lower, span_text)
            
        # 5. Specific fields
        if field_name == "tumeur_lateralite":
            for key, val in _LATERALITY_NORM.items():
                if key in text_lower:
                    return val
            return _LATERALITY_NORM.get(text_lower, span_text)
            
        if field_name == "ik_clinique":
            m = re.search(r"(\d{2,3})", text_lower)
            return str(int(m.group(1))) if m else span_text
            
        return span_text

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
            # We predict all labels at once using a low base threshold, then filter later
            entities = self._model.predict_entities(
                chunk, labels_to_extract, threshold=0.1
            )
            
            for ent in entities:
                label = ent["label"]
                score = ent["score"]
                span_text = ent["text"]
                
                field_name = self._REVERSE_LABEL_MAP.get(label)
                if not field_name:
                    continue
                
                # Check field-specific threshold
                threshold = self._CONFIDENCE_THRESHOLDS.get(field_name, self._confidence_threshold)
                if score < threshold:
                    continue
                
                # Post-process span to match schema
                norm_val = self._postprocess_span(field_name, span_text)
                
                # We simply keep the first match with best confidence for now
                if field_name not in results or score > results[field_name].confidence:
                    # Find offset in the original text (naive approach via find, but works for exact chunks)
                    start_char = text.find(span_text)
                    end_char = start_char + len(span_text) if start_char != -1 else None
                    if start_char == -1:
                        start_char = None
                    
                    results[field_name] = ExtractionValue(
                        value=norm_val,
                        source_span=span_text,
                        source_span_start=start_char,
                        source_span_end=end_char,
                        extraction_tier="gliner",
                        confidence=round(float(score), 4),
                        vocab_valid=True, # Needs specific post-processing validation if categorical
                    )
                    
        return results
