"""GLiNER-based primary extractor for clinical document fields.

Extracts all fields using a GLiNER model with semantic batching,
language routing, stateful context injection, and sliding-window
chunking to handle long documents within the 512-token limit.

Architecture
------------
- **Semantic Batches**: 21 batches optimized via advanced semantic anchoring.
- **Anchor Matrix**: Max 4 fields per batch (Intra-batch, Prior-context, Correlative-pivot).
- **Language Routing**: Detects language and selects EN/FR label maps.
- **Sliding Window**: 150-200 word chunks with 30-50 word overlap.

Tracked Features (Total: 111)
-----------------------------
1.  Identifiers & Dates (5): date_chir, num_labo, date_rcp, dn_date, date_deces
2.  Demographics (5): sexe, date_de_naissance, activite_professionnelle, antecedent_tumoral, ik_clinique
3.  Diagnosis (4): diag_histologique, diag_integre, classification_oms, grade
4.  Tumor Location (3): tumeur_lateralite, tumeur_position, dominance_cerebrale
5.  Radiology (5): exam_radio_date_decouverte, contraste_1er_symptome, prise_de_contraste, oedeme_1er_symptome, calcif_1er_symptome
6.  Symptoms Onset (6): date_1er_symptome, epilepsie_1er_symptome, ceph_hic_1er_symptome, deficit_1er_symptome, cognitif_1er_symptome, autre_trouble_1er_symptome
7.  Symptoms Current (5): epilepsie, ceph_hic, deficit, cognitif, autre_trouble
8.  Histology (4): histo_necrose, histo_pec, histo_mitoses, aspect_cellulaire
9.  IHC 1 (5): ihc_idh1, ihc_atrx, ihc_p53, ihc_fgfr3, ihc_braf
10. IHC 2 (7): ihc_gfap, ihc_olig2, ihc_ki67, ihc_hist_h3k27m, ihc_hist_h3k27me3, ihc_egfr_hirsch, ihc_mmr
11. Molecular 1 (7): mol_idh1, mol_idh2, mol_mgmt, mol_h3f3a, mol_hist1h3b, mol_tert, mol_CDKN2A
12. Chromosomal (9): ch1p, ch19q, ch1p19q_codel, ch7p, ch7q, ch10p, ch10q, ch9p, ch9q
13. Molecular 2 (9): mol_p53, mol_atrx, mol_cic, mol_fubp1, mol_fgfr1, mol_egfr_mut, mol_prkca, mol_pten, mol_braf
14. Amplifications & Fusions (8): ampli_egfr, ampli_mdm2, ampli_cdk4, ampli_met, ampli_mdm4, fusion_fgfr, fusion_ntrk, fusion_autre
15. Treatment Surgery (4): type_chirurgie, localisation_chir, qualite_exerese, chir_date
16. Treatment Chemo (5): chimios, chimio_protocole, chm_date_debut, chm_date_fin, chm_cycles
17. Treatment Radio (5): rx_date_debut, rx_date_fin, rx_dose, rx_fractionnement, localisation_radiotherapie
18. Adjunct (4): anti_epileptiques, essai_therapeutique, corticoides, optune
19. Evolution (5): evol_clinique, progress_clinique, progress_radiologique, reponse_radiologique, date_progression
20. Care Team (4): neuroncologue, neurochirurgien, radiotherapeute, anatomo_pathologiste
21. Outcome (2): infos_deces, survie_globale
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


# ---------------------------------------------------------------------------
# Semantic Batches — Consolidated 111 Fields (nip removed)
# ---------------------------------------------------------------------------

SEMANTIC_BATCHES: dict[str, dict] = {
    # 1. Identifiers & Dates
    "identifiers_dates": {
        "fields": {"date_chir", "num_labo", "date_rcp", "dn_date", "date_deces"},
        "anchors": {"date_chir", "num_labo", "date_rcp", "diag_histologique"},
        "labels_en": {
            "date_chir": "surgery date or operation date",
            "num_labo": "laboratory number or pathology reference",
            "date_rcp": "multidisciplinary meeting (RCP) date",
            "dn_date": "date of last news or follow-up",
            "date_deces": "date of death",
        },
        "labels_fr": {
            "date_chir": "date de chirurgie ou date opératoire",
            "num_labo": "numéro de laboratoire ou référence anatomo-pathologique",
            "date_rcp": "date de la Réunion de Concertation Pluridisciplinaire (RCP)",
            "dn_date": "date de dernière nouvelle ou suivi",
            "date_deces": "date de décès",
        },
    },

    # 2. Demographics
    "demographics": {
        "fields": {"sexe", "date_de_naissance", "activite_professionnelle", "antecedent_tumoral", "ik_clinique"},
        "anchors": {"sexe", "date_de_naissance", "ik_clinique", "diag_histologique"},
        "labels_en": {
            "sexe": "patient sex or gender",
            "date_de_naissance": "date of birth",
            "activite_professionnelle": "patient occupation",
            "antecedent_tumoral": "prior history of brain tumor",
            "ik_clinique": "Karnofsky performance status or WHO performance status score",
        },
        "labels_fr": {
            "sexe": "sexe du patient",
            "date_de_naissance": "date de naissance",
            "activite_professionnelle": "activité professionnelle",
            "antecedent_tumoral": "antécédent de tumeur cérébrale",
            "ik_clinique": "indice de Karnofsky ou score de performance OMS / PS",
        },
    },

    # 3. Diagnosis
    "diagnosis": {
        "fields": {"diag_histologique", "diag_integre", "classification_oms", "grade"},
        "anchors": {"diag_histologique", "grade", "diag_integre", "sexe"},
        "labels_en": {
            "diag_histologique": "histological diagnosis",
            "diag_integre": "integrated WHO 2021 diagnosis combining histology and molecular markers",
            "classification_oms": "WHO classification edition year",
            "grade": "CNS WHO tumor grade 1 to 4",
        },
        "labels_fr": {
            "diag_histologique": "diagnostic histologique",
            "diag_integre": "diagnostic intégré OMS 2021 combinant histologie et biologie moléculaire",
            "classification_oms": "année de classification OMS",
            "grade": "grade tumoral OMS de 1 à 4",
        },
    },

    # 4. Tumor Location
    "tumor_location": {
        "fields": {"tumeur_lateralite", "tumeur_position", "dominance_cerebrale"},
        "anchors": {"tumeur_position", "tumeur_lateralite", "dominance_cerebrale", "diag_histologique"},
        "labels_en": {
            "tumeur_lateralite": "tumor laterality (left, right, bilateral)",
            "tumeur_position": "anatomical location (frontal, temporal, etc.)",
            "dominance_cerebrale": "cerebral dominance or handedness (right-handed, left-handed)",
        },
        "labels_fr": {
            "tumeur_lateralite": "latéralité tumorale (gauche, droite, bilatéral)",
            "tumeur_position": "localisation anatomique (frontal, temporal, etc.)",
            "dominance_cerebrale": "latéralité manuelle ou dominance (droitier, gaucher)",
        },
    },

    # 5. Radiology
    "radiology": {
        "fields": {"exam_radio_date_decouverte", "contraste_1er_symptome", "prise_de_contraste", "oedeme_1er_symptome", "calcif_1er_symptome"},
        "anchors": {"contraste_1er_symptome", "prise_de_contraste", "tumeur_position", "diag_histologique"},
        "labels_en": {
            "exam_radio_date_decouverte": "radiological exam date at discovery",
            "contraste_1er_symptome": "contrast enhancement at discovery",
            "prise_de_contraste": "general mention of contrast enhancement",
            "oedeme_1er_symptome": "edema on initial imaging",
            "calcif_1er_symptome": "calcification on initial imaging",
        },
        "labels_fr": {
            "exam_radio_date_decouverte": "date de l'imagerie de découverte",
            "contraste_1er_symptome": "prise de contraste à la découverte",
            "prise_de_contraste": "mention générale de prise de contraste",
            "oedeme_1er_symptome": "oedème à l'imagerie initiale",
            "calcif_1er_symptome": "calcification à l'imagerie initiale",
        },
    },

    # 6. Symptoms Onset
    "symptoms_onset": {
        "fields": {"date_1er_symptome", "epilepsie_1er_symptome", "ceph_hic_1er_symptome", "deficit_1er_symptome", "cognitif_1er_symptome", "autre_trouble_1er_symptome"},
        "anchors": {"epilepsie_1er_symptome", "ceph_hic_1er_symptome", "tumeur_position", "diag_histologique"},
        "labels_en": {
            "date_1er_symptome": "date of first symptom",
            "epilepsie_1er_symptome": "epilepsy at onset",
            "ceph_hic_1er_symptome": "headache or HIC at onset",
            "deficit_1er_symptome": "neurological deficit at onset",
            "cognitif_1er_symptome": "cognitive disorder at onset",
            "autre_trouble_1er_symptome": "other symptoms at onset",
        },
        "labels_fr": {
            "date_1er_symptome": "date du premier symptôme",
            "epilepsie_1er_symptome": "épilepsie inaugurale",
            "ceph_hic_1er_symptome": "céphalées ou HIC initiale",
            "deficit_1er_symptome": "déficit neurologique initial",
            "cognitif_1er_symptome": "troubles cognitifs initiaux",
            "autre_trouble_1er_symptome": "autres symptômes initiaux",
        },
    },

    # 7. Symptoms Current
    "symptoms_current": {
        "fields": {"epilepsie", "ceph_hic", "deficit", "cognitif", "autre_trouble"},
        "anchors": {"epilepsie", "deficit", "epilepsie_1er_symptome", "tumeur_position"},
        "labels_en": {
            "epilepsie": "current epilepsy",
            "ceph_hic": "current headache or HIC",
            "deficit": "current neurological deficit",
            "cognitif": "current cognitive disorder",
            "autre_trouble": "other current symptoms",
        },
        "labels_fr": {
            "epilepsie": "épilepsie actuelle",
            "ceph_hic": "céphalées ou HIC actuelles",
            "deficit": "déficit neurologique actuel",
            "cognitif": "troubles cognitifs actuels",
            "autre_trouble": "autres troubles actuels",
        },
    },

    # 8. Histology
    "histology": {
        "fields": {"histo_necrose", "histo_pec", "histo_mitoses", "aspect_cellulaire"},
        "anchors": {"histo_necrose", "histo_pec", "aspect_cellulaire", "diag_histologique"},
        "labels_en": {
            "histo_necrose": "tumor necrosis",
            "histo_pec": "microvascular proliferation",
            "histo_mitoses": "mitotic count",
            "aspect_cellulaire": "cellular morphology description",
        },
        "labels_fr": {
            "histo_necrose": "nécrose tumorale",
            "histo_pec": "prolifération microvasculaire",
            "histo_mitoses": "nombre de mitoses",
            "aspect_cellulaire": "description de l'aspect cellulaire",
        },
    },

    # 9. IHC 1
    "ihc_1": {
        "fields": {"ihc_idh1", "ihc_atrx", "ihc_p53", "ihc_fgfr3", "ihc_braf"},
        "anchors": {"ihc_idh1", "ihc_atrx", "ihc_p53", "diag_histologique"},
        "labels_en": {
            "ihc_idh1": "IHC IDH1 R132H status",
            "ihc_atrx": "ATRX expression",
            "ihc_p53": "p53 expression",
            "ihc_fgfr3": "IHC FGFR3",
            "ihc_braf": "IHC BRAF V600E",
        },
        "labels_fr": {
            "ihc_idh1": "IHC IDH1 R132H",
            "ihc_atrx": "expression ATRX",
            "ihc_p53": "expression p53",
            "ihc_fgfr3": "IHC FGFR3",
            "ihc_braf": "IHC BRAF V600E",
        },
    },

    # 10. IHC 2
    "ihc_2": {
        "fields": {"ihc_gfap", "ihc_olig2", "ihc_ki67", "ihc_hist_h3k27m", "ihc_hist_h3k27me3", "ihc_egfr_hirsch", "ihc_mmr"},
        "anchors": {"ihc_gfap", "ihc_olig2", "ihc_ki67", "ihc_idh1"},
        "labels_en": {
            "ihc_gfap": "GFAP IHC",
            "ihc_olig2": "OLIG2 IHC",
            "ihc_ki67": "Ki67 index",
            "ihc_hist_h3k27m": "H3K27M IHC",
            "ihc_hist_h3k27me3": "H3K27me3 IHC",
            "ihc_egfr_hirsch": "EGFR Hirsch score",
            "ihc_mmr": "MMR IHC",
        },
        "labels_fr": {
            "ihc_gfap": "IHC GFAP",
            "ihc_olig2": "IHC OLIG2",
            "ihc_ki67": "index Ki67",
            "ihc_hist_h3k27m": "IHC H3K27M",
            "ihc_hist_h3k27me3": "IHC H3K27me3",
            "ihc_egfr_hirsch": "score de Hirsch EGFR",
            "ihc_mmr": "IHC MMR",
        },
    },

    # 11. Molecular 1
    "molecular_1": {
        "fields": {"mol_idh1", "mol_idh2", "mol_mgmt", "mol_h3f3a", "mol_hist1h3b", "mol_tert", "mol_CDKN2A"},
        "anchors": {"mol_idh1", "mol_idh2", "ihc_idh1", "diag_histologique"},
        "labels_en": {
            "mol_idh1": "molecular IDH1 mutation",
            "mol_idh2": "molecular IDH2 mutation",
            "mol_mgmt": "MGMT methylation",
            "mol_h3f3a": "H3F3A mutation",
            "mol_hist1h3b": "HIST1H3B mutation",
            "mol_tert": "TERT promoter mutation",
            "mol_CDKN2A": "CDKN2A deletion",
        },
        "labels_fr": {
            "mol_idh1": "mutation moléculaire IDH1",
            "mol_idh2": "mutation moléculaire IDH2",
            "mol_mgmt": "méthylation MGMT",
            "mol_h3f3a": "mutation H3F3A",
            "mol_hist1h3b": "mutation HIST1H3B",
            "mol_tert": "mutation promoteur TERT",
            "mol_CDKN2A": "délétion CDKN2A",
        },
    },

    # 12. Chromosomal
    "chromosomal": {
        "fields": {"ch1p", "ch19q", "ch1p19q_codel", "ch7p", "ch7q", "ch10p", "ch10q", "ch9p", "ch9q"},
        "anchors": {"ch1p", "ch19q", "ch1p19q_codel", "diag_histologique"},
        "labels_en": {
            "ch1p": "chr 1p status",
            "ch19q": "chr 19q status",
            "ch1p19q_codel": "1p/19q co-deletion",
            "ch7p": "chr 7p status",
            "ch7q": "chr 7q status",
            "ch10p": "chr 10p status",
            "ch10q": "chr 10q status",
            "ch9p": "chr 9p status",
            "ch9q": "chr 9q status",
        },
        "labels_fr": {
            "ch1p": "statut chr 1p",
            "ch19q": "statut chr 19q",
            "ch1p19q_codel": "codélétion 1p/19q",
            "ch7p": "statut chr 7p",
            "ch7q": "statut chr 7q",
            "ch10p": "statut chr 10p",
            "ch10q": "statut chr 10q",
            "ch9p": "statut chr 9p",
            "ch9q": "statut chr 9q",
        },
    },

    # 13. Molecular 2
    "molecular_2": {
        "fields": {"mol_p53", "mol_atrx", "mol_cic", "mol_fubp1", "mol_fgfr1", "mol_egfr_mut", "mol_prkca", "mol_pten", "mol_braf"},
        "anchors": {"mol_p53", "mol_atrx", "ihc_p53", "mol_idh1"},
        "labels_en": {
            "mol_p53": "TP53 molecular status",
            "mol_atrx": "ATRX molecular status",
            "mol_cic": "CIC mutation",
            "mol_fubp1": "FUBP1 mutation",
            "mol_fgfr1": "FGFR1 mutation",
            "mol_egfr_mut": "EGFR mutation",
            "mol_prkca": "PRKCA mutation",
            "mol_pten": "PTEN status",
            "mol_braf": "BRAF molecular mutation status",
        },
        "labels_fr": {
            "mol_p53": "statut moléculaire TP53",
            "mol_atrx": "statut moléculaire ATRX",
            "mol_cic": "mutation CIC",
            "mol_fubp1": "mutation FUBP1",
            "mol_fgfr1": "mutation FGFR1",
            "mol_egfr_mut": "mutation EGFR",
            "mol_prkca": "mutation PRKCA",
            "mol_pten": "statut PTEN",
            "mol_braf": "statut moléculaire de mutation BRAF",
        },
    },

    # 14. Amplifications & Fusions
    "amplifications_fusions": {
        "fields": {"ampli_egfr", "ampli_mdm2", "ampli_cdk4", "ampli_met", "ampli_mdm4", "fusion_fgfr", "fusion_ntrk", "fusion_autre"},
        "anchors": {"ampli_egfr", "mol_tert", "diag_histologique", "mol_idh1"},
        "labels_en": {
            "ampli_egfr": "EGFR amplification",
            "ampli_mdm2": "MDM2 amplification",
            "ampli_cdk4": "CDK4 amplification",
            "ampli_met": "MET amplification",
            "ampli_mdm4": "MDM4 amplification",
            "fusion_fgfr": "FGFR fusion",
            "fusion_ntrk": "NTRK fusion",
            "fusion_autre": "other fusion",
        },
        "labels_fr": {
            "ampli_egfr": "amplification EGFR",
            "ampli_mdm2": "amplification MDM2",
            "ampli_cdk4": "amplification CDK4",
            "ampli_met": "amplification MET",
            "ampli_mdm4": "amplification MDM4",
            "fusion_fgfr": "fusion FGFR",
            "fusion_ntrk": "fusion NTRK",
            "fusion_autre": "autre fusion",
        },
    },

    # 15. Treatment Surgery
    "treatment_surgery": {
        "fields": {"type_chirurgie", "localisation_chir", "qualite_exerese", "chir_date"},
        "anchors": {"type_chirurgie", "qualite_exerese", "tumeur_position", "diag_histologique"},
        "labels_en": {
            "type_chirurgie": "surgery type",
            "localisation_chir": "surgery location",
            "qualite_exerese": "resection quality",
            "chir_date": "surgery date",
        },
        "labels_fr": {
            "type_chirurgie": "type de chirurgie",
            "localisation_chir": "localisation chirurgicale",
            "qualite_exerese": "qualité de l'exérèse",
            "chir_date": "date de chirurgie",
        },
    },

    # 16. Treatment Chemo
    "treatment_chemo": {
        "fields": {"chimios", "chimio_protocole", "chm_date_debut", "chm_date_fin", "chm_cycles"},
        "anchors": {"chimios", "chimio_protocole", "mol_mgmt", "diag_histologique"},
        "labels_en": {
            "chimios": "chemotherapy drugs",
            "chimio_protocole": "chemo protocol",
            "chm_date_debut": "chemo start",
            "chm_date_fin": "chemo end",
            "chm_cycles": "chemo cycles",
        },
        "labels_fr": {
            "chimios": "chimiothérapies",
            "chimio_protocole": "protocole de chimie",
            "chm_date_debut": "début chimie",
            "chm_date_fin": "fin chimie",
            "chm_cycles": "cycles de chimie",
        },
    },

    # 17. Treatment Radio
    "treatment_radio": {
        "fields": {"rx_date_debut", "rx_date_fin", "rx_dose", "rx_fractionnement", "localisation_radiotherapie"},
        "anchors": {"rx_dose", "rx_fractionnement", "localisation_radiotherapie", "chimios"},
        "labels_en": {
            "rx_date_debut": "radio start",
            "rx_date_fin": "radio end",
            "rx_dose": "radio dose",
            "rx_fractionnement": "radio fractions",
            "localisation_radiotherapie": "radio location",
        },
        "labels_fr": {
            "rx_date_debut": "début radio",
            "rx_date_fin": "fin radio",
            "rx_dose": "dose radio",
            "rx_fractionnement": "fractionnement radio",
            "localisation_radiotherapie": "localisation radio",
        },
    },

    # 18. Adjunct
    "adjunct": {
        "fields": {"anti_epileptiques", "essai_therapeutique", "corticoides", "optune"},
        "anchors": {"anti_epileptiques", "corticoides", "epilepsie", "chimios"},
        "labels_en": {
            "anti_epileptiques": "AED drugs",
            "essai_therapeutique": "clinical trial",
            "corticoides": "steroids",
            "optune": "Optune device",
        },
        "labels_fr": {
            "anti_epileptiques": "anti-épileptiques",
            "essai_therapeutique": "essai thérapeutique",
            "corticoides": "corticoïdes",
            "optune": "dispositif Optune",
        },
    },

    # 19. Evolution
    "evolution": {
        "fields": {"evol_clinique", "progress_clinique", "progress_radiologique", "reponse_radiologique", "date_progression"},
        "anchors": {"evol_clinique", "progress_radiologique", "diag_histologique", "type_chirurgie"},
        "labels_en": {
            "evol_clinique": "clinical evolution",
            "progress_clinique": "clinical progression",
            "progress_radiologique": "radio progression",
            "reponse_radiologique": "radio response",
            "date_progression": "progression date",
        },
        "labels_fr": {
            "evol_clinique": "évolution clinique",
            "progress_clinique": "progression clinique",
            "progress_radiologique": "progression radio",
            "reponse_radiologique": "réponse radio",
            "date_progression": "date de progression",
        },
    },

    # 20. Care Team
    "care_team": {
        "fields": {"neuroncologue", "neurochirurgien", "radiotherapeute", "anatomo_pathologiste"},
        "anchors": {"neuroncologue", "neurochirurgien", "anatomo_pathologiste", "type_chirurgie"},
        "labels_en": {
            "neuroncologue": "neuro-oncologist",
            "neurochirurgien": "neurosurgeon",
            "radiotherapeute": "radiotherapist",
            "anatomo_pathologiste": "pathologist",
        },
        "labels_fr": {
            "neuroncologue": "neuro-oncologue",
            "neurochirurgien": "neurochirurgien",
            "radiotherapeute": "radiothérapeute",
            "anatomo_pathologiste": "pathologiste",
        },
    },

    # 21. Outcome
    "outcome": {
        "fields": {"infos_deces", "survie_globale"},
        "anchors": {"infos_deces", "survie_globale", "evol_clinique", "diag_histologique"},
        "labels_en": {
            "infos_deces": "death info",
            "survie_globale": "overall survival",
        },
        "labels_fr": {
            "infos_deces": "infos décès",
            "survie_globale": "survie globale",
        },
    },
}

# Build the complete set of all GLiNER-extractable fields
_ALL_GLINER_FIELDS: set[str] = set()
for _batch in SEMANTIC_BATCHES.values():
    _ALL_GLINER_FIELDS |= _batch["fields"]

# Build reverse map: label_text → field_name (for both EN and FR)
_REVERSE_LABEL_MAP: dict[str, str] = {}
for _batch in SEMANTIC_BATCHES.values():
    for field_name, label_text in _batch["labels_en"].items():
        _REVERSE_LABEL_MAP[label_text] = field_name
    for field_name, label_text in _batch["labels_fr"].items():
        _REVERSE_LABEL_MAP[label_text] = field_name


# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "evol_clinique": 0.5,
    "progress_clinique": 0.5,
    "progress_radiologique": 0.5,
}

_DEFAULT_THRESHOLD = 0.4


class GlinerExtractor:
    """Primary entity extractor using GLiNER with consolidated 111-field tracking."""

    GLINER_FIELDS: set[str] = _ALL_GLINER_FIELDS

    def __init__(
        self,
        model_name: str = "urchade/gliner_multi-v2.1",
        chunk_size: int = 180,
        chunk_overlap: int = 40,
    ):
        self._model_name = model_name
        self._model = None  # Lazy loading
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from gliner import GLiNER
        self._model = GLiNER.from_pretrained(self._model_name, load_onnx_model=False)

    @staticmethod
    def detect_language(text: str) -> str:
        try:
            from langdetect import detect
            return detect(text[:1500])
        except Exception:
            return "fr"

    def _chunk_text(self, text: str) -> list[tuple[str, int]]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self._chunk_size - self._chunk_overlap):
            chunk_text = " ".join(words[i : i + self._chunk_size])
            char_offset = text.find(chunk_text[:30]) if i > 0 else 0
            chunks.append((chunk_text, max(0, char_offset)))
        return chunks

    @staticmethod
    def _build_context_prefix(
        batch: dict,
        extraction_state: dict[str, tuple],
    ) -> str:
        anchors = batch.get("anchors", set())
        if not anchors:
            return ""

        parts = []
        for anchor_field in anchors:
            if anchor_field in extraction_state:
                value, _conf, *_ = extraction_state[anchor_field]
                parts.append(f"{anchor_field}: {value}")

        if not parts:
            return ""
        return "[Context: " + ", ".join(parts) + "]"

    _BINARY_FIELDS = {
        "ch1p19q_codel", "fusion_autre",
        # symptoms onset
        "epilepsie_1er_symptome", "ceph_hic_1er_symptome", "deficit_1er_symptome",
        "cognitif_1er_symptome", "autre_trouble_1er_symptome",
        # symptoms current
        "epilepsie", "ceph_hic", "deficit", "cognitif",
        # radiology
        "contraste_1er_symptome", "prise_de_contraste", "oedeme_1er_symptome",
        "calcif_1er_symptome",
        # demographics
        "antecedent_tumoral",
    }

    def _postprocess_span(self, field_name: str, span_text: str) -> Any:
        text_lower = span_text.lower().strip()

        # Binary logic for alterations, symptoms, radiology, etc.
        if field_name.startswith("ampli_") or field_name.startswith("fusion_") or field_name in self._BINARY_FIELDS:
            if any(neg in text_lower for neg in ["pas de", "absence", "non", "négatif", "neg"]):
                return "non"
            return "oui"

        # IHC normalization
        if field_name.startswith("ihc_"):
            for key, val in _IHC_VALUE_NORM.items():
                if key in text_lower: return val
            return span_text

        return span_text

    def extract(
        self,
        text: str,
        feature_subset: list[str],
        language: str | None = None,
    ) -> dict[str, ExtractionValue]:
        if language is None:
            language = self.detect_language(text)

        target_fields = set(feature_subset) & self.GLINER_FIELDS
        if not target_fields:
            return {}

        self._ensure_model()
        chunks = self._chunk_text(text)
        extraction_state: dict[str, tuple] = {}

        for chunk_text, _chunk_offset in chunks:
            for batch_name, batch_config in SEMANTIC_BATCHES.items():
                batch_fields = batch_config["fields"] & target_fields
                if not batch_fields:
                    continue

                context_prefix = self._build_context_prefix(batch_config, extraction_state)
                augmented_text = f"{context_prefix}\n\n{chunk_text}" if context_prefix else chunk_text

                label_key = "labels_fr" if language.startswith("fr") else "labels_en"
                label_map = batch_config.get(label_key, batch_config["labels_en"])
                labels = [label_map[f] for f in batch_fields if f in label_map]

                if not labels:
                    continue

                entities = self._model.predict_entities(augmented_text, labels, threshold=0.1)

                for ent in entities:
                    label_text = ent["label"]
                    score = ent["score"]
                    span_text = ent["text"]

                    field_name = _REVERSE_LABEL_MAP.get(label_text)
                    if not field_name or field_name not in batch_fields:
                        continue

                    threshold = _CONFIDENCE_THRESHOLDS.get(field_name, _DEFAULT_THRESHOLD)
                    if score < threshold:
                        continue

                    norm_val = self._postprocess_span(field_name, span_text)

                    existing = extraction_state.get(field_name)
                    if existing is None or score > existing[1]:
                        extraction_state[field_name] = (norm_val, score, span_text, None, None)

        results = {}
        for f, (v, s, span, _, _) in extraction_state.items():
            results[f] = ExtractionValue(
                value=v,
                source_span=span,
                extraction_tier="gliner",
                confidence=round(float(s), 4),
                vocab_valid=True
            )
        return results