"""GLiNER-based primary extractor for clinical document fields.

Extracts all fields using a GLiNER model with semantic batching,
language routing, stateful context injection, and sliding-window
chunking to handle long documents within the 512-token limit.

Architecture
------------
- **Semantic Macro-Batches**: 7 combined batches (11-18 labels each) to optimize computation speed while remaining safely under a 20-label cap.
- **Anchor Context**: Optimized anchors to inject key findings (e.g. diagnosis) into subsequent chunks.
- **Language Routing**: Detects language and selects EN/FR label maps.
- **Sliding Window**: 150-200 word chunks with 30-50 word overlap.

Tracked Features (Total: 111)
-----------------------------
Organized into 7 computational macro-batches of <= 20 labels.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from enum import Enum
from typing import Any

from src.extraction.schema import ExtractionValue, ALL_FIELDS_BY_NAME, MappingType
from src.extraction.rule_extraction import (
    _IHC_VALUE_NORM,
    _MOL_STATUS_NORM,
    _CHR_STATUS_NORM,
    _LATERALITY_NORM,
)
from src.extraction.similarity import match_to_vocab

logger = logging.getLogger(__name__)


class BatchingStrategy(str, Enum):
    """GLiNER field batching strategy."""

    SEMANTIC_CONTEXT = "semantic_context"    # Semantic macro-batches + anchor context injection
    SEMANTIC_ONLY = "semantic_only"          # Semantic macro-batches, no context injection
    HETEROGENEOUS = "heterogeneous"          # Mix fields from different domains for max discrimination


# ---------------------------------------------------------------------------
# Semantic Macro-Batches — Combined domains capped at 20 labels max
# ---------------------------------------------------------------------------

SEMANTIC_BATCHES: dict[str, dict] = {
    # Batch 1: Identifiers, Demographics, Diagnosis, Location (17 fields)
    "identifiers_demographics_diagnosis_location": {
        "fields": {
            "date_chir", "num_labo", "date_rcp", "dn_date", "date_deces",
            "sexe", "annee_de_naissance", "activite_professionnelle", "antecedent_tumoral", "ik_clinique",
            "diag_histologique", "diag_integre", "classification_oms", "grade",
            "tumeur_lateralite", "tumeur_position", "dominance_cerebrale"
        },
        "anchors": {"diag_histologique", "tumeur_position", "date_chir", "sexe"},
        "labels_en": {
            "date_chir": "neurosurgery operation date",
            "num_labo": "pathology lab number",
            "date_rcp": "MDT meeting date",
            "dn_date": "date of last news or follow-up",
            "date_deces": "patient death date",
            "sexe": "patient gender",
            "annee_de_naissance": "patient birth year",
            "activite_professionnelle": "patient occupation",
            "antecedent_tumoral": "prior brain tumor history",
            "ik_clinique": "Karnofsky or WHO performance score",
            "diag_histologique": "histological tumor diagnosis",
            "diag_integre": "integrated WHO 2021 diagnosis",
            "classification_oms": "WHO classification edition year",
            "grade": "CNS WHO tumor grade 1 to 4",
            "tumeur_lateralite": "tumor laterality",
            "tumeur_position": "tumor anatomical position",
            "dominance_cerebrale": "brain hemisphere dominance",
        },
        "labels_fr": {
            "date_chir": "date intervention chirurgicale",
            "num_labo": "numéro laboratoire anapath",
            "date_rcp": "date RCP",
            "dn_date": "date dernières nouvelles ou suivi",
            "date_deces": "date décès patient",
            "sexe": "sexe patient",
            "annee_de_naissance": "année naissance patient",
            "activite_professionnelle": "profession patient",
            "antecedent_tumoral": "antécédent tumeur cérébrale",
            "ik_clinique": "indice Karnofsky ou score performance OMS",
            "diag_histologique": "diagnostic histologique tumoral",
            "diag_integre": "diagnostic intégré OMS 2021",
            "classification_oms": "année classification OMS",
            "grade": "grade tumoral OMS de 1 à 4",
            "tumeur_lateralite": "latéralité tumorale",
            "tumeur_position": "position anatomique tumorale",
            "dominance_cerebrale": "dominance hémisphérique",
        },
    },

    # Batch 2: Radiology & Symptoms (16 fields)
    "radiology_symptoms": {
        "fields": {
            "exam_radio_date_decouverte", "contraste_1er_symptome", "prise_de_contraste", "oedeme_1er_symptome", "calcif_1er_symptome",
            "date_1er_symptome", "epilepsie_1er_symptome", "ceph_hic_1er_symptome", "deficit_1er_symptome", "cognitif_1er_symptome", "autre_trouble_1er_symptome",
            "epilepsie", "ceph_hic", "deficit", "cognitif", "autre_trouble"
        },
        "anchors": {"diag_histologique", "tumeur_position", "contraste_1er_symptome", "epilepsie", "deficit"},
        "labels_en": {
            "exam_radio_date_decouverte": "tumor discovery imaging date",
            "contraste_1er_symptome": "initial contrast enhancement",
            "prise_de_contraste": "general contrast enhancement mention",
            "oedeme_1er_symptome": "initial peritumoral edema",
            "calcif_1er_symptome": "initial tumor calcification",
            "date_1er_symptome": "first symptom onset date",
            "epilepsie_1er_symptome": "initial seizure symptom",
            "ceph_hic_1er_symptome": "initial intracranial hypertension",
            "deficit_1er_symptome": "initial neurological deficit",
            "cognitif_1er_symptome": "initial cognitive impairment",
            "autre_trouble_1er_symptome": "other initial clinical symptom",
            "epilepsie": "seizure or epilepsy mention",
            "ceph_hic": "intracranial hypertension sign",
            "deficit": "current neurological deficit",
            "cognitif": "cognitive impairment symptom",
            "autre_trouble": "other clinical symptom",
        },
        "labels_fr": {
            "exam_radio_date_decouverte": "date imagerie découverte",
            "contraste_1er_symptome": "prise contraste initiale",
            "prise_de_contraste": "mention générale prise de contraste",
            "oedeme_1er_symptome": "œdème péritumoral initial",
            "calcif_1er_symptome": "calcification tumorale initiale",
            "date_1er_symptome": "date apparition premier symptôme",
            "epilepsie_1er_symptome": "crise épilepsie initiale",
            "ceph_hic_1er_symptome": "hypertension intracrânienne initiale",
            "deficit_1er_symptome": "déficit neurologique initial",
            "cognitif_1er_symptome": "trouble cognitif initial",
            "autre_trouble_1er_symptome": "autre symptôme clinique initial",
            "epilepsie": "mention épilepsie ou crise",
            "ceph_hic": "signe hypertension intracrânienne",
            "deficit": "déficit neurologique actuel",
            "cognitif": "trouble ou déficit cognitif",
            "autre_trouble": "autre trouble clinique",
        },
    },

    # Batch 3: Histology & IHC (16 fields)
    "histology_ihc": {
        "fields": {
            "histo_necrose", "histo_pec", "histo_mitoses", "aspect_cellulaire",
            "ihc_idh1", "ihc_atrx", "ihc_p53", "ihc_fgfr3", "ihc_braf",
            "ihc_gfap", "ihc_olig2", "ihc_ki67", "ihc_hist_h3k27m", "ihc_hist_h3k27me3", "ihc_egfr_hirsch", "ihc_mmr"
        },
        "anchors": {"diag_histologique", "histo_necrose", "ihc_idh1", "ihc_p53", "ihc_ki67"},
        "labels_en": {
            "histo_necrose": "tumor necrosis presence",
            "histo_pec": "microvascular proliferation",
            "histo_mitoses": "mitotic count",
            "aspect_cellulaire": "cellular morphology",
            "ihc_idh1": "IDH1 IHC expression",
            "ihc_atrx": "ATRX IHC expression",
            "ihc_p53": "p53 IHC expression",
            "ihc_fgfr3": "FGFR3 IHC expression",
            "ihc_braf": "BRAF V600E IHC expression",
            "ihc_gfap": "GFAP protein expression",
            "ihc_olig2": "Olig2 IHC expression",
            "ihc_ki67": "Ki-67 proliferation index",
            "ihc_hist_h3k27m": "H3K27M IHC expression",
            "ihc_hist_h3k27me3": "H3K27me3 IHC expression",
            "ihc_egfr_hirsch": "EGFR Hirsch score",
            "ihc_mmr": "MMR proteins IHC",
        },
        "labels_fr": {
            "histo_necrose": "présence nécrose tumorale",
            "histo_pec": "prolifération microvasculaire",
            "histo_mitoses": "compte mitotique",
            "aspect_cellulaire": "morphologie cellulaire",
            "ihc_idh1": "expression IHC IDH1",
            "ihc_atrx": "expression IHC ATRX",
            "ihc_p53": "expression IHC p53",
            "ihc_fgfr3": "expression IHC FGFR3",
            "ihc_braf": "expression IHC BRAF V600E",
            "ihc_gfap": "expression protéine GFAP",
            "ihc_olig2": "expression IHC Olig2",
            "ihc_ki67": "index prolifération Ki-67",
            "ihc_hist_h3k27m": "expression IHC H3K27M",
            "ihc_hist_h3k27me3": "expression IHC H3K27me3",
            "ihc_egfr_hirsch": "score de Hirsch EGFR",
            "ihc_mmr": "IHC protéines MMR",
        },
    },

    # Batch 4: Molecular Mutations (16 fields)
    "molecular_basic": {
        "fields": {
            "mol_idh1", "mol_idh2", "mol_mgmt", "mol_h3f3a", "mol_hist1h3b", "mol_tert", "mol_CDKN2A",
            "mol_p53", "mol_atrx", "mol_cic", "mol_fubp1", "mol_fgfr1", "mol_egfr_mut", "mol_prkca", "mol_pten", "mol_braf"
        },
        "anchors": {"diag_histologique", "mol_idh1", "ihc_idh1", "mol_p53", "mol_atrx"},
        "labels_en": {
            "mol_idh1": "IDH1 gene mutation",
            "mol_idh2": "IDH2 gene mutation",
            "mol_mgmt": "MGMT promoter methylation",
            "mol_h3f3a": "H3F3A gene mutation",
            "mol_hist1h3b": "HIST1H3B gene mutation",
            "mol_tert": "TERT promoter mutation",
            "mol_CDKN2A": "CDKN2A gene deletion",
            "mol_p53": "TP53 gene mutation",
            "mol_atrx": "ATRX gene mutation",
            "mol_cic": "CIC gene mutation",
            "mol_fubp1": "FUBP1 gene mutation",
            "mol_fgfr1": "FGFR1 gene mutation",
            "mol_egfr_mut": "EGFR gene mutation",
            "mol_prkca": "PRKCA gene mutation",
            "mol_pten": "PTEN gene mutation",
            "mol_braf": "BRAF gene mutation",
        },
        "labels_fr": {
            "mol_idh1": "mutation gène IDH1",
            "mol_idh2": "mutation gène IDH2",
            "mol_mgmt": "méthylation promoteur MGMT",
            "mol_h3f3a": "mutation gène H3F3A",
            "mol_hist1h3b": "mutation gène HIST1H3B",
            "mol_tert": "mutation promoteur TERT",
            "mol_CDKN2A": "délétion gène CDKN2A",
            "mol_p53": "mutation gène TP53",
            "mol_atrx": "mutation gène ATRX",
            "mol_cic": "mutation gène CIC",
            "mol_fubp1": "mutation gène FUBP1",
            "mol_fgfr1": "mutation gène FGFR1",
            "mol_egfr_mut": "mutation gène EGFR",
            "mol_prkca": "mutation gène PRKCA",
            "mol_pten": "mutation gène PTEN",
            "mol_braf": "mutation gène BRAF",
        },
    },

    # Batch 5: Chromosomal & Amplifications/Fusions (17 fields)
    "chromosomal_amplifications": {
        "fields": {
            "ch1p", "ch19q", "ch1p19q_codel", "ch7p", "ch7q", "ch10p", "ch10q", "ch9p", "ch9q",
            "ampli_egfr", "ampli_mdm2", "ampli_cdk4", "ampli_met", "ampli_mdm4", "fusion_fgfr", "fusion_ntrk", "fusion_autre"
        },
        "anchors": {"diag_histologique", "ch1p19q_codel", "ampli_egfr", "mol_idh1"},
        "labels_en": {
            "ch1p": "1p chromosome status",
            "ch19q": "19q chromosome status",
            "ch1p19q_codel": "1p/19q co-deletion",
            "ch7p": "7p chromosome status",
            "ch7q": "7q chromosome status",
            "ch10p": "10p chromosome status",
            "ch10q": "10q chromosome status",
            "ch9p": "9p chromosome status",
            "ch9q": "9q chromosome status",
            "ampli_egfr": "EGFR gene amplification",
            "ampli_mdm2": "MDM2 gene amplification",
            "ampli_cdk4": "CDK4 gene amplification",
            "ampli_met": "MET gene amplification",
            "ampli_mdm4": "MDM4 gene amplification",
            "fusion_fgfr": "FGFR gene fusion",
            "fusion_ntrk": "NTRK gene fusion",
            "fusion_autre": "other gene fusion",
        },
        "labels_fr": {
            "ch1p": "statut chromosome 1p",
            "ch19q": "statut chromosome 19q",
            "ch1p19q_codel": "co-délétion 1p/19q",
            "ch7p": "statut chromosome 7p",
            "ch7q": "statut chromosome 7q",
            "ch10p": "statut chromosome 10p",
            "ch10q": "statut chromosome 10q",
            "ch9p": "statut chromosome 9p",
            "ch9q": "statut chromosome 9q",
            "ampli_egfr": "amplification gène EGFR",
            "ampli_mdm2": "amplification gène MDM2",
            "ampli_cdk4": "amplification gène CDK4",
            "ampli_met": "amplification gène MET",
            "ampli_mdm4": "amplification gène MDM4",
            "fusion_fgfr": "fusion gène FGFR",
            "fusion_ntrk": "fusion gène NTRK",
            "fusion_autre": "autre fusion génique",
        },
    },

    # Batch 6: Treatments (18 fields)
    "treatments": {
        "fields": {
            "type_chirurgie", "localisation_chir", "qualite_exerese", "chir_date",
            "chimios", "chimio_protocole", "chm_date_debut", "chm_date_fin", "chm_cycles",
            "rx_date_debut", "rx_date_fin", "rx_dose", "rx_fractionnement", "localisation_radiotherapie",
            "anti_epileptiques", "essai_therapeutique", "corticoides", "optune"
        },
        "anchors": {"diag_histologique", "type_chirurgie", "qualite_exerese", "chimios", "rx_dose"},
        "labels_en": {
            "type_chirurgie": "neurosurgery type",
            "localisation_chir": "surgery anatomical location",
            "qualite_exerese": "resection completeness",
            "chir_date": "tumor resection surgery date",
            "chimios": "chemotherapy drugs",
            "chimio_protocole": "chemotherapy protocol",
            "chm_date_debut": "chemotherapy start date",
            "chm_date_fin": "chemotherapy end date",
            "chm_cycles": "chemotherapy cycles count",
            "rx_date_debut": "radiotherapy start date",
            "rx_date_fin": "radiotherapy end date",
            "rx_dose": "radiotherapy total dose",
            "rx_fractionnement": "radiotherapy fractions count",
            "localisation_radiotherapie": "radiotherapy targeted region",
            "anti_epileptiques": "antiepileptic drugs",
            "essai_therapeutique": "clinical trial name",
            "corticoides": "corticosteroid treatment",
            "optune": "Optune treatment mention",
        },
        "labels_fr": {
            "type_chirurgie": "type chirurgie",
            "localisation_chir": "localisation anatomique chirurgie",
            "qualite_exerese": "qualité exérèse",
            "chir_date": "date opération résection tumorale",
            "chimios": "médicaments chimiothérapie",
            "chimio_protocole": "protocole chimiothérapie",
            "chm_date_debut": "date début chimiothérapie",
            "chm_date_fin": "date fin chimiothérapie",
            "chm_cycles": "nombre cycles chimiothérapie",
            "rx_date_debut": "date début radiothérapie",
            "rx_date_fin": "date fin radiothérapie",
            "rx_dose": "dose totale radiothérapie",
            "rx_fractionnement": "nombre fractions radiothérapie",
            "localisation_radiotherapie": "région ciblée radiothérapie",
            "anti_epileptiques": "médicaments antiépileptiques",
            "essai_therapeutique": "nom essai thérapeutique",
            "corticoides": "traitement corticoïde",
            "optune": "mention traitement Optune",
        },
    },

    # Batch 7: Evolution, Team & Outcome (11 fields)
    "evolution_team_outcome": {
        "fields": {
            "evol_clinique", "progress_clinique", "progress_radiologique", "reponse_radiologique", "date_progression",
            "neuroncologue", "neurochirurgien", "radiotherapeute", "anatomo_pathologiste",
            "infos_deces", "survie_globale"
        },
        "anchors": {"diag_histologique", "evol_clinique", "progress_radiologique", "type_chirurgie"},
        "labels_en": {
            "evol_clinique": "clinical evolution status",
            "progress_clinique": "clinical progression mention",
            "progress_radiologique": "radiological progression mention",
            "reponse_radiologique": "radiological response status",
            "date_progression": "cancer progression date",
            "neuroncologue": "neuro-oncologist family name",
            "neurochirurgien": "neurosurgeon surname",
            "radiotherapeute": "radiotherapist surname",
            "anatomo_pathologiste": "pathologist surname",
            "infos_deces": "death context or cause",
            "survie_globale": "overall survival duration",
        },
        "labels_fr": {
            "evol_clinique": "statut évolution clinique",
            "progress_clinique": "mention progression clinique",
            "progress_radiologique": "mention progression radiologique",
            "reponse_radiologique": "statut réponse radiologique",
            "date_progression": "date progression tumorale",
            "neuroncologue": "nom du neuro-oncologue",
            "neurochirurgien": "nom du neurochirurgien",
            "radiotherapeute": "nom du radiothérapeute",
            "anatomo_pathologiste": "nom anatomo-pathologiste",
            "infos_deces": "contexte ou cause décès",
            "survie_globale": "durée survie globale",
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
# Field descriptions — passed to GLiNER as entity labels for richer semantics
# Sourced from GlinerINPUTS.md
# ---------------------------------------------------------------------------

FIELD_DESCRIPTIONS_EN: dict[str, str] = {
    # 1. Identifiers & Dates
    "date_chir": "Neurosurgery or resection date.",
    "num_labo": "Pathology lab sample number.",
    "date_rcp": "Multidisciplinary team meeting date.",
    "dn_date": "Date of last news or follow-up contact.",
    "date_deces": "Patient date of death.",
    # 2. Demographics
    "sexe": "Patient sex or gender.",
    "annee_de_naissance": "Patient birth year.",
    "activite_professionnelle": "Patient job or occupation.",
    "antecedent_tumoral": "Previous personal brain tumor history.",
    "ik_clinique": "Karnofsky performance score (KPS) or WHO performance status (PS).",
    # 3. Diagnosis
    "diag_histologique": "Pathological brain tumor diagnosis.",
    "diag_integre": "Integrated WHO 2021 diagnosis combining histology and molecular markers.",
    "classification_oms": "WHO classification system used. Referred as the year. (2007, 2016 or 2021)",
    "grade": "CNS WHO tumor grade (1, 2, 3, 4).",
    # 4. Tumor Location
    "tumeur_lateralite": "Tumor laterality (left, right, bilateral).",
    "tumeur_position": "Specific brain lobe/structure containing tumor.",
    "dominance_cerebrale": "Brain hemisphere dominance (handedness).",
    # 5. Radiology
    "exam_radio_date_decouverte": "Date of first radiological exam discovering tumor (not MRI-specific).",
    "contraste_1er_symptome": "Contrast enhancement on first imaging.",
    "prise_de_contraste": "General mention of contrast enhancement (not restricted to current imaging).",
    "oedeme_1er_symptome": "Edema presence on first imaging.",
    "calcif_1er_symptome": "Calcification presence on first imaging.",
    # 6. Symptoms Onset
    "date_1er_symptome": "Date of first clinical symptoms.",
    "epilepsie_1er_symptome": "Seizures/convulsions as first symptom.",
    "ceph_hic_1er_symptome": "Headaches/ICP signs as first symptom.",
    "deficit_1er_symptome": "Neurological deficit at onset (motor, sensory, visual, speech — not motor-only).",
    "cognitif_1er_symptome": "Cognitive issues/confusion as first symptom.",
    "autre_trouble_1er_symptome": "Other non-specific first clinical symptoms.",
    # 7. Symptoms Current
    "epilepsie": "Mention of seizures, epilepsy, or convulsions.",
    "ceph_hic": "Mention of severe headaches or ICP signs.",
    "deficit": "Mention of neurological deficit (motor, sensory, visual, speech — not motor-only).",
    "cognitif": "Mention of cognitive impairment, memory loss, confusion.",
    "autre_trouble": "Mention of other current symptoms.",
    # 8. Histology
    "histo_necrose": "Necrosis presence in histological exam.",
    "histo_pec": "Microvascular/endothelial proliferation presence.",
    "histo_mitoses": "Number of mitoses or mitotic index.",
    "aspect_cellulaire": "Tumor cellular aspect (astrocytic, oligodendroglial).",
    # 9. IHC 1
    "ihc_idh1": "IDH1 R132H protein expression via IHC.",
    "ihc_atrx": "ATRX protein expression (retained/lost).",
    "ihc_p53": "p53 protein expression level/mutation.",
    "ihc_fgfr3": "FGFR3 protein expression status.",
    "ihc_braf": "BRAF V600E protein expression.",
    # 10. IHC 2
    "ihc_gfap": "GFAP protein expression status (positive/negative).",
    "ihc_olig2": "Olig2 protein expression status.",
    "ihc_ki67": "Ki-67 proliferation index percentage (0-100).",
    "ihc_hist_h3k27m": "Histone H3K27M protein expression.",
    "ihc_hist_h3k27me3": "Histone H3K27me3 protein expression.",
    "ihc_egfr_hirsch": "EGFR expression scored via Hirsch scoring system.",
    "ihc_mmr": "Mismatch repair (MMR) proteins expression.",
    # 11. Molecular 1
    "mol_idh1": "IDH1 gene mutation status.",
    "mol_idh2": "IDH2 gene mutation status.",
    "mol_mgmt": "MGMT promoter methylation status.",
    "mol_h3f3a": "H3F3A gene mutation status.",
    "mol_hist1h3b": "HIST1H3B gene mutation status.",
    "mol_tert": "TERT promoter mutation status.",
    "mol_CDKN2A": "CDKN2A homozygous deletion status.",
    # 12. Chromosomal
    "ch1p": "Deletion status of chromosome 1p.",
    "ch19q": "Deletion status of chromosome 19q.",
    "ch1p19q_codel": "Combined 1p/19q co-deletion status.",
    "ch7p": "Gain or loss of chromosome 7p.",
    "ch7q": "Gain or loss of chromosome 7q.",
    "ch10p": "Deletion status of chromosome 10p.",
    "ch10q": "Deletion status of chromosome 10q.",
    "ch9p": "Deletion status of chromosome 9p.",
    "ch9q": "Deletion status of chromosome 9q.",
    # 13. Molecular 2
    "mol_p53": "TP53 gene mutation status.",
    "mol_atrx": "ATRX gene mutation status.",
    "mol_cic": "CIC gene mutation status.",
    "mol_fubp1": "FUBP1 gene mutation status.",
    "mol_fgfr1": "FGFR1 gene mutation status.",
    "mol_egfr_mut": "EGFR gene mutation status.",
    "mol_prkca": "PRKCA gene mutation status.",
    "mol_pten": "PTEN gene mutation status.",
    "mol_braf": "BRAF gene mutation status.",
    # 14. Amplifications & Fusions
    "ampli_egfr": "EGFR gene amplification status.",
    "ampli_mdm2": "MDM2 gene amplification status.",
    "ampli_cdk4": "CDK4 gene amplification status.",
    "ampli_met": "MET gene amplification status.",
    "ampli_mdm4": "MDM4 gene amplification status.",
    "fusion_fgfr": "Fusion status involving FGFR.",
    "fusion_ntrk": "Fusion status involving NTRK.",
    "fusion_autre": "Any other relevant gene fusions.",
    # 15. Treatment Surgery
    "type_chirurgie": "Neurosurgical procedure type (biopsy, resection).",
    "localisation_chir": "Brain region targeted by surgery.",
    "qualite_exerese": "Tumor resection extent (total, partial).",
    "chir_date": "Neurosurgery or tumor resection date.",
    # 16. Treatment Chemo
    "chimios": "Chemotherapy agents administered.",
    "chimio_protocole": "Chemotherapy protocol name (e.g., Stupp).",
    "chm_date_debut": "Chemotherapy start date.",
    "chm_date_fin": "Chemotherapy end date.",
    "chm_cycles": "Number of completed chemotherapy cycles.",
    # 17. Treatment Radio
    "rx_date_debut": "Radiotherapy start date.",
    "rx_date_fin": "Radiotherapy end date.",
    "rx_dose": "Total radiotherapy dose in Grays.",
    "rx_fractionnement": "Number of radiotherapy fractions.",
    "localisation_radiotherapie": "Brain area targeted by radiotherapy.",
    # 18. Adjunct
    "anti_epileptiques": "Anticonvulsant/antiepileptic treatments.",
    "essai_therapeutique": "Clinical trial name.",
    "corticoides": "Corticosteroids administered (e.g., Dexamethasone).",
    "optune": "Mention of Optune/tumor treating fields.",
    # 19. Evolution
    "evol_clinique": "Overall clinical evolution (stable, progressing).",
    "progress_clinique": "Mention of worsening clinical symptoms.",
    "progress_radiologique": "Mention of tumor growth on imaging.",
    "reponse_radiologique": "Tumor response on imaging (shrinkage, stability).",
    "date_progression": "Tumor relapse or progression diagnosis date.",
    # 20. Care Team
    "neuroncologue": "Neuro-oncologist surname. Exclude titles/institutions.",
    "neurochirurgien": "Neurosurgeon surname. Exclude titles/hospitals.",
    "radiotherapeute": "Radiotherapy doctor surname. Exclude titles.",
    "anatomo_pathologiste": "Pathologist surname. Exclude titles.",
    # 21. Outcome
    "infos_deces": "Circumstances or cause of death.",
    "survie_globale": "Overall survival time in months/years.",
}

FIELD_DESCRIPTIONS_FR: dict[str, str] = {
    # 1. Identifiers & Dates
    "date_chir": "Date intervention neurochirurgicale ou résection.",
    "num_labo": "Numéro échantillon laboratoire anatomopathologie.",
    "date_rcp": "Date réunion concertation pluridisciplinaire.",
    "dn_date": "Date dernières nouvelles ou dernier suivi.",
    "date_deces": "Date décès patient.",
    # 2. Demographics
    "sexe": "Sexe ou genre patient.",
    "annee_de_naissance": "Année naissance patient.",
    "activite_professionnelle": "Profession ou métier patient.",
    "antecedent_tumoral": "Antécédent personnel tumeur cérébrale.",
    "ik_clinique": "Score performance Karnofsky (KPS) ou score performance OMS (PS).",
    # 3. Diagnosis
    "diag_histologique": "Diagnostic anatomopathologique tumeur cérébrale.",
    "diag_integre": "Diagnostic intégré OMS 2021 combinant histologie et biologie moléculaire.",
    "classification_oms": "Système de classification OMS utilisé. Référencé avec l'année. (2007, 2016 or 2021)",
    "grade": "Grade tumoral OMS (1, 2, 3, 4).",
    # 4. Tumor Location
    "tumeur_lateralite": "Latéralité tumeur (gauche, droite, bilatérale).",
    "tumeur_position": "Lobe ou structure cérébrale contenant tumeur.",
    "dominance_cerebrale": "Dominance hémisphérique cérébrale (droitier/gaucher).",
    # 5. Radiology
    "exam_radio_date_decouverte": "Date premier examen radiologique découvrant tumeur (non spécifique IRM).",
    "contraste_1er_symptome": "Prise contraste sur première imagerie.",
    "prise_de_contraste": "Mention générale prise de contraste (non restreinte à imagerie actuelle).",
    "oedeme_1er_symptome": "Présence œdème sur première imagerie.",
    "calcif_1er_symptome": "Présence calcification sur première imagerie.",
    # 6. Symptoms Onset
    "date_1er_symptome": "Date apparition premiers symptômes cliniques.",
    "epilepsie_1er_symptome": "Crises ou convulsions comme premier symptôme.",
    "ceph_hic_1er_symptome": "Céphalées ou signes HIC comme premier symptôme.",
    "deficit_1er_symptome": "Déficit neurologique initial (moteur, sensitif, visuel, phasique — pas uniquement moteur).",
    "cognitif_1er_symptome": "Troubles cognitifs/confusion comme premier symptôme.",
    "autre_trouble_1er_symptome": "Autres premiers symptômes cliniques non spécifiques.",
    # 7. Symptoms Current
    "epilepsie": "Mention épilepsie, crises ou convulsions.",
    "ceph_hic": "Mention céphalées sévères ou signes HIC.",
    "deficit": "Mention déficit neurologique (moteur, sensitif, visuel, phasique — pas uniquement moteur).",
    "cognitif": "Mention troubles cognitifs, perte mémoire, confusion.",
    "autre_trouble": "Mention autres symptômes actuels.",
    # 8. Histology
    "histo_necrose": "Présence nécrose dans examen histologique.",
    "histo_pec": "Présence prolifération endothéliale ou microvasculaire.",
    "histo_mitoses": "Nombre mitoses ou index mitotique.",
    "aspect_cellulaire": "Aspect cellulaire tumoral (astrocytaire, oligodendroglial).",
    # 9. IHC 1
    "ihc_idh1": "Expression protéine IDH1 R132H par IHC.",
    "ihc_atrx": "Expression protéine ATRX (conservée/perdue).",
    "ihc_p53": "Niveau expression ou mutation protéine p53.",
    "ihc_fgfr3": "Statut expression protéine FGFR3.",
    "ihc_braf": "Expression protéine BRAF V600E.",
    # 10. IHC 2
    "ihc_gfap": "Statut expression protéine GFAP (positif/négatif).",
    "ihc_olig2": "Statut expression protéine Olig2.",
    "ihc_ki67": "Pourcentage index prolifération Ki-67 (0-100).",
    "ihc_hist_h3k27m": "Expression protéine histone H3K27M.",
    "ihc_hist_h3k27me3": "Expression protéine histone H3K27me3.",
    "ihc_egfr_hirsch": "Expression EGFR évaluée par score de Hirsch.",
    "ihc_mmr": "Expression protéines réparation mésappariements (MMR).",
    # 11. Molecular 1
    "mol_idh1": "Statut mutation gène IDH1.",
    "mol_idh2": "Statut mutation gène IDH2.",
    "mol_mgmt": "Statut méthylation promoteur MGMT.",
    "mol_h3f3a": "Statut mutation gène H3F3A.",
    "mol_hist1h3b": "Statut mutation gène HIST1H3B.",
    "mol_tert": "Statut mutation promoteur TERT.",
    "mol_CDKN2A": "Statut délétion homozygote gène CDKN2A.",
    # 12. Chromosomal
    "ch1p": "Statut délétion chromosome 1p.",
    "ch19q": "Statut délétion chromosome 19q.",
    "ch1p19q_codel": "Statut co-délétion 1p/19q.",
    "ch7p": "Statut gain/perte chromosome 7p.",
    "ch7q": "Statut gain/perte chromosome 7q.",
    "ch10p": "Statut délétion chromosome 10p.",
    "ch10q": "Statut délétion chromosome 10q.",
    "ch9p": "Statut délétion chromosome 9p.",
    "ch9q": "Statut délétion chromosome 9q.",
    # 13. Molecular 2
    "mol_p53": "Statut mutation gène TP53.",
    "mol_atrx": "Statut mutation gène ATRX.",
    "mol_cic": "Statut mutation gène CIC.",
    "mol_fubp1": "Statut mutation gène FUBP1.",
    "mol_fgfr1": "Statut mutation gène FGFR1.",
    "mol_egfr_mut": "Statut mutation gène EGFR.",
    "mol_prkca": "Statut mutation gène PRKCA.",
    "mol_pten": "Statut mutation gène PTEN.",
    "mol_braf": "Statut mutation gène BRAF.",
    # 14. Amplifications & Fusions
    "ampli_egfr": "Statut amplification gène EGFR.",
    "ampli_mdm2": "Statut amplification gène MDM2.",
    "ampli_cdk4": "Statut amplification gène CDK4.",
    "ampli_met": "Statut amplification gène MET.",
    "ampli_mdm4": "Statut amplification gène MDM4.",
    "fusion_fgfr": "Statut fusion impliquant FGFR.",
    "fusion_ntrk": "Statut fusion impliquant NTRK.",
    "fusion_autre": "Autres fusions géniques pertinentes.",
    # 15. Treatment Surgery
    "type_chirurgie": "Type procédure neurochirurgicale (biopsie, exérèse).",
    "localisation_chir": "Région cérébrale ciblée par chirurgie.",
    "qualite_exerese": "Étendue résection tumorale (totale, partielle).",
    "chir_date": "Date opération résection tumorale.",
    # 16. Treatment Chemo
    "chimios": "Agents chimiothérapeutiques administrés.",
    "chimio_protocole": "Nom protocole chimiothérapie (ex: Stupp).",
    "chm_date_debut": "Date début traitement chimiothérapie.",
    "chm_date_fin": "Date fin traitement chimiothérapie.",
    "chm_cycles": "Nombre cycles chimiothérapie complétés.",
    # 17. Treatment Radio
    "rx_date_debut": "Date début traitement radiothérapie.",
    "rx_date_fin": "Date fin traitement radiothérapie.",
    "rx_dose": "Dose totale radiothérapie en Grays.",
    "rx_fractionnement": "Nombre fractions radiothérapie.",
    "localisation_radiotherapie": "Zone cérébrale ciblée par radiothérapie.",
    # 18. Adjunct
    "anti_epileptiques": "Traitements anticonvulsivants/antiépileptiques.",
    "essai_therapeutique": "Nom essai clinique ou thérapeutique.",
    "corticoides": "Corticoïdes administrés (ex: Dexaméthasone).",
    "optune": "Mention dispositif Optune/champs traitement tumoral.",
    # 19. Evolution
    "evol_clinique": "Évolution clinique globale (stable, progression).",
    "progress_clinique": "Mention aggravation symptômes cliniques.",
    "progress_radiologique": "Mention croissance tumorale sur imagerie.",
    "reponse_radiologique": "Réponse tumorale sur imagerie (réduction, stabilité).",
    "date_progression": "Date diagnostic récidive/progression tumorale.",
    # 20. Care Team
    "neuroncologue": "Nom neuro-oncologue traitant. Exclure titres/institutions.",
    "neurochirurgien": "Nom neurochirurgien. Exclure titres/hôpitaux.",
    "radiotherapeute": "Nom médecin radiothérapeute. Exclure titres.",
    "anatomo_pathologiste": "Nom anatomo-pathologiste. Exclure titres.",
    # 21. Outcome
    "infos_deces": "Circonstances ou cause décès.",
    "survie_globale": "Durée survie globale en mois/années.",
}

# Build reverse map: description_text → field_name (for both EN and FR)
_REVERSE_DESC_MAP: dict[str, str] = {}
for _f, _d in FIELD_DESCRIPTIONS_EN.items():
    _REVERSE_DESC_MAP[_d] = _f
for _f, _d in FIELD_DESCRIPTIONS_FR.items():
    _REVERSE_DESC_MAP[_d] = _f


# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "evol_clinique": 0.5,
    "progress_clinique": 0.5,
    "progress_radiologique": 0.5,
}

_DEFAULT_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Heterogeneous batch builder
# ---------------------------------------------------------------------------

# Reverse map: field_name → batch_name (domain)
_FIELD_TO_DOMAIN: dict[str, str] = {}
for _bname, _bconf in SEMANTIC_BATCHES.items():
    for _fname in _bconf["fields"]:
        _FIELD_TO_DOMAIN[_fname] = _bname


def _build_heterogeneous_batches(
    target_fields: set[str],
    language: str,
    max_per_batch: int = 20,  # Maintained at 20 for optimal performance within context limits
) -> list[dict]:
    """Build batches that maximise domain diversity per batch.

    Round-robin picks one field from each domain to build batches of
    *max_per_batch* fields, ensuring labels within each batch come from
    different semantic domains (maximising discrimination).
    """
    desc_map = FIELD_DESCRIPTIONS_FR if language.startswith("fr") else FIELD_DESCRIPTIONS_EN

    # Group target fields by their semantic domain
    domain_queues: dict[str, deque] = {}
    for field in sorted(target_fields):
        domain = _FIELD_TO_DOMAIN.get(field)
        if domain is None:
            continue
        domain_queues.setdefault(domain, deque()).append(field)

    batches: list[dict] = []
    domain_order = list(domain_queues.keys())

    while any(domain_queues.values()):
        batch_fields: set[str] = set()
        batch_labels: dict[str, str] = {}

        for domain_name in list(domain_order):
            q = domain_queues.get(domain_name)
            if not q:
                domain_order.remove(domain_name)
                continue
            if len(batch_fields) >= max_per_batch:
                break

            field = q.popleft()
            batch_fields.add(field)

            # Retrieve description for GLiNER
            batch_labels[field] = desc_map.get(field, field)

        if batch_fields:
            batches.append({
                "fields": batch_fields,
                "labels": batch_labels,
                "anchors": set(),  # no context injection
            })

    return batches


class GlinerExtractor:
    """Primary entity extractor using GLiNER with consolidated 111-field tracking."""

    GLINER_FIELDS: set[str] = _ALL_GLINER_FIELDS

    def __init__(
        self,
        model_name: str = "urchade/gliner_multi-v2.1",
        chunk_size: int = 220,
        chunk_overlap: int = 40,
        batching_strategy: str = "heterogeneous",
        backend: str = "pytorch", # "gliner2_onnx",
        quantize_int8: bool = False,
        use_disambiguator: bool = True,
    ):
        self._model_name = model_name
        self._model = None  # Lazy loading
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._batching_strategy = BatchingStrategy(batching_strategy)
        self._backend = backend
        self._quantize_int8 = quantize_int8
        
        self._disambiguator = None
        if use_disambiguator:
            from src.extraction.disambiguator import Disambiguator
            self._disambiguator = Disambiguator()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
            
        try:
            if self._backend == "gliner2_onnx":
                from gliner2_onnx import GLiNER2ONNXRuntime
                logger.info("Loading gliner2_onnx engine")
                self._model = GLiNER2ONNXRuntime.from_pretrained("lmo3/gliner2-multi-v1-onnx")
                if self._quantize_int8:
                    logger.warning("Dynamic INT8 quantization via initialization arg is currently optimized for the PyTorch backend.")
                return

        except Exception as e:
            logger.warning("Failed to load gliner2_onnx backend, falling back to pytorch: %s", e)
            self._backend = "pytorch"

        # PyTorch fallback
        logger.info("Loading pytorch engine")
        from gliner import GLiNER
        self._model = GLiNER.from_pretrained(self._model_name, load_onnx_model=False)
        
        # Apply dynamic quantization if enabled
        if self._quantize_int8:
            import torch
            logger.info("Applying dynamic INT8 quantization to PyTorch backend")
            self._model = torch.quantization.quantize_dynamic(
                self._model, {torch.nn.Linear}, dtype=torch.qint8
            )
            
        self._model.eval()

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

    def _postprocess_span(self, field_name: str, span_text: str, language: str | None = None) -> Any:
        """Map extracted span to a value based on the field's mapping_type."""
        field_def = ALL_FIELDS_BY_NAME.get(field_name)
        if field_def is None:
            return span_text

        text_lower = span_text.lower().strip()

        # Mode A: Presence logic — entity found → "oui" / negation keywords → "non"
        if field_def.mapping_type == MappingType.PRESENCE:
            _FR_NEG = ["pas ", "pas de", "pas d'", "pas d\u2019", "absence", "manque", "non", "négatif", "neg"]
            _EN_NEG = ["no ", "not ", "without", "absent", "negative", "lack"]
            if any(neg in text_lower for neg in _FR_NEG + _EN_NEG):
                return "non"
            return "oui"

        # Mode B: Similarity logic — match span to closest vocab option
        if field_def.mapping_type == MappingType.SIMILARITY:
            if field_def.allowed_values:
                matched, _score = match_to_vocab(
                    span_text, {str(v) for v in field_def.allowed_values}, 
                    field_name, language,
                )
                return matched
            return span_text

        # DIRECT: return as-is
        return span_text

    def _resolve_batches(
        self,
        target_fields: set[str],
        language: str,
    ) -> list[tuple[str | None, dict]]:
        """Return the batch list based on the selected batching strategy.

        Returns a list of ``(batch_name_or_None, batch_config)`` tuples.
        For HETEROGENEOUS strategy, batch_name is ``None``.
        """
        if self._batching_strategy == BatchingStrategy.HETEROGENEOUS:
            return [
                (None, batch)
                for batch in _build_heterogeneous_batches(target_fields, language)
            ]

        # SEMANTIC_CONTEXT and SEMANTIC_ONLY both use SEMANTIC_BATCHES
        return [
            (name, config) for name, config in SEMANTIC_BATCHES.items()
        ]

    def _process_entity(
        self, 
        label_text: str, 
        score: float, 
        orig_span: str,
        orig_start: int,
        orig_end: int,
        batch_fields: set[str], 
        language: str, 
        extraction_state: dict[str, tuple]
    ):
        field_name = _REVERSE_DESC_MAP.get(label_text)
        if not field_name or field_name not in batch_fields:
            return

        threshold = _CONFIDENCE_THRESHOLDS.get(field_name, _DEFAULT_THRESHOLD)
        if score < threshold:
            return

        norm_val = self._postprocess_span(field_name, orig_span, language)

        existing = extraction_state.get(field_name)
        if existing is None or score > existing[1]:
            extraction_state[field_name] = (norm_val, score, orig_span, orig_start, orig_end)

    def extract(
        self,
        text: str,
        feature_subset: list[str],
        language: str | None = None,
        verbose: bool = False,
    ) -> dict[str, ExtractionValue]:
        if language is None:
            language = self.detect_language(text)

        target_fields = set(feature_subset) & self.GLINER_FIELDS
        if not target_fields:
            return {}

        self._ensure_model()
        
        # 1) Disambiguation logic
        original_text = text
        offset_mapper = lambda x: x
        if getattr(self, "_disambiguator", None):
            text, offset_mapper = self._disambiguator.apply(text, language)
            
        chunks = self._chunk_text(text)
        extraction_state: dict[str, tuple] = {}

        use_context = self._batching_strategy == BatchingStrategy.SEMANTIC_CONTEXT
        batches = self._resolve_batches(target_fields, language)
        desc_map = FIELD_DESCRIPTIONS_FR if language.startswith("fr") else FIELD_DESCRIPTIONS_EN

        # Optional tqdm progress bars
        if verbose:
            try:
                from tqdm import tqdm
            except ImportError:
                tqdm = None
        else:
            tqdm = None

        # Sequential processing for other backends
        chunk_iter = enumerate(chunks)
        if tqdm is not None:
            chunk_iter = tqdm(chunk_iter, total=len(chunks), desc="GLiNER chunks", unit="chunk")

        for chunk_idx, (chunk_text, _chunk_offset) in chunk_iter:
            batch_iter = batches
            if tqdm is not None:
                batch_iter = tqdm(batches, desc=f"  Batches (chunk {chunk_idx+1}/{len(chunks)})", unit="batch", leave=False)

            for batch_name, batch_config in batch_iter:
                batch_fields = batch_config["fields"] & target_fields

                # Skip fields already extracted with high confidence
                batch_fields = {
                    f for f in batch_fields
                    if f not in extraction_state or extraction_state[f][1] < 0.8
                }

                if not batch_fields:
                    continue

                # Context injection only for SEMANTIC_CONTEXT strategy
                if use_context:
                    context_prefix = self._build_context_prefix(batch_config, extraction_state)
                    augmented_text = f"{context_prefix}\n\n{chunk_text}" if context_prefix else chunk_text
                    prefix_len = len(f"{context_prefix}\n\n") if context_prefix else 0
                else:
                    augmented_text = chunk_text
                    prefix_len = 0

                # Resolve descriptions to pass to GLiNER
                labels = [desc_map[f] for f in batch_fields if f in desc_map]

                if not labels:
                    continue

                if self._backend == "gliner2_onnx":
                    entities = self._model.extract_entities(augmented_text, labels, threshold=0.1)
                    for ent in entities:
                        st = ent.start - prefix_len
                        en = ent.end - prefix_len
                        if st < 0: continue
                        
                        glob_mod_start = _chunk_offset + st
                        glob_mod_end = _chunk_offset + en
                        
                        orig_start = offset_mapper(glob_mod_start)
                        orig_end = offset_mapper(glob_mod_end)
                        orig_span = original_text[orig_start:orig_end]
                        
                        self._process_entity(
                            ent.label, ent.score, orig_span, orig_start, orig_end, 
                            batch_fields, language, extraction_state
                        )
                else:
                    entities = self._model.predict_entities(augmented_text, labels, threshold=0.1)
                    for ent in entities:
                        st = ent["start"] - prefix_len
                        en = ent["end"] - prefix_len
                        if st < 0: continue
                        
                        glob_mod_start = _chunk_offset + st
                        glob_mod_end = _chunk_offset + en
                        
                        orig_start = offset_mapper(glob_mod_start)
                        orig_end = offset_mapper(glob_mod_end)
                        orig_span = original_text[orig_start:orig_end]
                        
                        self._process_entity(
                            ent["label"], ent["score"], orig_span, orig_start, orig_end, 
                            batch_fields, language, extraction_state
                        )

        results = {}
        for f, (v, s, span, o_start, o_end) in extraction_state.items():
            results[f] = ExtractionValue(
                value=v,
                source_span=span,
                source_span_start=o_start,
                source_span_end=o_end,
                extraction_tier="gliner",
                confidence=round(float(s), 4),
                vocab_valid=True
            )
        return results