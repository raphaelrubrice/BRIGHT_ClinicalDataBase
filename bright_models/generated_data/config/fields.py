"""111 field definitions, per-doc-type relevance, and structural guides."""

import sys
from pathlib import Path

# Import the canonical field list from profiles_validation.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from profiles_validation import ALL_111_FIELDS  # noqa: E402

# ═════════════════════════════════════════════════════════════════════════════
# FIELD DESCRIPTIONS (French) — used in LLM prompts
# ═════════════════════════════════════════════════════════════════════════════

FIELD_DESCRIPTIONS_FR: dict[str, str] = {
    # 1. Identifiants & Dates
    "date_chir": "Date intervention neurochirurgicale ou résection",
    "num_labo": "Numéro échantillon laboratoire anatomopathologie",
    "date_rcp": "Date réunion concertation pluridisciplinaire",
    "dn_date": "Date dernières nouvelles ou dernier suivi",
    "date_deces": "Date décès patient (seulement si décédé)",
    # 2. Démographie
    "sexe": "Sexe du patient (M ou F)",
    "annee_de_naissance": "Année de naissance (entier)",
    "activite_professionnelle": "Profession ou métier du patient",
    "antecedent_tumoral": "Antécédent personnel de tumeur cérébrale",
    "ik_clinique": "Score Karnofsky (KPS 0-100) ou score OMS (PS 0-4)",
    # 3. Diagnostic
    "diag_histologique": "Diagnostic anatomopathologique",
    "diag_integre": "Diagnostic intégré OMS 2021",
    "classification_oms": "Classification OMS utilisée (2007, 2016 ou 2021)",
    "grade": "Grade OMS (1, 2, 3 ou 4)",
    # 4. Localisation tumorale
    "tumeur_lateralite": "Latéralité (gauche, droite ou bilatérale)",
    "tumeur_position": "Lobe ou structure cérébrale",
    "dominance_cerebrale": "Dominance hémisphérique (droitier, gaucher)",
    # 5. Radiologie
    "exam_radio_date_decouverte": "Date premier examen découvrant la tumeur",
    "contraste_1er_symptome": "Prise de contraste sur première imagerie",
    "prise_de_contraste": "Mention générale prise de contraste",
    "oedeme_1er_symptome": "Présence œdème sur première imagerie",
    "calcif_1er_symptome": "Présence calcification sur première imagerie",
    # 6. Symptômes initiaux
    "date_1er_symptome": "Date apparition premiers symptômes",
    "epilepsie_1er_symptome": "Crises comme premier symptôme",
    "ceph_hic_1er_symptome": "Céphalées/HIC comme premier symptôme",
    "deficit_1er_symptome": "Déficit neurologique initial",
    "cognitif_1er_symptome": "Troubles cognitifs comme premier symptôme",
    "autre_trouble_1er_symptome": "Autres premiers symptômes",
    # 7. Symptômes actuels
    "epilepsie": "Mention épilepsie/crises",
    "ceph_hic": "Mention céphalées/HIC",
    "deficit": "Mention déficit neurologique",
    "cognitif": "Mention troubles cognitifs",
    "autre_trouble": "Autres symptômes actuels",
    # 8. Histologie
    "histo_necrose": "Présence nécrose",
    "histo_pec": "Prolifération endothéliale/microvasculaire",
    "histo_mitoses": "Nombre mitoses ou index mitotique",
    "aspect_cellulaire": "Aspect cellulaire (astrocytaire, oligodendroglial)",
    # 9. IHC partie 1
    "ihc_idh1": "Expression IDH1 R132H (positif/négatif)",
    "ihc_atrx": "ATRX (conservé/perdu)",
    "ihc_p53": "Expression p53",
    "ihc_fgfr3": "Expression FGFR3",
    "ihc_braf": "Expression BRAF V600E",
    # 10. IHC partie 2
    "ihc_gfap": "GFAP (positif/négatif)",
    "ihc_olig2": "Olig2",
    "ihc_ki67": "Index Ki-67 (0-100%)",
    "ihc_hist_h3k27m": "Expression H3K27M",
    "ihc_hist_h3k27me3": "Expression H3K27me3",
    "ihc_egfr_hirsch": "EGFR score de Hirsch",
    "ihc_mmr": "Protéines réparation mésappariements",
    # 11. Moléculaire partie 1
    "mol_idh1": "Statut mutation IDH1",
    "mol_idh2": "Statut mutation IDH2",
    "mol_mgmt": "Méthylation promoteur MGMT",
    "mol_h3f3a": "Mutation H3F3A",
    "mol_hist1h3b": "Mutation HIST1H3B",
    "mol_tert": "Mutation promoteur TERT",
    "mol_CDKN2A": "Délétion homozygote CDKN2A",
    # 12. Chromosomique
    "ch1p": "Délétion 1p",
    "ch19q": "Délétion 19q",
    "ch1p19q_codel": "Codélétion 1p/19q",
    "ch7p": "Gain/perte 7p",
    "ch7q": "Gain/perte 7q",
    "ch10p": "Délétion 10p",
    "ch10q": "Délétion 10q",
    "ch9p": "Délétion 9p",
    "ch9q": "Délétion 9q",
    # 13. Moléculaire partie 2
    "mol_p53": "Mutation p53",
    "mol_atrx": "Mutation ATRX",
    "mol_cic": "Mutation CIC",
    "mol_fubp1": "Mutation FUBP1",
    "mol_fgfr1": "Mutation FGFR1",
    "mol_egfr_mut": "Mutation EGFR",
    "mol_prkca": "Mutation PRKCA",
    "mol_pten": "Mutation PTEN",
    "mol_braf": "Mutation BRAF",
    # 14. Amplifications & Fusions
    "ampli_egfr": "Amplification EGFR",
    "ampli_mdm2": "Amplification MDM2",
    "ampli_cdk4": "Amplification CDK4",
    "ampli_met": "Amplification MET",
    "ampli_mdm4": "Amplification MDM4",
    "fusion_fgfr": "Fusion FGFR",
    "fusion_ntrk": "Fusion NTRK",
    "fusion_autre": "Autre fusion",
    # 15. Traitement chirurgie
    "type_chirurgie": "Type procédure (biopsie, exérèse partielle, exérèse totale)",
    "localisation_chir": "Région cérébrale ciblée par la chirurgie",
    "qualite_exerese": "Étendue résection (totale, subtotale, partielle)",
    "chir_date": "Date opération chirurgicale",
    # 16. Traitement chimiothérapie
    "chimios": "Agents chimiothérapeutiques",
    "chimio_protocole": "Protocole chimiothérapie (Stupp, PCV, etc.)",
    "chm_date_debut": "Date début chimiothérapie",
    "chm_date_fin": "Date fin chimiothérapie",
    "chm_cycles": "Nombre de cycles de chimiothérapie",
    # 17. Traitement radiothérapie
    "rx_date_debut": "Date début radiothérapie",
    "rx_date_fin": "Date fin radiothérapie",
    "rx_dose": "Dose totale en Grays",
    "rx_fractionnement": "Nombre de fractions",
    "localisation_radiotherapie": "Zone ciblée par la radiothérapie",
    # 18. Traitements adjuvants
    "anti_epileptiques": "Anticonvulsivants prescrits",
    "essai_therapeutique": "Nom essai clinique",
    "corticoides": "Corticoïdes prescrits",
    "optune": "Dispositif Optune/TTFields",
    # 19. Évolution
    "evol_clinique": "Évolution globale (stable, progression)",
    "progress_clinique": "Aggravation symptômes",
    "progress_radiologique": "Croissance tumorale imagerie",
    "reponse_radiologique": "Réponse tumorale imagerie",
    "date_progression": "Date récidive/progression",
    # 20. Équipe soignante
    "neuroncologue": "Nom du neuro-oncologue",
    "neurochirurgien": "Nom du neurochirurgien",
    "radiotherapeute": "Nom du radiothérapeute",
    "anatomo_pathologiste": "Nom de l'anatomopathologiste",
    # 21. Devenir
    "infos_deces": "Circonstances décès",
    "survie_globale": "Durée survie en mois",
}

# ═════════════════════════════════════════════════════════════════════════════
# PER-DOCUMENT-TYPE FIELD RELEVANCE
# ═════════════════════════════════════════════════════════════════════════════

# Fields expected to appear in each document type.
# A generated document should annotate a subset of these.

_DEMOGRAPHICS = [
    "sexe", "annee_de_naissance", "activite_professionnelle",
    "antecedent_tumoral", "ik_clinique",
]
_DIAGNOSIS = [
    "diag_histologique", "diag_integre", "classification_oms", "grade",
]
_TUMOR_LOC = [
    "tumeur_lateralite", "tumeur_position", "dominance_cerebrale",
]
_RADIOLOGY = [
    "exam_radio_date_decouverte", "contraste_1er_symptome",
    "prise_de_contraste", "oedeme_1er_symptome", "calcif_1er_symptome",
]
_SYMPTOMS_INITIAL = [
    "date_1er_symptome", "epilepsie_1er_symptome", "ceph_hic_1er_symptome",
    "deficit_1er_symptome", "cognitif_1er_symptome", "autre_trouble_1er_symptome",
]
_SYMPTOMS_CURRENT = [
    "epilepsie", "ceph_hic", "deficit", "cognitif", "autre_trouble",
]
_HISTOLOGY = [
    "histo_necrose", "histo_pec", "histo_mitoses", "aspect_cellulaire",
]
_IHC = [
    "ihc_idh1", "ihc_atrx", "ihc_p53", "ihc_fgfr3", "ihc_braf",
    "ihc_gfap", "ihc_olig2", "ihc_ki67", "ihc_hist_h3k27m",
    "ihc_hist_h3k27me3", "ihc_egfr_hirsch", "ihc_mmr",
]
_MOLECULAR = [
    "mol_idh1", "mol_idh2", "mol_mgmt", "mol_h3f3a", "mol_hist1h3b",
    "mol_tert", "mol_CDKN2A",
    "mol_p53", "mol_atrx", "mol_cic", "mol_fubp1", "mol_fgfr1",
    "mol_egfr_mut", "mol_prkca", "mol_pten", "mol_braf",
]
_CHROMOSOMAL = [
    "ch1p", "ch19q", "ch1p19q_codel", "ch7p", "ch7q",
    "ch10p", "ch10q", "ch9p", "ch9q",
]
_AMPLIFICATIONS = [
    "ampli_egfr", "ampli_mdm2", "ampli_cdk4", "ampli_met", "ampli_mdm4",
    "fusion_fgfr", "fusion_ntrk", "fusion_autre",
]
_SURGERY = [
    "type_chirurgie", "localisation_chir", "qualite_exerese",
    "chir_date", "date_chir", "num_labo",
]
_CHEMO = [
    "chimios", "chimio_protocole", "chm_date_debut", "chm_date_fin", "chm_cycles",
]
_RADIO = [
    "rx_date_debut", "rx_date_fin", "rx_dose", "rx_fractionnement",
    "localisation_radiotherapie",
]
_ADJUVANT = [
    "anti_epileptiques", "essai_therapeutique", "corticoides", "optune",
]
_EVOLUTION = [
    "evol_clinique", "progress_clinique", "progress_radiologique",
    "reponse_radiologique", "date_progression",
]
_TEAM = [
    "neuroncologue", "neurochirurgien", "radiotherapeute", "anatomo_pathologiste",
]
_OUTCOME = [
    "infos_deces", "survie_globale", "date_deces", "dn_date",
]
_DATES = [
    "date_chir", "date_rcp", "dn_date", "date_deces",
]

DOC_TYPE_FIELDS: dict[str, list[str]] = {
    "consultation": sorted(set(
        _DEMOGRAPHICS + _DIAGNOSIS + _TUMOR_LOC + _RADIOLOGY
        + _SYMPTOMS_INITIAL + _SYMPTOMS_CURRENT
        + _SURGERY + _CHEMO + _RADIO + _ADJUVANT + _EVOLUTION
        + _TEAM + _OUTCOME + _DATES
        # Consultation may mention key molecular results in discussion
        + ["ihc_idh1", "mol_idh1", "mol_mgmt", "ch1p19q_codel"]
    )),
    "rcp": sorted(set(ALL_111_FIELDS)),  # RCP discusses everything
    "anapath": sorted(set(
        _DIAGNOSIS + _HISTOLOGY + _IHC + _MOLECULAR + _CHROMOSOMAL
        + _AMPLIFICATIONS + _SURGERY
        + ["sexe", "annee_de_naissance", "tumeur_position", "tumeur_lateralite"]
    )),
}

# ═════════════════════════════════════════════════════════════════════════════
# CRITICAL FIELDS — zero tolerance for value_fallback in span resolution
# ═════════════════════════════════════════════════════════════════════════════

CRITICAL_FIELDS: list[str] = [
    "diag_histologique", "diag_integre", "grade",
    "ihc_idh1", "ch1p19q_codel", "mol_idh1",
]

# ═════════════════════════════════════════════════════════════════════════════
# DOCUMENT STRUCTURAL GUIDES — section ordering per document type
# ═════════════════════════════════════════════════════════════════════════════

DOC_STRUCTURES: dict[str, list[str]] = {
    "consultation": [
        "En-tête (hôpital, service, date, médecin)",
        "Identité patient (nom, prénom, date de naissance, âge)",
        "Motif de consultation",
        "Histoire de la maladie",
        "Examen clinique (poids, taille, IK/PS, déficits)",
        "Résultats récents (IRM, biologie moléculaire)",
        "Discussion / Proposition thérapeutique",
        "Conclusion et suivi prévu",
        "Signature médecin",
    ],
    "rcp": [
        "En-tête RCP (date, participants, lieu)",
        "Présentation du dossier (histoire résumée)",
        "Données histologiques et moléculaires",
        "Imagerie actuelle",
        "Discussion collégiale",
        "Proposition thérapeutique validée",
        "Participants et signatures",
    ],
    "anapath": [
        "En-tête laboratoire (service, adresse, n° de labo)",
        "Identité patient et chirurgien demandeur",
        "Renseignements cliniques",
        "Examen macroscopique (prélèvements reçus, taille)",
        "Examen microscopique (architecture, cellularité, mitoses)",
        "Étude immunohistochimique (IDH1, ATRX, p53, Ki67, GFAP, Olig2...)",
        "Biologie moléculaire (si résultats disponibles)",
        "Conclusion diagnostique (classification OMS, grade)",
    ],
}

# Few-shot example mapping: doc_type -> list of example filenames
FEW_SHOT_MAP: dict[str, list[str]] = {
    "consultation": ["consultation_example_1.json"],
    "rcp": ["rcp_example_1.json"],
    "anapath": ["anapath_example_1.json"],
}
