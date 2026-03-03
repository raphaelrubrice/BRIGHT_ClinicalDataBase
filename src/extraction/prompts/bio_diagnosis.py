"""Prompt templates for diagnosis and histological features extraction.

Contains the user prompt and system prompt for extracting diagnosis,
grade, and histological features from French neuro-oncology documents.
"""

# Fields targeted by this prompt group:
#   diag_histologique, diag_integre, classification_oms, grade,
#   histo_necrose, histo_pec, histo_mitoses

DIAGNOSIS_FIELDS: list[str] = [
    "diag_histologique", "diag_integre", "classification_oms", "grade",
    "histo_necrose", "histo_pec", "histo_mitoses",
]

DIAGNOSIS_SYSTEM = """\
Tu es un extracteur d'informations médicales spécialisé en neuropathologie. \
Tu extrais les informations diagnostiques et histologiques à partir de \
comptes rendus anatomopathologiques et de biologie moléculaire français. \
Tu ne FABRIQUES JAMAIS de données. Si une information n'est pas mentionnée \
dans le texte, retourne null. /no_think\
"""

DIAGNOSIS_PROMPT = """\
Extrais les informations diagnostiques et histologiques du texte suivant.

Pour chaque champ, retourne :
- La valeur extraite (parmi les valeurs autorisées quand applicable)
- Le passage exact du texte qui justifie ta réponse (dans le champ _source correspondant)

Si une information n'est PAS mentionnée dans le texte, retourne null. Ne FABRIQUE JAMAIS de valeur.

### Champs à extraire :
- diag_histologique: texte libre — le diagnostic histologique complet (ex: "glioblastome", "astrocytome diffus")
- diag_integre: texte libre — le diagnostic intégré selon la classification OMS (ex: "glioblastome, IDH wild-type")
- classification_oms: "2007" | "2016" | "2021" | null — la version de la classification OMS utilisée
- grade: 1 | 2 | 3 | 4 | null — le grade OMS (entier)
- histo_necrose: "oui" | "non" | null — présence de nécrose
- histo_pec: "oui" | "non" | null — présence de prolifération endothéliocapillaire (PEC)
- histo_mitoses: entier | null — nombre de mitoses (ex: 5, 12)

### Règles d'interprétation :
- Pour le grade, convertir les chiffres romains : I→1, II→2, III→3, IV→4
- "Grade IV" → 4
- "plages de nécrose", "nécrose palissadique" → histo_necrose="oui"
- "prolifération endothéliocapillaire", "PEC" → histo_pec="oui"
- "X mitoses" → histo_mitoses=X (entier)

### Exemples :

Texte : "glioblastome. PEC+, nécrose+, 6 mitoses/10 HPF, Ki67 15-20%.\nClassification OMS 2021, grade 4."
Réponse :
{"values": {"diag_histologique": "glioblastome", "classification_oms": "2021",
            "grade": 4, "histo_necrose": "oui", "histo_pec": "oui", "histo_mitoses": 6,
            "diag_integre": null},
 "_source": {"diag_histologique": "glioblastome", "classification_oms": "OMS 2021",
             "grade": "grade 4", "histo_necrose": "nécrose+", "histo_pec": "PEC+",
             "histo_mitoses": "6 mitoses/10 HPF"}}

RAPPEL CRITIQUE : Il vaut TOUJOURS mieux retourner null qu'une valeur
dont tu n'es pas sûr. Ne déduis pas, n'invente pas.

### Texte :
{section_text}
"""
