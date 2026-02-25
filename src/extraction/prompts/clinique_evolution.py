"""Prompt templates for clinical evolution and progression feature extraction.

Contains the user prompt and system prompt for extracting clinical
evolution, progression status, tumour location, and outcome information
from French neuro-oncology documents.
"""

# Fields targeted by this prompt group:
#   dn_date, evol_clinique,
#   progress_clinique, progress_radiologique, date_progression,
#   tumeur_lateralite, tumeur_position,
#   date_deces, infos_deces

EVOLUTION_FIELDS: list[str] = [
    "dn_date", "evol_clinique",
    "progress_clinique", "progress_radiologique", "date_progression",
    "tumeur_lateralite", "tumeur_position",
    "date_deces", "infos_deces",
]

EVOLUTION_SYSTEM = """\
Tu es un extracteur d'informations médicales spécialisé en neuro-oncologie. \
Tu extrais les informations d'évolution clinique, de progression tumorale, \
de localisation tumorale et de suivi à partir de comptes rendus de consultation \
et de RCP français. Tu ne FABRIQUES JAMAIS de données. Si une information \
n'est pas mentionnée dans le texte, retourne null.\
"""

EVOLUTION_PROMPT = """\
/no_think
Extrais les informations d'évolution clinique et de suivi du texte suivant.

Pour chaque champ, retourne :
- La valeur extraite (parmi les valeurs autorisées quand applicable)
- Le passage exact du texte qui justifie ta réponse (dans le champ _source correspondant)

Si une information n'est PAS mentionnée dans le texte, retourne null. Ne FABRIQUE JAMAIS de valeur.

### Évolution et suivi :
- dn_date: chaîne — date de dernière nouvelle au format JJ/MM/AAAA
- evol_clinique: "initial" | "P1" | "P2" | "P3" | ... | "terminal" | null — stade d'évolution

### Progression :
- progress_clinique: "oui" | "non" | null — progression clinique
- progress_radiologique: "oui" | "non" | null — progression radiologique
- date_progression: chaîne — date de progression au format JJ/MM/AAAA

### Localisation tumorale :
- tumeur_lateralite: "gauche" | "droite" | "bilateral" | "median" | null
- tumeur_position: texte libre — localisation anatomique (ex: "frontal droit", "temporale gauche")

### Décès :
- date_deces: chaîne — date de décès au format JJ/MM/AAAA
- infos_deces: texte libre — circonstances du décès

### Règles d'interprétation :
- "première récidive", "1ère progression" → evol_clinique = "P1"
- "deuxième récidive" → evol_clinique = "P2"
- "diagnostic initial", "découverte" → evol_clinique = "initial"
- "phase terminale", "soins palliatifs" → evol_clinique = "terminal"
- "hémisphère gauche", "côté gauche" → tumeur_lateralite = "gauche"
- "hémisphère droit", "côté droit" → tumeur_lateralite = "droite"
- "bilatéral", "deux hémisphères" → tumeur_lateralite = "bilateral"
- "médian", "ligne médiane", "vermis" → tumeur_lateralite = "median"
- Normaliser les dates au format JJ/MM/AAAA

### Texte :
{section_text}
"""
