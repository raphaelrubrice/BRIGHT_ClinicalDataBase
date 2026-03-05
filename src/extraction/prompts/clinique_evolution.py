"""Prompt templates for clinical evolution and progression feature extraction.

Contains the user prompt and system prompt for extracting clinical
evolution, progression status, tumour location, and outcome information
from French neuro-oncology documents.
"""

# Fields targeted by this prompt group:
#   infos_deces

EVOLUTION_FIELDS: list[str] = [
    "infos_deces",
]

EVOLUTION_SYSTEM = """\
Tu es un extracteur d'informations médicales spécialisé en neuro-oncologie. \
Tu extrais les informations de suivi à partir de comptes rendus de consultation \
et de RCP français. Tu ne FABRIQUES JAMAIS de données. Si une information \
n'est pas mentionnée dans le texte, retourne null. /no_think\
"""

EVOLUTION_PROMPT = """\
Extrais les informations de décès du texte suivant.

Pour chaque champ, retourne :
- La valeur extraite
- Le passage exact du texte qui justifie ta réponse (dans le champ _source correspondant)

Si une information n'est PAS mentionnée dans le texte, retourne null. Ne FABRIQUE JAMAIS de valeur.

### Décès :
- infos_deces: texte libre — circonstances du décès

### Règles sur les données pseudonymisées :
- Ne retourne JAMAIS un token de type [XXX_yyy] comme valeur extraite.

### Exemples :

Texte : "Patient vu le 15/03/2025 en consultation de suivi. Décédé le 20/03/2025 des suites de sa maladie."
Réponse :
{"values": {"infos_deces": "des suites de sa maladie"},
 "_source": {"infos_deces": "des suites de sa maladie"}}

RAPPEL CRITIQUE : Il vaut TOUJOURS mieux retourner null qu'une valeur
dont tu n'es pas sûr. Ne déduis pas, n'invente pas.

### Texte :
{section_text}
"""
