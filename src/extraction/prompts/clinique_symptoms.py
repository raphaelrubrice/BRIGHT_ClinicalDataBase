"""Prompt templates for symptoms feature extraction."""

SYMPTOMS_FIELDS: list[str] = [
    "ik_clinique",
]

SYMPTOMS_SYSTEM = """\
Tu es un extracteur d'informations médicales. Tu extrais l'indice de Karnofsky \
à partir de comptes rendus de neuro-oncologie français. Tu ne FABRIQUES JAMAIS de données. \
Si l'indice n'est pas mentionné, retourne null.\
"""

SYMPTOMS_PROMPT = """\
Extrais l'information clinique du texte suivant.

Pour chaque champ, retourne :
- La valeur extraite
- Le passage exact du texte qui justifie ta réponse (dans le champ _source correspondant)

Si une information n'est PAS mentionnée dans le texte, retourne null. Ne FABRIQUE JAMAIS de valeur.

### Champs à extraire :
- ik_clinique: entier (0 à 100) — Score ou Indice de performance de Karnofsky (IK, KPS)

### Exemples (extraits de documents réels) :

Texte : "Le patient est en bon état général, Karnofsky à 80%."
Réponse :
{"values": {"ik_clinique": 80},
 "_source": {"ik_clinique": "Karnofsky à 80%"}}

Texte : "IK évalué à 70."
Réponse :
{"values": {"ik_clinique": 70},
 "_source": {"ik_clinique": "IK évalué à 70."}}

Texte : "OMS 1, pas de plainte."
Réponse :
{"values": {"ik_clinique": null},
 "_source": {}}

RAPPEL CRITIQUE : Seul le score de Karnofsky doit être extrait pour ce champ. Ne convertis pas le score OMS.

### Texte :
{section_text}
"""
