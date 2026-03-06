"""Prompt templates for care team feature extraction."""

CARE_TEAM_FIELDS: list[str] = [
    "neuroncologue",
    "neurochirurgien",
    "radiotherapeute",
    "activite_professionnelle",
]

CARE_TEAM_SYSTEM = """\
Tu es un extracteur d'informations médicales. Tu extrais les noms des médecins \
et la profession du patient à partir de comptes rendus médicaux français. \
Tu ne FABRIQUES JAMAIS de données. Si une information n'est pas mentionnée \
dans le texte, retourne null.\
"""

CARE_TEAM_PROMPT = """\
Extrais les noms des médecins et la profession du patient à partir du texte suivant.

Pour chaque champ, retourne :
- La valeur extraite
- Le passage exact du texte qui justifie ta réponse (dans le champ _source correspondant)

Si une information n'est PAS mentionnée dans le texte, retourne null. Ne FABRIQUE JAMAIS de valeur.

### Champs à extraire :
- neuroncologue: texte libre — nom du neuro-oncologue
- neurochirurgien: texte libre — nom du neurochirurgien
- radiotherapeute: texte libre — nom du radiothérapeute
- activite_professionnelle: texte libre — profession ou activité du patient

### Règles de base :
- Pour les médecins, NE RETOURNE QUE LE NOM DE FAMILLE (retire "Dr", "Pr", le prénom).
- Exemple : "Dr Jean Dupont" -> "Dupont", "Pr. Martin" -> "Martin".
- NE RETOURNE JAMAIS un token pseudonymisé comme valeur extraite (ex: [NOM_123]). Si tu ne trouves qu'un token, retourne null.

### Exemples (extraits de documents réels) :

Texte : "Patient adressé par le Dr TOUAT (neuro-oncologie). Chirurgie réalisée par le Pr Mathon à la Salpêtrière. Patiente avocate."
Réponse :
{"values": {"neuroncologue": "TOUAT", "neurochirurgien": "Mathon", "radiotherapeute": null, "activite_professionnelle": "avocate"},
 "_source": {"neuroncologue": "Dr TOUAT", "neurochirurgien": "Pr Mathon", "activite_professionnelle": "avocate"}}

Texte : "Consultation post-op dr [NOM_456]. RT par Dr Hoang-Xuan."
Réponse :
{"values": {"neuroncologue": null, "neurochirurgien": null, "radiotherapeute": "Hoang-Xuan", "activite_professionnelle": null},
 "_source": {"radiotherapeute": "Dr Hoang-Xuan"}}

RAPPEL CRITIQUE : Seul le nom de famille des médecins doit être extrait. Si le nom est pseudonymisé, retourne null.

### Texte :
{section_text}
"""
