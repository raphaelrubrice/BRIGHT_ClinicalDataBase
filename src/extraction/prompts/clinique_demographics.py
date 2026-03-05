"""Prompt templates for demographics and care team feature extraction.

Contains the user prompt and system prompt for extracting patient
demographics and care team information from French neuro-oncology
consultation notes.
"""

# Fields targeted by this prompt group:
#   localisation_radiotherapie, localisation_chir

DEMOGRAPHICS_FIELDS: list[str] = [
    "localisation_radiotherapie", "localisation_chir",
]

DEMOGRAPHICS_SYSTEM = """\
Tu es un extracteur d'informations médicales. Tu extrais les informations \
démographiques et d'équipe soignante à partir de comptes rendus de consultation \
en neuro-oncologie français. Tu ne FABRIQUES JAMAIS de données. Si une \
information n'est pas mentionnée dans le texte, retourne null. /no_think\
"""

DEMOGRAPHICS_PROMPT = """\
Extrais les informations de traitement du texte suivant.

Pour chaque champ, retourne :
- La valeur extraite
- Le passage exact du texte qui justifie ta réponse (dans le champ _source correspondant)

Si une information n'est PAS mentionnée dans le texte, retourne null. Ne FABRIQUE JAMAIS de valeur.

### Champs à extraire :
- localisation_radiotherapie: texte libre — lieu de la radiothérapie
- localisation_chir: texte libre — lieu de la chirurgie

### Règles sur les données pseudonymisées :
- Les lieux peuvent apparaître sous forme [HOPITAL_xxx] ou [VILLE_xxx].
  Si tu ne trouves que des identifiants pseudonymisés pour localisation_radiotherapie ou
  localisation_chir, retourne null.
- Ne retourne JAMAIS un token de type [XXX_yyy] comme valeur extraite.

### Exemples (extraits de documents réels) :

Texte : "Référents :\n- Neurochirurgie : Dr Mathon, PSL\n- Neuro-oncologie : Dr Touat, PSL\n- Radiothérapie : Dr Assouline, Boulogne"
Réponse :
{"values": {"localisation_chir": "PSL", "localisation_radiotherapie": "Boulogne"},
 "_source": {"localisation_chir": "PSL", "localisation_radiotherapie": "Boulogne"}}

RAPPEL CRITIQUE : Il vaut TOUJOURS mieux retourner null qu'une valeur
dont tu n'es pas sûr. Ne déduis pas, n'invente pas.

### Texte :
{section_text}
"""
