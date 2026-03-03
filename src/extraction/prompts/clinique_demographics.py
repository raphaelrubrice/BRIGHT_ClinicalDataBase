"""Prompt templates for demographics and care team feature extraction.

Contains the user prompt and system prompt for extracting patient
demographics and care team information from French neuro-oncology
consultation notes.
"""

# Fields targeted by this prompt group:
#   nip, date_de_naissance, sexe, activite_professionnelle,
#   antecedent_tumoral, neuroncologue, neurochirurgien,
#   radiotherapeute, localisation_radiotherapie, localisation_chir

DEMOGRAPHICS_FIELDS: list[str] = [
    "nip", "date_de_naissance", "sexe", "activite_professionnelle",
    "antecedent_tumoral", "neuroncologue", "neurochirurgien",
    "radiotherapeute", "localisation_radiotherapie", "localisation_chir",
]

DEMOGRAPHICS_SYSTEM = """\
Tu es un extracteur d'informations médicales. Tu extrais les informations \
démographiques et d'équipe soignante à partir de comptes rendus de consultation \
en neuro-oncologie français. Tu ne FABRIQUES JAMAIS de données. Si une \
information n'est pas mentionnée dans le texte, retourne null. /no_think\
"""

DEMOGRAPHICS_PROMPT = """\
Extrais les informations démographiques et d'équipe soignante du texte suivant.

Pour chaque champ, retourne :
- La valeur extraite
- Le passage exact du texte qui justifie ta réponse (dans le champ _source correspondant)

Si une information n'est PAS mentionnée dans le texte, retourne null. Ne FABRIQUE JAMAIS de valeur.

### Champs à extraire :
- nip: chaîne — identifiant patient (NIP, IPP, numéro de dossier)
- date_de_naissance: chaîne — date de naissance au format JJ/MM/AAAA
- sexe: "M" | "F" | null
- activite_professionnelle: texte libre — profession du patient
- antecedent_tumoral: "oui" | "non" | null — antécédent de tumeur
- neuroncologue: texte libre — nom du neuro-oncologue
- neurochirurgien: texte libre — nom du neurochirurgien
- radiotherapeute: texte libre — nom du radiothérapeute
- localisation_radiotherapie: texte libre — lieu de la radiothérapie
- localisation_chir: texte libre — lieu de la chirurgie

### Règles d'interprétation :
- Pour le sexe : "homme", "masculin", "Mr", "M." → "M" ; "femme", "féminin", "Mme", "Mlle" → "F"
- Normaliser les dates au format JJ/MM/AAAA
- "Dr", "Pr", "Docteur", "Professeur" précédé d'un nom → nom du médecin

### Règles sur les données pseudonymisées :
- Les noms de médecins peuvent apparaître sous forme pseudonymisée : [NOM_xxx], [PRENOM_xxx].
  Si tu ne trouves que des identifiants pseudonymisés pour neuroncologue, neurochirurgien ou
  radiotherapeute, retourne null pour ces champs.
- Les lieux peuvent apparaître sous forme [HOPITAL_xxx] ou [VILLE_xxx].
  Si tu ne trouves que des identifiants pseudonymisés pour localisation_radiotherapie ou
  localisation_chir, retourne null.
- Les dates pseudonymisées (AAAA-??-??) doivent être retournées comme null.
- Ne retourne JAMAIS un token de type [XXX_yyy] comme valeur extraite.

### Exemples (extraits de documents réels) :

Texte : "Référents :\n- Neurochirurgie : Dr Mathon, PSL\n- Neuro-oncologie : Dr Touat, PSL\n- Radiothérapie : Dr Assouline, Boulogne\nMonsieur X, né le 1977-??-??, 47 ans, Agent immobilier"
Réponse :
{"values": {"neurochirurgien": "Mathon", "neuroncologue": "Touat",
            "radiotherapeute": "Assouline", "localisation_chir": "PSL",
            "localisation_radiotherapie": "Boulogne", "sexe": "M",
            "activite_professionnelle": "Agent immobilier",
            "nip": null, "date_de_naissance": null, "antecedent_tumoral": null},
 "_source": {"neurochirurgien": "Dr Mathon", "neuroncologue": "Dr Touat",
             "radiotherapeute": "Dr Assouline", "localisation_chir": "PSL",
             "localisation_radiotherapie": "Boulogne", "sexe": "Monsieur",
             "activite_professionnelle": "Agent immobilier"}}

Pour les dates : normalise au format JJ/MM/AAAA.
- Dates pseudonymisées (AAAA-??-??) → null
- "12 février 2024" → "12/02/2024"

RAPPEL CRITIQUE : Il vaut TOUJOURS mieux retourner null qu'une valeur
dont tu n'es pas sûr. Ne déduis pas, n'invente pas.

### Texte :
{section_text}
"""
