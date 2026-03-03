"""Prompt templates for treatment feature extraction.

Contains the user prompt and system prompt for extracting surgery,
chemotherapy, radiotherapy, and adjunct treatment information from
French neuro-oncology documents.
"""

# Fields targeted by this prompt group:
#   chimios, chm_date_debut, chm_date_fin, chm_cycles,
#   chir_date, type_chirurgie,
#   rx_date_debut, rx_date_fin, rx_dose,
#   anti_epileptiques, essai_therapeutique, corticoides, optune

TREATMENT_FIELDS: list[str] = [
    "chimios", "chm_date_debut", "chm_date_fin", "chm_cycles",
    "chir_date", "type_chirurgie",
    "rx_date_debut", "rx_date_fin", "rx_dose",
    "anti_epileptiques", "essai_therapeutique", "corticoides", "optune",
]

TREATMENT_SYSTEM = """\
Tu es un extracteur d'informations médicales spécialisé en neuro-oncologie. \
Tu extrais les informations thérapeutiques (chirurgie, chimiothérapie, \
radiothérapie, traitements adjuvants) à partir de comptes rendus de \
consultation et de RCP français. Tu distingues les traitements en cours \
des traitements historiques. Tu ne FABRIQUES JAMAIS de données. \
Si une information n'est pas mentionnée dans le texte, retourne null. /no_think\
"""

TREATMENT_PROMPT = """\
Extrais les informations thérapeutiques du texte suivant.

Pour chaque champ, retourne :
- La valeur extraite (parmi les valeurs autorisées quand applicable)
- Le passage exact du texte qui justifie ta réponse (dans le champ _source correspondant)

Si une information n'est PAS mentionnée dans le texte, retourne null. Ne FABRIQUE JAMAIS de valeur.
IMPORTANT : Distingue les traitements ACTUELS/EN COURS des traitements HISTORIQUES.

### Chirurgie :
- chir_date: chaîne — date de chirurgie au format JJ/MM/AAAA
- type_chirurgie: "exerese complete" | "exerese partielle" | "exerese" | "biopsie" | "en attente" | null

### Chimiothérapie :
- chimios: texte libre — nom(s) des chimiothérapies (ex: "témozolomide", "TMZ + avastin")
- chm_date_debut: chaîne — date début chimio au format JJ/MM/AAAA
- chm_date_fin: chaîne — date fin chimio au format JJ/MM/AAAA
- chm_cycles: entier | null — nombre de cycles de chimiothérapie

### Radiothérapie :
- rx_date_debut: chaîne — date début radiothérapie au format JJ/MM/AAAA
- rx_date_fin: chaîne — date fin radiothérapie au format JJ/MM/AAAA
- rx_dose: chaîne — dose en Gy (ex: "60", "59.4") ou "non"/"oui"/"en attente"

### Traitements adjuvants :
- anti_epileptiques: "oui" | "non" | null — sous anti-épileptiques
- essai_therapeutique: "oui" | "non" | null — inclusion dans un essai thérapeutique
- corticoides: "oui" | "non" | null — sous corticoïdes
- optune: "oui" | "non" | null — utilisation d'Optune (TTFields)

### Règles d'interprétation :
- "TMZ", "Témodal" → chimios = "témozolomide"
- "biopsie stéréotaxique" → type_chirurgie = "biopsie"
- "exérèse totale", "résection complète" → type_chirurgie = "exerese complete"
- "exérèse subtotale", "résection partielle" → type_chirurgie = "exerese partielle"
- Normaliser les dates au format JJ/MM/AAAA

### Règles sur les données pseudonymisées :
- Les dates pseudonymisées (AAAA-??-??) doivent être retournées comme null.
- Les lieux sous forme [HOPITAL_xxx] ou [VILLE_xxx] doivent être retournés comme null.
- Ne retourne JAMAIS un token de type [XXX_yyy] comme valeur extraite.

### Exemples :

Texte : "radiochimiothérapie (60 Gy en 30 fractions du 26/11/2024 au 09/01/2025), puis Temozolomide adjuvant 4 cycles, dispositif Optune."
Réponse :
{"values": {"rx_dose": "60", "rx_date_debut": "26/11/2024", "rx_date_fin": "09/01/2025",
            "chimios": "témozolomide", "chm_cycles": 4, "optune": "oui",
            "chir_date": null, "type_chirurgie": null, "chm_date_debut": null,
            "chm_date_fin": null, "anti_epileptiques": null, "essai_therapeutique": null,
            "corticoides": null},
 "_source": {"rx_dose": "60 Gy", "rx_date_debut": "du 26/11/2024",
             "rx_date_fin": "au 09/01/2025", "chimios": "Temozolomide adjuvant",
             "chm_cycles": "4 cycles", "optune": "Optune"}}

Pour les dates : normalise au format JJ/MM/AAAA.
- Dates pseudonymisées (AAAA-??-??) → null
- "12 février 2024" → "12/02/2024"

RAPPEL CRITIQUE : Il vaut TOUJOURS mieux retourner null qu'une valeur
dont tu n'es pas sûr. Ne déduis pas, n'invente pas.

### Texte :
{section_text}
"""
