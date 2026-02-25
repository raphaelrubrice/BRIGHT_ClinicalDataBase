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
Si une information n'est pas mentionnée dans le texte, retourne null.\
"""

TREATMENT_PROMPT = """\
/no_think
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

### Texte :
{section_text}
"""
