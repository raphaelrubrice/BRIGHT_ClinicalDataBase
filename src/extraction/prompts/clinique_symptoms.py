"""Prompt templates for symptoms feature extraction.

Contains the user prompt and system prompt for extracting first symptoms,
current symptoms, and clinical state from French neuro-oncology documents.
"""

# Fields targeted by this prompt group:
#   date_1er_symptome, epilepsie_1er_symptome, ceph_hic_1er_symptome,
#   deficit_1er_symptome, cognitif_1er_symptome, autre_trouble_1er_symptome,
#   exam_radio_date_decouverte, contraste_1er_symptome,
#   oedeme_1er_symptome, calcif_1er_symptome,
#   epilepsie, ceph_hic, deficit, cognitif, autre_trouble, ik_clinique

SYMPTOMS_FIELDS: list[str] = [
    "date_1er_symptome", "epilepsie_1er_symptome", "ceph_hic_1er_symptome",
    "deficit_1er_symptome", "cognitif_1er_symptome", "autre_trouble_1er_symptome",
    "exam_radio_date_decouverte", "contraste_1er_symptome",
    "oedeme_1er_symptome", "calcif_1er_symptome",
    "epilepsie", "ceph_hic", "deficit", "cognitif", "autre_trouble",
    "ik_clinique",
]

SYMPTOMS_SYSTEM = """\
Tu es un extracteur d'informations médicales spécialisé en neuro-oncologie. \
Tu extrais les symptômes initiaux et actuels à partir de comptes rendus de \
consultation français. Tu distingues soigneusement les symptômes au moment \
du diagnostic initial des symptômes actuels. Tu ne FABRIQUES JAMAIS de données. \
Si une information n'est pas mentionnée dans le texte, retourne null.\
"""

SYMPTOMS_PROMPT = """\
/no_think
Extrais les symptômes initiaux (au diagnostic) et actuels du texte suivant.

Pour chaque champ, retourne :
- La valeur extraite (parmi les valeurs autorisées quand applicable)
- Le passage exact du texte qui justifie ta réponse (dans le champ _source correspondant)

Si une information n'est PAS mentionnée dans le texte, retourne null. Ne FABRIQUE JAMAIS de valeur.
IMPORTANT : Distingue bien les symptômes au moment du PREMIER diagnostic des symptômes ACTUELS.

### Symptômes au premier diagnostic :
- date_1er_symptome: chaîne — date au format JJ/MM/AAAA
- epilepsie_1er_symptome: "oui" | "non" | null — épilepsie au diagnostic
- ceph_hic_1er_symptome: "oui" | "non" | null — céphalées/HTIC au diagnostic
- deficit_1er_symptome: "oui" | "non" | null — déficit neurologique au diagnostic
- cognitif_1er_symptome: "oui" | "non" | null — troubles cognitifs au diagnostic
- autre_trouble_1er_symptome: "oui" | "non" | null — autres troubles au diagnostic

### Imagerie au diagnostic :
- exam_radio_date_decouverte: chaîne — date de l'imagerie de découverte (JJ/MM/AAAA)
- contraste_1er_symptome: "oui" | "non" | null — prise de contraste à la découverte
- oedeme_1er_symptome: "oui" | "non" | null — œdème à la découverte
- calcif_1er_symptome: "oui" | "non" | null — calcifications à la découverte

### État clinique actuel :
- epilepsie: "oui" | "non" | null — épilepsie actuelle
- ceph_hic: "oui" | "non" | null — céphalées/HTIC actuelle
- deficit: "oui" | "non" | null — déficit neurologique actuel
- cognitif: "oui" | "non" | null — troubles cognitifs actuels
- autre_trouble: texte libre | null — autre trouble actuel (description)
- ik_clinique: entier (0-100) | null — indice de Karnofsky

### Règles d'interprétation :
- "crises comitiales", "crises convulsives", "crise épileptique" → épilepsie = "oui"
- "pas d'épilepsie", "absence de crise" → épilepsie = "non"
- "IK", "Karnofsky", "KPS" suivi d'un nombre → ik_clinique
- Normaliser les dates au format JJ/MM/AAAA

### Texte :
{section_text}
"""
