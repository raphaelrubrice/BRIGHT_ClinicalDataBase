"""Prompt templates for chromosomal, amplification, and fusion feature extraction.

Contains the user prompt and system prompt for extracting chromosomal
alterations, gene amplifications, and gene fusions from French
neuro-oncology documents.
"""

# Fields targeted by this prompt group:
#   ch1p, ch19q, ch10p, ch10q, ch7p, ch7q, ch9p, ch9q,
#   ampli_mdm2, ampli_cdk4, ampli_egfr, ampli_met, ampli_mdm4,
#   fusion_fgfr, fusion_ntrk, fusion_autre

CHROMOSOMAL_FIELDS: list[str] = [
    "ch1p", "ch19q", "ch10p", "ch10q", "ch7p", "ch7q", "ch9p", "ch9q",
    "ampli_mdm2", "ampli_cdk4", "ampli_egfr", "ampli_met", "ampli_mdm4",
    "fusion_fgfr", "fusion_ntrk", "fusion_autre",
]

CHROMOSOMAL_SYSTEM = """\
Tu es un extracteur d'informations médicales spécialisé en cytogénétique \
et génomique tumorale neuro-oncologique. Tu extrais les altérations \
chromosomiques, amplifications et fusions à partir de comptes rendus \
de CGH-array et de biologie moléculaire en français. Tu ne FABRIQUES \
JAMAIS de données. Si une information n'est pas mentionnée, retourne null.\
"""

CHROMOSOMAL_PROMPT = """\
/no_think
Extrais les altérations chromosomiques, amplifications géniques et fusions du texte suivant.

Pour chaque item, retourne :
- La valeur extraite (parmi les valeurs autorisées)
- Le passage exact du texte qui justifie ta réponse (dans le champ _source correspondant)

Si une altération n'est PAS mentionnée dans le texte, retourne null. Ne FABRIQUE JAMAIS de valeur.

### Altérations chromosomiques :
- ch1p: "gain" | "perte" | "perte partielle" | null
- ch19q: "gain" | "perte" | "perte partielle" | null
- ch10p: "gain" | "perte" | "perte partielle" | null
- ch10q: "gain" | "perte" | "perte partielle" | null
- ch7p: "gain" | "perte" | "perte partielle" | null
- ch7q: "gain" | "perte" | "perte partielle" | null
- ch9p: "gain" | "perte" | "perte partielle" | null
- ch9q: "gain" | "perte" | "perte partielle" | null

### Amplifications géniques :
- ampli_mdm2: "oui" | "non" | null
- ampli_cdk4: "oui" | "non" | null
- ampli_egfr: "oui" | "non" | null
- ampli_met: "oui" | "non" | null
- ampli_mdm4: "oui" | "non" | null

### Fusions géniques :
- fusion_fgfr: "oui" | "non" | null
- fusion_ntrk: "oui" | "non" | null
- fusion_autre: "oui" | "non" | null (toute autre fusion identifiée)

### Règles d'interprétation :
- "codélétion 1p/19q" → ch1p="perte", ch19q="perte"
- "délétion", "deleted" → "perte"
- "perte homozygote" → "perte"
- "perte hétérozygote" → "perte partielle"
- "amplification de X" → ampli_X="oui"
- "pas d'amplification de X" → ampli_X="non"
- "fusion X" ou "réarrangement X" → fusion_X="oui"
- "pas de fusion" → fusion_X="non"

### Texte :
{section_text}
"""
