"""Prompt templates for IHC (immunohistochemistry) feature extraction.

Contains the user prompt and system prompt for extracting IHC markers
from French neuro-oncology documents using Qwen3-4B via Ollama.
"""

# Fields targeted by this prompt group:
#   ihc_idh1, ihc_p53, ihc_atrx, ihc_fgfr3, ihc_braf,
#   ihc_hist_h3k27m, ihc_hist_h3k27me3, ihc_egfr_hirsch,
#   ihc_gfap, ihc_olig2, ihc_ki67, ihc_mmr

IHC_FIELDS: list[str] = [
    "ihc_idh1", "ihc_p53", "ihc_atrx", "ihc_fgfr3", "ihc_braf",
    "ihc_hist_h3k27m", "ihc_hist_h3k27me3", "ihc_egfr_hirsch",
    "ihc_gfap", "ihc_olig2", "ihc_ki67", "ihc_mmr",
]

IHC_SYSTEM = """\
Tu es un extracteur d'informations médicales spécialisé en neuropathologie. \
Tu extrais les résultats d'immunohistochimie (IHC) à partir de comptes rendus \
anatomopathologiques français. Tu ne FABRIQUES JAMAIS de données. \
Si une information n'est pas mentionnée dans le texte, retourne null. /no_think\
"""

IHC_PROMPT = """\
Extrais les résultats d'immunohistochimie (IHC) du texte suivant.

Pour chaque marqueur, retourne :
- La valeur extraite (parmi les valeurs autorisées)
- Le passage exact du texte qui justifie ta réponse (dans le champ _source correspondant)

Si un marqueur n'est PAS mentionné dans le texte, retourne null. Ne FABRIQUE JAMAIS de valeur.
Distingue les résultats actuels des résultats historiques (antérieurs).

### Marqueurs à extraire :
- ihc_idh1: "positif" | "negatif" | null
- ihc_p53: "positif" | "negatif" | null
- ihc_atrx: "positif" | "negatif" | "maintenu" | null
- ihc_fgfr3: "positif" | "negatif" | null
- ihc_braf: "positif" | "negatif" | null
- ihc_hist_h3k27m: "positif" | "negatif" | null
- ihc_hist_h3k27me3: "positif" | "negatif" | "maintenu" | null
- ihc_egfr_hirsch: score Hirsch (chaîne, entier 0-3, ou "positif"/"negatif") | null
- ihc_gfap: "positif" | "negatif" | null
- ihc_olig2: "positif" | "negatif" | null
- ihc_ki67: pourcentage (chaîne, ex: "15", "5-10", "<5") | null
- ihc_mmr: "positif" | "negatif" | "maintenu" | null

### Règles d'interprétation :
- "perte d'expression" → "negatif"
- "expression conservée" ou "maintenu" → "maintenu" (pour ATRX, H3K27me3, MMR)
- "expression conservée" → "positif" (pour les autres marqueurs)
- "+", "positive" → "positif"
- "-", "negative", "négative" → "negatif"

### Exemples :

Texte : "IHC : IDH1 R132H positif, ATRX perte d'expression, p53 positif diffus (>50%), GFAP positif, Olig2 négatif, Ki67 15-20%."
Réponse :
{"values": {"ihc_idh1": "positif", "ihc_atrx": "negatif", "ihc_p53": "positif",
            "ihc_gfap": "positif", "ihc_olig2": "negatif", "ihc_ki67": "15-20",
            "ihc_fgfr3": null, "ihc_braf": null, "ihc_hist_h3k27m": null,
            "ihc_hist_h3k27me3": null, "ihc_egfr_hirsch": null, "ihc_mmr": null},
 "_source": {"ihc_idh1": "IDH1 R132H positif", "ihc_atrx": "ATRX perte d'expression",
             "ihc_p53": "p53 positif diffus (>50%)", "ihc_gfap": "GFAP positif",
             "ihc_olig2": "Olig2 négatif", "ihc_ki67": "Ki67 15-20%"}}

RAPPEL CRITIQUE : Il vaut TOUJOURS mieux retourner null qu'une valeur
dont tu n'es pas sûr. Ne déduis pas, n'invente pas.

### Texte :
{section_text}
"""
