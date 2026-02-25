"""Prompt templates for molecular biology feature extraction.

Contains the user prompt and system prompt for extracting molecular
biology results from French neuro-oncology documents.
"""

# Fields targeted by this prompt group:
#   mol_idh1, mol_idh2, mol_tert, mol_CDKN2A, mol_h3f3a,
#   mol_hist1h3b, mol_braf, mol_mgmt, mol_fgfr1, mol_egfr_mut,
#   mol_prkca, mol_p53, mol_pten, mol_cic, mol_fubp1, mol_atrx

MOLECULAR_FIELDS: list[str] = [
    "mol_idh1", "mol_idh2", "mol_tert", "mol_CDKN2A", "mol_h3f3a",
    "mol_hist1h3b", "mol_braf", "mol_mgmt", "mol_fgfr1", "mol_egfr_mut",
    "mol_prkca", "mol_p53", "mol_pten", "mol_cic", "mol_fubp1", "mol_atrx",
]

MOLECULAR_SYSTEM = """\
Tu es un extracteur d'informations médicales spécialisé en biologie moléculaire \
neuro-oncologique. Tu extrais les statuts moléculaires (mutations, variants, \
méthylation) à partir de comptes rendus français. Tu ne FABRIQUES JAMAIS de données. \
Si une information n'est pas mentionnée dans le texte, retourne null.\
"""

MOLECULAR_PROMPT = """\
/no_think
Extrais les résultats de biologie moléculaire du texte suivant.

Pour chaque gène, retourne :
- La valeur extraite (parmi les valeurs autorisées)
- Le passage exact du texte qui justifie ta réponse (dans le champ _source correspondant)

Si un gène n'est PAS mentionné dans le texte, retourne null. Ne FABRIQUE JAMAIS de valeur.
Distingue les résultats actuels des résultats historiques.

### Gènes à extraire :
- mol_idh1: "wt" | "mute" | variant (ex: "R132H") | null
- mol_idh2: "wt" | "mute" | variant | null
- mol_tert: "wt" | "mute" | variant (ex: "C228T", "C250T") | null
- mol_CDKN2A: "wt" | "mute" | "mute + delete" | null
- mol_h3f3a: "wt" | "mute" | variant (ex: "K27M", "G34R") | null
- mol_hist1h3b: "wt" | "mute" | variant | null
- mol_braf: "wt" | "mute" | variant (ex: "V600E") | null
- mol_mgmt: "methyle" | "non methyle" | null
- mol_fgfr1: "wt" | "mute" | variant | null
- mol_egfr_mut: "wt" | "mute" | variant | null
- mol_prkca: "wt" | "mute" | variant | null
- mol_p53: "wt" | "mute" | variant | null
- mol_pten: "wt" | "mute" | variant | null
- mol_cic: "wt" | "mute" | variant | null
- mol_fubp1: "wt" | "mute" | variant | null
- mol_atrx: "wt" | "mute" | variant | null

### Règles d'interprétation :
- "wild-type", "sauvage", "type sauvage", "non muté(e)", "absence de mutation" → "wt"
- "muté(e)", "mutation", "présence de mutation" → "mute"
- Si un variant spécifique est mentionné (ex: "IDH1 R132H"), retourne "mute"
- Pour MGMT : "méthylé" → "methyle", "non méthylé" → "non methyle"
- "pas de mutation" → "wt"

### Texte :
{section_text}
"""
