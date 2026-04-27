"""Per-field vocabulary definitions for the ControlledExtractor.

Each field with a controlled vocabulary gets a ``FieldVocabConfig`` that
specifies:

* **identification_list**, surface forms to *find* in the document
  (marker names, gene names, chromosome arms, …).
* **category_specific_dict**, for each allowed category value, a list
  of surface-form terms expected in the surrounding context.

Two registries are provided:

* ``CONTROLLED_REGISTRY_FR``, French clinical documents.
* ``CONTROLLED_REGISTRY_EN``, English clinical documents (unused, all
  documents are French; kept for potential future use).

Categories with value ``"autre"`` or ``"NA"`` are intentionally **not**
included: ``"autre"`` gets an empty list (free-text → no robust
heuristic) and ``"NA"`` is the implicit fallback when no category
matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ───────────────────────────────────────────────────────────────────────
# Data-class
# ───────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FieldVocabConfig:
    """Configuration for one controlled-vocabulary field."""

    field_name: str
    """Canonical field name (e.g. ``"ch1p"``)."""

    identification_list: tuple[str, ...]
    """Terms used to *locate* this field's mention in the document."""

    category_specific_dict: dict[str, tuple[str, ...]]
    """Mapping ``category_value → tuple_of_surface_forms``."""

    context_half_window: int = 80
    """Characters before/after an identification hit to scan for
    category terms."""

    id_fuzzy_threshold: int = 80
    """Minimum ``partial_ratio`` score for the Find step."""

    cat_fuzzy_threshold: int = 75
    """Minimum ``partial_ratio`` score for the Check step."""


# ═══════════════════════════════════════════════════════════════════════
# Helper: invert a {surface_form: canonical} dict → {canonical: [forms]}
# ═══════════════════════════════════════════════════════════════════════

def _invert(norm_dict: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for surface, canonical in norm_dict.items():
        out.setdefault(canonical, []).append(surface)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Shared category term pools (FR)
# ═══════════════════════════════════════════════════════════════════════

_IHC_CATS_FR: dict[str, tuple[str, ...]] = {
    "positif": (
        "positif", "positive", "positifs", "+",
        "surexprime", "surexpression", "exprime", "present",
        "surexprimé", "exprimé", "présent",
        "marquage diffus", "marquage intense",
        "forte reactivite", "forte réactivité",
        "marquage net", "marquage franc", "expression franche",
    ),
    "negatif": (
        "negatif", "negative", "négatif", "négative", "-",
        "perte d'expression", "absent", "absence d'expression",
        "perte", "non exprime", "non exprimé",
        "non detecte", "non détecté", "non retrouve", "non retrouvé",
        "perdu",
        "perte du marquage nucleaire", "perte du marquage nucléaire",
        "perte d'expression nucleaire", "perte d'expression nucléaire",
        "negativation", "négativation",
        "extinction", "perte totale", "marquage negatif", "marquage négatif",
    ),
    "maintenu": (
        "maintenu", "maintenue", "conserve", "conservé",
        "conservée", "conservee",
        "expression conservee", "expression conservée",
        "expression maintenue",
        "preserve", "préservé", "normal",
        "marquage nucleaire conserve", "marquage nucléaire conservé",
        "expression nucleaire conservee", "expression nucléaire conservée",
        "marquage preserve", "marquage préservé",
        "marquage non altere", "marquage non altéré", "sans anomalie de marquage",
    ),
    # "autre" intentionally empty
}

_MOL_CATS_FR: dict[str, tuple[str, ...]] = {
    "wt": (
        "wt", "wild-type", "wild type", "sauvage", "type sauvage",
        "sequence sauvage", "séquence sauvage",
        "non mute", "non muté", "non mutée", "non mutee",
        "absence de mutation", "pas de mutation",
        "pas de mutation detectee", "pas de mutation détectée",
        "absence de mutation detectee", "absence de mutation détectée",
        "statut wt", "status wt",
        "sans argument pour une mutation",
        "ne met pas en evidence de mutation", "ne met pas en évidence de mutation",
        "aucun variant pathogene", "aucun variant pathogène",
        "absence d'anomalie",
        "profil non mute", "profil non muté",
        "aucune mutation",
        "sans anomalie", "profil sauvage", "non porteuse", "non porteur",
        "aucun variant détecté", "aucun variant detecte",
    ),
    "mute": (
        "mute", "muté", "mutée", "mutee", "mutation",
        "presence de mutation", "présence de mutation",
        "mutation detectee", "mutation détectée",
        "mutation identifiee", "mutation identifiée",
        "variant pathogene", "variant pathogène",
        "altere", "altéré", "alteree", "altérée",
        "mutation faux sens", "mutation non sens", "positif pour la mutation",
    ),
    # "vus": (
    #     "vus", "variant de signification indeterminee", "variant de signification indéterminée",
    #     "variant de signification incertaine", "polymorphisme",
    #     "non pathogene", "non pathogène",
    #     "variante de signification inconnue", "variant atypique", "signification clinique incertaine",
    # ),
    # "autre" intentionally empty
}

_MOL_MGMT_CATS_FR: dict[str, tuple[str, ...]] = {
    **_MOL_CATS_FR,
    "methyle": (
        "methyle", "méthylé", "methylé",
        "methylation positive", "méthylation positive",
        "promoteur methyle", "promoteur méthylé",
        "methylation du promoteur", "méthylation du promoteur",
        "hypermethyle", "hyperméthylé",
        "hyperméthylation", "hypermethylation", "profil methylé", "profil méthylé",
    ),
    "non methyle": (
        "non methyle", "non méthylé", "non methylé",
        "methylation negative", "méthylation négative",
        "non methylation",
        "absence de methylation", "absence de méthylation",
        "absence de methylation du promoteur",
        "absence de méthylation du promoteur",
        "methylation absente", "méthylation absente",
        "non hypermethyle", "non hyperméthylé",
        "promoteur non methyle", "promoteur non méthylé",
        "profil non methylé", "profil non méthylé", 
        "absence d'hyperméthylation", "absence d'hypermethylation",
    ),
}

_CHR_CATS_FR: dict[str, tuple[str, ...]] = {
    "gain": (
        "gain", "gain de signal", "polysomie", "trisomie",
        "normal", "normale", "sur-représentation", "sur-representation",
    ),
    "perte": (
        "perte", "deletion", "délétion", "deleted", "del",
        "monosomie", "perte de signal",
        "perte d'heterozygotie", "perte d'hétérozygotie",
        "loh", "perte allelique", "perte allélique",
        "perte homozygote",
        "perte d'un allele", "perte d'un allèle",
        "sous-représentation", "sous-representation",
    ),
    "perte partielle": (
        "perte partielle", "perte heterozygote", "perte hétérozygote",
        "deletion focale", "délétion focale",
        "deletion partielle", "délétion partielle",
        "perte focale",
        "perte hemizygote", "perte hémizygote",
    ),
}

_AMPLI_CATS_FR: dict[str, tuple[str, ...]] = {
    "oui": (
        "amplification", "amplifie", "amplifié", "amplifiée",
        "amplification detectee", "amplification détectée",
        "presence d'amplification", "présence d'amplification",
        "sur-amplification", "co-amplification",
    ),
    "non": (
        "pas d'amplification", "absence d'amplification",
        "non amplifie", "non amplifié",
        "pas d'amplification detectee", "pas d'amplification détectée",
        "sans argument pour une amplification",
        "ne met pas en evidence d'amplification", "ne met pas en évidence d'amplification",
        "aucune amplification", "absence d'anomalie",
        "pas de sur-amplification", "non amplifiee", "non amplifiée",
    ),
}

_FUSION_CATS_FR: dict[str, tuple[str, ...]] = {
    "oui": (
        "fusion", "rearrangement", "réarrangement", "translocation",
        "fusion detectee", "fusion détectée",
        "presence de fusion", "présence de fusion",
    ),
    "non": (
        "pas de fusion", "absence de fusion",
        "pas de rearrangement", "pas de réarrangement",
        "absence de rearrangement", "absence de réarrangement",
        "aucune fusion", "sans argument pour une fusion",
    ),
}

_BINARY_CATS_FR: dict[str, tuple[str, ...]] = {
    "oui": (
        "oui", "présent", "present", "positif", "positive",
        "retrouve", "retrouvé", "mis en evidence", "mis en évidence",
    ),
    "non": (
        "non", "absent", "absence", "négatif", "negatif", "negative",
        "pas de", "aucun", "sans",
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# Shared category term pools (EN)
# ═══════════════════════════════════════════════════════════════════════

_IHC_CATS_EN: dict[str, tuple[str, ...]] = {
    "positif": (
        "positive", "pos", "+", "overexpressed", "overexpression",
        "expressed", "present", "detected",
        "diffuse staining", "intense staining", "strong reactivity",
        "frank expression", "strong expression", "positive staining",
    ),
    "negatif": (
        "negative", "neg", "-", "loss of expression", "absent",
        "not expressed", "not detected", "lost", "loss",
        "loss of nuclear expression", "loss of staining", "negative staining",
        "loss of reactivity", "no staining",
    ),
    "maintenu": (
        "retained", "preserved", "maintained", "intact", "normal",
        "retained nuclear expression", "preserved nuclear expression",
        "ubiquitous expression", "wild-type expression", "unaltered staining",
    ),
}

_MOL_CATS_EN: dict[str, tuple[str, ...]] = {
    "wt": (
        "wt", "wild-type", "wild type", "wildtype",
        "not mutated", "no mutation", "no mutation detected",
        "absence of mutation", "unmutated",
        "no evidence of mutation", "failed to demonstrate a mutation",
        "negative for mutation", "no pathogenic variant",
        "no alterations", "wild type profile", "negative for pathogenic variants", 
        "no variants detected",
    ),
    "mute": (
        "mutated", "mutant", "mutation", "mutation detected",
        "mutation identified", "pathogenic variant",
        "altered", "alteration",
        "missense mutation", "nonsense mutation", "frameshift", "positive for mutation",
    ),
    # "vus": (
    #     "vus", "variant of uncertain significance",
    #     "variant of unknown significance", "polymorphism",
    #     "non-pathogenic", "non pathogenic",
    #     "variant of unknown clinical significance", "atypical variant",
    # ),
}

_MOL_MGMT_CATS_EN: dict[str, tuple[str, ...]] = {
    **_MOL_CATS_EN,
    "methyle": (
        "methylated", "methylation positive",
        "promoter methylated", "promoter methylation",
        "hypermethylated",
        "hypermethylation", "methylated profile",
    ),
    "non methyle": (
        "unmethylated", "not methylated", "methylation negative",
        "no methylation", "absence of methylation",
        "promoter unmethylated",
        "unmethylated profile", "no hypermethylation",
    ),
}

_CHR_CATS_EN: dict[str, tuple[str, ...]] = {
    "gain": (
        "gain", "gain of signal", "polysomy", "trisomy", "normal",
        "overrepresentation",
    ),
    "perte": (
        "loss", "deletion", "deleted", "del", "monosomy",
        "loss of signal", "loss of heterozygosity", "loh",
        "allelic loss", "homozygous loss",
        "underrepresentation", "loss of one allele",
    ),
    "perte partielle": (
        "partial loss", "heterozygous loss",
        "focal deletion", "partial deletion",
        "focal loss", "hemizygous loss",
    ),
}

_AMPLI_CATS_EN: dict[str, tuple[str, ...]] = {
    "oui": (
        "amplification", "amplified", "amplification detected",
    ),
    "non": (
        "no amplification", "not amplified",
        "absence of amplification", "no evidence of amplification",
    ),
}

_FUSION_CATS_EN: dict[str, tuple[str, ...]] = {
    "oui": (
        "fusion", "rearrangement", "translocation",
        "fusion detected",
    ),
    "non": (
        "no fusion", "absence of fusion",
        "no rearrangement", "absence of rearrangement",
        "no evidence of fusion",
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# FR REGISTRY
# ═══════════════════════════════════════════════════════════════════════

def _cfg(name: str, ids: tuple[str, ...],
         cats: dict[str, tuple[str, ...]],
         half_win: int = 80,
         id_thr: int = 80,
         cat_thr: int = 75) -> FieldVocabConfig:
    return FieldVocabConfig(
        field_name=name,
        identification_list=ids,
        category_specific_dict=cats,
        context_half_window=half_win,
        id_fuzzy_threshold=id_thr,
        cat_fuzzy_threshold=cat_thr,
    )


CONTROLLED_REGISTRY_FR: dict[str, FieldVocabConfig] = {

    # ── IHC fields ────────────────────────────────────────────────────
    "ihc_idh1": _cfg("ihc_idh1",
        ids=("idh1", "idh-1", "idh 1"),
        cats=_IHC_CATS_FR),

    "ihc_p53": _cfg("ihc_p53",
        ids=("p53", "tp53", "proteine p53", "protéine p53", "anti-p53"),
        cats=_IHC_CATS_FR),

    "ihc_atrx": _cfg("ihc_atrx",
        ids=("atrx", "anti-atrx"),
        cats=_IHC_CATS_FR),

    "ihc_fgfr3": _cfg("ihc_fgfr3",
        ids=("fgfr3", "fgfr-3"),
        cats=_IHC_CATS_FR),

    "ihc_braf": _cfg("ihc_braf",
        ids=("braf",),
        cats=_IHC_CATS_FR),

    "ihc_hist_h3k27m": _cfg("ihc_hist_h3k27m",
        ids=("h3k27m", "h3 k27m", "h3.3 k27m", "h3.3k27m",
             "histone h3 k27m", "h3 k27"),
        cats=_IHC_CATS_FR),

    "ihc_hist_h3k27me3": _cfg("ihc_hist_h3k27me3",
        ids=("h3k27me3", "h3 k27me3"),
        cats=_IHC_CATS_FR),

    "ihc_egfr_hirsch": _cfg("ihc_egfr_hirsch",
        ids=("egfr hirsch", "score hirsch", "hirsch"),
        cats=_IHC_CATS_FR),

    "ihc_gfap": _cfg("ihc_gfap",
        ids=("gfap",),
        cats=_IHC_CATS_FR),

    "ihc_olig2": _cfg("ihc_olig2",
        ids=("olig2",),
        cats=_IHC_CATS_FR),

    "ihc_ki67": _cfg("ihc_ki67",
        ids=("ki67", "ki-67", "ki 67", "index de proliferation",
             "index de prolifération", "mib1", "mib-1"),
        cats=_IHC_CATS_FR),

    "ihc_mmr": _cfg("ihc_mmr",
        ids=("mmr", "mlh1", "msh2", "msh6", "pms2",
             "dmmr", "pmmr", "deficit mmr", "déficit mmr"),
        cats=_IHC_CATS_FR),

    # ── Molecular fields ──────────────────────────────────────────────
    "mol_idh1": _cfg("mol_idh1",
        ids=("idh1", "idh-1", "idh 1", "r132h", "r132c", "r132s",
             "r132g", "r132l"),
        cats=_MOL_CATS_FR),

    "mol_idh2": _cfg("mol_idh2",
        ids=("idh2", "idh-2", "idh 2", "r172k", "r172m", "r172w"),
        cats=_MOL_CATS_FR),

    "mol_tert": _cfg("mol_tert",
        ids=("tert", "promoteur tert", "promoteur du tert",
             "c228t", "c250t"),
        cats=_MOL_CATS_FR),

    "mol_CDKN2A": _cfg("mol_CDKN2A",
        ids=("cdkn2a", "p16", "cdkn2a/b",
             "deletion homozygote cdkn2a",
             "délétion homozygote cdkn2a"),
        cats=_MOL_CATS_FR),

    "mol_h3f3a": _cfg("mol_h3f3a",
        ids=("h3f3a", "h3.3", "k27m", "g34r", "g34v"),
        cats=_MOL_CATS_FR),

    "mol_hist1h3b": _cfg("mol_hist1h3b",
        ids=("hist1h3b", "h3.1"),
        cats=_MOL_CATS_FR),

    "mol_braf": _cfg("mol_braf",
        ids=("braf", "v600e", "v600"),
        cats=_MOL_CATS_FR),

    "mol_mgmt": _cfg("mol_mgmt",
        ids=("mgmt", "promoteur mgmt", "promoteur du mgmt",
             "methylation mgmt", "méthylation mgmt", "o6-methylguanine"),
        cats=_MOL_MGMT_CATS_FR),

    "mol_fgfr1": _cfg("mol_fgfr1",
        ids=("fgfr1",),
        cats=_MOL_CATS_FR),

    "mol_egfr_mut": _cfg("mol_egfr_mut",
        ids=("egfr",),
        cats=_MOL_CATS_FR),

    "mol_prkca": _cfg("mol_prkca",
        ids=("prkca", "prkc alpha"),
        cats=_MOL_CATS_FR),

    "mol_p53": _cfg("mol_p53",
        ids=("p53", "tp53"),
        cats=_MOL_CATS_FR),

    "mol_pten": _cfg("mol_pten",
        ids=("pten",),
        cats=_MOL_CATS_FR),

    "mol_cic": _cfg("mol_cic",
        ids=("cic",),
        cats=_MOL_CATS_FR),

    "mol_fubp1": _cfg("mol_fubp1",
        ids=("fubp1",),
        cats=_MOL_CATS_FR),

    "mol_atrx": _cfg("mol_atrx",
        ids=("atrx",),
        cats=_MOL_CATS_FR),

    # ── Chromosomal fields ────────────────────────────────────────────
    "ch1p": _cfg("ch1p",
        ids=("1p", "chromosome 1p", "del(1p)", "bras court du chromosome 1",
             "codeletion 1p", "codélétion 1p",
             "1p/19q", "1p19q", "codeletion 1p/19q", "codélétion 1p/19q"),
        cats=_CHR_CATS_FR),

    "ch19q": _cfg("ch19q",
        ids=("19q", "chromosome 19q", "del(19q)", "bras long du chromosome 19",
             "codeletion 19q", "codélétion 19q",
             "1p/19q", "1p19q", "codeletion 1p/19q", "codélétion 1p/19q"),
        cats=_CHR_CATS_FR),

    "ch10p": _cfg("ch10p",
        ids=("10p", "chromosome 10p"),
        cats=_CHR_CATS_FR),

    "ch10q": _cfg("ch10q",
        ids=("10q", "chromosome 10q"),
        cats=_CHR_CATS_FR),

    "ch7p": _cfg("ch7p",
        ids=("7p", "chromosome 7p"),
        cats=_CHR_CATS_FR),

    "ch7q": _cfg("ch7q",
        ids=("7q", "chromosome 7q"),
        cats=_CHR_CATS_FR),

    "ch9p": _cfg("ch9p",
        ids=("9p", "chromosome 9p"),
        cats=_CHR_CATS_FR),

    "ch9q": _cfg("ch9q",
        ids=("9q", "chromosome 9q"),
        cats=_CHR_CATS_FR),

    # ── Amplification fields ──────────────────────────────────────────
    "ampli_mdm2": _cfg("ampli_mdm2",
        ids=("amplification mdm2", "mdm2 amplifie", "mdm2 amplifié", "mdm2"),
        cats=_AMPLI_CATS_FR),

    "ampli_cdk4": _cfg("ampli_cdk4",
        ids=("amplification cdk4", "cdk4 amplifie", "cdk4 amplifié", "cdk4"),
        cats=_AMPLI_CATS_FR),

    "ampli_egfr": _cfg("ampli_egfr",
        ids=("amplification egfr", "egfr amplifie", "egfr amplifié", "egfr"),
        cats=_AMPLI_CATS_FR),

    "ampli_met": _cfg("ampli_met",
        ids=("amplification met", "met amplifie", "met amplifié", "met"),
        cats=_AMPLI_CATS_FR),

    "ampli_mdm4": _cfg("ampli_mdm4",
        ids=("amplification mdm4", "mdm4 amplifie", "mdm4 amplifié", "mdm4"),
        cats=_AMPLI_CATS_FR),

    # ── Fusion fields ─────────────────────────────────────────────────
    "fusion_fgfr": _cfg("fusion_fgfr",
        ids=("fusion fgfr", "rearrangement fgfr", "réarrangement fgfr",
             "translocation fgfr", "fgfr"),
        cats=_FUSION_CATS_FR),

    "fusion_ntrk": _cfg("fusion_ntrk",
        ids=("fusion ntrk", "rearrangement ntrk", "réarrangement ntrk",
             "translocation ntrk", "ntrk"),
        cats=_FUSION_CATS_FR),

    "fusion_autre": _cfg("fusion_autre",
        ids=("fusion alk", "fusion ros1", "fusion met", "fusion braf",
             "rearrangement alk", "réarrangement alk",
             "translocation alk", "translocation ros1"),
        cats=_FUSION_CATS_FR),

    # ── Other categorical fields ──────────────────────────────────────
    "sexe": _cfg("sexe",
        ids=("sexe", "genre", "monsieur", "madame", "mme", "mr",
             "homme", "femme", "masculin", "féminin", "feminin"),
        cats={
            "M": ("homme", "masculin", "monsieur", "mr", "male", "mâle"),
            "F": ("femme", "féminin", "feminin", "madame", "mme",
                  "female", "fille"),
        },
        half_win=50),

    "tumeur_lateralite": _cfg("tumeur_lateralite",
        ids=("lateralite", "latéralité", "cote", "côté",
             "hemisphere", "hémisphère", "gauche", "droit", "droite",
             "bilateral", "bilatéral", "median", "médian"),
        cats={
            "gauche": ("gauche", "left", "g"),
            "droite": ("droite", "droit", "right", "d"),
            "bilateral": ("bilateral", "bilatéral", "bilatérale",
                          "bilaterale", "bilateral"),
            "median": ("median", "médian", "médiane", "mediane",
                       "ligne mediane", "ligne médiane"),
        },
        half_win=100),

    "type_chirurgie": _cfg("type_chirurgie",
        ids=("chirurgie", "intervention", "exerese", "exérèse",
             "resection", "résection", "biopsie", "craniotomie",
             "craniectomie", "gtr", "str", "bst"),
        cats={
            "exerese complete": (
                "exerese complete", "exérèse complète",
                "resection complete", "résection complète",
                "resection totale", "résection totale",
                "exerese en totalite", "exérèse en totalité",
                "gtr", "complete", "complète", "totale",
                "exerese macroscopiquement complete", "exérèse macroscopiquement complète",
                "resection macroscopiquement complete", "résection macroscopiquement complète",
                "absence de residu", "absence de résidu",
                "resection supramarginale", "résection supramarginale",
                "résection macroscopique complète", "exérèse totale", "macroscopiquement complet",
            ),
            "exerese partielle": (
                "exerese partielle", "exérèse partielle",
                "resection partielle", "résection partielle",
                "resection subtotale", "résection subtotale",
                "exerese incomplete", "exérèse incomplète",
                "str", "partielle", "subtotale", "incomplete",
                "incomplète",
                "resection partielle avec reliquat", "résection partielle avec reliquat",
                "exerese sub-totale", "exérèse sub-totale",
                "debulking", "morcellement",
                "exerese fragmentee", "exérèse fragmentée",
                "exérèse incomplète", "exerese incomplete", 
                "réduction tumorale", "reduction tumorale",
            ),
            "biopsie": (
                "biopsie", "biopsie stereotaxique",
                "biopsie stéréotaxique", "bst",
                "biopsie chirurgicale",
                "biopsie en conditions stereotaxiques", "biopsie en conditions stéréotaxiques",
                "biopsie a l'aiguille", "biopsie à l'aiguille",
                "stereotaxie", "stéréotaxie",
            ),
            "exerese": (
                "exerese", "exérèse", "resection", "résection",
            ),
            "en attente": (
                "en attente", "chirurgie a planifier",
                "chirurgie à planifier", "chirurgie prevue",
                "chirurgie prévue", "non realisee",
                "non réalisée",
            ),
            # "autre" intentionally empty
        },
        half_win=60),

    "evol_clinique": _cfg("evol_clinique",
        ids=("evolution", "évolution", "etape", "étape",
             "timepoint", "initial", "terminal", "progression",
             "recidive", "récidive", "rechute"),
        cats={
            "initial": (
                "initial", "initiale", "premiere consultation",
                "première consultation", "bilan initial",
                "diagnostic initial",
                "diagnostic de certitude", "pre-operatoire", "pré-opératoire",
                "presentation initiale", "présentation initiale",
                "découverte", "decouverte", "diagnostic de novo", "primo-diagnostic",
            ),
            "progression": (
                "progression", "recidive", "récidive", "rechute",
                "echappement", "échappement", "aggravation",
                "évolution péjorative", "evolution pejorative", "reprise évolutive",
                "reprise evolutive", "majoration",
            ),
            "reponse / stable": (
                "maladie stable", "reponse partielle", "réponse partielle",
                "pseudo-progression", "radionecrose", "radionécrose",
                "réponse complète", "reponse complete", "rémission", "remission", "stabilisation",
            ),
            "terminal": (
                "terminal", "terminale", "fin de vie",
                "soins palliatifs", "phase terminale",
                "deces", "décès", "exitus", "soins de confort",
                "soins de support", "dégradation de l'état général", "degradation de l'etat general",
            ),
            # "autre" intentionally empty
        },
        half_win=100),

    "classification_oms": _cfg("classification_oms",
        ids=("classification oms", "classification who",
             "oms 20", "who 20", "classification 20"),
        cats={
            "2007": ("2007", "oms 2007", "who 2007"),
            "2016": ("2016", "oms 2016", "who 2016"),
            "2021": ("2021", "oms 2021", "who 2021"),
        },
        half_win=60),

    "lateralite_main": _cfg("lateralite_main",
        ids=("lateralite main", "latéralité main", "main dominante",
             "droitier", "gaucher", "ambidextre"),
        cats={
            "droitier": ("droitier", "droitière", "right-handed"),
            "gaucher": ("gaucher", "gauchère", "left-handed"),
            "ambidextre": ("ambidextre", "ambidextrous"),
            "droitier contrarie": (
                "droitier contrarie", "droitier contrarié",
            ),
            "gaucher contrarie": (
                "gaucher contrarie", "gaucher contrarié",
            ),
        },
        half_win=60),
}


# ═══════════════════════════════════════════════════════════════════════
# EN REGISTRY
# ═══════════════════════════════════════════════════════════════════════

CONTROLLED_REGISTRY_EN: dict[str, FieldVocabConfig] = {

    # ── IHC fields ────────────────────────────────────────────────────
    "ihc_idh1": _cfg("ihc_idh1",
        ids=("idh1", "idh-1", "idh 1"),
        cats=_IHC_CATS_EN),

    "ihc_p53": _cfg("ihc_p53",
        ids=("p53", "tp53", "p53 protein", "anti-p53"),
        cats=_IHC_CATS_EN),

    "ihc_atrx": _cfg("ihc_atrx",
        ids=("atrx", "anti-atrx"),
        cats=_IHC_CATS_EN),

    "ihc_fgfr3": _cfg("ihc_fgfr3",
        ids=("fgfr3", "fgfr-3"),
        cats=_IHC_CATS_EN),

    "ihc_braf": _cfg("ihc_braf",
        ids=("braf",),
        cats=_IHC_CATS_EN),

    "ihc_hist_h3k27m": _cfg("ihc_hist_h3k27m",
        ids=("h3k27m", "h3 k27m", "h3.3 k27m", "h3.3k27m",
             "histone h3 k27m"),
        cats=_IHC_CATS_EN),

    "ihc_hist_h3k27me3": _cfg("ihc_hist_h3k27me3",
        ids=("h3k27me3", "h3 k27me3"),
        cats=_IHC_CATS_EN),

    "ihc_egfr_hirsch": _cfg("ihc_egfr_hirsch",
        ids=("egfr hirsch", "hirsch score", "hirsch"),
        cats=_IHC_CATS_EN),

    "ihc_gfap": _cfg("ihc_gfap",
        ids=("gfap",),
        cats=_IHC_CATS_EN),

    "ihc_olig2": _cfg("ihc_olig2",
        ids=("olig2",),
        cats=_IHC_CATS_EN),

    "ihc_ki67": _cfg("ihc_ki67",
        ids=("ki67", "ki-67", "ki 67", "proliferation index", "mib1", "mib-1"),
        cats=_IHC_CATS_EN),

    "ihc_mmr": _cfg("ihc_mmr",
        ids=("mmr", "mlh1", "msh2", "msh6", "pms2",
             "dmmr", "pmmr", "mmr deficient"),
        cats=_IHC_CATS_EN),

    # ── Molecular fields ──────────────────────────────────────────────
    "mol_idh1": _cfg("mol_idh1",
        ids=("idh1", "idh-1", "idh 1", "r132h", "r132c", "r132s"),
        cats=_MOL_CATS_EN),

    "mol_idh2": _cfg("mol_idh2",
        ids=("idh2", "idh-2", "idh 2", "r172k", "r172m"),
        cats=_MOL_CATS_EN),

    "mol_tert": _cfg("mol_tert",
        ids=("tert", "tert promoter", "c228t", "c250t"),
        cats=_MOL_CATS_EN),

    "mol_CDKN2A": _cfg("mol_CDKN2A",
        ids=("cdkn2a", "p16", "cdkn2a/b",
             "homozygous deletion cdkn2a"),
        cats=_MOL_CATS_EN),

    "mol_h3f3a": _cfg("mol_h3f3a",
        ids=("h3f3a", "h3.3", "k27m", "g34r", "g34v"),
        cats=_MOL_CATS_EN),

    "mol_hist1h3b": _cfg("mol_hist1h3b",
        ids=("hist1h3b", "h3.1"),
        cats=_MOL_CATS_EN),

    "mol_braf": _cfg("mol_braf",
        ids=("braf", "v600e", "v600"),
        cats=_MOL_CATS_EN),

    "mol_mgmt": _cfg("mol_mgmt",
        ids=("mgmt", "mgmt promoter", "mgmt methylation", "o6-methylguanine"),
        cats=_MOL_MGMT_CATS_EN),

    "mol_fgfr1": _cfg("mol_fgfr1",
        ids=("fgfr1",),
        cats=_MOL_CATS_EN),

    "mol_egfr_mut": _cfg("mol_egfr_mut",
        ids=("egfr",),
        cats=_MOL_CATS_EN),

    "mol_prkca": _cfg("mol_prkca",
        ids=("prkca",),
        cats=_MOL_CATS_EN),

    "mol_p53": _cfg("mol_p53",
        ids=("p53", "tp53"),
        cats=_MOL_CATS_EN),

    "mol_pten": _cfg("mol_pten",
        ids=("pten",),
        cats=_MOL_CATS_EN),

    "mol_cic": _cfg("mol_cic",
        ids=("cic",),
        cats=_MOL_CATS_EN),

    "mol_fubp1": _cfg("mol_fubp1",
        ids=("fubp1",),
        cats=_MOL_CATS_EN),

    "mol_atrx": _cfg("mol_atrx",
        ids=("atrx",),
        cats=_MOL_CATS_EN),

    # ── Chromosomal fields ────────────────────────────────────────────
    "ch1p": _cfg("ch1p",
        ids=("1p", "chromosome 1p", "del(1p)", "short arm of chromosome 1",
             "1p codeletion", "1p/19q", "1p19q", "1p/19q codeletion"),
        cats=_CHR_CATS_EN),

    "ch19q": _cfg("ch19q",
        ids=("19q", "chromosome 19q", "del(19q)", "long arm of chromosome 19",
             "19q codeletion", "1p/19q", "1p19q", "1p/19q codeletion"),
        cats=_CHR_CATS_EN),

    "ch10p": _cfg("ch10p",
        ids=("10p", "chromosome 10p"),
        cats=_CHR_CATS_EN),

    "ch10q": _cfg("ch10q",
        ids=("10q", "chromosome 10q"),
        cats=_CHR_CATS_EN),

    "ch7p": _cfg("ch7p",
        ids=("7p", "chromosome 7p"),
        cats=_CHR_CATS_EN),

    "ch7q": _cfg("ch7q",
        ids=("7q", "chromosome 7q"),
        cats=_CHR_CATS_EN),

    "ch9p": _cfg("ch9p",
        ids=("9p", "chromosome 9p"),
        cats=_CHR_CATS_EN),

    "ch9q": _cfg("ch9q",
        ids=("9q", "chromosome 9q"),
        cats=_CHR_CATS_EN),

    # ── Amplification fields ──────────────────────────────────────────
    "ampli_mdm2": _cfg("ampli_mdm2",
        ids=("amplification mdm2", "mdm2 amplified", "mdm2"),
        cats=_AMPLI_CATS_EN),

    "ampli_cdk4": _cfg("ampli_cdk4",
        ids=("amplification cdk4", "cdk4 amplified", "cdk4"),
        cats=_AMPLI_CATS_EN),

    "ampli_egfr": _cfg("ampli_egfr",
        ids=("amplification egfr", "egfr amplified", "egfr"),
        cats=_AMPLI_CATS_EN),

    "ampli_met": _cfg("ampli_met",
        ids=("amplification met", "met amplified", "met"),
        cats=_AMPLI_CATS_EN),

    "ampli_mdm4": _cfg("ampli_mdm4",
        ids=("amplification mdm4", "mdm4 amplified", "mdm4"),
        cats=_AMPLI_CATS_EN),

    # ── Fusion fields ─────────────────────────────────────────────────
    "fusion_fgfr": _cfg("fusion_fgfr",
        ids=("fgfr fusion", "fgfr rearrangement", "fgfr translocation",
             "fgfr"),
        cats=_FUSION_CATS_EN),

    "fusion_ntrk": _cfg("fusion_ntrk",
        ids=("ntrk fusion", "ntrk rearrangement", "ntrk translocation",
             "ntrk"),
        cats=_FUSION_CATS_EN),

    "fusion_autre": _cfg("fusion_autre",
        ids=("alk fusion", "ros1 fusion", "met fusion", "braf fusion",
             "alk rearrangement", "ros1 rearrangement"),
        cats=_FUSION_CATS_EN),

    # ── Other categorical fields ──────────────────────────────────────
    "sexe": _cfg("sexe",
        ids=("sex", "gender", "mr", "mrs", "ms",
             "male", "female", "man", "woman"),
        cats={
            "M": ("male", "man", "mr", "gentleman"),
            "F": ("female", "woman", "mrs", "ms", "lady"),
        },
        half_win=50),

    "tumeur_lateralite": _cfg("tumeur_lateralite",
        ids=("laterality", "side", "hemisphere",
             "left", "right", "bilateral", "midline"),
        cats={
            "gauche": ("left",),
            "droite": ("right",),
            "bilateral": ("bilateral", "both sides"),
            "median": ("midline", "median", "central"),
        },
        half_win=100),

    "type_chirurgie": _cfg("type_chirurgie",
        ids=("surgery", "resection", "excision", "biopsy",
             "craniotomy", "craniectomy", "gtr", "str"),
        cats={
            "exerese complete": (
                "gross total resection", "gtr", "complete resection",
                "total resection", "complete excision",
                "macroscopically complete",
                "gross total excision",
            ),
            "exerese partielle": (
                "subtotal resection", "str", "partial resection",
                "incomplete resection", "partial excision",
                "near total resection", "ntr", "debulking",
                "tumor reduction", "incomplete excision",
            ),
            "biopsie": (
                "biopsy", "stereotactic biopsy", "needle biopsy",
            ),
            "exerese": (
                "resection", "excision",
            ),
            "en attente": (
                "awaiting surgery", "surgery planned",
                "surgery pending", "not yet operated",
            ),
        },
        half_win=60),

    "evol_clinique": _cfg("evol_clinique",
        ids=("evolution", "stage", "timepoint", "initial",
             "terminal", "progression", "recurrence", "relapse"),
        cats={
            "initial": (
                "initial", "first consultation",
                "initial workup", "initial diagnosis",
                "newly diagnosed", "de novo",
            ),
            "progression": (
                "progression", "recurrence", "relapse", "worsening",
                "disease progression", "tumor growth", "clinical deterioration",
            ),
            "reponse / stable": (
                "stable disease", "partial response",
                "pseudoprogression", "radionecrosis",
                "stable", "complete response", "remission", "disease stabilization",
            ),
            "terminal": (
                "terminal", "end of life", "palliative care",
                "terminal phase",
                "supportive care", "hospice",
            ),
        },
        half_win=100),

    "classification_oms": _cfg("classification_oms",
        ids=("who classification", "who 20", "classification 20"),
        cats={
            "2007": ("2007", "who 2007"),
            "2016": ("2016", "who 2016"),
            "2021": ("2021", "who 2021"),
        },
        half_win=60),

    "lateralite_main": _cfg("lateralite_main",
        ids=("handedness", "dominant hand",
             "right-handed", "left-handed", "ambidextrous"),
        cats={
            "droitier": ("right-handed", "right hand dominant"),
            "gaucher": ("left-handed", "left hand dominant"),
            "ambidextre": ("ambidextrous",),
        },
        half_win=60),
}