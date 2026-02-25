"""Feature schema definitions, controlled vocabularies, and JSON schemas.

Defines all 102 clinical + biological fields with their value types,
constraints, and controlled vocabulary references.  Field names are
derived from the REQ_BIO.csv and REQ_CLINIQUE.csv annotation files in
``test_annotated/ANNOTATIONS_RE MAJ Infos cliniques Braincap/``.

Public API
----------
- ``ControlledVocab``          – Enum-like namespace of allowed value sets.
- ``ExtractionValue``          – Single extracted value with provenance.
- ``BiologicalFeatures``       – All 54 biological fields.
- ``ClinicalFeatures``         – All 48 clinical fields.
- ``DocumentExtraction``       – Full extraction result for one document.
- ``FEATURE_ROUTING``          – Document-type → extractable feature subsets.
- ``get_json_schema(group)``   – JSON Schema dict for Ollama ``format`` param.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

class ControlledVocab:
    """Namespace of allowed value sets for constrained fields."""

    BINARY: set[str] = {"oui", "non"}

    IHC_STATUS: set[str] = {"positif", "negatif", "maintenu"}

    MOLECULAR_STATUS: set[str] = {"wt", "mute"}
    # Note: free variant strings (e.g. "R132H", "C228T") are also accepted
    # for molecular fields.  ``is_valid_molecular`` performs the check.

    CHROMOSOMAL: set[str] = {"gain", "perte", "perte partielle"}

    METHYLATION: set[str] = {"methyle", "non methyle"}
    # MGMT also accepts "wt" and "mute" in the annotation data.
    MGMT_STATUS: set[str] = {"methyle", "non methyle", "wt", "mute"}

    GRADE: set[int] = {1, 2, 3, 4}

    WHO_CLASSIFICATION: set[str] = {"2007", "2016", "2021"}

    SURGERY_TYPE: set[str] = {
        "exerese complete",
        "exerese partielle",
        "exerese",
        "biopsie",
        "en attente",
    }

    EVOLUTION: set[str] = {"initial", "terminal"}
    # Plus P1, P2, P3, … (validated by ``is_valid_evolution``).

    SEX: set[str] = {"M", "F"}

    LATERALITY: set[str] = {"gauche", "droite", "bilateral", "median"}

    @staticmethod
    def is_valid_evolution(value: str) -> bool:
        """Return *True* if *value* is a valid evolution label.

        Accepted formats: ``initial``, ``terminal``, ``P<k>`` where *k* is a
        positive integer.
        """
        if value in ("initial", "terminal"):
            return True
        return bool(re.fullmatch(r"P\d+", value))

    @staticmethod
    def is_valid_molecular(value: str) -> bool:
        """Return *True* if *value* is an accepted molecular status.

        Accepts ``wt``, ``mute``, or specific variant strings (e.g.
        ``R132H``, ``C228T``, ``V600E``, ``mute + delete``).
        """
        if value in ControlledVocab.MOLECULAR_STATUS:
            return True
        # Accept specific variant patterns (alphanumeric + optional combinators)
        if re.fullmatch(r"[A-Za-z0-9_+/ .-]+", value) and len(value) <= 50:
            return True
        return False


# ---------------------------------------------------------------------------
# ExtractionValue — single extracted field with provenance
# ---------------------------------------------------------------------------

class ExtractionValue(BaseModel):
    """Single extracted value with provenance metadata."""

    value: Optional[str | int | float] = None
    source_span: Optional[str] = None  # exact text from document
    source_span_start: Optional[int] = None  # character offset
    source_span_end: Optional[int] = None
    extraction_tier: Literal["rule", "llm", "manual"] = "rule"
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    section: Optional[str] = None  # which document section
    vocab_valid: bool = True  # passed controlled vocabulary check
    flagged: bool = False  # needs human review

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "value": "positif",
                    "source_span": "IDH1 : positif",
                    "extraction_tier": "rule",
                    "vocab_valid": True,
                }
            ]
        }
    )


# ---------------------------------------------------------------------------
# Field definitions — metadata per feature
# ---------------------------------------------------------------------------

class FieldType(str, Enum):
    """Data-type tag for each feature."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DATE = "date"  # DD/MM/YYYY or abbreviated
    CATEGORICAL = "categorical"  # value from a ControlledVocab set
    FREE_TEXT = "free_text"


class FieldDefinition(BaseModel):
    """Metadata for a single schema field."""

    name: str
    display_name: str = ""  # human-readable label
    field_type: FieldType = FieldType.STRING
    allowed_values: Optional[set[str | int]] = None  # None ⇒ free / unconstrained
    nullable: bool = True
    group: str = ""  # feature group (e.g. "ihc", "molecular", "demographics")
    description: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Biological feature fields  (54 fields from REQ_BIO.csv)
# ---------------------------------------------------------------------------

BIO_FIELDS: list[FieldDefinition] = [
    # ── Identifiers / context ──
    FieldDefinition(name="nip",       display_name="NIP (patient ID)",        field_type=FieldType.STRING,      group="identifiers"),
    FieldDefinition(name="date_chir", display_name="Date chirurgie",          field_type=FieldType.DATE,        group="identifiers"),
    FieldDefinition(name="num_labo",  display_name="Numéro laboratoire",      field_type=FieldType.STRING,      group="identifiers"),

    # ── Diagnosis ──
    FieldDefinition(name="diag_histologique",   display_name="Diagnostic histologique",     field_type=FieldType.FREE_TEXT,     group="diagnosis"),
    FieldDefinition(name="diag_integre",        display_name="Diagnostic intégré",          field_type=FieldType.FREE_TEXT,     group="diagnosis"),
    FieldDefinition(name="classification_oms",  display_name="Classification OMS",          field_type=FieldType.CATEGORICAL,  group="diagnosis",  allowed_values=ControlledVocab.WHO_CLASSIFICATION),
    FieldDefinition(name="grade",               display_name="Grade OMS",                   field_type=FieldType.INTEGER,      group="diagnosis",  allowed_values=ControlledVocab.GRADE),

    # ── IHC ──
    FieldDefinition(name="ihc_idh1",            display_name="IHC IDH1",             field_type=FieldType.CATEGORICAL, group="ihc", allowed_values=ControlledVocab.IHC_STATUS),
    FieldDefinition(name="ihc_p53",             display_name="IHC p53",              field_type=FieldType.CATEGORICAL, group="ihc", allowed_values=ControlledVocab.IHC_STATUS),
    FieldDefinition(name="ihc_atrx",            display_name="IHC ATRX",             field_type=FieldType.CATEGORICAL, group="ihc", allowed_values=ControlledVocab.IHC_STATUS),
    FieldDefinition(name="ihc_fgfr3",           display_name="IHC FGFR3",            field_type=FieldType.CATEGORICAL, group="ihc", allowed_values=ControlledVocab.IHC_STATUS),
    FieldDefinition(name="ihc_braf",            display_name="IHC BRAF",             field_type=FieldType.CATEGORICAL, group="ihc", allowed_values=ControlledVocab.IHC_STATUS),
    FieldDefinition(name="ihc_hist_h3k27m",     display_name="IHC H3K27M",           field_type=FieldType.CATEGORICAL, group="ihc", allowed_values=ControlledVocab.IHC_STATUS),
    FieldDefinition(name="ihc_hist_h3k27me3",   display_name="IHC H3K27me3",         field_type=FieldType.CATEGORICAL, group="ihc", allowed_values=ControlledVocab.IHC_STATUS),
    FieldDefinition(name="ihc_egfr_hirsch",     display_name="IHC EGFR (Hirsch / status)", field_type=FieldType.STRING, group="ihc"),  # can be score or status string
    FieldDefinition(name="ihc_gfap",            display_name="IHC GFAP",             field_type=FieldType.CATEGORICAL, group="ihc", allowed_values=ControlledVocab.IHC_STATUS),
    FieldDefinition(name="ihc_olig2",           display_name="IHC Olig2",            field_type=FieldType.CATEGORICAL, group="ihc", allowed_values=ControlledVocab.IHC_STATUS),
    FieldDefinition(name="ihc_ki67",            display_name="IHC Ki67 (%)",          field_type=FieldType.STRING,      group="ihc"),  # can be "15-20", "5", "<5%"
    FieldDefinition(name="ihc_mmr",             display_name="IHC MMR",              field_type=FieldType.CATEGORICAL, group="ihc", allowed_values=ControlledVocab.IHC_STATUS),

    # ── Histology assessment ──
    FieldDefinition(name="histo_necrose",  display_name="Nécrose",         field_type=FieldType.CATEGORICAL, group="histology", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="histo_pec",      display_name="Prise de contraste endothéliocapillaire", field_type=FieldType.CATEGORICAL, group="histology", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="histo_mitoses",  display_name="Mitoses (count)", field_type=FieldType.INTEGER,     group="histology"),

    # ── Molecular biology ──
    FieldDefinition(name="mol_idh1",       display_name="IDH1 moléculaire",       field_type=FieldType.STRING, group="molecular"),
    FieldDefinition(name="mol_idh2",       display_name="IDH2 moléculaire",       field_type=FieldType.STRING, group="molecular"),
    FieldDefinition(name="mol_tert",       display_name="TERT moléculaire",       field_type=FieldType.STRING, group="molecular"),
    FieldDefinition(name="mol_CDKN2A",     display_name="CDKN2A moléculaire",     field_type=FieldType.STRING, group="molecular"),
    FieldDefinition(name="mol_h3f3a",      display_name="H3F3A moléculaire",      field_type=FieldType.STRING, group="molecular"),
    FieldDefinition(name="mol_hist1h3b",   display_name="HIST1H3B moléculaire",   field_type=FieldType.STRING, group="molecular"),
    FieldDefinition(name="mol_braf",       display_name="BRAF moléculaire",       field_type=FieldType.STRING, group="molecular"),
    FieldDefinition(name="mol_mgmt",       display_name="MGMT méthylation",       field_type=FieldType.STRING, group="molecular"),
    FieldDefinition(name="mol_fgfr1",      display_name="FGFR1 moléculaire",      field_type=FieldType.STRING, group="molecular"),
    FieldDefinition(name="mol_egfr_mut",   display_name="EGFR mutation",          field_type=FieldType.STRING, group="molecular"),
    FieldDefinition(name="mol_prkca",      display_name="PRKCA moléculaire",      field_type=FieldType.STRING, group="molecular"),
    FieldDefinition(name="mol_p53",        display_name="TP53 moléculaire",       field_type=FieldType.STRING, group="molecular"),
    FieldDefinition(name="mol_pten",       display_name="PTEN moléculaire",       field_type=FieldType.STRING, group="molecular"),
    FieldDefinition(name="mol_cic",        display_name="CIC moléculaire",        field_type=FieldType.STRING, group="molecular"),
    FieldDefinition(name="mol_fubp1",      display_name="FUBP1 moléculaire",      field_type=FieldType.STRING, group="molecular"),
    FieldDefinition(name="mol_atrx",       display_name="ATRX moléculaire",       field_type=FieldType.STRING, group="molecular"),

    # ── Chromosomal alterations ──
    FieldDefinition(name="ch1p",   display_name="1p",  field_type=FieldType.CATEGORICAL, group="chromosomal", allowed_values=ControlledVocab.CHROMOSOMAL),
    FieldDefinition(name="ch19q",  display_name="19q", field_type=FieldType.CATEGORICAL, group="chromosomal", allowed_values=ControlledVocab.CHROMOSOMAL),
    FieldDefinition(name="ch10p",  display_name="10p", field_type=FieldType.CATEGORICAL, group="chromosomal", allowed_values=ControlledVocab.CHROMOSOMAL),
    FieldDefinition(name="ch10q",  display_name="10q", field_type=FieldType.CATEGORICAL, group="chromosomal", allowed_values=ControlledVocab.CHROMOSOMAL),
    FieldDefinition(name="ch7p",   display_name="7p",  field_type=FieldType.CATEGORICAL, group="chromosomal", allowed_values=ControlledVocab.CHROMOSOMAL),
    FieldDefinition(name="ch7q",   display_name="7q",  field_type=FieldType.CATEGORICAL, group="chromosomal", allowed_values=ControlledVocab.CHROMOSOMAL),
    FieldDefinition(name="ch9p",   display_name="9p",  field_type=FieldType.CATEGORICAL, group="chromosomal", allowed_values=ControlledVocab.CHROMOSOMAL),
    FieldDefinition(name="ch9q",   display_name="9q",  field_type=FieldType.CATEGORICAL, group="chromosomal", allowed_values=ControlledVocab.CHROMOSOMAL),

    # ── Amplifications ──
    FieldDefinition(name="ampli_mdm2",  display_name="Amplification MDM2", field_type=FieldType.CATEGORICAL, group="amplification", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="ampli_cdk4",  display_name="Amplification CDK4", field_type=FieldType.CATEGORICAL, group="amplification", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="ampli_egfr",  display_name="Amplification EGFR", field_type=FieldType.CATEGORICAL, group="amplification", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="ampli_met",   display_name="Amplification MET",  field_type=FieldType.CATEGORICAL, group="amplification", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="ampli_mdm4",  display_name="Amplification MDM4", field_type=FieldType.CATEGORICAL, group="amplification", allowed_values=ControlledVocab.BINARY),

    # ── Fusions ──
    FieldDefinition(name="fusion_fgfr",   display_name="Fusion FGFR",   field_type=FieldType.CATEGORICAL, group="fusion", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="fusion_ntrk",   display_name="Fusion NTRK",   field_type=FieldType.CATEGORICAL, group="fusion", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="fusion_autre",  display_name="Fusion autre",   field_type=FieldType.CATEGORICAL, group="fusion", allowed_values=ControlledVocab.BINARY),
]

# ── All BIO field names (convenience list) ──
ALL_BIO_FIELD_NAMES: list[str] = [f.name for f in BIO_FIELDS]


# ---------------------------------------------------------------------------
# Clinical feature fields  (48 fields from REQ_CLINIQUE.csv)
# ---------------------------------------------------------------------------

CLINIQUE_FIELDS: list[FieldDefinition] = [
    # ── Identifiers / demographics ──
    FieldDefinition(name="nip",                       display_name="NIP (patient ID)",             field_type=FieldType.STRING,      group="demographics"),
    FieldDefinition(name="date_de_naissance",          display_name="Date de naissance",            field_type=FieldType.DATE,        group="demographics"),
    FieldDefinition(name="sexe",                       display_name="Sexe",                         field_type=FieldType.CATEGORICAL, group="demographics", allowed_values=ControlledVocab.SEX),
    FieldDefinition(name="activite_professionnelle",   display_name="Activité professionnelle",     field_type=FieldType.FREE_TEXT,   group="demographics"),
    FieldDefinition(name="antecedent_tumoral",         display_name="Antécédent tumoral",           field_type=FieldType.CATEGORICAL, group="demographics", allowed_values={"Oui", "Non", "oui", "non"}),

    # ── Care team ──
    FieldDefinition(name="neuroncologue",              display_name="Neuro-oncologue",              field_type=FieldType.FREE_TEXT,   group="care_team"),
    FieldDefinition(name="neurochirurgien",            display_name="Neurochirurgien",              field_type=FieldType.FREE_TEXT,   group="care_team"),
    FieldDefinition(name="radiotherapeute",            display_name="Radiothérapeute",              field_type=FieldType.FREE_TEXT,   group="care_team"),
    FieldDefinition(name="localisation_radiotherapie", display_name="Localisation radiothérapie",   field_type=FieldType.FREE_TEXT,   group="care_team"),
    FieldDefinition(name="localisation_chir",          display_name="Localisation chirurgie",       field_type=FieldType.FREE_TEXT,   group="care_team"),

    # ── Outcome ──
    FieldDefinition(name="date_deces",    display_name="Date décès",    field_type=FieldType.DATE,      group="outcome"),
    FieldDefinition(name="infos_deces",   display_name="Infos décès",   field_type=FieldType.FREE_TEXT, group="outcome"),

    # ── First symptoms ──
    FieldDefinition(name="date_1er_symptome",          display_name="Date 1er symptôme",          field_type=FieldType.DATE,        group="first_symptoms"),
    FieldDefinition(name="epilepsie_1er_symptome",     display_name="Épilepsie 1er symptôme",     field_type=FieldType.CATEGORICAL, group="first_symptoms", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="ceph_hic_1er_symptome",      display_name="Céphalées/HTIC 1er symptôme", field_type=FieldType.CATEGORICAL, group="first_symptoms", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="deficit_1er_symptome",       display_name="Déficit 1er symptôme",       field_type=FieldType.CATEGORICAL, group="first_symptoms", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="cognitif_1er_symptome",      display_name="Cognitif 1er symptôme",      field_type=FieldType.CATEGORICAL, group="first_symptoms", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="autre_trouble_1er_symptome", display_name="Autre trouble 1er symptôme", field_type=FieldType.CATEGORICAL, group="first_symptoms", allowed_values=ControlledVocab.BINARY),

    # ── Radiology / imaging at discovery ──
    FieldDefinition(name="exam_radio_date_decouverte", display_name="Date découverte radiologique", field_type=FieldType.DATE,        group="radiology"),
    FieldDefinition(name="contraste_1er_symptome",     display_name="Prise de contraste initiale",  field_type=FieldType.CATEGORICAL, group="radiology", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="oedeme_1er_symptome",        display_name="Œdème initial",                field_type=FieldType.CATEGORICAL, group="radiology", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="calcif_1er_symptome",        display_name="Calcification initiale",       field_type=FieldType.CATEGORICAL, group="radiology", allowed_values=ControlledVocab.BINARY),

    # ── Tumour location ──
    FieldDefinition(name="tumeur_lateralite",  display_name="Latéralité tumeur",   field_type=FieldType.CATEGORICAL, group="tumour_location", allowed_values=ControlledVocab.LATERALITY),
    FieldDefinition(name="tumeur_position",    display_name="Position tumeur",     field_type=FieldType.FREE_TEXT,   group="tumour_location"),

    # ── Clinical timepoint ──
    FieldDefinition(name="dn_date",         display_name="Date dernière nouvelle",  field_type=FieldType.DATE,        group="evolution"),
    FieldDefinition(name="evol_clinique",   display_name="Évolution clinique",      field_type=FieldType.STRING,      group="evolution"),  # validated via is_valid_evolution

    # ── Treatment — chemotherapy ──
    FieldDefinition(name="chimios",           display_name="Chimiothérapie(s)",       field_type=FieldType.FREE_TEXT,    group="treatment_chemo"),
    FieldDefinition(name="chm_date_debut",    display_name="Date début chimio",       field_type=FieldType.DATE,         group="treatment_chemo"),
    FieldDefinition(name="chm_date_fin",      display_name="Date fin chimio",         field_type=FieldType.DATE,         group="treatment_chemo"),
    FieldDefinition(name="chm_cycles",        display_name="Nombre cycles chimio",    field_type=FieldType.INTEGER,      group="treatment_chemo"),

    # ── Current clinical state ──
    FieldDefinition(name="ik_clinique",             display_name="Indice de Karnofsky",    field_type=FieldType.INTEGER,     group="clinical_state"),
    FieldDefinition(name="progress_clinique",       display_name="Progression clinique",   field_type=FieldType.CATEGORICAL, group="clinical_state", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="progress_radiologique",   display_name="Progression radiologique", field_type=FieldType.CATEGORICAL, group="clinical_state", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="date_progression",        display_name="Date progression",       field_type=FieldType.DATE,        group="clinical_state"),

    FieldDefinition(name="epilepsie",     display_name="Épilepsie actuelle",      field_type=FieldType.CATEGORICAL, group="current_symptoms", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="ceph_hic",      display_name="Céphalées/HTIC actuelle", field_type=FieldType.CATEGORICAL, group="current_symptoms", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="deficit",       display_name="Déficit actuel",          field_type=FieldType.CATEGORICAL, group="current_symptoms", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="cognitif",      display_name="Trouble cognitif",        field_type=FieldType.CATEGORICAL, group="current_symptoms", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="autre_trouble", display_name="Autre trouble",           field_type=FieldType.FREE_TEXT,   group="current_symptoms"),

    # ── Adjunct medications / devices ──
    FieldDefinition(name="anti_epileptiques",       display_name="Anti-épileptiques",       field_type=FieldType.CATEGORICAL, group="adjunct", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="essai_therapeutique",      display_name="Essai thérapeutique",     field_type=FieldType.CATEGORICAL, group="adjunct", allowed_values=ControlledVocab.BINARY),

    # ── Surgery ──
    FieldDefinition(name="chir_date",       display_name="Date chirurgie",    field_type=FieldType.DATE,        group="surgery"),
    FieldDefinition(name="type_chirurgie",  display_name="Type chirurgie",    field_type=FieldType.CATEGORICAL, group="surgery", allowed_values=ControlledVocab.SURGERY_TYPE),

    # ── Treatment — radiotherapy ──
    FieldDefinition(name="rx_date_debut",  display_name="Date début radiothérapie",   field_type=FieldType.DATE,   group="treatment_radio"),
    FieldDefinition(name="rx_date_fin",    display_name="Date fin radiothérapie",     field_type=FieldType.DATE,   group="treatment_radio"),
    FieldDefinition(name="rx_dose",        display_name="Dose radiothérapie (Gy)",    field_type=FieldType.STRING, group="treatment_radio"),  # "60", "non", "oui", "en attente"

    # ── Other ──
    FieldDefinition(name="corticoides",  display_name="Corticoïdes",  field_type=FieldType.CATEGORICAL, group="adjunct", allowed_values=ControlledVocab.BINARY),
    FieldDefinition(name="optune",       display_name="Optune (TTFields)", field_type=FieldType.CATEGORICAL, group="adjunct", allowed_values=ControlledVocab.BINARY),
]

# ── All CLINIQUE field names (convenience list) ──
ALL_CLINIQUE_FIELD_NAMES: list[str] = [f.name for f in CLINIQUE_FIELDS]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

_BIO_FIELDS_BY_NAME: dict[str, FieldDefinition] = {f.name: f for f in BIO_FIELDS}
_CLINIQUE_FIELDS_BY_NAME: dict[str, FieldDefinition] = {f.name: f for f in CLINIQUE_FIELDS}
ALL_FIELDS_BY_NAME: dict[str, FieldDefinition] = {**_BIO_FIELDS_BY_NAME, **_CLINIQUE_FIELDS_BY_NAME}


def get_field(name: str) -> FieldDefinition:
    """Return the ``FieldDefinition`` for a given field name."""
    try:
        return ALL_FIELDS_BY_NAME[name]
    except KeyError:
        raise KeyError(f"Unknown field: {name!r}. Available: {sorted(ALL_FIELDS_BY_NAME)}")


# ---------------------------------------------------------------------------
# Pydantic models — structured extraction results
# ---------------------------------------------------------------------------

class BiologicalFeatures(BaseModel):
    """All 54 biological fields, each wrapped in an ``ExtractionValue``."""

    # Identifiers
    nip: Optional[ExtractionValue] = None
    date_chir: Optional[ExtractionValue] = None
    num_labo: Optional[ExtractionValue] = None

    # Diagnosis
    diag_histologique: Optional[ExtractionValue] = None
    diag_integre: Optional[ExtractionValue] = None
    classification_oms: Optional[ExtractionValue] = None
    grade: Optional[ExtractionValue] = None

    # IHC
    ihc_idh1: Optional[ExtractionValue] = None
    ihc_p53: Optional[ExtractionValue] = None
    ihc_atrx: Optional[ExtractionValue] = None
    ihc_fgfr3: Optional[ExtractionValue] = None
    ihc_braf: Optional[ExtractionValue] = None
    ihc_hist_h3k27m: Optional[ExtractionValue] = None
    ihc_hist_h3k27me3: Optional[ExtractionValue] = None
    ihc_egfr_hirsch: Optional[ExtractionValue] = None
    ihc_gfap: Optional[ExtractionValue] = None
    ihc_olig2: Optional[ExtractionValue] = None
    ihc_ki67: Optional[ExtractionValue] = None
    ihc_mmr: Optional[ExtractionValue] = None

    # Histology
    histo_necrose: Optional[ExtractionValue] = None
    histo_pec: Optional[ExtractionValue] = None
    histo_mitoses: Optional[ExtractionValue] = None

    # Molecular
    mol_idh1: Optional[ExtractionValue] = None
    mol_idh2: Optional[ExtractionValue] = None
    mol_tert: Optional[ExtractionValue] = None
    mol_CDKN2A: Optional[ExtractionValue] = None
    mol_h3f3a: Optional[ExtractionValue] = None
    mol_hist1h3b: Optional[ExtractionValue] = None
    mol_braf: Optional[ExtractionValue] = None
    mol_mgmt: Optional[ExtractionValue] = None
    mol_fgfr1: Optional[ExtractionValue] = None
    mol_egfr_mut: Optional[ExtractionValue] = None
    mol_prkca: Optional[ExtractionValue] = None
    mol_p53: Optional[ExtractionValue] = None
    mol_pten: Optional[ExtractionValue] = None
    mol_cic: Optional[ExtractionValue] = None
    mol_fubp1: Optional[ExtractionValue] = None
    mol_atrx: Optional[ExtractionValue] = None

    # Chromosomal
    ch1p: Optional[ExtractionValue] = None
    ch19q: Optional[ExtractionValue] = None
    ch10p: Optional[ExtractionValue] = None
    ch10q: Optional[ExtractionValue] = None
    ch7p: Optional[ExtractionValue] = None
    ch7q: Optional[ExtractionValue] = None
    ch9p: Optional[ExtractionValue] = None
    ch9q: Optional[ExtractionValue] = None

    # Amplifications
    ampli_mdm2: Optional[ExtractionValue] = None
    ampli_cdk4: Optional[ExtractionValue] = None
    ampli_egfr: Optional[ExtractionValue] = None
    ampli_met: Optional[ExtractionValue] = None
    ampli_mdm4: Optional[ExtractionValue] = None

    # Fusions
    fusion_fgfr: Optional[ExtractionValue] = None
    fusion_ntrk: Optional[ExtractionValue] = None
    fusion_autre: Optional[ExtractionValue] = None


class ClinicalFeatures(BaseModel):
    """All 48 clinical fields, each wrapped in an ``ExtractionValue``."""

    # Demographics
    nip: Optional[ExtractionValue] = None
    date_de_naissance: Optional[ExtractionValue] = None
    sexe: Optional[ExtractionValue] = None
    activite_professionnelle: Optional[ExtractionValue] = None
    antecedent_tumoral: Optional[ExtractionValue] = None

    # Care team
    neuroncologue: Optional[ExtractionValue] = None
    neurochirurgien: Optional[ExtractionValue] = None
    radiotherapeute: Optional[ExtractionValue] = None
    localisation_radiotherapie: Optional[ExtractionValue] = None
    localisation_chir: Optional[ExtractionValue] = None

    # Outcome
    date_deces: Optional[ExtractionValue] = None
    infos_deces: Optional[ExtractionValue] = None

    # First symptoms
    date_1er_symptome: Optional[ExtractionValue] = None
    epilepsie_1er_symptome: Optional[ExtractionValue] = None
    ceph_hic_1er_symptome: Optional[ExtractionValue] = None
    deficit_1er_symptome: Optional[ExtractionValue] = None
    cognitif_1er_symptome: Optional[ExtractionValue] = None
    autre_trouble_1er_symptome: Optional[ExtractionValue] = None

    # Radiology at discovery
    exam_radio_date_decouverte: Optional[ExtractionValue] = None
    contraste_1er_symptome: Optional[ExtractionValue] = None
    oedeme_1er_symptome: Optional[ExtractionValue] = None
    calcif_1er_symptome: Optional[ExtractionValue] = None

    # Tumour location
    tumeur_lateralite: Optional[ExtractionValue] = None
    tumeur_position: Optional[ExtractionValue] = None

    # Evolution
    dn_date: Optional[ExtractionValue] = None
    evol_clinique: Optional[ExtractionValue] = None

    # Treatment — chemo
    chimios: Optional[ExtractionValue] = None
    chm_date_debut: Optional[ExtractionValue] = None
    chm_date_fin: Optional[ExtractionValue] = None
    chm_cycles: Optional[ExtractionValue] = None

    # Clinical state
    ik_clinique: Optional[ExtractionValue] = None
    progress_clinique: Optional[ExtractionValue] = None
    progress_radiologique: Optional[ExtractionValue] = None
    date_progression: Optional[ExtractionValue] = None

    # Current symptoms
    epilepsie: Optional[ExtractionValue] = None
    ceph_hic: Optional[ExtractionValue] = None
    deficit: Optional[ExtractionValue] = None
    cognitif: Optional[ExtractionValue] = None
    autre_trouble: Optional[ExtractionValue] = None

    # Adjunct
    anti_epileptiques: Optional[ExtractionValue] = None
    essai_therapeutique: Optional[ExtractionValue] = None

    # Surgery
    chir_date: Optional[ExtractionValue] = None
    type_chirurgie: Optional[ExtractionValue] = None

    # Treatment — radio
    rx_date_debut: Optional[ExtractionValue] = None
    rx_date_fin: Optional[ExtractionValue] = None
    rx_dose: Optional[ExtractionValue] = None

    # Other adjuncts
    corticoides: Optional[ExtractionValue] = None
    optune: Optional[ExtractionValue] = None


class DocumentExtraction(BaseModel):
    """Full extraction result for one document."""

    document_id: str = ""
    document_type: str = ""  # one of DOCUMENT_TYPES
    patient_id: str = ""
    clinical: ClinicalFeatures = Field(default_factory=ClinicalFeatures)
    biological: BiologicalFeatures = Field(default_factory=BiologicalFeatures)
    raw_text: str = ""
    sections: dict[str, str] = Field(default_factory=dict)  # section_name → section_text


# ---------------------------------------------------------------------------
# Document types
# ---------------------------------------------------------------------------

DOCUMENT_TYPES: list[str] = [
    "anapath",
    "molecular_report",
    "consultation",
    "rcp",
    "radiology",
]


# ---------------------------------------------------------------------------
# Feature routing — document type → extractable feature subsets
# ---------------------------------------------------------------------------

# Group names used in FieldDefinition.group
_BIO_GROUP_NAMES = {f.group for f in BIO_FIELDS}
_CLINIQUE_GROUP_NAMES = {f.group for f in CLINIQUE_FIELDS}

# Helper: field names by group
def _fields_in_groups(fields: list[FieldDefinition], groups: set[str]) -> list[str]:
    return [f.name for f in fields if f.group in groups]

def _fields_matching(fields: list[FieldDefinition], pattern: str) -> list[str]:
    """Return field names matching a prefix pattern (e.g. ``mol_*``)."""
    if pattern.endswith("*"):
        prefix = pattern[:-1]
        return [f.name for f in fields if f.name.startswith(prefix)]
    return [f.name for f in fields if f.name == pattern]

def _resolve_patterns(fields: list[FieldDefinition], patterns: list[str]) -> list[str]:
    """Resolve a list of field name patterns (with optional ``*`` suffix) to concrete names."""
    result: list[str] = []
    for pat in patterns:
        result.extend(_fields_matching(fields, pat))
    return sorted(set(result))


# Partial subsets for RCP (gets both bio and clinical, but not everything)
_RCP_BIO_FIELDS = _resolve_patterns(BIO_FIELDS, [
    "diag_histologique", "diag_integre", "classification_oms", "grade",
    "ihc_*", "mol_*", "ch*", "ampli_*", "fusion_*",
    "histo_necrose", "histo_pec", "histo_mitoses",
])

_RCP_CLINIQUE_FIELDS = _resolve_patterns(CLINIQUE_FIELDS, [
    "nip", "sexe", "date_de_naissance",
    "chimios", "chm_*",
    "rx_*",
    "chir_date", "type_chirurgie",
    "ik_clinique",
    "tumeur_lateralite", "tumeur_position",
    "evol_clinique",
    "progress_clinique", "progress_radiologique", "date_progression",
])

_RADIOLOGY_CLINIQUE_FIELDS = _resolve_patterns(CLINIQUE_FIELDS, [
    "tumeur_lateralite", "tumeur_position",
    "exam_radio_date_decouverte",
    "contraste_1er_symptome", "oedeme_1er_symptome", "calcif_1er_symptome",
    "progress_radiologique",
])


FEATURE_ROUTING: dict[str, dict[str, list[str]]] = {
    "anapath": {
        "bio": ALL_BIO_FIELD_NAMES,
        "clinique": [],
    },
    "molecular_report": {
        "bio": _resolve_patterns(BIO_FIELDS, [
            "mol_*", "ch*", "ampli_*", "fusion_*", "mol_mgmt",
        ]),
        "clinique": [],
    },
    "consultation": {
        "bio": [],
        "clinique": ALL_CLINIQUE_FIELD_NAMES,
    },
    "rcp": {
        "bio": _RCP_BIO_FIELDS,
        "clinique": _RCP_CLINIQUE_FIELDS,
    },
    "radiology": {
        "bio": [],
        "clinique": _RADIOLOGY_CLINIQUE_FIELDS,
    },
}


def get_extractable_fields(document_type: str) -> list[str]:
    """Return the list of field names extractable from *document_type*."""
    if document_type not in FEATURE_ROUTING:
        raise ValueError(
            f"Unknown document type: {document_type!r}. "
            f"Expected one of {DOCUMENT_TYPES}"
        )
    routing = FEATURE_ROUTING[document_type]
    return sorted(set(routing["bio"] + routing["clinique"]))


# ---------------------------------------------------------------------------
# Feature groups for prompt organisation
# ---------------------------------------------------------------------------

FEATURE_GROUPS: dict[str, list[str]] = {
    "ihc": _resolve_patterns(BIO_FIELDS, ["ihc_*"]),
    "molecular": _resolve_patterns(BIO_FIELDS, ["mol_*"]),
    "chromosomal": (
        _resolve_patterns(BIO_FIELDS, ["ch*"])
        + _resolve_patterns(BIO_FIELDS, ["ampli_*"])
        + _resolve_patterns(BIO_FIELDS, ["fusion_*"])
    ),
    "diagnosis": _resolve_patterns(BIO_FIELDS, [
        "diag_histologique", "diag_integre", "classification_oms", "grade",
        "histo_necrose", "histo_pec", "histo_mitoses",
    ]),
    "demographics": _resolve_patterns(CLINIQUE_FIELDS, [
        "nip", "date_de_naissance", "sexe", "activite_professionnelle",
        "antecedent_tumoral", "neuroncologue", "neurochirurgien",
        "radiotherapeute", "localisation_radiotherapie", "localisation_chir",
    ]),
    "symptoms": (
        _resolve_patterns(CLINIQUE_FIELDS, [
            "date_1er_symptome", "epilepsie_1er_symptome",
            "ceph_hic_1er_symptome", "deficit_1er_symptome",
            "cognitif_1er_symptome", "autre_trouble_1er_symptome",
            "exam_radio_date_decouverte",
            "contraste_1er_symptome", "oedeme_1er_symptome", "calcif_1er_symptome",
        ])
        + _resolve_patterns(CLINIQUE_FIELDS, [
            "epilepsie", "ceph_hic", "deficit", "cognitif", "autre_trouble",
            "ik_clinique",
        ])
    ),
    "treatment": _resolve_patterns(CLINIQUE_FIELDS, [
        "chimios", "chm_*",
        "chir_date", "type_chirurgie",
        "rx_*",
        "anti_epileptiques", "essai_therapeutique",
        "corticoides", "optune",
    ]),
    "evolution": _resolve_patterns(CLINIQUE_FIELDS, [
        "dn_date", "evol_clinique",
        "progress_clinique", "progress_radiologique", "date_progression",
        "tumeur_lateralite", "tumeur_position",
        "date_deces", "infos_deces",
    ]),
}


# ---------------------------------------------------------------------------
# JSON schema generation for Ollama constrained decoding
# ---------------------------------------------------------------------------

def _json_type_for_field(field: FieldDefinition) -> dict[str, Any]:
    """Return the JSON Schema type object for a single field."""
    if field.allowed_values:
        # Any field with an explicit allowed_values set gets an enum
        enum_values: list[Any] = sorted(field.allowed_values, key=str)
        base: dict[str, Any] = {"enum": enum_values + [None]}  # type: ignore[arg-type]
    elif field.field_type == FieldType.INTEGER:
        base = {"type": ["integer", "null"]}
    elif field.field_type == FieldType.FLOAT:
        base = {"type": ["number", "null"]}
    else:
        base = {"type": ["string", "null"]}

    base["description"] = field.display_name or field.name
    return base


def _build_group_schema(field_names: list[str]) -> dict[str, Any]:
    """Build a JSON Schema ``object`` for a group of fields.

    Includes a parallel ``_source`` object so the LLM can cite text spans
    alongside each extracted value.
    """
    value_properties: dict[str, Any] = {}
    source_properties: dict[str, Any] = {}

    for name in field_names:
        field = ALL_FIELDS_BY_NAME.get(name)
        if field is None:
            continue
        value_properties[name] = _json_type_for_field(field)
        source_properties[name] = {
            "type": ["string", "null"],
            "description": f"Exact source text span for {name}",
        }

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "values": {
                "type": "object",
                "properties": value_properties,
                "required": list(value_properties.keys()),
            },
            "_source": {
                "type": "object",
                "properties": source_properties,
                "description": "Exact text spans from the document justifying each value.",
            },
        },
        "required": ["values", "_source"],
    }
    return schema


def get_json_schema(feature_group: str) -> dict[str, Any]:
    """Return a JSON Schema dict for the given feature group.

    Parameters
    ----------
    feature_group : str
        One of the keys in ``FEATURE_GROUPS``:
        ``ihc``, ``molecular``, ``chromosomal``, ``diagnosis``,
        ``demographics``, ``symptoms``, ``treatment``, ``evolution``.

    Returns
    -------
    dict
        A valid JSON Schema (draft-compatible) suitable for passing to
        Ollama's ``format`` parameter for constrained decoding.

    Raises
    ------
    KeyError
        If *feature_group* is not recognised.
    """
    if feature_group not in FEATURE_GROUPS:
        raise KeyError(
            f"Unknown feature group: {feature_group!r}. "
            f"Available: {sorted(FEATURE_GROUPS)}"
        )
    return _build_group_schema(FEATURE_GROUPS[feature_group])


def get_all_json_schemas() -> dict[str, dict[str, Any]]:
    """Return a dict mapping each feature group name to its JSON Schema."""
    return {group: get_json_schema(group) for group in FEATURE_GROUPS}
