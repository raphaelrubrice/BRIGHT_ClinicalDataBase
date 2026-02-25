"""Tests for src/extraction/schema.py — feature schema and controlled vocabularies.

Validates:
- All 102 fields (54 BIO + 48 CLINIQUE) are defined.
- Controlled vocabulary coverage.
- ExtractionValue creation and validation.
- BiologicalFeatures / ClinicalFeatures / DocumentExtraction models.
- FEATURE_ROUTING maps every document type.
- get_json_schema() returns valid JSON Schema for each feature group.
- Sample extraction outputs parse successfully.
"""

import json

import pytest

from src.extraction.schema import (
    ALL_BIO_FIELD_NAMES,
    ALL_CLINIQUE_FIELD_NAMES,
    ALL_FIELDS_BY_NAME,
    BIO_FIELDS,
    CLINIQUE_FIELDS,
    DOCUMENT_TYPES,
    FEATURE_GROUPS,
    FEATURE_ROUTING,
    BiologicalFeatures,
    ClinicalFeatures,
    ControlledVocab,
    DocumentExtraction,
    ExtractionValue,
    FieldDefinition,
    FieldType,
    get_all_json_schemas,
    get_extractable_fields,
    get_field,
    get_json_schema,
)


# ======================================================================
# Field count tests
# ======================================================================


class TestFieldCounts:
    """Verify that the schema defines the expected number of fields."""

    def test_bio_field_count(self):
        """54 biological fields (from REQ_BIO.csv)."""
        assert len(BIO_FIELDS) == 54, (
            f"Expected 54 BIO fields, got {len(BIO_FIELDS)}. "
            f"Names: {ALL_BIO_FIELD_NAMES}"
        )

    def test_clinique_field_count(self):
        """48 clinical fields (from REQ_CLINIQUE.csv)."""
        assert len(CLINIQUE_FIELDS) == 48, (
            f"Expected 48 CLINIQUE fields, got {len(CLINIQUE_FIELDS)}. "
            f"Names: {ALL_CLINIQUE_FIELD_NAMES}"
        )

    def test_total_field_count(self):
        """102 total fields minus the shared 'nip' → 101 unique names."""
        total_unique = len(ALL_FIELDS_BY_NAME)
        # 'nip' appears in both BIO and CLINIQUE, so we expect
        # 54 + 48 - 1 = 101 unique field names (nip is shared)
        assert total_unique == 101, (
            f"Expected 101 unique field names, got {total_unique}"
        )

    def test_no_duplicate_bio_fields(self):
        """Each BIO field name should be unique within the BIO list."""
        names = [f.name for f in BIO_FIELDS]
        assert len(names) == len(set(names)), (
            f"Duplicate BIO field names: "
            f"{[n for n in names if names.count(n) > 1]}"
        )

    def test_no_duplicate_clinique_fields(self):
        """Each CLINIQUE field name should be unique within the CLINIQUE list."""
        names = [f.name for f in CLINIQUE_FIELDS]
        assert len(names) == len(set(names)), (
            f"Duplicate CLINIQUE field names: "
            f"{[n for n in names if names.count(n) > 1]}"
        )


# ======================================================================
# Specific field existence tests  (cross-checked against REQ CSVs)
# ======================================================================


class TestBioFieldNames:
    """Verify all expected BIO field names from REQ_BIO.csv exist."""

    EXPECTED_BIO = [
        "nip", "date_chir", "num_labo",
        "diag_histologique", "diag_integre", "classification_oms", "grade",
        "ihc_idh1", "ihc_p53", "ihc_atrx", "ihc_fgfr3", "ihc_braf",
        "ihc_hist_h3k27m", "ihc_hist_h3k27me3", "ihc_egfr_hirsch",
        "ihc_gfap", "ihc_olig2", "ihc_ki67", "ihc_mmr",
        "histo_necrose", "histo_pec", "histo_mitoses",
        "mol_idh1", "mol_idh2", "mol_tert", "mol_CDKN2A", "mol_h3f3a",
        "mol_hist1h3b", "mol_braf", "mol_mgmt", "mol_fgfr1",
        "mol_egfr_mut", "mol_prkca", "mol_p53", "mol_pten",
        "mol_cic", "mol_fubp1", "mol_atrx",
        "ch1p", "ch19q", "ch10p", "ch10q", "ch7p", "ch7q", "ch9p", "ch9q",
        "ampli_mdm2", "ampli_cdk4", "ampli_egfr", "ampli_met", "ampli_mdm4",
        "fusion_fgfr", "fusion_ntrk", "fusion_autre",
    ]

    @pytest.mark.parametrize("field_name", EXPECTED_BIO)
    def test_bio_field_exists(self, field_name: str):
        assert field_name in ALL_BIO_FIELD_NAMES, (
            f"BIO field {field_name!r} missing from schema"
        )


class TestCliniqueFieldNames:
    """Verify all expected CLINIQUE field names from REQ_CLINIQUE.csv exist."""

    EXPECTED_CLINIQUE = [
        "nip", "date_de_naissance", "sexe", "activite_professionnelle",
        "antecedent_tumoral",
        "neuroncologue", "neurochirurgien", "radiotherapeute",
        "localisation_radiotherapie", "localisation_chir",
        "date_deces", "infos_deces",
        "date_1er_symptome", "epilepsie_1er_symptome",
        "ceph_hic_1er_symptome", "deficit_1er_symptome",
        "cognitif_1er_symptome", "autre_trouble_1er_symptome",
        "exam_radio_date_decouverte",
        "contraste_1er_symptome", "oedeme_1er_symptome", "calcif_1er_symptome",
        "tumeur_lateralite", "tumeur_position",
        "dn_date", "evol_clinique",
        "chimios", "chm_date_debut", "chm_date_fin", "chm_cycles",
        "ik_clinique", "progress_clinique", "progress_radiologique",
        "date_progression",
        "epilepsie", "ceph_hic", "deficit", "cognitif", "autre_trouble",
        "anti_epileptiques", "essai_therapeutique",
        "chir_date", "type_chirurgie",
        "rx_date_debut", "rx_date_fin", "rx_dose",
        "corticoides", "optune",
    ]

    @pytest.mark.parametrize("field_name", EXPECTED_CLINIQUE)
    def test_clinique_field_exists(self, field_name: str):
        assert field_name in ALL_CLINIQUE_FIELD_NAMES, (
            f"CLINIQUE field {field_name!r} missing from schema"
        )


# ======================================================================
# Controlled vocabulary tests
# ======================================================================


class TestControlledVocab:
    """Test controlled vocabulary definitions and validators."""

    def test_binary_values(self):
        assert ControlledVocab.BINARY == {"oui", "non"}

    def test_ihc_status_values(self):
        assert "positif" in ControlledVocab.IHC_STATUS
        assert "negatif" in ControlledVocab.IHC_STATUS
        assert "maintenu" in ControlledVocab.IHC_STATUS

    def test_molecular_status_values(self):
        assert "wt" in ControlledVocab.MOLECULAR_STATUS
        assert "mute" in ControlledVocab.MOLECULAR_STATUS

    def test_chromosomal_values(self):
        assert "gain" in ControlledVocab.CHROMOSOMAL
        assert "perte" in ControlledVocab.CHROMOSOMAL
        assert "perte partielle" in ControlledVocab.CHROMOSOMAL

    def test_grade_values(self):
        assert ControlledVocab.GRADE == {1, 2, 3, 4}

    def test_who_classification(self):
        assert ControlledVocab.WHO_CLASSIFICATION == {"2007", "2016", "2021"}

    def test_surgery_types(self):
        expected = {"exerese complete", "exerese partielle", "exerese", "biopsie", "en attente"}
        assert ControlledVocab.SURGERY_TYPE == expected

    def test_sex_values(self):
        assert ControlledVocab.SEX == {"M", "F"}

    def test_laterality_values(self):
        assert "gauche" in ControlledVocab.LATERALITY
        assert "droite" in ControlledVocab.LATERALITY

    # Evolution validator
    def test_evolution_initial(self):
        assert ControlledVocab.is_valid_evolution("initial")

    def test_evolution_terminal(self):
        assert ControlledVocab.is_valid_evolution("terminal")

    def test_evolution_p1(self):
        assert ControlledVocab.is_valid_evolution("P1")

    def test_evolution_p10(self):
        assert ControlledVocab.is_valid_evolution("P10")

    def test_evolution_invalid(self):
        assert not ControlledVocab.is_valid_evolution("foo")
        assert not ControlledVocab.is_valid_evolution("")
        assert not ControlledVocab.is_valid_evolution("p1")  # lowercase p not accepted

    # Molecular validator
    def test_molecular_wt(self):
        assert ControlledVocab.is_valid_molecular("wt")

    def test_molecular_mute(self):
        assert ControlledVocab.is_valid_molecular("mute")

    def test_molecular_variant(self):
        assert ControlledVocab.is_valid_molecular("R132H")
        assert ControlledVocab.is_valid_molecular("C228T")
        assert ControlledVocab.is_valid_molecular("V600E")

    def test_molecular_complex_variant(self):
        assert ControlledVocab.is_valid_molecular("mute + delete")

    def test_molecular_invalid(self):
        assert not ControlledVocab.is_valid_molecular("")


# ======================================================================
# FieldDefinition and get_field tests
# ======================================================================


class TestFieldDefinition:
    """Test FieldDefinition model and lookup helpers."""

    def test_get_field_existing(self):
        field = get_field("ihc_idh1")
        assert field.name == "ihc_idh1"
        assert field.group == "ihc"
        assert field.field_type == FieldType.CATEGORICAL
        assert field.allowed_values == ControlledVocab.IHC_STATUS

    def test_get_field_unknown(self):
        with pytest.raises(KeyError, match="Unknown field"):
            get_field("nonexistent_field_xyz")

    def test_all_fields_have_group(self):
        for name, field in ALL_FIELDS_BY_NAME.items():
            assert field.group, f"Field {name!r} has empty group"

    def test_categorical_fields_have_allowed_values(self):
        """All CATEGORICAL fields should have allowed_values set."""
        for name, field in ALL_FIELDS_BY_NAME.items():
            if field.field_type == FieldType.CATEGORICAL:
                assert field.allowed_values is not None, (
                    f"CATEGORICAL field {name!r} has no allowed_values"
                )


# ======================================================================
# ExtractionValue tests
# ======================================================================


class TestExtractionValue:
    """Test ExtractionValue pydantic model."""

    def test_minimal_creation(self):
        ev = ExtractionValue(value="positif")
        assert ev.value == "positif"
        assert ev.extraction_tier == "rule"
        assert ev.vocab_valid is True
        assert ev.flagged is False

    def test_full_creation(self):
        ev = ExtractionValue(
            value="negatif",
            source_span="IDH1 : négatif",
            source_span_start=42,
            source_span_end=56,
            extraction_tier="llm",
            confidence=0.95,
            section="ihc",
            vocab_valid=True,
            flagged=False,
        )
        assert ev.value == "negatif"
        assert ev.confidence == 0.95

    def test_null_value(self):
        ev = ExtractionValue(value=None)
        assert ev.value is None

    def test_integer_value(self):
        ev = ExtractionValue(value=4)
        assert ev.value == 4

    def test_float_value(self):
        ev = ExtractionValue(value=60.0)
        assert ev.value == 60.0

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            ExtractionValue(value="x", confidence=1.5)
        with pytest.raises(Exception):
            ExtractionValue(value="x", confidence=-0.1)

    def test_json_roundtrip(self):
        ev = ExtractionValue(value="positif", extraction_tier="rule")
        data = ev.model_dump()
        ev2 = ExtractionValue(**data)
        assert ev2.value == ev.value


# ======================================================================
# Pydantic feature model tests
# ======================================================================


class TestBiologicalFeatures:
    """Test BiologicalFeatures model."""

    def test_empty_creation(self):
        bio = BiologicalFeatures()
        assert bio.nip is None
        assert bio.ihc_idh1 is None

    def test_partial_population(self):
        bio = BiologicalFeatures(
            ihc_idh1=ExtractionValue(value="negatif"),
            grade=ExtractionValue(value=4),
        )
        assert bio.ihc_idh1.value == "negatif"
        assert bio.grade.value == 4
        assert bio.mol_tert is None  # unpopulated

    def test_all_fields_match_schema(self):
        """Every field in BiologicalFeatures must correspond to a BIO_FIELDS entry."""
        model_fields = set(BiologicalFeatures.model_fields.keys())
        schema_names = set(ALL_BIO_FIELD_NAMES)
        assert model_fields == schema_names, (
            f"Mismatch between BiologicalFeatures model and BIO_FIELDS: "
            f"model-only={model_fields - schema_names}, "
            f"schema-only={schema_names - model_fields}"
        )

    def test_json_roundtrip(self):
        bio = BiologicalFeatures(
            ihc_idh1=ExtractionValue(value="positif"),
            mol_idh1=ExtractionValue(value="R132H"),
        )
        data = bio.model_dump()
        bio2 = BiologicalFeatures(**data)
        assert bio2.ihc_idh1.value == "positif"
        assert bio2.mol_idh1.value == "R132H"


class TestClinicalFeatures:
    """Test ClinicalFeatures model."""

    def test_empty_creation(self):
        clin = ClinicalFeatures()
        assert clin.nip is None
        assert clin.epilepsie is None

    def test_partial_population(self):
        clin = ClinicalFeatures(
            sexe=ExtractionValue(value="M"),
            ik_clinique=ExtractionValue(value=90),
            type_chirurgie=ExtractionValue(value="exerese complete"),
        )
        assert clin.sexe.value == "M"
        assert clin.ik_clinique.value == 90

    def test_all_fields_match_schema(self):
        """Every field in ClinicalFeatures must correspond to a CLINIQUE_FIELDS entry."""
        model_fields = set(ClinicalFeatures.model_fields.keys())
        schema_names = set(ALL_CLINIQUE_FIELD_NAMES)
        assert model_fields == schema_names, (
            f"Mismatch between ClinicalFeatures model and CLINIQUE_FIELDS: "
            f"model-only={model_fields - schema_names}, "
            f"schema-only={schema_names - model_fields}"
        )


class TestDocumentExtraction:
    """Test DocumentExtraction model."""

    def test_creation(self):
        doc = DocumentExtraction(
            document_id="doc_001",
            document_type="anapath",
            patient_id="8003373720",
        )
        assert doc.document_id == "doc_001"
        assert doc.document_type == "anapath"
        assert isinstance(doc.clinical, ClinicalFeatures)
        assert isinstance(doc.biological, BiologicalFeatures)

    def test_with_features(self):
        doc = DocumentExtraction(
            document_id="doc_001",
            document_type="anapath",
            biological=BiologicalFeatures(
                grade=ExtractionValue(value=4),
            ),
        )
        assert doc.biological.grade.value == 4


# ======================================================================
# Feature routing tests
# ======================================================================


class TestFeatureRouting:
    """Test FEATURE_ROUTING and get_extractable_fields."""

    def test_all_document_types_routed(self):
        for doc_type in DOCUMENT_TYPES:
            assert doc_type in FEATURE_ROUTING, (
                f"Document type {doc_type!r} missing from FEATURE_ROUTING"
            )

    def test_anapath_gets_all_bio(self):
        fields = get_extractable_fields("anapath")
        for bio_name in ALL_BIO_FIELD_NAMES:
            assert bio_name in fields, f"Anapath should extract BIO field {bio_name!r}"

    def test_anapath_no_clinique(self):
        routing = FEATURE_ROUTING["anapath"]
        assert routing["clinique"] == []

    def test_consultation_gets_all_clinique(self):
        fields = get_extractable_fields("consultation")
        for clin_name in ALL_CLINIQUE_FIELD_NAMES:
            assert clin_name in fields, (
                f"Consultation should extract CLINIQUE field {clin_name!r}"
            )

    def test_consultation_no_bio(self):
        routing = FEATURE_ROUTING["consultation"]
        assert routing["bio"] == []

    def test_molecular_report_subset(self):
        fields = get_extractable_fields("molecular_report")
        # Should include molecular fields
        assert "mol_idh1" in fields
        assert "mol_tert" in fields
        # Should include chromosomal fields
        assert "ch1p" in fields
        # Should NOT include IHC or diagnosis
        assert "ihc_idh1" not in fields

    def test_rcp_has_both(self):
        routing = FEATURE_ROUTING["rcp"]
        assert len(routing["bio"]) > 0
        assert len(routing["clinique"]) > 0

    def test_radiology_limited_scope(self):
        fields = get_extractable_fields("radiology")
        assert "tumeur_lateralite" in fields
        assert "tumeur_position" in fields
        # Should not include treatment or bio fields
        assert "chimios" not in fields
        assert "mol_idh1" not in fields

    def test_invalid_document_type(self):
        with pytest.raises(ValueError, match="Unknown document type"):
            get_extractable_fields("unknown_type")

    def test_routed_fields_exist_in_schema(self):
        """Every field name in FEATURE_ROUTING must exist in ALL_FIELDS_BY_NAME."""
        for doc_type, routing in FEATURE_ROUTING.items():
            for field_name in routing["bio"] + routing["clinique"]:
                assert field_name in ALL_FIELDS_BY_NAME, (
                    f"FEATURE_ROUTING[{doc_type!r}] references unknown field "
                    f"{field_name!r}"
                )


# ======================================================================
# Feature groups tests
# ======================================================================


class TestFeatureGroups:
    """Test FEATURE_GROUPS organisation."""

    def test_all_groups_exist(self):
        expected = {
            "ihc", "molecular", "chromosomal", "diagnosis",
            "demographics", "symptoms", "treatment", "evolution",
        }
        assert set(FEATURE_GROUPS.keys()) == expected

    def test_ihc_group_fields(self):
        ihc = FEATURE_GROUPS["ihc"]
        assert "ihc_idh1" in ihc
        assert "ihc_ki67" in ihc
        assert "ihc_mmr" in ihc
        assert len(ihc) == 12  # 12 IHC fields

    def test_molecular_group_fields(self):
        mol = FEATURE_GROUPS["molecular"]
        assert "mol_idh1" in mol
        assert "mol_mgmt" in mol
        assert len(mol) == 16  # 16 molecular fields

    def test_groups_reference_real_fields(self):
        """All fields referenced by groups must exist."""
        for group, fields in FEATURE_GROUPS.items():
            for name in fields:
                assert name in ALL_FIELDS_BY_NAME, (
                    f"FEATURE_GROUPS[{group!r}] references unknown field {name!r}"
                )


# ======================================================================
# JSON schema generation tests
# ======================================================================


class TestJSONSchemaGeneration:
    """Test get_json_schema() and get_all_json_schemas()."""

    @pytest.mark.parametrize("group", list(FEATURE_GROUPS.keys()))
    def test_schema_is_valid_json(self, group: str):
        schema = get_json_schema(group)
        # Must be serialisable to JSON
        json_str = json.dumps(schema, indent=2)
        parsed = json.loads(json_str)
        assert parsed == schema

    @pytest.mark.parametrize("group", list(FEATURE_GROUPS.keys()))
    def test_schema_structure(self, group: str):
        schema = get_json_schema(group)
        assert schema["type"] == "object"
        assert "values" in schema["properties"]
        assert "_source" in schema["properties"]
        assert schema["required"] == ["values", "_source"]

    def test_ihc_schema_values(self):
        schema = get_json_schema("ihc")
        values_props = schema["properties"]["values"]["properties"]
        # Check a categorical field
        assert "ihc_idh1" in values_props
        idh1 = values_props["ihc_idh1"]
        assert "enum" in idh1
        assert None in idh1["enum"]
        assert "positif" in idh1["enum"]
        assert "negatif" in idh1["enum"]

    def test_ihc_schema_source_spans(self):
        schema = get_json_schema("ihc")
        source_props = schema["properties"]["_source"]["properties"]
        assert "ihc_idh1" in source_props
        assert source_props["ihc_idh1"]["type"] == ["string", "null"]

    def test_diagnosis_schema_integer_field(self):
        schema = get_json_schema("diagnosis")
        values_props = schema["properties"]["values"]["properties"]
        grade = values_props["grade"]
        # Grade is an integer with enum values {1, 2, 3, 4}
        assert "enum" in grade
        assert 1 in grade["enum"]
        assert 4 in grade["enum"]
        assert None in grade["enum"]

    def test_unknown_group_raises(self):
        with pytest.raises(KeyError, match="Unknown feature group"):
            get_json_schema("nonexistent_group")

    def test_get_all_schemas(self):
        all_schemas = get_all_json_schemas()
        assert set(all_schemas.keys()) == set(FEATURE_GROUPS.keys())
        for schema in all_schemas.values():
            assert schema["type"] == "object"


# ======================================================================
# Sample extraction parsing tests
# ======================================================================


class TestSampleExtractionParsing:
    """Validate that the schema can parse sample extraction outputs
    matching REQ_BIO.csv / REQ_CLINIQUE.csv annotation data."""

    def test_parse_bio_sample(self):
        """Parse a sample BIO extraction matching REQ_BIO patient 1."""
        bio = BiologicalFeatures(
            nip=ExtractionValue(value="8003373720"),
            date_chir=ExtractionValue(value="15/10/2024"),
            num_labo=ExtractionValue(value="24EN01638"),
            diag_histologique=ExtractionValue(value="glioblastome"),
            diag_integre=ExtractionValue(value="glioblastome IDHwt"),
            classification_oms=ExtractionValue(value="2021"),
            grade=ExtractionValue(value=4),
            ihc_idh1=ExtractionValue(value="negatif"),
            ihc_p53=ExtractionValue(value="negatif"),
            ihc_atrx=ExtractionValue(value="maintenu"),
            ihc_fgfr3=ExtractionValue(value="negatif"),
            ihc_egfr_hirsch=ExtractionValue(value="negatif"),
            ihc_olig2=ExtractionValue(value="positif"),
            ihc_ki67=ExtractionValue(value="15-20"),
            histo_necrose=ExtractionValue(value="oui"),
            histo_pec=ExtractionValue(value="oui"),
            histo_mitoses=ExtractionValue(value=6),
            mol_idh1=ExtractionValue(value="wt"),
            mol_idh2=ExtractionValue(value="wt"),
            mol_tert=ExtractionValue(value="mute"),
            mol_CDKN2A=ExtractionValue(value="wt"),
            mol_h3f3a=ExtractionValue(value="wt"),
            mol_hist1h3b=ExtractionValue(value="wt"),
            mol_braf=ExtractionValue(value="V600E"),
            mol_fgfr1=ExtractionValue(value="wt"),
            mol_egfr_mut=ExtractionValue(value="wt"),
            mol_prkca=ExtractionValue(value="wt"),
            mol_p53=ExtractionValue(value="mute"),
            mol_pten=ExtractionValue(value="mute"),
            mol_cic=ExtractionValue(value="wt"),
            mol_fubp1=ExtractionValue(value="wt"),
            mol_atrx=ExtractionValue(value="wt"),
            ch10p=ExtractionValue(value="perte"),
            ch10q=ExtractionValue(value="perte"),
            ch7p=ExtractionValue(value="gain"),
            ch7q=ExtractionValue(value="gain"),
            ampli_mdm2=ExtractionValue(value="non"),
            ampli_cdk4=ExtractionValue(value="oui"),
            ampli_egfr=ExtractionValue(value="non"),
            ampli_met=ExtractionValue(value="non"),
            ampli_mdm4=ExtractionValue(value="non"),
            fusion_fgfr=ExtractionValue(value="non"),
            fusion_ntrk=ExtractionValue(value="non"),
            fusion_autre=ExtractionValue(value="non"),
        )
        assert bio.grade.value == 4
        assert bio.ihc_idh1.value == "negatif"
        assert bio.mol_braf.value == "V600E"

    def test_parse_clinique_sample(self):
        """Parse a sample CLINIQUE extraction matching REQ_CLINIQUE patient 1."""
        clin = ClinicalFeatures(
            nip=ExtractionValue(value="8003373720"),
            date_de_naissance=ExtractionValue(value="26/08/1977"),
            sexe=ExtractionValue(value="M"),
            activite_professionnelle=ExtractionValue(value="Agent immobilier"),
            antecedent_tumoral=ExtractionValue(value="Non"),
            neuroncologue=ExtractionValue(value="Touat"),
            neurochirurgien=ExtractionValue(value="Mathon"),
            radiotherapeute=ExtractionValue(value="Assouline"),
            localisation_radiotherapie=ExtractionValue(value="Boulogne"),
            localisation_chir=ExtractionValue(value="PSL"),
            date_deces=ExtractionValue(value="NA"),
            infos_deces=ExtractionValue(value="NA"),
            date_1er_symptome=ExtractionValue(value="03/10/2024"),
            epilepsie_1er_symptome=ExtractionValue(value="oui"),
            ceph_hic_1er_symptome=ExtractionValue(value="non"),
            deficit_1er_symptome=ExtractionValue(value="non"),
            cognitif_1er_symptome=ExtractionValue(value="non"),
            autre_trouble_1er_symptome=ExtractionValue(value="non"),
            exam_radio_date_decouverte=ExtractionValue(value="oct-24"),
            tumeur_lateralite=ExtractionValue(value="gauche"),
            tumeur_position=ExtractionValue(value="pariétale"),
            dn_date=ExtractionValue(value="13/01/2026"),
            evol_clinique=ExtractionValue(value="initial"),
            chimios=ExtractionValue(value="Temozolomide"),
            chm_date_debut=ExtractionValue(value="06/02/2025"),
            chm_date_fin=ExtractionValue(value="13/05/2025"),
            chm_cycles=ExtractionValue(value=4),
            ik_clinique=ExtractionValue(value=100),
            progress_clinique=ExtractionValue(value="non"),
            progress_radiologique=ExtractionValue(value="oui"),
            date_progression=ExtractionValue(value="09/05/2025"),
            epilepsie=ExtractionValue(value="oui"),
            ceph_hic=ExtractionValue(value="non"),
            deficit=ExtractionValue(value="non"),
            cognitif=ExtractionValue(value="non"),
            autre_trouble=ExtractionValue(value="non"),
            anti_epileptiques=ExtractionValue(value="oui"),
            essai_therapeutique=ExtractionValue(value="non"),
            chir_date=ExtractionValue(value="15/10/2024"),
            type_chirurgie=ExtractionValue(value="exerese complete"),
            rx_date_debut=ExtractionValue(value="26/11/2024"),
            rx_date_fin=ExtractionValue(value="09/01/2025"),
            rx_dose=ExtractionValue(value="60"),
            optune=ExtractionValue(value="oui"),
        )
        assert clin.sexe.value == "M"
        assert clin.ik_clinique.value == 100
        assert clin.type_chirurgie.value == "exerese complete"

    def test_document_extraction_roundtrip(self):
        """Full DocumentExtraction with both BIO and CLINIQUE parses and roundtrips."""
        doc = DocumentExtraction(
            document_id="test_001",
            document_type="anapath",
            patient_id="8003373720",
            biological=BiologicalFeatures(
                grade=ExtractionValue(value=4),
                ihc_idh1=ExtractionValue(
                    value="negatif",
                    source_span="IDH1 : négatif",
                    extraction_tier="rule",
                ),
            ),
            clinical=ClinicalFeatures(
                sexe=ExtractionValue(value="M"),
            ),
            raw_text="Sample anapath report text…",
            sections={"ihc": "IDH1 : négatif, Ki67 : 15-20%"},
        )
        # JSON roundtrip
        json_str = doc.model_dump_json()
        doc2 = DocumentExtraction.model_validate_json(json_str)
        assert doc2.biological.grade.value == 4
        assert doc2.biological.ihc_idh1.source_span == "IDH1 : négatif"
        assert doc2.clinical.sexe.value == "M"
        assert doc2.sections["ihc"] == "IDH1 : négatif, Ki67 : 15-20%"
