"""Tests for src/extraction/rule_extraction.py — Tier 1 rule-based extractors.

Each extractor has at least 5 positive and 3 negative examples per the
Phase 4 acceptance criteria.
"""

import pytest

from src.extraction.rule_extraction import (
    extract_dates,
    extract_ihc,
    extract_molecular,
    extract_chromosomal,
    extract_binary,
    extract_numerical,
    extract_amplifications,
    extract_fusions,
    run_rule_extraction,
)
from src.extraction.negation import AssertionAnnotator
from src.extraction.schema import ExtractionValue


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.1  Date extraction
# ═══════════════════════════════════════════════════════════════════════════

class TestDateExtraction:
    """Date extraction and normalisation to DD/MM/YYYY format."""

    # --- Positive examples (6+ formats) ---

    def test_dd_mm_yyyy_slash(self):
        results = extract_dates("Consultation du 15/03/2024.")
        assert len(results) >= 1
        assert results[0][0] == "15/03/2024"

    def test_dd_mm_yyyy_dot(self):
        results = extract_dates("Date : 01.12.2023")
        assert len(results) >= 1
        assert results[0][0] == "01/12/2023"

    def test_yyyy_mm_dd(self):
        results = extract_dates("Date : 2024/03/15")
        assert len(results) >= 1
        assert results[0][0] == "15/03/2024"

    def test_dd_month_yyyy_french(self):
        results = extract_dates("Le 3 février 2025, le patient consulte.")
        assert len(results) >= 1
        assert results[0][0] == "03/02/2025"

    def test_abbreviated_month_year(self):
        results = extract_dates("Depuis janv-25 le patient présente…")
        assert len(results) >= 1
        assert results[0][0] == "01/01/2025"

    def test_year_only(self):
        results = extract_dates("Chirurgie initiale en 2018.")
        assert len(results) >= 1
        assert results[0][0] == "01/01/2018"

    def test_dd_mm_yyyy_dash(self):
        results = extract_dates("Le 07-09-2022 : IRM cérébrale")
        assert len(results) >= 1
        assert results[0][0] == "07/09/2022"

    def test_multiple_dates(self):
        text = "Consultation du 10/01/2024, IRM du 05/02/2024."
        results = extract_dates(text)
        assert len(results) >= 2
        dates = [r[0] for r in results]
        assert "10/01/2024" in dates
        assert "05/02/2024" in dates

    # --- Negative examples ---

    def test_no_date_in_text(self):
        results = extract_dates("Le patient se porte bien.")
        assert len(results) == 0

    def test_number_not_date(self):
        """Pure numbers that look like dates should not match."""
        results = extract_dates("Ki67 : 30%")
        assert len(results) == 0

    def test_phone_number_not_date(self):
        results = extract_dates("Tel: 01 23 45 67 89")
        assert len(results) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.2  IHC extraction
# ═══════════════════════════════════════════════════════════════════════════

class TestIHCExtraction:
    """IHC marker extraction."""

    # --- Positive examples ---

    def test_idh1_positif(self):
        results = extract_ihc("IDH1 : positif")
        assert "ihc_idh1" in results
        assert results["ihc_idh1"].value == "positif"

    def test_atrx_perte_expression(self):
        results = extract_ihc("ATRX : perte d'expression")
        assert "ihc_atrx" in results
        assert results["ihc_atrx"].value == "negatif"

    def test_ki67_percentage(self):
        results = extract_ihc("Ki67 : 30%")
        assert "ihc_ki67" in results
        assert results["ihc_ki67"].value == "30"

    def test_p53_negatif(self):
        results = extract_ihc("p53 : négatif")
        assert "ihc_p53" in results
        assert results["ihc_p53"].value == "negatif"

    def test_gfap_positif(self):
        results = extract_ihc("GFAP = positif")
        assert "ihc_gfap" in results
        assert results["ihc_gfap"].value == "positif"

    def test_olig2_positif(self):
        results = extract_ihc("Olig2 : positif")
        assert "ihc_olig2" in results
        assert results["ihc_olig2"].value == "positif"

    def test_h3k27me3_maintenu(self):
        results = extract_ihc("H3K27me3 : maintenu")
        assert "ihc_hist_h3k27me3" in results
        assert results["ihc_hist_h3k27me3"].value == "maintenu"

    def test_braf_negatif(self):
        results = extract_ihc("BRAF - négatif")
        assert "ihc_braf" in results
        assert results["ihc_braf"].value == "negatif"

    def test_h3k27m_negatif(self):
        results = extract_ihc("H3K27M: négatif")
        assert "ihc_hist_h3k27m" in results
        assert results["ihc_hist_h3k27m"].value == "negatif"

    def test_atrx_conserve(self):
        results = extract_ihc("ATRX : conservé")
        assert "ihc_atrx" in results
        assert results["ihc_atrx"].value == "maintenu"

    # --- Negative examples ---

    def test_no_markers(self):
        results = extract_ihc("Le patient présente des céphalées.")
        assert len(results) == 0

    def test_marker_name_only(self):
        """Marker name without a result value should not match."""
        results = extract_ihc("Nous avons analysé IDH1.")
        assert len(results) == 0

    def test_unrelated_numbers(self):
        results = extract_ihc("Consultation du 15/03/2024.")
        assert len(results) == 0

    # --- Source span ---

    def test_source_span_preserved(self):
        text = "L'immunohistochimie révèle : IDH1 : positif, p53 : négatif."
        results = extract_ihc(text)
        assert "ihc_idh1" in results
        assert results["ihc_idh1"].source_span is not None
        assert "IDH1" in results["ihc_idh1"].source_span


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.3  Molecular status extraction
# ═══════════════════════════════════════════════════════════════════════════

class TestMolecularExtraction:
    """Molecular biology result extraction."""

    # --- Positive examples ---

    def test_idh1_r132h(self):
        results = extract_molecular("IDH1 : R132H")
        assert "mol_idh1" in results
        assert results["mol_idh1"].value == "mute"

    def test_tert_mute_c228t(self):
        results = extract_molecular("TERT muté C228T")
        assert "mol_tert" in results
        assert results["mol_tert"].value == "mute"

    def test_mgmt_non_methyle(self):
        results = extract_molecular("MGMT : non méthylé")
        assert "mol_mgmt" in results
        assert results["mol_mgmt"].value == "non methyle"

    def test_mgmt_methyle(self):
        results = extract_molecular("MGMT : méthylé")
        assert "mol_mgmt" in results
        assert results["mol_mgmt"].value == "methyle"

    def test_idh2_wt(self):
        results = extract_molecular("IDH2 : wt")
        assert "mol_idh2" in results
        assert results["mol_idh2"].value == "wt"

    def test_braf_sauvage(self):
        results = extract_molecular("BRAF : sauvage")
        assert "mol_braf" in results
        assert results["mol_braf"].value == "wt"

    def test_pten_mute(self):
        results = extract_molecular("PTEN muté")
        assert "mol_pten" in results
        assert results["mol_pten"].value == "mute"

    def test_pas_de_mutation_idh1(self):
        """Negated mutation pattern -> wt."""
        results = extract_molecular("pas de mutation IDH1")
        assert "mol_idh1" in results
        assert results["mol_idh1"].value == "wt"

    def test_mutation_du_promoteur_tert(self):
        results = extract_molecular("mutation du promoteur TERT (C228T)")
        assert "mol_tert" in results
        assert results["mol_tert"].value == "mute"

    # --- Negative examples ---

    def test_no_molecular(self):
        results = extract_molecular("Le patient va bien.")
        assert len(results) == 0

    def test_gene_name_only(self):
        results = extract_molecular("Analyse de IDH1.")
        assert len(results) == 0

    def test_unrelated_text(self):
        results = extract_molecular("IRM cérébrale normale.")
        assert len(results) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.4  Chromosomal alteration extraction
# ═══════════════════════════════════════════════════════════════════════════

class TestChromosomalExtraction:
    """Chromosomal alteration extraction."""

    # --- Positive examples ---

    def test_1p_perte(self):
        results = extract_chromosomal("1p : perte")
        assert "ch1p" in results
        assert results["ch1p"].value == "perte"

    def test_19q_perte(self):
        results = extract_chromosomal("19q = perte")
        assert "ch19q" in results
        assert results["ch19q"].value == "perte"

    def test_7p_gain(self):
        results = extract_chromosomal("7p : gain")
        assert "ch7p" in results
        assert results["ch7p"].value == "gain"

    def test_10q_perte_partielle(self):
        results = extract_chromosomal("10q perte partielle")
        assert "ch10q" in results
        assert results["ch10q"].value == "perte partielle"

    def test_codeletion_1p_19q(self):
        results = extract_chromosomal("codélétion 1p/19q")
        assert "ch1p" in results
        assert "ch19q" in results
        assert results["ch1p"].value == "perte"
        assert results["ch19q"].value == "perte"

    def test_9p_deletion(self):
        results = extract_chromosomal("9p : délétion")
        assert "ch9p" in results
        assert results["ch9p"].value == "perte"

    # --- Negative examples ---

    def test_no_chromosomal(self):
        results = extract_chromosomal("Le patient va bien.")
        assert len(results) == 0

    def test_chromosome_word_only(self):
        results = extract_chromosomal("Analyse chromosomique demandée.")
        assert len(results) == 0

    def test_unrelated_numbers(self):
        results = extract_chromosomal("Ki67 : 10%")
        assert len(results) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.5  Binary field extraction
# ═══════════════════════════════════════════════════════════════════════════

class TestBinaryExtraction:
    """Binary (oui/non) field extraction with negation awareness."""

    # --- Positive examples ---

    def test_epilepsie_present(self):
        results = extract_binary("Le patient présente une épilepsie.")
        assert "epilepsie" in results
        assert results["epilepsie"].value == "oui"

    def test_crises_comitiales_synonym(self):
        """Synonym detection: 'crises comitiales' → epilepsie=oui."""
        results = extract_binary("Crises comitiales depuis 2020.")
        assert "epilepsie" in results
        assert results["epilepsie"].value == "oui"

    def test_epilepsie_negated(self):
        results = extract_binary("Pas d'épilepsie.")
        assert "epilepsie" in results
        assert results["epilepsie"].value == "non"

    def test_deficit_present(self):
        results = extract_binary("Déficit moteur droit.")
        assert "deficit" in results
        assert results["deficit"].value == "oui"

    def test_cephalees_present(self):
        results = extract_binary("Céphalées intenses depuis une semaine.")
        assert "ceph_hic" in results
        assert results["ceph_hic"].value == "oui"

    def test_absence_deficit(self):
        results = extract_binary("Absence de déficit neurologique.")
        assert "deficit" in results
        assert results["deficit"].value == "non"

    def test_necrose_present(self):
        results = extract_binary("Présence de nécrose.")
        assert "histo_necrose" in results
        assert results["histo_necrose"].value == "oui"

    def test_corticoides_present(self):
        results = extract_binary("Le patient est sous dexaméthasone.")
        assert "corticoides" in results
        assert results["corticoides"].value == "oui"

    # --- Negative examples ---

    def test_no_keywords(self):
        results = extract_binary("IRM cérébrale normale.")
        assert len(results) == 0

    def test_unrelated_medical_text(self):
        results = extract_binary("IDH1 : positif, p53 : négatif")
        assert "epilepsie" not in results

    def test_not_applicable_text(self):
        results = extract_binary("Consultation de suivi programmée le 15/03.")
        assert len(results) == 0

    # --- With annotator ---

    def test_with_annotator_negated(self):
        annotator = AssertionAnnotator(use_edsnlp=False)
        results = extract_binary("Pas d'épilepsie.", annotator)
        assert "epilepsie" in results
        assert results["epilepsie"].value == "non"

    def test_with_annotator_positive(self):
        annotator = AssertionAnnotator(use_edsnlp=False)
        results = extract_binary("Épilepsie depuis 2020.", annotator)
        assert "epilepsie" in results
        assert results["epilepsie"].value == "oui"


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.6  Numerical extraction
# ═══════════════════════════════════════════════════════════════════════════

class TestNumericalExtraction:
    """Numerical value extraction."""

    # --- Positive examples ---

    def test_ki67_percentage(self):
        results = extract_numerical("Ki67 : 30%")
        assert "ihc_ki67" in results
        assert results["ihc_ki67"].value == "30"

    def test_ki67_low_percentage(self):
        results = extract_numerical("Ki67 = 5%")
        assert "ihc_ki67" in results
        assert results["ihc_ki67"].value == "5"

    def test_karnofsky_score(self):
        results = extract_numerical("IK à 80%")
        assert "ik_clinique" in results
        assert results["ik_clinique"].value == 80

    def test_karnofsky_kps(self):
        results = extract_numerical("KPS : 90")
        assert "ik_clinique" in results
        assert results["ik_clinique"].value == 90

    def test_mitoses_count(self):
        results = extract_numerical("12 mitoses / 10 HPF")
        assert "histo_mitoses" in results
        assert results["histo_mitoses"].value == 12

    def test_grade_arabic(self):
        results = extract_numerical("grade 4")
        assert "grade" in results
        assert results["grade"].value == 4

    def test_grade_roman(self):
        results = extract_numerical("Grade IV")
        assert "grade" in results
        assert results["grade"].value == 4

    def test_grade_2(self):
        results = extract_numerical("Grade II")
        assert "grade" in results
        assert results["grade"].value == 2

    def test_dose_gy(self):
        results = extract_numerical("60 Gy en 30 fractions")
        assert "rx_dose" in results
        assert results["rx_dose"].value == "60"

    def test_chemo_cycles(self):
        results = extract_numerical("6 cycles de témozolomide")
        assert "chm_cycles" in results
        assert results["chm_cycles"].value == 6

    # --- Negative examples ---

    def test_no_numbers(self):
        results = extract_numerical("Le patient va bien.")
        assert len(results) == 0

    def test_phone_number(self):
        results = extract_numerical("Tel: 01 23 45 67 89")
        assert "ik_clinique" not in results

    def test_random_percentage(self):
        """Percentage not associated with Ki67 should not be extracted."""
        results = extract_numerical("Compliance à 95%")
        assert "ihc_ki67" not in results


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.7  Amplification extraction
# ═══════════════════════════════════════════════════════════════════════════

class TestAmplificationExtraction:
    """Gene amplification extraction."""

    # --- Positive examples ---

    def test_amplification_egfr(self):
        results = extract_amplifications("amplification EGFR")
        assert "ampli_egfr" in results
        assert results["ampli_egfr"].value == "oui"

    def test_amplification_mdm2(self):
        results = extract_amplifications("amplification de MDM2")
        assert "ampli_mdm2" in results
        assert results["ampli_mdm2"].value == "oui"

    def test_cdk4_amplifie(self):
        results = extract_amplifications("CDK4 amplifié")
        assert "ampli_cdk4" in results
        assert results["ampli_cdk4"].value == "oui"

    def test_amplification_met(self):
        results = extract_amplifications("Amplification MET objectivée")
        assert "ampli_met" in results
        assert results["ampli_met"].value == "oui"

    def test_pas_amplification_egfr(self):
        results = extract_amplifications("pas d'amplification EGFR")
        assert "ampli_egfr" in results
        assert results["ampli_egfr"].value == "non"

    def test_absence_amplification_mdm2(self):
        results = extract_amplifications("Absence d'amplification MDM2")
        assert "ampli_mdm2" in results
        assert results["ampli_mdm2"].value == "non"

    # --- Negative examples ---

    def test_no_amplification(self):
        results = extract_amplifications("Le patient va bien.")
        assert len(results) == 0

    def test_gene_name_only(self):
        results = extract_amplifications("Analyse de EGFR par NGS.")
        assert len(results) == 0

    def test_unrelated_text(self):
        results = extract_amplifications("IRM cérébrale normale.")
        assert len(results) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 4.2.8  Fusion extraction
# ═══════════════════════════════════════════════════════════════════════════

class TestFusionExtraction:
    """Gene fusion extraction."""

    # --- Positive examples ---

    def test_fusion_fgfr(self):
        results = extract_fusions("fusion FGFR détectée")
        assert "fusion_fgfr" in results
        assert results["fusion_fgfr"].value == "oui"

    def test_fusion_ntrk(self):
        results = extract_fusions("réarrangement NTRK identifié")
        assert "fusion_ntrk" in results
        assert results["fusion_ntrk"].value == "oui"

    def test_fusion_braf(self):
        results = extract_fusions("fusion BRAF")
        assert "fusion_autre" in results
        assert results["fusion_autre"].value == "oui"

    def test_pas_de_fusion_fgfr(self):
        results = extract_fusions("pas de fusion FGFR")
        assert "fusion_fgfr" in results
        assert results["fusion_fgfr"].value == "non"

    def test_absence_de_rearrangement_ntrk(self):
        results = extract_fusions("absence de réarrangement NTRK")
        assert "fusion_ntrk" in results
        assert results["fusion_ntrk"].value == "non"

    # --- Negative examples ---

    def test_no_fusion(self):
        results = extract_fusions("Consultation de suivi.")
        assert len(results) == 0

    def test_gene_name_only(self):
        results = extract_fusions("FGFR analysé.")
        assert len(results) == 0

    def test_unrelated_text(self):
        results = extract_fusions("Ki67 : 30%")
        assert len(results) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 4.3   Master extraction function
# ═══════════════════════════════════════════════════════════════════════════

class TestRunRuleExtraction:
    """Tests for the master ``run_rule_extraction`` function."""

    def test_anapath_report(self):
        """Simulate an anapath report with IHC and molecular sections."""
        text = (
            "Immunohistochimie:\n"
            "IDH1 : positif\n"
            "p53 : négatif\n"
            "ATRX : perte d'expression\n"
            "Ki67 : 15%\n"
            "\n"
            "Biologie moléculaire:\n"
            "IDH1 muté R132H\n"
            "TERT : wt\n"
            "MGMT : méthylé\n"
            "\n"
            "Conclusion:\n"
            "Grade IV\n"
        )
        sections = {
            "ihc": "IDH1 : positif\np53 : négatif\nATRX : perte d'expression\nKi67 : 15%",
            "molecular": "IDH1 muté R132H\nTERT : wt\nMGMT : méthylé",
            "conclusion": "Grade IV",
        }
        feature_subset = [
            "ihc_idh1", "ihc_p53", "ihc_atrx", "ihc_ki67",
            "mol_idh1", "mol_tert", "mol_mgmt",
            "grade",
        ]

        results = run_rule_extraction(text, sections, feature_subset)

        assert "ihc_idh1" in results
        assert results["ihc_idh1"].value == "positif"

        assert "ihc_p53" in results
        assert results["ihc_p53"].value == "negatif"

        assert "ihc_atrx" in results
        assert results["ihc_atrx"].value == "negatif"

        assert "mol_idh1" in results
        assert results["mol_idh1"].value == "mute"

        assert "mol_tert" in results
        assert results["mol_tert"].value == "wt"

        assert "mol_mgmt" in results
        assert results["mol_mgmt"].value == "methyle"

        assert "grade" in results
        assert results["grade"].value == 4

    def test_full_text_fallback(self):
        """When no sections, extractors run on full text."""
        text = (
            "IDH1 : positif\n"
            "Grade III\n"
            "Amplification EGFR\n"
        )
        sections = {"full_text": text}
        feature_subset = ["ihc_idh1", "grade", "ampli_egfr"]

        results = run_rule_extraction(text, sections, feature_subset)

        assert "ihc_idh1" in results
        assert "grade" in results
        assert "ampli_egfr" in results

    def test_feature_filtering(self):
        """Only features in feature_subset should be returned."""
        text = "IDH1 : positif, p53 : négatif"
        sections = {"full_text": text}
        feature_subset = ["ihc_idh1"]  # Only IDH1

        results = run_rule_extraction(text, sections, feature_subset)

        assert "ihc_idh1" in results
        assert "ihc_p53" not in results

    def test_section_assignment(self):
        """Extracted values should have section metadata."""
        text = "IDH1 : positif"
        sections = {"ihc": text}
        feature_subset = ["ihc_idh1"]

        results = run_rule_extraction(text, sections, feature_subset)
        assert results["ihc_idh1"].section == "ihc"

    def test_empty_text(self):
        """Empty text should produce no results."""
        results = run_rule_extraction("", {"full_text": ""}, ["ihc_idh1"])
        assert len(results) == 0

    def test_consultation_note(self):
        """Test extraction from a consultation-like document."""
        text = (
            "Patient présentant une épilepsie depuis janvier 2023.\n"
            "IK à 80%.\n"
            "Pas de déficit neurologique.\n"
            "Sous dexaméthasone.\n"
        )
        sections = {
            "clinical_exam": text,
        }
        feature_subset = [
            "epilepsie", "ik_clinique", "deficit", "corticoides",
        ]

        results = run_rule_extraction(text, sections, feature_subset)

        assert "epilepsie" in results
        assert results["epilepsie"].value == "oui"

        assert "ik_clinique" in results
        assert results["ik_clinique"].value == 80

        assert "corticoides" in results
        assert results["corticoides"].value == "oui"

    def test_chromosomal_with_codeletion(self):
        """Test codeletion pattern in chromosomal section."""
        text = "Codélétion 1p/19q confirmée. 7p : gain."
        sections = {"chromosomal": text}
        feature_subset = ["ch1p", "ch19q", "ch7p"]

        results = run_rule_extraction(text, sections, feature_subset)

        assert "ch1p" in results
        assert results["ch1p"].value == "perte"
        assert "ch19q" in results
        assert results["ch19q"].value == "perte"
        assert "ch7p" in results
        assert results["ch7p"].value == "gain"

    def test_extraction_tier_is_rule(self):
        """All extractions should have extraction_tier='rule'."""
        text = "IDH1 : positif"
        sections = {"full_text": text}
        results = run_rule_extraction(text, sections, ["ihc_idh1"])
        assert results["ihc_idh1"].extraction_tier == "rule"
