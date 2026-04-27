"""Tests for Phase A/B/C new rule-based extractors.

Covers: sexe, tumeur_lateralite, evol_clinique, type_chirurgie,
classification_oms, chimios, tumeur_position, diag_histologique.
"""

import pytest

from src.extraction.rule_extraction import (
    extract_sexe,
    extract_tumeur_lateralite,
    extract_evol_clinique,
    extract_type_chirurgie,
    extract_classification_oms,
    extract_chimios,
    extract_tumeur_position,
    extract_diag_histologique,
    extract_ihc,
    extract_molecular,
    extract_chromosomal,
    run_rule_extraction,
)
from src.extraction.schema import ExtractionValue


# ═══════════════════════════════════════════════════════════════════════════
# A1, sexe
# ═══════════════════════════════════════════════════════════════════════════

class TestSexeExtraction:
    """Patient sex extraction."""

    def test_sexe_header_f(self):
        """Header line pipe format."""
        result = extract_sexe("Pat.: Smith | F | 45 ans")
        assert "sexe" in result
        assert result["sexe"].value == "F"

    def test_sexe_header_m(self):
        result = extract_sexe("Pat.: Dupont | M | 60 ans")
        assert result["sexe"].value == "M"

    def test_sexe_salutation_mme(self):
        result = extract_sexe("Madame Dupont vue en consultation")
        assert result["sexe"].value == "F"

    def test_sexe_salutation_monsieur(self):
        result = extract_sexe("Monsieur Martin, né le 12/03/1960")
        assert result["sexe"].value == "M"

    def test_sexe_nee_le(self):
        result = extract_sexe("Patiente née le 05/04/1975")
        assert result["sexe"].value == "F"

    def test_sexe_ne_le(self):
        result = extract_sexe("Patient né le 12/03/1960")
        assert result["sexe"].value == "M"

    def test_sexe_agreement_vue(self):
        result = extract_sexe("La patiente a été vue en consultation le 15/03")
        assert result["sexe"].value == "F"

    def test_sexe_agreement_vu(self):
        result = extract_sexe("Le patient a été vu en consultation le 15/03")
        assert result["sexe"].value == "M"

    def test_sexe_token_femme(self):
        result = extract_sexe("Sexe : femme")
        assert result["sexe"].value == "F"

    def test_sexe_token_homme(self):
        result = extract_sexe("Sexe : homme")
        assert result["sexe"].value == "M"

    def test_sexe_none(self):
        """Generic text with no gender marker."""
        result = extract_sexe("Consultation de suivi programmée.")
        assert len(result) == 0

    def test_sexe_extraction_tier(self):
        result = extract_sexe("Pat.: X | F |")
        assert result["sexe"].extraction_tier == "rule"

    def test_sexe_confidence_header_highest(self):
        result = extract_sexe("Pat.: X | F |")
        assert result["sexe"].confidence == 0.95


# ═══════════════════════════════════════════════════════════════════════════
# A2, tumeur_lateralite
# ═══════════════════════════════════════════════════════════════════════════

class TestLateraliteExtraction:
    """Tumour laterality extraction."""

    def test_lateralite_explicit_gauche(self):
        result = extract_tumeur_lateralite("Latéralité : gauche")
        assert result["tumeur_lateralite"].value == "gauche"

    def test_lateralite_explicit_droite(self):
        result = extract_tumeur_lateralite("côté : droite")
        assert result["tumeur_lateralite"].value == "droite"

    def test_lateralite_anatomical_droit(self):
        result = extract_tumeur_lateralite("lobe frontal droit fortement envahi")
        assert result["tumeur_lateralite"].value == "droite"

    def test_lateralite_anatomical_gauche(self):
        result = extract_tumeur_lateralite("hémisphère gauche atteint")
        assert result["tumeur_lateralite"].value == "gauche"

    def test_lateralite_typo_no_accent(self):
        """Lateralite without accent should still match."""
        result = extract_tumeur_lateralite("lateralite: gauche")
        assert result["tumeur_lateralite"].value == "gauche"

    def test_lateralite_bilateral(self):
        result = extract_tumeur_lateralite("Latéralité : bilatéral")
        assert result["tumeur_lateralite"].value == "bilateral"

    def test_lateralite_median(self):
        result = extract_tumeur_lateralite("Latéralité : médian")
        assert result["tumeur_lateralite"].value == "median"

    def test_lateralite_near_tumour(self):
        """Side keyword near tumour anchor within 200 chars."""
        result = extract_tumeur_lateralite("La tumeur mesure 3cm. Elle est gauche.")
        assert result["tumeur_lateralite"].value == "gauche"

    def test_lateralite_none_no_context(self):
        """Side keyword far from tumour context should not match."""
        result = extract_tumeur_lateralite("Le patient mange de la main gauche." + " " * 300 + "Rien à signaler.")
        # The word "gauche" is there but no tumour anchor
        assert len(result) == 0 or result.get("tumeur_lateralite") is None

    def test_lateralite_normalisation(self):
        result = extract_tumeur_lateralite("lobe temporal droit")
        assert result["tumeur_lateralite"].value == "droite"


# ═══════════════════════════════════════════════════════════════════════════
# A3, evol_clinique
# ═══════════════════════════════════════════════════════════════════════════

class TestEvolCliniqueExtraction:
    """Clinical evolution label extraction."""

    def test_evol_explicit_p2(self):
        result = extract_evol_clinique("Évolution : P2")
        assert result["evol_clinique"].value == "P2"

    def test_evol_explicit_initial(self):
        result = extract_evol_clinique("Étape : initial")
        assert result["evol_clinique"].value == "initial"

    def test_evol_explicit_terminal(self):
        result = extract_evol_clinique("Évolution : terminal")
        assert result["evol_clinique"].value == "terminal"

    def test_evol_header_initial(self):
        """Keyword 'initial' in document header (first 200 chars)."""
        result = extract_evol_clinique("Point Initial - Consultation du 15/03/2024")
        assert result["evol_clinique"].value == "initial"

    def test_evol_header_p1(self):
        result = extract_evol_clinique("RCP P1 - Dossier 12345")
        assert result["evol_clinique"].value == "P1"

    def test_evol_ordinal_deuxieme(self):
        result = extract_evol_clinique("deuxième progression documentée en mars 2024")
        assert result["evol_clinique"].value == "P2"

    def test_evol_ordinal_premiere(self):
        result = extract_evol_clinique("première progression sous traitement")
        assert result["evol_clinique"].value == "P1"

    def test_evol_context_bilan_initial(self):
        result = extract_evol_clinique("Le patient est adressé pour bilan initial.")
        assert result["evol_clinique"].value == "initial"
        assert result["evol_clinique"].flagged is True

    def test_evol_none(self):
        result = extract_evol_clinique("IRM cérébrale de contrôle.")
        assert len(result) == 0

    def test_evol_ordinal_3eme(self):
        result = extract_evol_clinique("3ème progression visible sur l'imagerie")
        assert result["evol_clinique"].value == "P3"


# ═══════════════════════════════════════════════════════════════════════════
# A4, type_chirurgie
# ═══════════════════════════════════════════════════════════════════════════

class TestTypeChirurgieExtraction:
    """Surgery type extraction."""

    def test_chir_exerese_complete(self):
        result = extract_type_chirurgie("Résection complète de la tumeur (GTR)")
        assert result["type_chirurgie"].value == "exerese complete"

    def test_chir_gtr(self):
        result = extract_type_chirurgie("GTR réalisée le 15/03")
        assert result["type_chirurgie"].value == "exerese complete"

    def test_chir_exerese_partielle(self):
        result = extract_type_chirurgie("exérèse partielle de la lésion")
        assert result["type_chirurgie"].value == "exerese partielle"

    def test_chir_str(self):
        result = extract_type_chirurgie("STR réalisée")
        assert result["type_chirurgie"].value == "exerese partielle"

    def test_chir_subtotale(self):
        result = extract_type_chirurgie("Résection sub-totale effectuée")
        assert result["type_chirurgie"].value == "exerese partielle"

    def test_chir_biopsie(self):
        result = extract_type_chirurgie("biopsie stéréotaxique réalisée le 10/02")
        assert result["type_chirurgie"].value == "biopsie"

    def test_chir_bst(self):
        result = extract_type_chirurgie("BST réalisée")
        assert result["type_chirurgie"].value == "biopsie"

    def test_chir_attente(self):
        result = extract_type_chirurgie("chirurgie prévue ultérieurement")
        assert result["type_chirurgie"].value == "en attente"

    def test_chir_typo_no_accent(self):
        """'exerese complete' without accents should match."""
        result = extract_type_chirurgie("exerese complete de la tumeur")
        assert result["type_chirurgie"].value == "exerese complete"

    def test_chir_bare_exerese(self):
        """Bare 'exérèse' without qualifier."""
        result = extract_type_chirurgie("exérèse de la lésion")
        assert result["type_chirurgie"].value == "exerese"

    def test_chir_none(self):
        result = extract_type_chirurgie("IRM cérébrale de contrôle.")
        assert len(result) == 0

    def test_chir_priority_complete_over_bare(self):
        """Complete should take priority over bare exerese."""
        result = extract_type_chirurgie("Exérèse complète de la tumeur après exérèse")
        assert result["type_chirurgie"].value == "exerese complete"


# ═══════════════════════════════════════════════════════════════════════════
# A5, classification_oms
# ═══════════════════════════════════════════════════════════════════════════

class TestClassificationOMSExtraction:
    """WHO classification year extraction."""

    def test_oms_explicit_2021(self):
        result = extract_classification_oms("classification OMS 2021")
        assert result["classification_oms"].value == "2021"

    def test_oms_who_2016(self):
        result = extract_classification_oms("WHO 2016 glioma classification")
        assert result["classification_oms"].value == "2016"

    def test_oms_2007(self):
        result = extract_classification_oms("classification WHO de 2007")
        assert result["classification_oms"].value == "2007"

    def test_oms_class_abbreviation(self):
        result = extract_classification_oms("class. OMS 2021")
        assert result["classification_oms"].value == "2021"

    def test_oms_invalid_year(self):
        """Year 2019 is not a valid OMS edition."""
        result = extract_classification_oms("classification OMS 2019")
        assert len(result) == 0

    def test_oms_classification_only(self):
        result = extract_classification_oms("classification 2021")
        assert result["classification_oms"].value == "2021"

    def test_oms_none(self):
        result = extract_classification_oms("Le patient va bien.")
        assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════════════
# B1, chimios
# ═══════════════════════════════════════════════════════════════════════════

class TestChimiosExtraction:
    """Chemotherapy drug dictionary extraction."""

    def test_chimios_tmz(self):
        result = extract_chimios("TMZ 75 mg/m²")
        assert result["chimios"].value == "temozolomide"

    def test_chimios_temozolomide_full(self):
        result = extract_chimios("temozolomide 200mg")
        assert result["chimios"].value == "temozolomide"

    def test_chimios_multi(self):
        result = extract_chimios("bevacizumab + TMZ en traitement")
        assert "bevacizumab" in result["chimios"].value
        assert "temozolomide" in result["chimios"].value

    def test_chimios_avastin(self):
        result = extract_chimios("Protocole par avastin")
        assert result["chimios"].value == "bevacizumab"

    def test_chimios_lomustine(self):
        result = extract_chimios("CCNU administré")
        assert result["chimios"].value == "lomustine"

    def test_chimios_none(self):
        result = extract_chimios("IRM cérébrale normale.")
        assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════════════
# B2, tumeur_position
# ═══════════════════════════════════════════════════════════════════════════

class TestTumeurPositionExtraction:
    """Tumour position extraction."""

    def test_position_frontal(self):
        result = extract_tumeur_position("tumeur du lobe frontal gauche")
        assert "tumeur_position" in result
        assert "frontal" in result["tumeur_position"].value

    def test_position_thalamus(self):
        result = extract_tumeur_position("localisée au thalamus")
        assert "thalamus" in result["tumeur_position"].value

    def test_position_corps_calleux(self):
        result = extract_tumeur_position("lésion du corps calleux")
        assert "corps calleux" in result["tumeur_position"].value

    def test_position_unmatched(self):
        result = extract_tumeur_position("zone mal définie")
        assert len(result) == 0

    def test_position_cervelet(self):
        result = extract_tumeur_position("tumeur du cervelet")
        assert "cervelet" in result["tumeur_position"].value


# ═══════════════════════════════════════════════════════════════════════════
# C5, diag_histologique
# ═══════════════════════════════════════════════════════════════════════════

class TestDiagHistologiqueExtraction:
    """Histological diagnosis extraction."""

    def test_diag_gbm(self):
        result = extract_diag_histologique("GBM grade IV")
        assert result["diag_histologique"].value == "glioblastome"

    def test_diag_glioblastome(self):
        result = extract_diag_histologique("Conclusion : glioblastome de haut grade")
        assert result["diag_histologique"].value == "glioblastome"

    def test_diag_typo(self):
        result = extract_diag_histologique("glioblasatome multiforme")
        assert result["diag_histologique"].value == "glioblastome"

    def test_diag_oligo(self):
        result = extract_diag_histologique("oligo grade II")
        assert result["diag_histologique"].value == "oligodendrogliome"

    def test_diag_meningiome(self):
        result = extract_diag_histologique("méningiome de grade I")
        assert result["diag_histologique"].value == "meningiome"

    def test_diag_raw_fallback(self):
        result = extract_diag_histologique("Diagnostic : tumeur neuroépithéliale rare")
        assert result["diag_histologique"].value is not None
        assert result["diag_histologique"].flagged is True

    def test_diag_none(self):
        result = extract_diag_histologique("Le patient va bien.")
        assert len(result) == 0

    def test_diag_metastase(self):
        result = extract_diag_histologique("localisation secondaire cérébrale")
        assert result["diag_histologique"].value == "metastase"


# ═══════════════════════════════════════════════════════════════════════════
# Integration: run_rule_extraction with new extractors
# ═══════════════════════════════════════════════════════════════════════════

class TestRunRuleExtractionNewExtractors:
    """Integration tests: new extractors via run_rule_extraction."""

    def test_sexe_in_full_text(self):
        text = "Madame Dupont, 55 ans, vue en consultation."
        result = run_rule_extraction(text, {"full_text": text}, ["sexe"])
        assert "sexe" in result
        assert result["sexe"].value == "F"

    def test_type_chirurgie_in_treatment(self):
        text = "Biopsie stéréotaxique réalisée le 10/02/2024."
        result = run_rule_extraction(
            text, {"treatment": text}, ["type_chirurgie"]
        )
        assert "type_chirurgie" in result
        assert result["type_chirurgie"].value == "biopsie"

    def test_evol_in_conclusion(self):
        text = "Évolution : P2. Progression radiologique confirmée."
        result = run_rule_extraction(
            text, {"conclusion": text}, ["evol_clinique"]
        )
        assert "evol_clinique" in result
        assert result["evol_clinique"].value == "P2"

    def test_classification_in_conclusion(self):
        text = "Glioblastome, classification OMS 2021, grade IV."
        result = run_rule_extraction(
            text, {"conclusion": text}, ["classification_oms", "grade"]
        )
        assert "classification_oms" in result
        assert result["classification_oms"].value == "2021"

    def test_combined_document(self):
        """Full document with multiple new fields."""
        text = (
            "Pat.: Dupont | F | 55 ans\n"
            "Latéralité : gauche\n"
            "Résection complète de la tumeur (GTR)\n"
            "classification OMS 2021\n"
            "Évolution : P1\n"
            "TMZ 6 cycles\n"
            "Glioblastome grade IV\n"
        )
        features = [
            "sexe", "tumeur_lateralite", "type_chirurgie",
            "classification_oms", "evol_clinique", "chimios",
            "diag_histologique",
        ]
        result = run_rule_extraction(text, {"full_text": text}, features)

        assert result["sexe"].value == "F"
        assert result["tumeur_lateralite"].value == "gauche"
        assert result["type_chirurgie"].value == "exerese complete"
        assert result["classification_oms"].value == "2021"
        assert result["evol_clinique"].value == "P1"
        assert result["chimios"].value == "temozolomide"
        assert result["diag_histologique"].value == "glioblastome"

    def test_fallback_to_full_text(self):
        """When no sections match, fall back to full text."""
        text = "Patient homme de 60 ans. Exérèse partielle. WHO 2016."
        result = run_rule_extraction(
            text, {}, ["sexe", "type_chirurgie", "classification_oms"]
        )
        assert "sexe" in result
        assert result["sexe"].value == "M"
        assert "type_chirurgie" in result
        assert result["type_chirurgie"].value == "exerese partielle"
        assert "classification_oms" in result
        assert result["classification_oms"].value == "2016"


# ═══════════════════════════════════════════════════════════════════════════
# C2, Extended IHC synonym tables
# ═══════════════════════════════════════════════════════════════════════════

class TestIHCExtendedAliases:
    """Phase C2: extended IHC marker and value aliases."""

    def test_h3k27m_dotted_alias(self):
        """H3.3 K27M alias should map to ihc_hist_h3k27m."""
        result = extract_ihc("H3.3 K27M : positif")
        assert "ihc_hist_h3k27m" in result
        assert result["ihc_hist_h3k27m"].value == "positif"

    def test_h3k27m_histone_alias(self):
        result = extract_ihc("histone H3 K27M : négatif")
        assert "ihc_hist_h3k27m" in result
        assert result["ihc_hist_h3k27m"].value == "negatif"

    def test_egfr_hirsch_alias(self):
        result = extract_ihc("score Hirsch : score de 3")
        assert "ihc_egfr_hirsch" in result

    def test_value_surexprime(self):
        """surexprimé should map to positif."""
        result = extract_ihc("p53 : surexprimé")
        assert result["ihc_p53"].value == "positif"

    def test_value_absent(self):
        """absent should map to negatif."""
        result = extract_ihc("ATRX : absent")
        assert result["ihc_atrx"].value == "negatif"

    def test_value_normal_as_maintenu(self):
        """normal should map to maintenu."""
        result = extract_ihc("ATRX : normal")
        assert result["ihc_atrx"].value == "maintenu"

    def test_value_expression_conservee(self):
        result = extract_ihc("ATRX : expression conservée")
        assert result["ihc_atrx"].value == "maintenu"

    def test_value_non_detecte(self):
        result = extract_ihc("BRAF : non détecté")
        assert result["ihc_braf"].value == "negatif"

    def test_ki67_range(self):
        """Ki67 range '15 à 20%' should normalise to '15-20'."""
        result = extract_ihc("Ki67 : 15 à 20%")
        assert result["ihc_ki67"].value == "15-20"

    def test_ki67_less_than(self):
        """Ki67 '<5%' should normalise to '<5'."""
        result = extract_ihc("Ki67 : <5%")
        assert result["ihc_ki67"].value == "<5"

    def test_dmmr_alias(self):
        result = extract_ihc("dMMR : positif")
        assert "ihc_mmr" in result
        assert result["ihc_mmr"].value == "positif"


# ═══════════════════════════════════════════════════════════════════════════
# C3, Extended molecular synonym tables
# ═══════════════════════════════════════════════════════════════════════════

class TestMolecularExtendedAliases:
    """Phase C3: extended molecular status aliases and MGMT pattern."""

    def test_sequence_sauvage(self):
        """'séquence sauvage' should map to wt."""
        result = extract_molecular("IDH1 : séquence sauvage")
        assert "mol_idh1" in result
        assert result["mol_idh1"].value == "wt"

    def test_mutation_identifiee(self):
        """'mutation identifiée' should map to mute."""
        result = extract_molecular("BRAF : mutation identifiée")
        assert "mol_braf" in result
        assert result["mol_braf"].value.startswith("mute")

    def test_mgmt_methyle(self):
        """MGMT méthylé should be captured."""
        result = extract_molecular("MGMT : méthylé")
        assert "mol_mgmt" in result
        assert result["mol_mgmt"].value == "methyle"

    def test_mgmt_non_methyle(self):
        result = extract_molecular("MGMT : non méthylé")
        assert "mol_mgmt" in result
        assert result["mol_mgmt"].value == "non methyle"

    def test_promoteur_methyle(self):
        result = extract_molecular("MGMT : promoteur méthylé")
        assert "mol_mgmt" in result
        assert result["mol_mgmt"].value == "methyle"

    def test_variant_r132h(self):
        """IDH1 R132H variant should be detected as mute."""
        result = extract_molecular("IDH1 : R132H")
        assert "mol_idh1" in result
        assert result["mol_idh1"].value.startswith("mute")

    def test_statut_wt(self):
        result = extract_molecular("IDH1 : statut WT")
        assert "mol_idh1" in result
        assert result["mol_idh1"].value == "wt"

    def test_absence_mutation_detectee(self):
        result = extract_molecular("pas de mutation détectée IDH1")
        assert "mol_idh1" in result
        assert result["mol_idh1"].value == "wt"

    def test_methylation_absente(self):
        result = extract_molecular("MGMT : méthylation absente")
        assert "mol_mgmt" in result
        assert result["mol_mgmt"].value == "non methyle"


# ═══════════════════════════════════════════════════════════════════════════
# C4, Extended chromosomal synonym tables
# ═══════════════════════════════════════════════════════════════════════════

class TestChromosomalExtendedAliases:
    """Phase C4: extended chromosomal status aliases, CGH notation, CDKN2A."""

    def test_monosomie(self):
        """monosomie should map to perte."""
        result = extract_chromosomal("10q : monosomie")
        assert "ch10q" in result
        assert result["ch10q"].value == "perte"

    def test_loh(self):
        """LOH should map to perte."""
        result = extract_chromosomal("1p : LOH")
        assert "ch1p" in result
        assert result["ch1p"].value == "perte"

    def test_polysomie(self):
        """polysomie should map to gain."""
        result = extract_chromosomal("7q : polysomie")
        assert "ch7q" in result
        assert result["ch7q"].value == "gain"

    def test_trisomie(self):
        result = extract_chromosomal("7p : trisomie")
        assert "ch7p" in result
        assert result["ch7p"].value == "gain"

    def test_cgh_short_notation_loss(self):
        """CGH array short notation: '1p -' should mean perte."""
        result = extract_chromosomal("CGH array: 1p - 19q - 10q -")
        assert "ch1p" in result
        assert result["ch1p"].value == "perte"

    def test_cgh_short_notation_gain(self):
        """CGH array short notation: '7p +' should mean gain."""
        result = extract_chromosomal("CGH array: 7p + 7q +")
        assert "ch7p" in result
        assert result["ch7p"].value == "gain"

    def test_reversed_codeletion(self):
        """'1p19q codélétion' (reversed order) should set both arms."""
        result = extract_chromosomal("1p19q codélétion confirmée")
        assert "ch1p" in result
        assert result["ch1p"].value == "perte"
        assert "ch19q" in result
        assert result["ch19q"].value == "perte"

    def test_cdkn2a_homozygous_deletion(self):
        """Homozygous CDKN2A deletion should set mol_CDKN2A and ch9p."""
        result = extract_chromosomal("délétion homozygote de CDKN2A")
        assert "mol_CDKN2A" in result
        assert result["mol_CDKN2A"].value == "mute"
        assert "ch9p" in result
        assert result["ch9p"].value == "perte"

    def test_deletion_focale(self):
        result = extract_chromosomal("9p : délétion focale")
        assert "ch9p" in result
        assert result["ch9p"].value == "perte partielle"


# ═══════════════════════════════════════════════════════════════════════════
# C6, Improved date context assignment
# ═══════════════════════════════════════════════════════════════════════════

class TestDateContextImproved:
    """Phase C6: expanded context window and additional keywords."""

    def test_date_expanded_window(self):
        """Date 150 chars before keyword should now be assigned (was too far at 120)."""
        padding = "x" * 140
        text = f"15/03/2024 {padding} chirurgie réalisée"
        result = run_rule_extraction(text, {"full_text": text}, ["date_chir"])
        assert "date_chir" in result
        assert result["date_chir"].value == "15/03/2024"

    def test_date_new_keyword_craniotomie(self):
        """New keyword 'craniotomie' should assign date_chir."""
        text = "craniotomie le 10/02/2024"
        result = run_rule_extraction(text, {"full_text": text}, ["date_chir"])
        assert "date_chir" in result
        assert result["date_chir"].value == "10/02/2024"

    def test_date_new_keyword_dn_du(self):
        """New keyword 'DN du' should assign dn_date."""
        text = "DN du 15/01/2025"
        result = run_rule_extraction(text, {"full_text": text}, ["dn_date"])
        assert "dn_date" in result
        assert result["dn_date"].value == "15/01/2025"

    def test_date_nearest_keyword_wins(self):
        """When two keywords compete, nearest should win."""
        text = "chirurgie 01/03/2024 récidive confirmée le 15/06/2024"
        result = run_rule_extraction(
            text, {"full_text": text}, ["date_chir", "date_progression"]
        )
        assert "date_chir" in result
        assert result["date_chir"].value == "01/03/2024"
        assert "date_progression" in result
        assert result["date_progression"].value == "15/06/2024"
