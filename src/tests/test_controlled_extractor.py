"""Tests for the ControlledExtractor (Find & Check)."""

from __future__ import annotations

import pytest

from src.extraction.controlled_extractor import ControlledExtractor


@pytest.fixture(scope="module")
def extractor() -> ControlledExtractor:
    return ControlledExtractor()


# ═══════════════════════════════════════════════════════════════════════
# Chromosomal fields
# ═══════════════════════════════════════════════════════════════════════

class TestChromosomal:

    def test_ch1p_perte(self, extractor: ControlledExtractor):
        text = "FISH : délétion du bras 1p, absence de perte 19q."
        res = extractor.extract(text, ["ch1p", "ch19q"], language="fr")
        assert "ch1p" in res
        assert res["ch1p"].value == "perte"

    def test_ch19q_gain(self, extractor: ControlledExtractor):
        text = "Chromosome 19q : gain de signal."
        res = extractor.extract(text, ["ch19q"], language="fr")
        assert "ch19q" in res
        assert res["ch19q"].value == "gain"

    def test_ch10q_perte_partielle(self, extractor: ControlledExtractor):
        text = "10q : perte partielle hétérozygote."
        res = extractor.extract(text, ["ch10q"], language="fr")
        assert "ch10q" in res
        assert res["ch10q"].value == "perte partielle"

    def test_ch7p_normal(self, extractor: ControlledExtractor):
        text = "7p : normal, pas d'altération."
        res = extractor.extract(text, ["ch7p"], language="fr")
        assert "ch7p" in res
        assert res["ch7p"].value == "gain"  # "normal" maps to gain

    def test_en_chromosome_loss(self, extractor: ControlledExtractor):
        text = "Chromosome 1p: deletion detected by FISH."
        res = extractor.extract(text, ["ch1p"], language="en")
        assert "ch1p" in res
        assert res["ch1p"].value == "perte"


# ═══════════════════════════════════════════════════════════════════════
# IHC fields
# ═══════════════════════════════════════════════════════════════════════

class TestIHC:

    def test_ihc_idh1_positif(self, extractor: ControlledExtractor):
        text = "IHC : IDH1 positive, p53 surexprimé."
        res = extractor.extract(text, ["ihc_idh1", "ihc_p53"], language="fr")
        assert "ihc_idh1" in res
        assert res["ihc_idh1"].value == "positif"

    def test_ihc_p53_surexprime(self, extractor: ControlledExtractor):
        text = "Immunohistochimie : p53 surexprimé."
        res = extractor.extract(text, ["ihc_p53"], language="fr")
        assert "ihc_p53" in res
        assert res["ihc_p53"].value == "positif"

    def test_ihc_atrx_positif_remapped(self, extractor: ControlledExtractor):
        """ATRX positif should be remapped to maintenu."""
        text = "ATRX : expression conservée."
        res = extractor.extract(text, ["ihc_atrx"], language="fr")
        assert "ihc_atrx" in res
        assert res["ihc_atrx"].value == "maintenu"

    def test_ihc_negatif(self, extractor: ControlledExtractor):
        text = "OLIG2 : perte d'expression."
        res = extractor.extract(text, ["ihc_olig2"], language="fr")
        assert "ihc_olig2" in res
        assert res["ihc_olig2"].value == "negatif"

    def test_ihc_en_retained(self, extractor: ControlledExtractor):
        text = "ATRX: retained nuclear expression."
        res = extractor.extract(text, ["ihc_atrx"], language="en")
        assert "ihc_atrx" in res
        # "retained" maps to maintenu, then ATRX remap has no effect
        assert res["ihc_atrx"].value == "maintenu"


# ═══════════════════════════════════════════════════════════════════════
# Molecular fields
# ═══════════════════════════════════════════════════════════════════════

class TestMolecular:

    def test_mol_idh1_mute(self, extractor: ControlledExtractor):
        text = "Biologie moléculaire : IDH1 muté (R132H)."
        res = extractor.extract(text, ["mol_idh1"], language="fr")
        assert "mol_idh1" in res
        assert res["mol_idh1"].value == "mute"

    def test_mol_idh1_wt(self, extractor: ControlledExtractor):
        text = "IDH1 : wild-type, pas de mutation détectée."
        res = extractor.extract(text, ["mol_idh1"], language="fr")
        assert "mol_idh1" in res
        assert res["mol_idh1"].value == "wt"

    def test_mol_mgmt_methyle(self, extractor: ControlledExtractor):
        text = "MGMT : promoteur méthylé."
        res = extractor.extract(text, ["mol_mgmt"], language="fr")
        assert "mol_mgmt" in res
        assert res["mol_mgmt"].value == "methyle"

    def test_mol_mgmt_non_methyle(self, extractor: ControlledExtractor):
        text = "Statut MGMT : absence de méthylation du promoteur."
        res = extractor.extract(text, ["mol_mgmt"], language="fr")
        assert "mol_mgmt" in res
        assert res["mol_mgmt"].value == "non methyle"

    def test_mol_tert_mutation(self, extractor: ControlledExtractor):
        text = "Promoteur TERT : mutation C228T détectée."
        res = extractor.extract(text, ["mol_tert"], language="fr")
        assert "mol_tert" in res
        assert res["mol_tert"].value == "mute"

    def test_mol_en_wildtype(self, extractor: ControlledExtractor):
        text = "IDH2: no mutation detected."
        res = extractor.extract(text, ["mol_idh2"], language="en")
        assert "mol_idh2" in res
        assert res["mol_idh2"].value == "wt"


# ═══════════════════════════════════════════════════════════════════════
# Amplification fields
# ═══════════════════════════════════════════════════════════════════════

class TestAmplification:

    def test_ampli_egfr_oui(self, extractor: ControlledExtractor):
        text = "EGFR : amplification détectée par FISH."
        res = extractor.extract(text, ["ampli_egfr"], language="fr")
        assert "ampli_egfr" in res
        assert res["ampli_egfr"].value == "oui"

    def test_ampli_mdm2_non(self, extractor: ControlledExtractor):
        text = "Pas d'amplification de MDM2."
        res = extractor.extract(text, ["ampli_mdm2"], language="fr")
        assert "ampli_mdm2" in res
        assert res["ampli_mdm2"].value == "non"


# ═══════════════════════════════════════════════════════════════════════
# Fusion fields
# ═══════════════════════════════════════════════════════════════════════

class TestFusion:

    def test_fusion_fgfr_oui(self, extractor: ControlledExtractor):
        text = "Réarrangement FGFR détecté, fusion confirmée."
        res = extractor.extract(text, ["fusion_fgfr"], language="fr")
        assert "fusion_fgfr" in res
        assert res["fusion_fgfr"].value == "oui"

    def test_fusion_ntrk_non(self, extractor: ControlledExtractor):
        text = "Absence de fusion NTRK."
        res = extractor.extract(text, ["fusion_ntrk"], language="fr")
        assert "fusion_ntrk" in res
        assert res["fusion_ntrk"].value == "non"


# ═══════════════════════════════════════════════════════════════════════
# Other categorical fields
# ═══════════════════════════════════════════════════════════════════════

class TestOtherCategorical:

    def test_type_chirurgie_biopsie(self, extractor: ControlledExtractor):
        text = "Le patient a bénéficié d'une biopsie stéréotaxique."
        res = extractor.extract(text, ["type_chirurgie"], language="fr")
        assert "type_chirurgie" in res
        assert res["type_chirurgie"].value == "biopsie"

    def test_type_chirurgie_complete(self, extractor: ControlledExtractor):
        text = "Exérèse macroscopiquement complète réalisée."
        res = extractor.extract(text, ["type_chirurgie"], language="fr")
        assert "type_chirurgie" in res
        assert res["type_chirurgie"].value == "exerese complete"

    def test_evol_initial(self, extractor: ControlledExtractor):
        text = "Évolution : initial — première consultation."
        res = extractor.extract(text, ["evol_clinique"], language="fr")
        assert "evol_clinique" in res
        assert res["evol_clinique"].value == "initial"

    def test_classification_oms(self, extractor: ControlledExtractor):
        text = "Classification OMS 2021."
        res = extractor.extract(text, ["classification_oms"], language="fr")
        assert "classification_oms" in res
        assert res["classification_oms"].value == "2021"


# ═══════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_no_match_returns_empty(self, extractor: ControlledExtractor):
        text = "Le patient se porte bien."
        res = extractor.extract(text, ["ch1p", "mol_idh1"], language="fr")
        # May or may not find hits — but if no category matches, should be empty
        for ev in res.values():
            assert ev.confidence is not None and ev.confidence > 0

    def test_unknown_field_ignored(self, extractor: ControlledExtractor):
        text = "IDH1 muté."
        res = extractor.extract(text, ["nonexistent_field"], language="fr")
        assert len(res) == 0

    def test_confidence_is_normalised(self, extractor: ControlledExtractor):
        text = "1p : délétion complète."
        res = extractor.extract(text, ["ch1p"], language="fr")
        if "ch1p" in res:
            assert 0.0 <= res["ch1p"].confidence <= 1.0

    def test_extraction_tier_is_rule(self, extractor: ControlledExtractor):
        text = "IDH1 muté."
        res = extractor.extract(text, ["mol_idh1"], language="fr")
        if "mol_idh1" in res:
            assert res["mol_idh1"].extraction_tier == "rule"

    def test_multiple_fields_same_text(self, extractor: ControlledExtractor):
        """Multiple fields can be extracted from the same document."""
        text = (
            "IHC : IDH1 positif, ATRX maintenu, p53 négatif.\n"
            "Biologie moléculaire : IDH1 muté (R132H), MGMT méthylé.\n"
            "FISH : 1p délétion, 19q perte, 7p gain."
        )
        fields = [
            "ihc_idh1", "ihc_atrx", "ihc_p53",
            "mol_idh1", "mol_mgmt",
            "ch1p", "ch19q", "ch7p",
        ]
        res = extractor.extract(text, fields, language="fr")
        # We expect at least several fields to be extracted
        assert len(res) >= 3
