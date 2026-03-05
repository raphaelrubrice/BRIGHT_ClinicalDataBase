import pytest
from src.extraction.eds_extractor import EDSExtractor

def test_eds_extractor_demographics():
    extractor = EDSExtractor()
    text = "Le patient est né le 15/05/1980. | M | Patient pris en charge."
    res = extractor.extract(text, {}, ["date_de_naissance", "sexe"])
    
    assert "date_de_naissance" in res and res.get("date_de_naissance").value == "15/05/1980"
    assert "sexe" in res and res.get("sexe").value == "M"

def test_eds_extractor_dates():
    extractor = EDSExtractor()
    text = "Il a été opéré le 10 juin 2023 pour une exérèse complète. Début de chimiothérapie le 01 juillet 2023. Fin de radiothérapie le 15 août 2023."
    fields = ["chir_date", "chm_date_debut", "rx_date_fin"]
    res = extractor.extract(text, {}, fields)
    
    assert "chir_date" in res and res.get("chir_date").value == "10/06/2023"
    assert "chm_date_debut" in res and res.get("chm_date_debut").value == "01/07/2023"
    assert "rx_date_fin" in res and res.get("rx_date_fin").value == "15/08/2023"

def test_eds_extractor_booleans():
    extractor = EDSExtractor()
    text = "Le patient n'a pas de traitement par corticoïdes. Traitement par Keppra initié. Porteur du dispositif optune."
    fields = ["corticoides", "anti_epileptiques", "optune"]
    res = extractor.extract(text, {}, fields)
    
    assert "corticoides" in res and res.get("corticoides").value == "non"
    assert "anti_epileptiques" in res and res.get("anti_epileptiques").value == "oui"
    assert "optune" in res and res.get("optune").value == "oui"

def test_eds_extractor_numericals():
    extractor = EDSExtractor()
    text = "Indice de Karnofsky: 80%. Dose radiothérapie 60 Gy. 6 cycles de TMZ."
    fields = ["ik_clinique", "rx_dose", "chm_cycles"]
    res = extractor.extract(text, {}, fields)
    
    assert "ik_clinique" in res and res.get("ik_clinique").value == 80
    assert "rx_dose" in res and res.get("rx_dose").value == "60"
    assert "chm_cycles" in res and res.get("chm_cycles").value == 6

def test_eds_extractor_categorical():
    extractor = EDSExtractor()
    text = "Tumeur de grade IV, classification OMS 2021. Lésion siégeant dans le lobe frontal droit. Le patient a bénéficié d'une exérèse partielle."
    fields = ["grade", "classification_oms", "tumeur_lateralite", "type_chirurgie"]
    res = extractor.extract(text, {}, fields)
    
    assert "grade" in res and res.get("grade").value == 4
    assert "classification_oms" in res and res.get("classification_oms").value == "2021"
    assert "tumeur_lateralite" in res and res.get("tumeur_lateralite").value == "droite"
    assert "type_chirurgie" in res and res.get("type_chirurgie").value == "exerese partielle"

def test_eds_extractor_ihc():
    extractor = EDSExtractor()
    text = "IDH1 : positif. Le pourcentage de Ki-67 est estimé à 15-20%. ATRX : perte d'expression. p53 : +."
    fields = ["ihc_idh1", "ihc_ki67", "ihc_atrx", "ihc_p53"]
    res = extractor.extract(text, {}, fields)
    
    assert "ihc_idh1" in res and res.get("ihc_idh1").value == "positif"
    assert "ihc_ki67" in res and res.get("ihc_ki67").value == "15-20"
    assert "ihc_atrx" in res and res.get("ihc_atrx").value == "perte"
    assert "ihc_p53" in res and res.get("ihc_p53").value == "positif"

def test_eds_extractor_molecular_chromosomal():
    extractor = EDSExtractor()
    text = "Absence de mutation du gène IDH1. Délétion homozygote de CDKN2A. Codélétion 1p/19q présente."
    fields = ["mol_idh1", "mol_CDKN2A", "ch9p", "ch1p", "ch19q"]
    res = extractor.extract(text, {}, fields)
    
    assert "mol_idh1" in res and res.get("mol_idh1").value == "wt"
    assert "mol_CDKN2A" in res and res.get("mol_CDKN2A").value == "mute"
    assert "ch9p" in res and res.get("ch9p").value == "perte"  # Triggered by CDKN2A homodel
    assert "ch1p" in res and res.get("ch1p").value == "perte"
    assert "ch19q" in res and res.get("ch19q").value == "perte"
    
def test_eds_extractor_negation_and_family():
    extractor = EDSExtractor()
    text = "Pas de signe d'épilepsie inaugurale. Le père est décédé d'un glioblastome."
    fields = ["epilepsie_1er_symptome", "diag_histologique"]
    res = extractor.extract(text, {}, fields)
    
    assert "epilepsie_1er_symptome" in res and res.get("epilepsie_1er_symptome").value == "non"
    # "glioblastome" refers to family, should be skipped
    assert "diag_histologique" not in res

if __name__ == "__main__":
    pytest.main(["-v", "src/tests/test_eds_extractor.py"])
