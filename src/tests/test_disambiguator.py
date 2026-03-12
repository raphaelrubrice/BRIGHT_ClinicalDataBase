from __future__ import annotations

import pytest

from src.extraction.disambiguator import Disambiguator


@pytest.fixture
def disambiguator():
    return Disambiguator()


def test_disambiguator_mapping(disambiguator):
    text = "Le patient a été inclus dans l'essai clinique onconeurotek 2 le 12 mai. Prise de temodal et de keppra (250mg). Suivi à la pitié."
    
    mod_text, offset_mapper = disambiguator.apply(text, language="fr")
    
    assert len(mod_text) > len(text)
    
    # Check that injected contexts are present in modified text
    assert "(essai thérapeutique)" in mod_text
    assert "(chimiothérapie)" in mod_text
    assert "(traitement antiépileptique)" in mod_text
    assert "(hôpital)" in mod_text
    
    # 1. Test mapping of original un-modified text
    # "Le patient a été inclus dans l'essai clinique "
    assert offset_mapper(0) == 0
    assert offset_mapper(10) == 10
    
    # Find "onconeurotek 2" in both
    term1 = "onconeurotek 2"
    start_orig_1 = text.find(term1)
    
    start_mod_1 = mod_text.find(term1)
    end_mod_1 = start_mod_1 + len(term1)
    
    # The term itself should map perfectly back
    assert offset_mapper(start_mod_1) == start_orig_1
    assert offset_mapper(end_mod_1) == start_orig_1 + len(term1)
    
    # 2. Test mapping of an INJECTED context
    # (essai thérapeutique) was injected right after term1
    ctx1 = " (essai thérapeutique)"
    ctx_start_mod = end_mod_1
    ctx_end_mod = ctx_start_mod + len(ctx1)
    
    # Every index inside the injected context should map back to the END of the term in original text
    orig_anchor_1 = start_orig_1 + len(term1)
    assert offset_mapper(ctx_start_mod) == orig_anchor_1
    assert offset_mapper(ctx_end_mod) == orig_anchor_1
    assert offset_mapper(ctx_start_mod + 5) == orig_anchor_1
    
    # 3. Test mapping of text AFTER an injection
    # " le 12 mai. Prise de "
    text_between = " le 12 mai. Prise de "
    start_orig_between = text.find(text_between)
    start_mod_between = mod_text.find(text_between)
    
    assert offset_mapper(start_mod_between) == start_orig_between
    assert offset_mapper(start_mod_between + 5) == start_orig_between + 5
    
    # Find "temodal"
    term2 = "temodal"
    start_mod_2 = mod_text.find(term2)
    end_mod_2 = start_mod_2 + len(term2)
    start_orig_2 = text.find(term2)
    
    assert offset_mapper(start_mod_2) == start_orig_2
    assert offset_mapper(end_mod_2) == start_orig_2 + len(term2)
    
    # 4. Test mapping out of bounds
    assert offset_mapper(-5) == 0
    assert offset_mapper(len(mod_text) + 10) == len(text)


def test_no_matches(disambiguator):
    text = "Pas de traitement particulier."
    mod_text, offset_mapper = disambiguator.apply(text, language="fr")
    
    assert mod_text == text
    assert offset_mapper(10) == 10
    assert offset_mapper(100) == 100


def test_english_repertoire(disambiguator):
    text = "Patient enrolled in eortc 26981 protocol. Started on tmz."
    
    mod_text, offset_mapper = disambiguator.apply(text, language="en")
    
    assert "(clinical trial)" in mod_text
    assert "(chemotherapy agent)" in mod_text


def test_multiple_injections_offsets(disambiguator):
    # This text has multiple consecutive hits that will be injected
    text = "Traitement par temodal et keppra instauré à la pitié."
    # temodal -> (chimiothérapie)
    # keppra -> (traitement antiépileptique)
    # pitié -> (hôpital)
    
    mod_text, offset_mapper = disambiguator.apply(text, language="fr")
    
    # Let's find "et" which is between temodal and keppra injections
    orig_et_idx = text.find(" et ")
    mod_et_idx = mod_text.find(" et ")
    
    # "et" should map perfectly back
    assert offset_mapper(mod_et_idx) == orig_et_idx
    assert offset_mapper(mod_et_idx + 1) == orig_et_idx + 1
    
    # "instauré" which is after keppra
    orig_inst_idx = text.find("instauré")
    mod_inst_idx = mod_text.find("instauré")
    
    assert offset_mapper(mod_inst_idx) == orig_inst_idx
    assert offset_mapper(mod_inst_idx + 5) == orig_inst_idx + 5
    
    # Validate the injected context maps to the anchor
    term_keppra = "keppra"
    orig_keppra_end = text.find(term_keppra) + len(term_keppra)
    
    # The injection for keppra is ` (traitement antiépileptique)`
    injection = " (traitement antiépileptique)"
    mod_inj_start = mod_text.find(injection)
    mod_inj_end = mod_inj_start + len(injection)
    
    assert offset_mapper(mod_inj_start) == orig_keppra_end
    assert offset_mapper(mod_inj_end) == orig_keppra_end
    assert offset_mapper(mod_inj_start + 4) == orig_keppra_end
