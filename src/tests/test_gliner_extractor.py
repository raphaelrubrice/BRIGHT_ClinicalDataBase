import pytest
from src.extraction.gliner_extractor import GlinerExtractor

def test_gliner_extractor():
    extractor = GlinerExtractor(model_name="urchade/gliner_multi-v2.1")
    
    text = "Le patient a présenté une crise d'épilepsie inaugurale. Il travaillait comme boulanger avant la maladie. La tumeur est située dans le lobe temporal droit."
    fields = ["epilepsie_1er_symptome", "activite_professionnelle", "tumeur_position"]
    
    # We mock out the GLiNER predict_entities to avoid downloading the multi-GB model during testing
    # but still test the chunking and extraction logic wrapper.
    class MockModel:
        def predict_entities(self, text, labels, threshold=0.5):
            print(f"Mock analyzing text: {text}")
            results = []
            if "crise d'épilepsie" in text:
                results.append({
                    "text": "crise d'épilepsie", "label": "crise d'epilepsie ou convulsion inaugurale", "score": 0.99
                })
            if "boulanger" in text:
                results.append({
                    "text": "boulanger", "label": "profession ou activite professionnelle", "score": 0.95
                })
            if "lobe temporal droit" in text:
                results.append({
                    "text": "lobe temporal droit", "label": "localisation anatomique de la tumeur", "score": 0.98
                })
            return results

    # Inject mock model
    extractor._model = MockModel()
    
    res = extractor.extract(text, {}, fields)
    
    assert "epilepsie_1er_symptome" in res and res["epilepsie_1er_symptome"].value == "crise d'épilepsie"
    assert "activite_professionnelle" in res and res["activite_professionnelle"].value == "boulanger"
    assert "tumeur_position" in res and res["tumeur_position"].value == "lobe temporal droit"

if __name__ == "__main__":
    pytest.main(["-v", "src/tests/test_gliner_extractor.py"])
