import os
import sys
import edsnlp
import importlib
import pkgutil
from huggingface_hub import snapshot_download

# --- CONFIGURATION ---
cache_dir = os.path.join(os.getcwd(), "hf_cache")

# --- STEP 1: DOWNLOAD ---
# (We keep this to ensure files are present, but it will skip if already there)
print(f"Checking model files in: {cache_dir} ...")
snapshot_download(
    repo_id="AP-HP/eds-pseudo-public", 
    local_dir=cache_dir, 
    local_dir_use_symlinks=False,  # This parameter is now ignored by new HF versions, but harmless to keep
    ignore_patterns=["*.git*"] 
)

# --- STEP 2: FORCE LOAD CUSTOM MODULES ---
# Add the cache folder to the system path
abs_cache_dir = os.path.abspath(cache_dir)
if abs_cache_dir not in sys.path:
    sys.path.insert(0, abs_cache_dir)

def import_recursive(package_name):
    """
    Manually walks through a package and imports every submodule found.
    This ensures that all @registry decorators are executed.
    """
    try:
        # Import the top-level package first
        package = importlib.import_module(package_name)
        print(f"✅ Imported top-level package: {package_name}")
    except ImportError as e:
        print(f"❌ Could not import {package_name}: {e}")
        return

    # Walk through all sub-files and import them
    if hasattr(package, "__path__"):
        for _, name, _ in pkgutil.walk_packages(package.__path__, package_name + "."):
            try:
                importlib.import_module(name)
                # print(f"   -> Loaded submodule: {name}") # Uncomment to debug
            except Exception as e:
                print(f"   ⚠️ Warning: Failed to load {name}: {e}")

print("Registering custom components...")
import_recursive("eds_pseudo")

# --- STEP 3: LOAD PIPELINE ---
model_path = os.path.join(cache_dir, "artifacts")

print("\nLoading EDSNLP pipeline...")
# We use auto_update=False because we are managing the download manually
nlp = edsnlp.load(model_path, auto_update=False)

# --- STEP 4: TEST ---
text = (
    "En 2015, M. Charles-François-Bienvenu "
    "Myriel était évêque de Digne. C’était un vieillard "
    "d’environ soixante-quinze ans ; il occupait le "
    "siège de Digne depuis 2006."
)

doc = nlp(text)

print("\n" + "="*50)
print("RESULTS")
print("="*50)

found_entities = False
for ent in doc.ents:
    found_entities = True
    date_val = getattr(ent._, "date", "N/A")
    print(f"Entity: {ent.text: <30} | Label: {ent.label_: <10} | Date: {date_val}")

if not found_entities:
    print("No entities found (Check if the model logic is running correctly).")