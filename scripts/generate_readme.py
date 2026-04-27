# DEPRECATED: This script uses hardcoded absolute paths tied to the original
# developer's machine (c:\Users\rapha\...) and cannot be run on other systems.
# The README.md is now maintained by hand. Do not run this script.
import sys
import os

sys.path.append(r"c:\Users\rapha\OneDrive\Bureau\MVA\BRIGHT\BRIGHT_ClinicalDataBase")

from src.extraction.schema import BIO_FIELDS, CLINIQUE_FIELDS, FieldType

def format_allowed(allowed, ftype, name=""):
    if allowed:
        return ", ".join(f"'{x}'" if isinstance(x, str) else str(x) for x in sorted(list(allowed), key=str))
    if ftype == FieldType.DATE:
        return "DD/MM/YYYY or similar"
    if name == "annee_de_naissance":
        return "Year (YYYY)"
    if ftype == FieldType.INTEGER:
        return "Integer (e.g., 1, 60)"
    if ftype == FieldType.FLOAT:
        return "Float (e.g., 1.5)"
    if ftype == FieldType.CATEGORICAL:
         return "Categorical string"
    return "Free text / String"

def generate_table(fields):
    lines = ["| Field Name | Description | Group | Expected Values / Examples |", "|---|---|---|---|"]
    for f in fields:
        allowed = format_allowed(f.allowed_values, f.field_type, f.name)
        desc = f.display_name or f.description or f.name
        lines.append(f"| `{f.name}` | {desc} | {f.group} | {allowed} |")
    return "\n".join(lines)

bio_table = generate_table(BIO_FIELDS)
clinique_table = generate_table(CLINIQUE_FIELDS)

readme_content = f"""# **BRIGHT Clinical Features DataBase**
To build a clinical database from medical reports for the BRIGHT team at Institut de Neurologie at Hopital Pitié-Salpêtrière.

## **Architecture: GLiNER-First Extraction**

The extraction pipeline leverages a **GLiNER-first** approach, operating purely on discriminative encoder models and rule-based NLP to ensure fast, deterministic, and resource-efficient processing on standard hardware.

### **Key Extraction Strategies**

#### **1. Semantic Batching & Prior-Context Injection**
Extracting 111 fields simultaneously exceeds the capacity of standard GLiNER models without severe performance degradation. We split extraction into **21 semantic batches** (e.g., *Demographics*, *IHC 1*, *Treatment Chemo*). 
To maintain context across related fields without overcrowding the prompt, we use an **Anchor Matrix**: up to 4 previously extracted high-confidence fields are injected as prior context `[Context: Field: Value, ...]` at the beginning of the text chunk for the current batch. This balances semantic cohesion with optimal processing speed (batched inference is significantly faster while preserving accuracy).

#### **2. Smart Chunking**
GLiNER has a strict 512-token context limit. To handle long clinical documents:
- Documents are processed using a **sliding window** chunking strategy.
- Chunks are sized at **150-200 words** with a **30-50 word overlap**.
- This overlapping ensures that entity boundaries aren't arbitrarily cut and context is preserved across chunk borders.

#### **3. Bilingual Description Handling (EN/FR)**
The system dynamically routes language based on document content (`langdetect`):
- Uses specific French descriptions (`labels_fr`) for French documents and English descriptions (`labels_en`) otherwise.
- Allows flexibility to use a multilingual GLiNER model or gracefully fallback to a fine-tuned French clinical model.

#### **4. Synergistic Merge with EDS-NLP**
**EDS-NLP** (a rule-based clinical NLP framework) is used in tandem with GLiNER:
- **Qualifier Check**: Acts as a robust secondary validator post-GLiNER to handle complex negation and hypothesis detection explicitly (e.g. `absence de mutation`).
- **Alternative Extractor**: Extracts deterministic fields (dates, simple regex patterns, specific drug names) via highly tuned rule-based pipelines.
- Results from GLiNER and EDS-NLP are subsequently merged: if both agree, confidence is boosted; if they conflict, rule-based logic or confidence scores dictate the final output.

## **Demo & Usage**

### Launch the Desktop Interface
```bash
python -m src.ui.app_qt
```

### Run the CLI Demo
Run the full extraction pipeline from the command line:
```bash
python scripts/full_demo_test.py
```

### Run Tests
```bash
pytest src/tests/
```

## **Tracked Features (111 Fields)**

The database extracts 111 specific clinical and biological features.

### **Biological Features (55 Fields)**
{bio_table}

### **Clinical Features (56 Fields)**
{clinique_table}

## **Installation**
1) Ensure you have a working `conda` and `pip` installed.
2) Create a virtual environment and activate it:
```bash
conda create -n bright_db python=3.12
conda activate bright_db
```
3) Clone this repo:
```bash
git clone https://github.com/raphaelrubrice/BRIGHT_ClinicalDataBase.git
cd BRIGHT_ClinicalDataBase
```
4) Ensure that you have the appropriate channels ready first:
```bash
conda config --add channels conda-forge
```
5) Run the setup script to install dependencies (PyMuPDF, PyTorch, EDS-pseudo, GLiNER, EDS-NLP):
```bash
bash scripts/setup.sh
pip install -r requirements.txt
```
> **Note:** You may need access to specific Hugging Face models (like AP-HP/eds-pseudo-public) depending on your pseudonymization usage.
"""

with open(r"c:\Users\rapha\OneDrive\Bureau\MVA\BRIGHT\BRIGHT_ClinicalDataBase\README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("README generated successfully.")
