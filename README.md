# **BRIGHT Clinical Features DataBase**
To build a clinical database from medical reports for the BRIGHT team at Institut de Neurologie at Hopital Pitié-Salpêtrière.

## **Installation**

The extraction pipeline uses `onnxruntime` and `gliner2-onnx` for maximum inference speed compared to standard PyTorch.

### 1. Prerequisites
- **conda** and **pip** installed.
- **Git** installed.
- Access to Hugging Face models (you will be prompted for an HF token).

### 2. Base Environment

Create and activate a virtual environment:
```bash
conda create -n bright_db python=3.12
conda activate bright_db
conda config --add channels conda-forge
```

Clone this repository:
```bash
git clone https://github.com/raphaelrubrice/BRIGHT_ClinicalDataBase.git
cd BRIGHT_ClinicalDataBase
```

### 3. Setup Scripts

Run the appropriate setup script for your operating system. These scripts will:
- Install base Python dependencies (PyTorch, PyMuPDF, `edsnlp`, etc.)
- Prompt for your Hugging Face token.
- Clone and install `eds-pseudo`.

**For UNIX (Linux / macOS):**
```bash
bash scripts/setup.sh
pip install -r requirements.txt
```

**For Windows (PowerShell):**
```powershell
# Ensure execution of scripts is allowed
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

.\scripts\setup.ps1
pip install -r requirements.txt
```

### 4. PyTorch Fallback
The `gliner2-onnx` backend is configured to automatically download and run the ONNX model optimized for inference. If downloading or loading the ONNX model fails for any reason, the pipeline is designed to gracefully fall back to standard `gliner` via PyTorch eager-execution automatically.

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
| Field Name | Description | Group | Expected Values / Examples |
|---|---|---|---|
| `date_chir` | Date chirurgie | identifiers | DD/MM/YYYY or similar |
| `num_labo` | Numéro laboratoire | identifiers | Free text / String |
| `diag_histologique` | Diagnostic histologique | diagnosis | Free text / String |
| `diag_integre` | Diagnostic intégré | diagnosis | Free text / String |
| `classification_oms` | Classification OMS | diagnosis | '2007', '2016', '2021' |
| `grade` | Grade OMS | diagnosis | 1, 2, 3, 4 |
| `ihc_idh1` | IHC IDH1 | ihc | 'maintenu', 'negatif', 'positif' |
| `ihc_p53` | IHC p53 | ihc | 'maintenu', 'negatif', 'positif' |
| `ihc_atrx` | IHC ATRX | ihc | 'maintenu', 'negatif', 'positif' |
| `ihc_fgfr3` | IHC FGFR3 | ihc | 'maintenu', 'negatif', 'positif' |
| `ihc_braf` | IHC BRAF | ihc | 'maintenu', 'negatif', 'positif' |
| `ihc_hist_h3k27m` | IHC H3K27M | ihc | 'maintenu', 'negatif', 'positif' |
| `ihc_hist_h3k27me3` | IHC H3K27me3 | ihc | 'maintenu', 'negatif', 'positif' |
| `ihc_egfr_hirsch` | IHC EGFR (Hirsch / status) | ihc | Free text / String |
| `ihc_gfap` | IHC GFAP | ihc | 'maintenu', 'negatif', 'positif' |
| `ihc_olig2` | IHC Olig2 | ihc | 'maintenu', 'negatif', 'positif' |
| `ihc_ki67` | IHC Ki67 (%) | ihc | Free text / String |
| `ihc_mmr` | IHC MMR | ihc | 'maintenu', 'negatif', 'positif' |
| `histo_necrose` | Nécrose | histology | 'non', 'oui' |
| `histo_pec` | Prise de contraste endothéliocapillaire | histology | 'non', 'oui' |
| `histo_mitoses` | Mitoses (count) | histology | Integer (e.g., 1, 60) |
| `aspect_cellulaire` | Aspect cellulaire | histology | Free text / String |
| `mol_idh1` | IDH1 moléculaire | molecular | Free text / String |
| `mol_idh2` | IDH2 moléculaire | molecular | Free text / String |
| `mol_tert` | TERT moléculaire | molecular | Free text / String |
| `mol_CDKN2A` | CDKN2A moléculaire | molecular | Free text / String |
| `mol_h3f3a` | H3F3A moléculaire | molecular | Free text / String |
| `mol_hist1h3b` | HIST1H3B moléculaire | molecular | Free text / String |
| `mol_braf` | BRAF moléculaire | molecular | Free text / String |
| `mol_mgmt` | MGMT méthylation | molecular | Free text / String |
| `mol_fgfr1` | FGFR1 moléculaire | molecular | Free text / String |
| `mol_egfr_mut` | EGFR mutation | molecular | Free text / String |
| `mol_prkca` | PRKCA moléculaire | molecular | Free text / String |
| `mol_p53` | TP53 moléculaire | molecular | Free text / String |
| `mol_pten` | PTEN moléculaire | molecular | Free text / String |
| `mol_cic` | CIC moléculaire | molecular | Free text / String |
| `mol_fubp1` | FUBP1 moléculaire | molecular | Free text / String |
| `mol_atrx` | ATRX moléculaire | molecular | Free text / String |
| `ch1p` | 1p | chromosomal | 'gain', 'perte', 'perte partielle' |
| `ch19q` | 19q | chromosomal | 'gain', 'perte', 'perte partielle' |
| `ch10p` | 10p | chromosomal | 'gain', 'perte', 'perte partielle' |
| `ch10q` | 10q | chromosomal | 'gain', 'perte', 'perte partielle' |
| `ch7p` | 7p | chromosomal | 'gain', 'perte', 'perte partielle' |
| `ch7q` | 7q | chromosomal | 'gain', 'perte', 'perte partielle' |
| `ch9p` | 9p | chromosomal | 'gain', 'perte', 'perte partielle' |
| `ch9q` | 9q | chromosomal | 'gain', 'perte', 'perte partielle' |
| `ch1p19q_codel` | Codélétion 1p/19q | chromosomal | 'non', 'oui' |
| `ampli_mdm2` | Amplification MDM2 | amplification | 'non', 'oui' |
| `ampli_cdk4` | Amplification CDK4 | amplification | 'non', 'oui' |
| `ampli_egfr` | Amplification EGFR | amplification | 'non', 'oui' |
| `ampli_met` | Amplification MET | amplification | 'non', 'oui' |
| `ampli_mdm4` | Amplification MDM4 | amplification | 'non', 'oui' |
| `fusion_fgfr` | Fusion FGFR | fusion | 'non', 'oui' |
| `fusion_ntrk` | Fusion NTRK | fusion | 'non', 'oui' |
| `fusion_autre` | Fusion autre | fusion | 'non', 'oui' |

### **Clinical Features (56 Fields)**
| Field Name | Description | Group | Expected Values / Examples |
|---|---|---|---|
| `date_rcp` | Date RCP | demographics | DD/MM/YYYY or similar |
| `annee_de_naissance` | Année de naissance | demographics | Year (YYYY) |
| `sexe` | Sexe | demographics | 'F', 'M' |
| `activite_professionnelle` | Activité professionnelle | demographics | Free text / String |
| `antecedent_tumoral` | Antécédent tumoral | demographics | 'Non', 'Oui', 'non', 'oui' |
| `neuroncologue` | Neuro-oncologue | care_team | Free text / String |
| `neurochirurgien` | Neurochirurgien | care_team | Free text / String |
| `radiotherapeute` | Radiothérapeute | care_team | Free text / String |
| `anatomo_pathologiste` | Anatomo-pathologiste | care_team | Free text / String |
| `localisation_radiotherapie` | Localisation radiothérapie | care_team | Free text / String |
| `localisation_chir` | Localisation chirurgie | care_team | Free text / String |
| `date_deces` | Date décès | outcome | DD/MM/YYYY or similar |
| `infos_deces` | Infos décès | outcome | Free text / String |
| `survie_globale` | Survie globale | outcome | Free text / String |
| `date_1er_symptome` | Date 1er symptôme | first_symptoms | DD/MM/YYYY or similar |
| `epilepsie_1er_symptome` | Épilepsie 1er symptôme | first_symptoms | 'non', 'oui' |
| `ceph_hic_1er_symptome` | Céphalées/HTIC 1er symptôme | first_symptoms | 'non', 'oui' |
| `deficit_1er_symptome` | Déficit 1er symptôme | first_symptoms | 'non', 'oui' |
| `cognitif_1er_symptome` | Cognitif 1er symptôme | first_symptoms | 'non', 'oui' |
| `autre_trouble_1er_symptome` | Autre trouble 1er symptôme | first_symptoms | 'non', 'oui' |
| `exam_radio_date_decouverte` | Date découverte radiologique | radiology | DD/MM/YYYY or similar |
| `contraste_1er_symptome` | Prise de contraste initiale | radiology | 'non', 'oui' |
| `prise_de_contraste` | Prise de contraste | radiology | 'non', 'oui' |
| `oedeme_1er_symptome` | Œdème initial | radiology | 'non', 'oui' |
| `calcif_1er_symptome` | Calcification initiale | radiology | 'non', 'oui' |
| `tumeur_lateralite` | Latéralité tumeur | tumour_location | 'bilateral', 'droite', 'gauche', 'median' |
| `tumeur_position` | Position tumeur | tumour_location | Free text / String |
| `dominance_cerebrale` | Dominance cérébrale | tumour_location | 'ambidextre', 'droitier', 'droitier contrarié', 'gaucher', 'gaucher contrarié' |
| `dn_date` | Date dernière nouvelle | evolution | DD/MM/YYYY or similar |
| `evol_clinique` | Évolution clinique | evolution | Free text / String |
| `reponse_radiologique` | Réponse radiologique | evolution | Free text / String |
| `chimios` | Chimiothérapie(s) | treatment_chemo | Free text / String |
| `chimio_protocole` | Protocole chimiothérapie | treatment_chemo | Free text / String |
| `chm_date_debut` | Date début chimio | treatment_chemo | DD/MM/YYYY or similar |
| `chm_date_fin` | Date fin chimio | treatment_chemo | DD/MM/YYYY or similar |
| `chm_cycles` | Nombre cycles chimio | treatment_chemo | Integer (e.g., 1, 60) |
| `ik_clinique` | Indice de Karnofsky | clinical_state | Integer (e.g., 1, 60) |
| `progress_clinique` | Progression clinique | clinical_state | 'non', 'oui' |
| `progress_radiologique` | Progression radiologique | clinical_state | 'non', 'oui' |
| `date_progression` | Date progression | clinical_state | DD/MM/YYYY or similar |
| `epilepsie` | Épilepsie actuelle | current_symptoms | 'non', 'oui' |
| `ceph_hic` | Céphalées/HTIC actuelle | current_symptoms | 'non', 'oui' |
| `deficit` | Déficit actuel | current_symptoms | 'non', 'oui' |
| `cognitif` | Trouble cognitif | current_symptoms | 'non', 'oui' |
| `autre_trouble` | Autre trouble | current_symptoms | Free text / String |
| `anti_epileptiques` | Anti-épileptiques | adjunct | 'non', 'oui' |
| `essai_therapeutique` | Essai thérapeutique | adjunct | 'non', 'oui' |
| `chir_date` | Date chirurgie | surgery | DD/MM/YYYY or similar |
| `type_chirurgie` | Type chirurgie | surgery | 'biopsie', 'en attente', 'exerese', 'exerese complete', 'exerese partielle' |
| `qualite_exerese` | Qualité de l'exérèse | surgery | Free text / String |
| `rx_date_debut` | Date début radiothérapie | treatment_radio | DD/MM/YYYY or similar |
| `rx_date_fin` | Date fin radiothérapie | treatment_radio | DD/MM/YYYY or similar |
| `rx_dose` | Dose radiothérapie (Gy) | treatment_radio | Free text / String |
| `rx_fractionnement` | Fractionnement radiothérapie | treatment_radio | Free text / String |
| `corticoides` | Corticoïdes | adjunct | 'non', 'oui' |
| `optune` | Optune (TTFields) | adjunct | 'non', 'oui' |
