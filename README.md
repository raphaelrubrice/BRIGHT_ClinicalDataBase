# **BRIGHT Clinical Features DataBase**
To build a clinical database from medical reports for the BRIGHT team at Institut de Neurologie at Hopital Pitié-Salpêtrière.

## **Demo**

Try the extraction pipeline directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphaelrubrice/BRIGHT_ClinicalDataBase/blob/main/notebooks/run_demo.ipynb)

## **Installation**
1) Ensure you have a working `conda` and `pip` installed.
2) Create a virtual environment and activate it
```bash
conda create -n bright_db python=3.12
conda activate bright_db
```
3) Clone this repo
```bash
git clone https://github.com/raphaelrubrice/BRIGHT_ClinicalDataBase.git
cd BRIGHT_ClinicalDataBase
```

4) Install dependencies

Ensure that you have the appropriate channels ready first:
```bash
conda config --add channels conda-forge
```

5) Run the setup script

The setup script installs remaining dependencies (PyMuPDF, PyTorch, EDS-pseudo), prompts for your Hugging Face token, and verifies the installation:
```bash
bash scripts/setup.sh
pip install -r requirements.txt
```
> **Note:** You need access to the [AP-HP/eds-pseudo-public](https://hf.co/AP-HP/eds-pseudo-public) model and a [Hugging Face token](https://huggingface.co/settings/tokens?new_token=true) before running the script.

## **Usage**

### Launch the Desktop Interface
```bash
python -m src.ui.app_qt
```

### Run the CLI Demo
Run the full extraction pipeline from the command line with an Ollama model:
```bash
python scripts/full_demo_test.py --model qwen3:4b-instruct
```

### Run Tests
```bash
pytest src/tests/
```

## **Project Structure**
```
BRIGHT_ClinicalDataBase/
├── src/
│   ├── extraction/    # Two-tier extraction pipeline (rules + LLM)
│   ├── database/      # CSV database operations & pseudonymization
│   ├── aggregation/   # Patient timeline building
│   ├── evaluation/    # Benchmarking against gold standard
│   ├── ui/            # PySide6 Qt desktop application
│   └── tests/         # Test suite
├── scripts/
│   ├── setup.sh       # Environment setup automation
│   └── full_demo_test.py  # CLI demo runner
├── notebooks/
│   └── run_demo.ipynb # Google Colab demo notebook
├── data/
│   └── gold_standard/ # Gold standard annotations
└── requirements.txt
```
