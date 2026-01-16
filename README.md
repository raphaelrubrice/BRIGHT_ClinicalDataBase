# **BRIGHT Clinical Features DataBase**
To build a clinical database from medical reports for the BRIGHT team at Institut de Neurologie at Hopital Pitié-Salpêtrière.

# **Installation**
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

4) Install depedencies
Ensure that you have the appropriate channels ready first:
```bash
conda config --add channels conda-forge
```
Then:
```bash
conda install --file requirements.txt
```

5) Install PyMuPDF
```bash
pip install pymupdf
```

6) Install pytorch
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
```

7) To help stabilize the env
```bash
pip install --force-reinstall markupsafe jinja2 "numpy<2.0" pillow
```

8) Install [EDS-pseudo](https://github.com/aphp/eds-pseudo/tree/main)
```bash
git clone https://github.com/aphp/eds-pseudo.git
pip install "edsnlp[ml]" -U
```
- Get access to the model at [AP-HP/eds-pseudo-public](https://hf.co/AP-HP/eds-pseudo-public)
- Create and copy a huggingface token https://huggingface.co/settings/tokens?new_token=true
- Register the token (only once) on your machine by executing in the python console:  
```python
import huggingface_hub

huggingface_hub.login(token=YOUR_TOKEN, new_session=False, add_to_git_credential=True)
```

- Test setup by running:
```bash
python test_eds.py
```

# **Launch the Database interface**
```bash
python .\src\ui\app_qt.py
```
Or
```bash
python -m src.ui.app_qt
```