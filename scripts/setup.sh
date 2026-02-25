pip install pymupdf
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
pip install --force-reinstall markupsafe jinja2 "numpy<2.0" pillow
git clone https://github.com/aphp/eds-pseudo.git
pip install "edsnlp[ml]" -U

read -p "Enter your Hugging Face token: " HF_TOKEN
python -c "import huggingface_hub; huggingface_hub.login(token='${HF_TOKEN}', new_session=False, add_to_git_credential=True)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python "$SCRIPT_DIR/../test_eds.py"