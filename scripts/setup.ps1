$ErrorActionPreference = "Stop"

pip install pymupdf
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
pip install --force-reinstall markupsafe jinja2 "numpy<2.0" pillow

python -m spacy download en_core_web_lg
python -m spacy download fr_core_news_lg

if (-Not (Test-Path "eds-pseudo")) {
    git clone https://github.com/aphp/eds-pseudo.git
}
pip install "edsnlp[ml]" -U

$HF_TOKEN = Read-Host -Prompt "Enter your Hugging Face token (input will be hidden)" -AsSecureString
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($HF_TOKEN)
$TokenString = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

python -c "import huggingface_hub; huggingface_hub.login(token='$TokenString', add_to_git_credential=True)"

# Clear token from memory
[System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($BSTR)
$TokenString = $null
$HF_TOKEN = $null

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$SCRIPT_DIR\..\test_eds.py"
