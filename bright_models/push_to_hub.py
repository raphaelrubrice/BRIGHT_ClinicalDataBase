import os
import sys
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi

from utils import GROUPS, FIELD_DESCRIPTIONS, GROUP_NAMES

def generate_readme(group: str, method: str, df: pd.DataFrame, username: str) -> str:
    """Generates the README Model Card text based on metrics from results.csv"""
    method_name = "GLiNER2" if method == "gliner" else "EDS-NLP (CamemBERT + CRF)"
    repo_name = f"bright-{method}-{group}"
    
    # Filter the exact row group for this method/group
    group_df = df[(df["group"] == group) & (df["method"] == method)]
    if group_df.empty:
        return ""
        
    macro = group_df[group_df["label"] == "macro"].iloc[0]
    micro = group_df[group_df["label"] == "micro"].iloc[0]
    
    labels = GROUPS[group]
    
    # Format Fields Descriptions List
    fields_list = "\n".join([f"- **{l}**: {FIELD_DESCRIPTIONS.get(l, l)}" for l in labels])
    
    # Format Per-Label Metric Table
    perf_table_rows = []
    for l in labels:
        row = group_df[group_df["label"] == l]
        if not row.empty:
            perf_table_rows.append(f"| {l} | {row.iloc[0]['precision']:.4f} | {row.iloc[0]['recall']:.4f} | {row.iloc[0]['f1']:.4f} |")
    
    perf_table = "\n".join([
        "| Label | Precision | Recall | F1 |",
        "|---|---|---|---|",
        *perf_table_rows
    ])
    
    # Construct Inference Block dynamically depending on the model pipeline
    inference_code = f"""
```python
# Inference Code
"""
    if method == "gliner":
        inference_code += f"""from gliner2 import GLiNER2

model = GLiNER2.from_pretrained("{username}/{repo_name}")
text = "Patient presenting with epileptic seizures..."
entities = model.extract_entities(text)

for entity in entities:
    print(entity["text"], "=>", entity["label"])
```"""
    else:
        inference_code += f"""import edsnlp

nlp = edsnlp.load("{username}/{repo_name}")
doc = nlp("Patient presenting with epileptic seizures...")

for ent in doc.ents:
    print(ent.text, "=>", ent.label_)
```"""

    # Model Card Assembly
    readme = f"""
---
tags:
- {method}
- ner
- medical
- french
language:
- fr
---
# BRIGHT NER: {method_name} fine-tuned for {group}

## Description
This is a {method_name} architecture fine-tuned to extract clinical neuro-oncology entities related to the `{group}` semantic group. It was trained on a synthetic dataset generated for the properly de-identified BRIGHT project dataset (see the `generated_data` folder in the primary repository).

This model repository was specifically designed to fit within the `bright_db` overarching namespace.

## Fields
It extracts the following fields (described in French):
{fields_list}

## Performance on Validation Set
**Aggregates**:
- Macro F1: {macro['f1']:.4f} (Precision: {macro['precision']:.4f}, Recall: {macro['recall']:.4f})
- Micro F1: {micro['f1']:.4f} (Precision: {micro['precision']:.4f}, Recall: {micro['recall']:.4f})

**Per-Label Breakdowns**:
{perf_table}

## Usage
{inference_code}
"""
    return readme.strip()


def push_models(username: str, output_dir_str: str="./output"):
    """Validates the output directory and pushes all found validated models to HF"""
    api = HfApi()
    output_dir = Path(output_dir_str)
    results_csv = output_dir / "results.csv"
    
    if not results_csv.exists():
        print(f"Failed to find {results_csv}. Please run training script first!")
        return
        
    df = pd.read_csv(results_csv)
    
    for group in GROUP_NAMES:
        for method in ["gliner", "eds"]:
            
            # Resolve physical model folder path, prioritizing the Classifier Re-Trained architecture
            if method == "gliner":
                model_folder = output_dir / group / "gliner" / "best_merged_crt"
                if not model_folder.exists():
                    model_folder = output_dir / group / "gliner" / "best_merged"
            else:
                model_folder = output_dir / group / "eds" / "model-balanced-best"
                if not model_folder.exists():
                    model_folder = output_dir / group / "eds" / "model-best"
                    
            if not model_folder.exists():
                print(f"Skipping {group} / {method}: local model artifacts not found.")
                continue
                
            # Compile README and write it to disk natively inside the local directory
            readme_text = generate_readme(group, method, df, username)
            
            readme_path = model_folder / "README.md"
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_text)
                
            # Create huggingface repository mapped identically to our architecture logic above
            repo_id = f"{username}/bright-{method}-{group}"
            print(f"Pushing {repo_id} to Hugging Face Hub...")
            
            try:
                # Set up the repo and push the entire folder (this natively includes our generated config files alongside the README)
                api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
                api.upload_folder(
                    folder_path=str(model_folder),
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"-- Pushed {repo_id} successfully.")
            except Exception as e:
                print(f"-- Failed to push {repo_id}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python push_to_hub.py <your_huggingface_username> <output_dir>")
        sys.exit(1)
        
    hf_username = sys.argv[1]
    output_dir = sys.argv[2]
    push_models(hf_username, output_dir)
