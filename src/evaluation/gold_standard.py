import json
from pathlib import Path
from typing import Any

def load_gold_standard(directory: str | Path) -> list[dict[str, Any]]:
    """
    Load all gold standard JSON files from a directory.
    
    Returns
    -------
    list[dict[str, Any]]
        A list of parsed JSON documents containing annotations.
    """
    directory = Path(directory)
    results = []
    if not directory.exists() or not directory.is_dir():
        return results
        
    for file_path in directory.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure annotations dict exists
            if "annotations" not in data:
                data["annotations"] = {}
            results.append(data)
    return results

def save_gold_standard(document: dict[str, Any], file_path: str | Path) -> None:
    """
    Helper to save a document to the gold standard format.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=4, ensure_ascii=False)
