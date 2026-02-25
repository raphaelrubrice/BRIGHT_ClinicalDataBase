"""Gold standard annotation loading and comparison.

Loads manually annotated ground truth JSON files and compares
them against pipeline extraction outputs.

Gold standard format (one JSON file per patient-visit):

    {
        "document_id": "8003373720_initial",
        "patient_id": "8003373720",
        "date_chir": "15/10/2024",
        "evol_clinique": "initial",
        "has_bio_annotations": true,
        "has_clinique_annotations": true,
        "annotations": {
            "ihc_idh1": {"value": "negatif"},
            "grade": {"value": 4},
            "mol_tert": {"value": "mute"},
            ...
        }
    }

Annotations can optionally include a "source_span" key alongside "value"
to record the exact text passage that justifies the annotation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Default gold standard directory (relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GOLD_DIR = _PROJECT_ROOT / "data" / "gold_standard"


@dataclass
class GoldAnnotation:
    """A single ground-truth annotation for one field."""

    field_name: str
    value: Any
    source_span: str | None = None

    def matches(self, predicted_value: Any) -> bool:
        """Check if a predicted value matches this annotation.

        Comparison is case-insensitive for strings.
        """
        if self.value is None and predicted_value is None:
            return True
        if self.value is None or predicted_value is None:
            return False
        if isinstance(self.value, str) and isinstance(predicted_value, str):
            return self.value.strip().lower() == predicted_value.strip().lower()
        return self.value == predicted_value


@dataclass
class GoldDocument:
    """Ground-truth annotations for a single patient-visit."""

    document_id: str
    patient_id: str
    date_chir: str | None
    evol_clinique: str | None
    has_bio_annotations: bool
    has_clinique_annotations: bool
    annotations: dict[str, GoldAnnotation] = field(default_factory=dict)

    @property
    def n_annotations(self) -> int:
        return len(self.annotations)

    @property
    def annotated_fields(self) -> list[str]:
        return sorted(self.annotations.keys())

    def get(self, field_name: str) -> GoldAnnotation | None:
        return self.annotations.get(field_name)


def load_gold_document(filepath: Path) -> GoldDocument:
    """Load a single gold standard JSON file into a GoldDocument."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    annotations = {}
    for field_name, ann in data.get("annotations", {}).items():
        annotations[field_name] = GoldAnnotation(
            field_name=field_name,
            value=ann.get("value"),
            source_span=ann.get("source_span"),
        )

    return GoldDocument(
        document_id=data["document_id"],
        patient_id=data["patient_id"],
        date_chir=data.get("date_chir"),
        evol_clinique=data.get("evol_clinique"),
        has_bio_annotations=data.get("has_bio_annotations", False),
        has_clinique_annotations=data.get("has_clinique_annotations", False),
        annotations=annotations,
    )


def load_gold_standard(
    gold_dir: Path | str | None = None,
) -> list[GoldDocument]:
    """Load all gold standard documents from a directory.

    Args:
        gold_dir: Path to the gold standard directory. If None, uses
            the default ``data/gold_standard/`` directory.

    Returns:
        List of GoldDocument objects, sorted by document_id.
    """
    gold_dir = Path(gold_dir) if gold_dir else DEFAULT_GOLD_DIR
    if not gold_dir.exists():
        raise FileNotFoundError(f"Gold standard directory not found: {gold_dir}")

    documents = []
    for filepath in sorted(gold_dir.glob("*.json")):
        if filepath.name == "manifest.json":
            continue
        documents.append(load_gold_document(filepath))

    return documents


def load_manifest(gold_dir: Path | str | None = None) -> dict:
    """Load the gold standard manifest file.

    The manifest provides a summary of all entries without loading
    the full annotation data.
    """
    gold_dir = Path(gold_dir) if gold_dir else DEFAULT_GOLD_DIR
    manifest_path = gold_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def compare_extraction(
    gold: GoldDocument,
    predicted: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compare pipeline extraction output against gold standard.

    Args:
        gold: The ground-truth GoldDocument.
        predicted: Dict mapping field_name -> predicted value (or
            dict with a "value" key).

    Returns:
        Dict mapping field_name to comparison result:
            {
                "gold_value": ...,
                "predicted_value": ...,
                "match": bool,
                "category": "TP" | "FP" | "FN" | "TN",
            }
    """
    results = {}
    all_fields = set(gold.annotations.keys()) | set(predicted.keys())

    for field_name in sorted(all_fields):
        gold_ann = gold.annotations.get(field_name)
        pred_raw = predicted.get(field_name)

        # Unwrap predicted value if it's a dict with a "value" key
        if isinstance(pred_raw, dict):
            pred_value = pred_raw.get("value")
        else:
            pred_value = pred_raw

        gold_value = gold_ann.value if gold_ann else None
        has_gold = gold_ann is not None and gold_value is not None
        has_pred = pred_value is not None

        if has_gold and has_pred:
            match = gold_ann.matches(pred_value)
            category = "TP" if match else "FP"  # FP = wrong value
        elif has_gold and not has_pred:
            match = False
            category = "FN"  # missed
        elif not has_gold and has_pred:
            match = False
            category = "FP"  # hallucinated
        else:
            match = True
            category = "TN"  # both absent

        results[field_name] = {
            "gold_value": gold_value,
            "predicted_value": pred_value,
            "match": match,
            "category": category,
        }

    return results
