"""Shared utilities for BRIGHT NER training pipeline.

Data loading, semantic group definitions, format conversion (GLiNER / EDS),
metric computation, and result persistence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE = Path(__file__).resolve().parent
GENERATED_DATA_DIR = _BASE / "generated_data" / "data"
DEFAULT_DATASET = GENERATED_DATA_DIR / "generated_dataset.jsonl"
DEFAULT_DATASET_SMALL = GENERATED_DATA_DIR / "generated_dataset_small.jsonl"

# ---------------------------------------------------------------------------
# Field descriptions (French) — used as GLiNER entity_descriptions
# ---------------------------------------------------------------------------

# Import canonical descriptions from the generation config
import sys as _sys

_sys.path.insert(0, str(_BASE / "generated_data" / "config"))
from fields import FIELD_DESCRIPTIONS_FR as FIELD_DESCRIPTIONS  # noqa: E402

_sys.path.pop(0)

# ---------------------------------------------------------------------------
# 10 semantic groups
# ---------------------------------------------------------------------------

GROUPS: dict[str, list[str]] = {
    "diagnosis": [
        "diag_histologique", "diag_integre", "classification_oms",
        "grade", "num_labo",
    ],
    "ihc": [
        "ihc_idh1", "ihc_atrx", "ihc_p53", "ihc_fgfr3", "ihc_braf",
        "ihc_gfap", "ihc_olig2", "ihc_ki67", "ihc_hist_h3k27m",
        "ihc_hist_h3k27me3", "ihc_egfr_hirsch", "ihc_mmr",
    ],
    "histology": [
        "histo_necrose", "histo_pec", "histo_mitoses", "aspect_cellulaire",
    ],
    "molecular": [
        "mol_idh1", "mol_idh2", "mol_mgmt", "mol_h3f3a", "mol_hist1h3b",
        "mol_tert", "mol_CDKN2A", "mol_atrx", "mol_cic", "mol_fubp1",
        "mol_fgfr1", "mol_egfr_mut", "mol_prkca", "mol_pten", "mol_p53",
        "mol_braf",
    ],
    "chromosomal": [
        "ch1p", "ch19q", "ch1p19q_codel", "ch7p", "ch7q",
        "ch10p", "ch10q", "ch9p", "ch9q",
        "ampli_egfr", "ampli_cdk4", "ampli_mdm2", "ampli_mdm4", "ampli_met",
        "fusion_fgfr", "fusion_ntrk", "fusion_autre",
    ],
    "demographics": [
        "sexe", "annee_de_naissance", "activite_professionnelle",
        "antecedent_tumoral", "ik_clinique", "dominance_cerebrale",
        "neuroncologue", "neurochirurgien", "radiotherapeute",
        "anatomo_pathologiste",
    ],
    "tumor_location": [
        "tumeur_lateralite", "tumeur_position", "localisation_chir",
    ],
    "treatment": [
        "chimios", "chimio_protocole", "chm_cycles",
        "chm_date_debut", "chm_date_fin",
        "type_chirurgie", "qualite_exerese", "chir_date",
        "rx_dose", "rx_fractionnement", "rx_date_debut", "rx_date_fin",
        "localisation_radiotherapie",
        "corticoides", "anti_epileptiques", "optune",
        "essai_therapeutique",
    ],
    "symptoms_evolution": [
        "epilepsie_1er_symptome", "ceph_hic_1er_symptome", "ceph_hic",
        "deficit_1er_symptome", "deficit",
        "cognitif_1er_symptome", "cognitif",
        "autre_trouble_1er_symptome",
        "contraste_1er_symptome", "prise_de_contraste",
        "oedeme_1er_symptome", "calcif_1er_symptome",
        "epilepsie", "autre_trouble",
        "evol_clinique", "progress_clinique",
        "progress_radiologique", "reponse_radiologique",
    ],
    "dates_outcomes": [
        "date_chir", "date_rcp", "dn_date", "date_deces",
        "date_1er_symptome", "exam_radio_date_decouverte",
        "date_progression", "survie_globale", "infos_deces",
    ],
}

GROUP_NAMES: list[str] = list(GROUPS.keys())

ALL_LABELS: list[str] = sorted({l for labels in GROUPS.values() for l in labels})

LABEL_TO_GROUP: dict[str, str] = {
    label: group for group, labels in GROUPS.items() for label in labels
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(path: str | Path | None = None) -> list[dict]:
    """Load a JSONL dataset of annotated documents.

    Each line: {"note_id": str, "note_text": str,
                "entities": [{"start": int, "end": int, "label": str, "value": str}]}
    """
    path = Path(path) if path else DEFAULT_DATASET
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def filter_by_group(docs: list[dict], group_name: str) -> list[dict]:
    """Keep only entities whose label belongs to *group_name*.

    Documents with zero matching entities are dropped.
    """
    labels = set(GROUPS[group_name])
    filtered = []
    for doc in docs:
        ents = [e for e in doc["entities"] if e["label"] in labels]
        if ents:
            filtered.append({**doc, "entities": ents})
    return filtered


def create_balanced_dataset(docs: list[dict], group_name: str, target_count: int = 150) -> list[dict]:
    """Create a class-balanced dataset using Classifier Re-Training (cRT) sampling strategy.
    
    Under-samples frequent classes and over-samples rare classes to approach `target_count`.
    """
    import random
    import copy
    
    labels = set(GROUPS[group_name])
    
    # Map entities to documents
    doc_by_label = {l: [] for l in labels}
    for doc in docs:
        doc_labels = set(e["label"] for e in doc["entities"] if e["label"] in labels)
        for l in doc_labels:
            doc_by_label[l].append(doc)
            
    balanced_docs = []
    seen_ids = set()
    
    # Process rare classes first
    sorted_labels = sorted(labels, key=lambda l: len(doc_by_label[l]))
    
    for l in sorted_labels:
        docs_for_l = doc_by_label[l]
        if not docs_for_l:
            continue
            
        current_count = sum(1 for d in balanced_docs for e in d["entities"] if e["label"] == l)
        needed = max(0, target_count - current_count)
        
        if needed == 0:
            continue
            
        if needed > len(docs_for_l):
            sampled = random.choices(docs_for_l, k=needed)
        else:
            sampled = random.sample(docs_for_l, k=needed)
            
        for i, doc in enumerate(sampled):
            # Only generate a deepcopy with a new ID if we are explicitly over-sampling and we already have it
            if doc["note_id"] in seen_ids and needed > len(docs_for_l):
                new_doc = copy.deepcopy(doc)
                new_doc["note_id"] = f"{doc['note_id']}_cRT_{l}_{i}"
                balanced_docs.append(new_doc)
            else:
                if doc["note_id"] not in seen_ids:
                    balanced_docs.append(doc)
                    seen_ids.add(doc["note_id"])
                    
    return balanced_docs


# ---------------------------------------------------------------------------
# Patient-level split
# ---------------------------------------------------------------------------

_PATIENT_RE = re.compile(r"synth-(P\d+)") # synthetic patient IDs


def _patient_id(note_id: str) -> str:
    m = _PATIENT_RE.search(note_id)
    return m.group(1) if m else note_id


def split_dataset(
    docs: list[dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Patient-level train / val split."""
    import random

    patients = sorted({_patient_id(d["note_id"]) for d in docs})
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_train = max(1, int(len(patients) * train_ratio))
    train_patients = set(patients[:n_train])

    train, val = [], []
    for doc in docs:
        (train if _patient_id(doc["note_id"]) in train_patients else val).append(doc)
    return train, val


# ---------------------------------------------------------------------------
# Chunking (for GLiNER — long documents)
# ---------------------------------------------------------------------------

# Maximum span length in characters before we fall back to value
_MAX_SPAN_CHARS = 300


def _estimate_tokens(text: str) -> int:
    """Rough token count (whitespace-based, good enough for chunking)."""
    return len(text.split())


def chunk_document(
    doc: dict,
    max_tokens: int = 1800,
    overlap_tokens: int = 200,
) -> list[dict]:
    """Split a long document into overlapping chunks with their entities.

    Each chunk is a new doc dict with adjusted entity offsets.
    Entities in the overlap zone are assigned to the *first* chunk only.
    Entities whose span exceeds _MAX_SPAN_CHARS are capped (use value instead).
    """
    text = doc["note_text"]
    if _estimate_tokens(text) <= max_tokens:
        return [doc]

    # Build character-level chunk boundaries from token boundaries
    words = text.split()
    # Compute char offset of each word
    word_starts: list[int] = []
    pos = 0
    for w in words:
        idx = text.index(w, pos)
        word_starts.append(idx)
        pos = idx + len(w)

    stride = max_tokens - overlap_tokens
    chunks = []
    i = 0
    while i < len(words):
        end_word = min(i + max_tokens, len(words))
        char_start = word_starts[i]
        char_end = (word_starts[end_word] if end_word < len(words)
                    else len(text))

        # Overlap boundary: entities starting before this are "overlap" for
        # the previous chunk and should not be duplicated.
        overlap_char_start = (word_starts[min(i + stride, len(words) - 1)]
                              if i > 0 else char_start)

        chunk_text = text[char_start:char_end]

        # Assign entities to this chunk
        chunk_ents = []
        for e in doc["entities"]:
            es, ee = e["start"], e["end"]
            # Entity must be fully within this chunk
            if es >= char_start and ee <= char_end:
                # Skip if entity starts in the overlap zone and this isn't
                # the first chunk that covers it
                if i > 0 and es < overlap_char_start:
                    continue
                chunk_ents.append({
                    **e,
                    "start": es - char_start,
                    "end": ee - char_start,
                })

        chunks.append({
            "note_id": f"{doc['note_id']}_chunk{len(chunks)}",
            "note_text": chunk_text,
            "entities": chunk_ents,
        })

        if end_word >= len(words):
            break
        i += stride

    return chunks


# ---------------------------------------------------------------------------
# GLiNER format conversion
# ---------------------------------------------------------------------------


def to_gliner_examples(
    docs: list[dict],
    group_name: str,
) -> list[dict]:
    """Convert annotated docs to GLiNER2 JSONL dicts.

    Output format per example:
        {"input": text, "output": {"entities": {label: [span, ...]},
                                   "entity_descriptions": {label: desc}}}

    Long documents are chunked first. Spans longer than _MAX_SPAN_CHARS
    fall back to the entity's ``value`` field.
    """
    labels = set(GROUPS[group_name])
    descriptions = {l: FIELD_DESCRIPTIONS.get(l, l) for l in labels}

    examples = []
    for doc in docs:
        for chunk in chunk_document(doc):
            text = chunk["note_text"]
            ents_by_label: dict[str, list[str]] = {}

            for e in chunk["entities"]:
                if e["label"] not in labels:
                    continue
                span_text = text[e["start"]:e["end"]]
                # Fall back to value for absurdly long spans
                if len(span_text) > _MAX_SPAN_CHARS:
                    span_text = str(e["value"])
                ents_by_label.setdefault(e["label"], []).append(span_text)

            if not ents_by_label:
                continue

            examples.append({
                "input": text,
                "output": {
                    "entities": ents_by_label,
                    "entity_descriptions": {
                        l: descriptions[l] for l in ents_by_label
                    },
                },
            })
    return examples


# ---------------------------------------------------------------------------
# EDS-NLP format conversion
# ---------------------------------------------------------------------------


def to_eds_spans(
    docs: list[dict],
    group_name: str,
) -> list[dict]:
    """Convert annotated docs to a simple span-based format for EDS-NLP.

    Returns list of {"note_id", "note_text", "spans": [{"start", "end", "label"}]}.
    No chunking needed — EDS-NLP handles long docs via sliding window.
    """
    labels = set(GROUPS[group_name])
    result = []
    for doc in docs:
        spans = []
        for e in doc["entities"]:
            if e["label"] not in labels:
                continue
            # Skip unreasonably long spans
            if (e["end"] - e["start"]) > _MAX_SPAN_CHARS:
                continue
            spans.append({"start": e["start"], "end": e["end"], "label": e["label"]})
        if spans:
            result.append({
                "note_id": doc["note_id"],
                "note_text": doc["note_text"],
                "spans": spans,
            })
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    predictions: list[dict],
    ground_truth: list[dict],
    labels: list[str],
) -> dict:
    """Compute per-label and micro/macro P/R/F1.

    Both ``predictions`` and ``ground_truth`` are lists of
    {"note_id": str, "entities": {label: [span, ...]}} dicts.
    We evaluate at entity-set level: for each (doc, label) pair,
    y_true=1 if any entity of that label exists, y_pred=1 likewise.
    """
    # Build lookup: note_id -> set of labels present
    gt_map: dict[str, set[str]] = {}
    for d in ground_truth:
        gt_map[d["note_id"]] = set(d.get("entities", {}).keys())

    pred_map: dict[str, set[str]] = {}
    for d in predictions:
        pred_map[d["note_id"]] = set(d.get("entities", {}).keys())

    all_ids = sorted(gt_map.keys() | pred_map.keys())

    y_true_all, y_pred_all = [], []
    per_label: dict[str, dict] = {}

    for label in labels:
        yt = [1 if label in gt_map.get(nid, set()) else 0 for nid in all_ids]
        yp = [1 if label in pred_map.get(nid, set()) else 0 for nid in all_ids]
        p, r, f1, sup = precision_recall_fscore_support(
            yt, yp, average="binary", zero_division=0,
        )
        per_label[label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": int(sum(yt)),
        }
        y_true_all.extend(yt)
        y_pred_all.extend(yp)

    # Micro average (binary on concatenated per-label arrays)
    mic_p, mic_r, mic_f1, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="binary", zero_division=0,
    )

    # Macro average: mean of per-label scores
    n_labels = len(per_label)
    if n_labels > 0:
        mac_p = sum(m["precision"] for m in per_label.values()) / n_labels
        mac_r = sum(m["recall"] for m in per_label.values()) / n_labels
        mac_f1 = sum(m["f1"] for m in per_label.values()) / n_labels
    else:
        mac_p = mac_r = mac_f1 = 0.0

    return {
        "per_label": per_label,
        "micro": {"precision": round(mic_p, 4), "recall": round(mic_r, 4), "f1": round(mic_f1, 4)},
        "macro": {"precision": round(mac_p, 4), "recall": round(mac_r, 4), "f1": round(mac_f1, 4)},
    }


# ---------------------------------------------------------------------------
# Result dataclass & persistence
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    group: str
    method: str  # "gliner" or "eds"
    per_label: dict[str, dict] = field(default_factory=dict)
    micro: dict = field(default_factory=dict)
    macro: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)  # training time, epochs, etc.


def results_to_dataframe(results: list[ModelResult]) -> pd.DataFrame:
    """Flatten a list of ModelResult into a tidy DataFrame.

    Columns: group, method, label, precision, recall, f1, support
    Plus aggregate rows with label="__micro__" / "__macro__".
    """
    rows = []
    for r in results:
        for label, m in r.per_label.items():
            rows.append({
                "group": r.group, "method": r.method, "label": label, **m,
            })
        for agg_name, m in [("micro", r.micro), ("macro", r.macro)]:
            rows.append({
                "group": r.group, "method": r.method, "label": agg_name, **m,
            })
    return pd.DataFrame(rows)


def save_results(results: list[ModelResult], path: str | Path) -> None:
    """Save results to CSV."""
    df = results_to_dataframe(results)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_results(path: str | Path) -> pd.DataFrame:
    """Load a previously saved results CSV."""
    return pd.read_csv(path)
