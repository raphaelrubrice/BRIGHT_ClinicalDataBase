"""EDS-NLP (CamemBERT + CRF) fine-tuning for one semantic group.

Simplified from old_code/scripts/train.py.  Key simplifications:
  - No custom batch sampler or sub-batch collater
  - Fixed sample-based batch size
  - Uses edsnlp's built-in preprocessing and collate
  - Minimal dependencies on custom adapters

Usage:
    from eds_bright.training_eds import train_eds, predict_eds
    result = train_eds("diagnosis", train_docs, val_docs, output_dir="./output")
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import os
import spacy
import torch
from accelerate import Accelerator
from tqdm import tqdm

# Synchronous CUDA errors for better debugging
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import edsnlp
from edsnlp.pipes.trainable.embeddings.transformer.transformer import Transformer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    GROUPS,
    ModelResult,
    to_eds_spans,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

DEFAULT_BACKBONE = "almanach/camembert-bio-base"


def build_pipeline(
    labels: list[str],
    backbone: str = DEFAULT_BACKBONE,
    window: int = 510,    # < 512 to leave room for CLS/SEP within CamemBERT's 514 max positions
    stride: int = 382,
) -> edsnlp.Pipeline:
    """Build an edsnlp pipeline with transformer + NER-CRF."""
    nlp = edsnlp.blank("fr")
    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe(
        "eds.transformer",
        name="transformer",
        config={
            "model": backbone,
            "window": window,
            "stride": stride,
        },
    )
    nlp.add_pipe(
        "eds.text_cnn",
        name="text_cnn",
        config={
            "embedding": nlp.get_pipe("transformer"),
            "kernel_sizes": (3,12),
            "residual": True, 
            'normalize': 'pre',
        },
    )
    nlp.add_pipe(
        "eds.ner_crf",
        name="ner",
        config={
            "embedding": nlp.get_pipe("text_cnn"),
            "mode": "joint",
            "window": 0,  # windowed CRF inference had index bug; 0 = full Viterbi
            "target_span_getter": {"ents": True},
            "labels": labels,
        },
    )
    return nlp


# ---------------------------------------------------------------------------
# Data conversion: dict -> spaCy Doc
# ---------------------------------------------------------------------------


def _dicts_to_docs(
    nlp: edsnlp.Pipeline,
    span_dicts: list[dict],
) -> list[spacy.tokens.Doc]:
    """Convert span dicts to spaCy Docs with entity annotations."""
    docs = []
    for sd in span_dicts:
        doc = nlp.make_doc(sd["note_text"])
        # Run normalizer + sentence segmenter
        for name in ("eds.normalizer", "eds.sentences"):
            if nlp.has_pipe(name):
                nlp.get_pipe(name)(doc)

        spans = []
        for s in sd["spans"]:
            span = doc.char_span(
                s["start"], s["end"], label=s["label"],
                alignment_mode="expand",
            )
            if span is not None:
                spans.append(span)

        doc.ents = spacy.util.filter_spans(spans)

        # Skip docs where no spans survived alignment
        if not doc.ents:
            continue

        if not spacy.tokens.Doc.has_extension("note_id"):
            spacy.tokens.Doc.set_extension("note_id", default=None)
        doc._.note_id = sd["note_id"]
        docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_live(
    nlp: edsnlp.Pipeline,
    val_docs: list[dict],
    group_name: str,
) -> dict:
    """Run inference on val docs and compute text-level entity F1/P/R exactly as compute_metrics."""
    from utils import compute_metrics
    labels = GROUPS[group_name]
    label_set = set(labels)

    docs_to_pipe = [d["note_text"] for d in val_docs]
    predictions = []
    
    for doc_dict, doc in zip(val_docs, nlp.pipe(docs_to_pipe)):
        ents: dict[str, list[str]] = {}
        for ent in doc.ents:
            if ent.label_ in label_set:
                ents.setdefault(ent.label_, []).append(ent.text)
        predictions.append({"note_id": doc_dict["note_id"], "entities": ents})

    ground_truth = _docs_to_entity_sets(val_docs, group_name)
    metrics = compute_metrics(predictions, ground_truth, labels)
    return {
        "macro_f": metrics["macro"]["f1"],
        "macro_p": metrics["macro"]["precision"],
        "macro_r": metrics["macro"]["recall"],
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_eds(
    group_name: str,
    train_docs: list[dict],
    val_docs: list[dict],
    output_dir: str | Path = "./output",
    *,
    backbone: str = DEFAULT_BACKBONE,
    num_epochs: int = 15,
    batch_size: int = 4,
    embedding_lr: float = 5e-5,  # matches old config
    task_lr: float = 5e-5,  # aligned with old config (was 3e-4, 6x too high)
    grad_max_norm: float = 5.0,  # aligned with old config (was 1.0)
    patience: int = 5,
    seed: int = 42,
    window: int = 510,
    stride: int = 382,
    use_crt: bool = False,
) -> ModelResult:
    """Fine-tune CamemBERT+CRF for a single semantic group.

    Returns a ModelResult with metrics computed on *val_docs*.
    Best model is saved to ``output_dir/{group_name}/eds/model-best``.
    """
    labels = GROUPS[group_name]
    model_dir = Path(output_dir) / group_name / "eds"
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- Reset CUDA state (clean up after prior models e.g. GLiNER) ---
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # --- Build pipeline ---
    nlp = build_pipeline(labels, backbone=backbone, window=window, stride=stride)

    # --- Convert data ---
    train_spans = to_eds_spans(train_docs, group_name)
    val_spans = to_eds_spans(val_docs, group_name)

    print(f"[EDS] {group_name}: {len(train_spans)} train, "
          f"{len(val_spans)} val docs, {len(labels)} labels")

    if not train_spans:
        print(f"[EDS] WARNING: no training data for {group_name}")
        return ModelResult(group=group_name, method="eds")

    max_steps = max(10, num_epochs * len(train_spans) // batch_size)
    validation_interval = max(1, max_steps // num_epochs)
    print(f"[EDS] Computed {max_steps} max_steps for {num_epochs} epochs (eval every {validation_interval} steps).")

    train_spacy = _dicts_to_docs(nlp, train_spans)
    val_spacy = _dicts_to_docs(nlp, val_spans)

    # --- Diagnostic: verify gold entities survived alignment ---
    total_ents = sum(len(d.ents) for d in train_spacy)
    label_counts: dict[str, int] = defaultdict(int)
    for d in train_spacy:
        for e in d.ents:
            label_counts[e.label_] += 1
    print(f"[EDS] Gold entities in train: {total_ents} across {len(train_spacy)} docs")
    print(f"[EDS] Per-label counts: {dict(label_counts)}")

    # --- Initialize ---
    random.seed(seed)
    torch.manual_seed(seed)
    nlp.post_init(train_spacy)

    # Preprocess
    print("[EDS] Preprocessing training data...")
    preprocessed = list(nlp.preprocess_many(train_spacy, supervision=True))
    print(f"[EDS] {len(preprocessed)} preprocessed samples")

    # --- CPU sanity check (catches index/shape errors before GPU obscures them) ---
    if val_spacy:
        print("[EDS] Running CPU sanity check...")
        nlp.train(False)
        try:
            _test = val_spacy[0].copy()
            _test.ents = []
            list(nlp.pipe([_test]))
            print("[EDS] CPU sanity check passed")
        except Exception as e:
            print(f"[EDS] CPU sanity check FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise
        nlp.train(True)

    # --- Optimizer ---
    trf_pipe = nlp.get_pipe("transformer")
    trf_params = set(trf_pipe.parameters())
    all_params = set(nlp.parameters())

    optimizer = torch.optim.AdamW([
        {"params": list(all_params - trf_params), "lr": task_lr},
        {"params": list(trf_params), "lr": embedding_lr},
    ])

    # Asymmetric schedules (matches old_code behaviour):
    #   - Task head (CRF): no warmup, start at full task_lr, then linear decay
    #   - Transformer: linear warmup from 0, then linear decay
    warmup_steps = int(max_steps * 0.1)

    def task_lr_lambda(step):
        return max(0.0, 1.0 - step / max(1, max_steps))

    def embed_lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, max_steps - warmup_steps))

    # param groups: [0]=task/CRF, [1]=transformer
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [task_lr_lambda, embed_lr_lambda]
    )

    # --- Accelerate ---
    accelerator = Accelerator()
    device = accelerator.device
    print(f"[EDS] Device: {device}")

    # Move model components to device
    for name, pipe in nlp.torch_components():
        pipe.to(device)

    # --- Training loop ---
    best_f1 = -1.0  # ensures model-best is saved at the very first validation
    best_step = 0
    steps_without_improvement = 0
    all_metrics = []

    print("\n" + "="*50)
    print("== STAGE 1: Natural Representation Flow ==")
    print("="*50 + "\n")

    nlp.train(True)
    t0 = time.time()

    with tqdm(range(1, max_steps + 1), desc=f"[EDS] {group_name}", leave=True) as bar:
        data_iter = _infinite_shuffled_batches(preprocessed, batch_size, seed)

        for step in bar:
            # --- Validation ---
            if step == 1 or step % validation_interval == 0:
                nlp.train(False)
                scores = _evaluate_live(nlp, val_docs, group_name)
                macro_f1 = scores.get("macro_f", 0.0)
                macro_p = scores.get("macro_p", 0.0)
                macro_r = scores.get("macro_r", 0.0)

                all_metrics.append({
                    "step": step,
                    "f1": macro_f1, "precision": macro_p, "recall": macro_r,
                    "lr": optimizer.param_groups[0]["lr"],
                })
                # Using Macro metrics for reporting
                bar.set_postfix(f1=f"{macro_f1:.2%}", p=f"{macro_p:.2%}", r=f"{macro_r:.2%}")

                # Save last
                nlp.to_disk(model_dir / "model-last")

                # Track best using Macro metric
                if macro_f1 > best_f1:
                    best_f1 = macro_f1
                    best_step = step
                    nlp.to_disk(model_dir / "model-best")
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                # Early stopping
                if patience > 0 and steps_without_improvement >= patience and step > 1:
                    print(f"\n[EDS] Early stop at step {step}: "
                          f"best F1={best_f1:.2%} at step {best_step}")
                    break

                nlp.train(True)

            # --- Training step ---
            batch = next(data_iter)
            optimizer.zero_grad()

            loss = torch.zeros((), device=device)
            try:
                with nlp.cache():
                    collated = nlp.collate(batch)
                    collated = _to_device(collated, device)
                    for name, pipe in nlp.torch_components():
                        output = pipe.module_forward(collated[name])
                        if "loss" in output:
                            loss += output["loss"]
            except RuntimeError as e:
                print(f"[EDS] Step {step} forward FAILED: {e}")
                print(f"[EDS] Batch size: {len(batch)}")
                for i, sample in enumerate(batch[:2]):
                    print(f"[EDS] Sample {i} keys: {list(sample.keys())}")
                raise

            # --- Diagnostic: loss value (first 3 steps + every 50) ---
            if step <= 3 or step % 50 == 0:
                print(f"[EDS] Step {step} loss={loss.item():.6f}")

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(nlp.parameters(), grad_max_norm)
            optimizer.step()
            scheduler.step()

    elapsed = time.time() - t0
    print(f"[EDS] {group_name}: done in {elapsed / 60:.1f} min, best F1={best_f1:.2%}")

    # Save metrics log
    (model_dir / "train_metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )

    # --- Final evaluation for ModelResult ---
    # Reload best model
    best_path = model_dir / "model-best"
    last_path = model_dir / "model-last"
    if best_path.exists():
        predictions = predict_eds(str(best_path), val_docs, group_name)
    elif last_path.exists():
        predictions = predict_eds(str(last_path), val_docs, group_name)
    else:
        print(f"[EDS] WARNING: no saved model found in {model_dir}, skipping evaluation")
        predictions = [{"note_id": d["note_id"], "entities": {}} for d in val_docs]

    ground_truth = _docs_to_entity_sets(val_docs, group_name)
    metrics_stage1 = compute_metrics(predictions, ground_truth, labels)
    print(f"[EDS] Stage 1 Macro F1 = {metrics_stage1['macro']['f1']:.2%}")

    if not use_crt:
        return ModelResult(
            group=group_name,
            method="eds",
            per_label=metrics_stage1["per_label"],
            micro=metrics_stage1["micro"],
            macro=metrics_stage1["macro"],
            extra={
                "training_time_s": round(elapsed, 1),
                "num_train": len(train_spans),
                "num_val": len(val_spans),
                "best_f1": round(best_f1, 4),
                "best_step": best_step,
            },
        )
        
    print("\n" + "="*50)
    print("== STAGE 2: Classifier Re-Training (cRT) ==")
    print("="*50 + "\n")
    
    from utils import create_balanced_dataset
    print("[EDS] cRT: Generating class-balanced dataloader subset...")
    balanced_docs = create_balanced_dataset(train_docs, group_name, target_count=150)
    
    balanced_spans = to_eds_spans(balanced_docs, group_name)
    print(f"[EDS] cRT: Balanced docs count = {len(balanced_spans)}")
    
    print("[EDS] cRT: Loading Stage 1 model and freezing Transformer/CNN...")
    nlp_crt = edsnlp.load(str(model_dir / "model-best"))
    
    # Freeze everything except ner pipe
    for name, pipe in nlp_crt.torch_components():
        if name != "ner":
            for param in pipe.parameters():
                param.requires_grad = False
                
    # New optimizer
    ner_pipe = nlp_crt.get_pipe("ner")
    optimizer_crt = torch.optim.AdamW(filter(lambda p: p.requires_grad, nlp_crt.parameters()), lr=task_lr)
    
    crt_epochs = 4
    crt_max_steps = max(10, crt_epochs * len(balanced_spans) // batch_size)
    print(f"[EDS] cRT: Computed {crt_max_steps} max_steps for {crt_epochs} epochs.")
    
    balanced_spacy = _dicts_to_docs(nlp_crt, balanced_spans)
    preprocessed_crt = list(nlp_crt.preprocess_many(balanced_spacy, supervision=True))
    data_iter_crt = _infinite_shuffled_batches(preprocessed_crt, batch_size, seed)
    
    # Move new model to device
    for name, pipe in nlp_crt.torch_components():
        pipe.to(device)
        
    def task_lr_lambda_crt(step):
        return max(0.0, 1.0 - step / max(1, crt_max_steps))
    scheduler_crt = torch.optim.lr_scheduler.LambdaLR(optimizer_crt, task_lr_lambda_crt)
    
    best_crt_f1 = -1.0
    best_crt_step = 0
    steps_without_improvement = 0
    crt_patience = max(5, int(crt_max_steps * 0.1))
    crt_val_interval = max(1, crt_max_steps // crt_epochs)
    
    print(f"[EDS] cRT: Initiating re-training for {crt_max_steps} steps (eval every {crt_val_interval}).")
    nlp_crt.train(True)
    t0_crt = time.time()
    
    for step in range(1, crt_max_steps + 1):
        if step == 1 or step % crt_val_interval == 0:
            nlp_crt.train(False)
            scores = _evaluate_live(nlp_crt, val_docs, group_name)
            macro_f = scores.get("macro_f", 0.0)
            
            if macro_f > best_crt_f1:
                best_crt_f1 = macro_f
                best_crt_step = step
                nlp_crt.to_disk(model_dir / "model-balanced-best")
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                
            if steps_without_improvement >= crt_patience and step > 1:
                print(f"[EDS] cRT Early stop at step {step}: best Macro F1={best_crt_f1:.2%}")
                break
            nlp_crt.train(True)
            
        batch = next(data_iter_crt)
        optimizer_crt.zero_grad()
        loss = torch.zeros((), device=device)
        with nlp_crt.cache():
            collated = nlp_crt.collate(batch)
            collated = _to_device(collated, device)
            for name, pipe in nlp_crt.torch_components():
                output = pipe.module_forward(collated[name])
                if "loss" in output:
                    loss += output["loss"]
                    
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, nlp_crt.parameters()), grad_max_norm)
        optimizer_crt.step()
        scheduler_crt.step()
        
    elapsed_crt = time.time() - t0_crt
    
    best_crt_path = model_dir / "model-balanced-best"
    print(f"[EDS] cRT: saved final balanced model to {best_crt_path}")
    
    # Final evaluation for cRT
    predictions_crt = predict_eds(str(best_crt_path), val_docs, group_name)
    metrics_stage2 = compute_metrics(predictions_crt, ground_truth, labels)
    
    print(f"[EDS] cRT Pipeline finished. Stage 1 Macro F1 = {metrics_stage1['macro']['f1']:.2%} | Stage 2 Macro F1 = {metrics_stage2['macro']['f1']:.2%}")

    if metrics_stage2['macro']['f1'] > metrics_stage1['macro']['f1']:
        print("[EDS] cRT successfully improved model performance. Returning cRT model.")
        return ModelResult(
            group=group_name,
            method="eds_crt",
            per_label=metrics_stage2["per_label"],
            micro=metrics_stage2["micro"],
            macro=metrics_stage2["macro"],
            extra={
                "training_time_s": round(elapsed + elapsed_crt, 1),
                "num_train_stage1": len(train_spans),
                "num_train_stage2": len(balanced_spans),
                "best_f1": round(best_crt_f1, 4),
                "best_step": best_crt_step,
            },
        )
    else:
        print("[EDS] Stage 2 cRT failed to improve Macro F1. Falling back to Stage 1 model.")
        return ModelResult(
            group=group_name,
            method="eds",
            per_label=metrics_stage1["per_label"],
            micro=metrics_stage1["micro"],
            macro=metrics_stage1["macro"],
            extra={
                "training_time_s": round(elapsed, 1),
                "num_train_stage1": len(train_spans),
                "num_train_stage2": len(balanced_spans),
                "best_f1": round(best_f1, 4), # Best F1 from stage 1
                "best_step": best_step,
            },
        )

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_eds(
    model_path: str | Path,
    docs: list[dict],
    group_name: str,
) -> list[dict]:
    """Load a saved EDS model and predict on docs.

    Returns list of {"note_id": str, "entities": {label: [span_text, ...]}}.
    """
    nlp = edsnlp.load(str(model_path))
    labels = set(GROUPS[group_name])

    predictions = []
    for doc_dict in docs:
        doc = nlp(doc_dict["note_text"])
        ents: dict[str, list[str]] = {}
        for ent in doc.ents:
            if ent.label_ in labels:
                ents.setdefault(ent.label_, []).append(ent.text)
        predictions.append({"note_id": doc_dict["note_id"], "entities": ents})
    return predictions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_device(obj, device):
    """Recursively move tensors in nested dicts/lists to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


def _infinite_shuffled_batches(data: list, batch_size: int, seed: int):
    """Yield batches from data, reshuffling each epoch."""
    rng = random.Random(seed)
    indices = list(range(len(data)))
    while True:
        rng.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            yield [data[j] for j in batch_idx]


def _docs_to_entity_sets(docs: list[dict], group_name: str) -> list[dict]:
    """Convert raw docs to unified format for metric computation."""
    labels = set(GROUPS[group_name])
    result = []
    for doc in docs:
        ents: dict[str, list[str]] = {}
        for e in doc["entities"]:
            if e["label"] in labels:
                span = doc["note_text"][e["start"]:e["end"]]
                ents.setdefault(e["label"], []).append(span)
        result.append({"note_id": doc["note_id"], "entities": ents})
    return result
