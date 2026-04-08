"""GLiNER2 fine-tuning for one semantic group.

Usage:
    from gliner_bright.training_gliner import train_gliner, predict_gliner
    result = train_gliner("diagnosis", train_docs, val_docs, output_dir="./output")
"""

from __future__ import annotations

import json
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


# ---------------------------------------------------------------------------
# Monkey-patch for GLiNER2 batch_size > 1 bug (upstream PR #96)
# In gliner2 1.2.4, compute_span_rep_batched slices span_rep per sample but
# leaves spans_idx / span_mask at the padded batch-max length.  This causes
# a tensor-size mismatch in compute_struct_loss when samples in a batch have
# different token lengths.  The patch below trims all three tensors to the
# per-sample span count so they stay consistent.
# Remove this once gliner2 ships the fix from PR #96.
# ---------------------------------------------------------------------------

def _patched_compute_span_rep_batched(
    self, token_embs_list: List[torch.Tensor]
) -> List[Dict[str, Any]]:
    if not token_embs_list:
        return []

    device = token_embs_list[0].device
    text_lengths = [len(t) for t in token_embs_list]
    max_text_len = max(text_lengths)
    batch_size = len(token_embs_list)
    hidden = token_embs_list[0].shape[-1]

    padded = torch.zeros(
        batch_size, max_text_len, hidden,
        device=device, dtype=token_embs_list[0].dtype,
    )
    for i, emb in enumerate(token_embs_list):
        padded[i, : text_lengths[i]] = emb

    text_len_t = torch.tensor(text_lengths, device=device)
    span_rep, safe_spans, span_mask = self._compute_span_rep_core(
        padded, text_len_t,
    )

    results: List[Dict[str, Any]] = []
    for i in range(batch_size):
        tl = text_lengths[i]
        n_spans = tl * self.max_width
        results.append({
            "span_rep": span_rep[i, :tl, :, :],
            "spans_idx": safe_spans[i : i + 1, :n_spans, :],
            "span_mask": span_mask[i : i + 1, :n_spans],
        })
    return results

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    GROUPS,
    FIELD_DESCRIPTIONS,
    ModelResult,
    chunk_document,
    to_gliner_examples,
    compute_metrics,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BASE_MODEL = "fastino/gliner2-multi-v1"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_gliner(
    group_name: str,
    train_docs: list[dict],
    val_docs: list[dict],
    output_dir: str | Path = "./output",
    *,
    base_model: str = BASE_MODEL,
    num_epochs: int = 15,
    batch_size: int = 8,
    task_lr: float = 5e-4,
    encoder_lr: float = 1e-5,
    use_lora: bool = True,
    lora_r: int = 64,
    lora_dropout: float = 0.1,
    lora_target_modules: list[str] | None = None,
    fp16: bool = True,
    early_stopping_patience: int | str = "auto",
    report_to_wandb: bool = False,
    wandb_project: str = "bright_gliner",
    use_crt: bool = False,
) -> ModelResult:
    """Fine-tune a GLiNER2 model for a single semantic group.

    Returns a ModelResult with per-label and aggregate metrics on *val_docs*.
    The best model is saved to ``output_dir/{group_name}/gliner/best``.
    """
    lora_alpha = 2 * lora_r
    if lora_target_modules is None:
        # Fine-tune all linear blocks (encoder + task heads)
        lora_target_modules = [
            "encoder", "span_rep", "classifier", "count_embed", "count_pred",
        ]

    if early_stopping_patience == "auto":
        early_stopping_patience = max(5, int(0.05 * num_epochs))
    labels = GROUPS[group_name]
    model_dir = Path(output_dir) / group_name / "gliner"
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- Convert data ---
    train_examples = to_gliner_examples(train_docs, group_name)
    val_examples = to_gliner_examples(val_docs, group_name)

    print(f"[GLiNER] {group_name}: {len(train_examples)} train, "
          f"{len(val_examples)} val examples, {len(labels)} labels")

    if not train_examples:
        print(f"[GLiNER] WARNING: no training examples for {group_name}")
        return ModelResult(group=group_name, method="gliner")

    # Save JSONL for reproducibility
    for split_name, exs in [("train", train_examples), ("val", val_examples)]:
        p = model_dir / f"{split_name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for ex in exs:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # --- Load model ---
    model = GLiNER2.from_pretrained(base_model)

    # Apply batch-size bug fix (see module-level docstring)
    model.compute_span_rep_batched = types.MethodType(
        _patched_compute_span_rep_batched, model,
    )

    # --- Configure training ---
    config = TrainingConfig(
        output_dir=str(model_dir),
        experiment_name=f"bright_{group_name}",
        num_epochs=num_epochs,
        batch_size=batch_size,
        encoder_lr=encoder_lr,
        task_lr=task_lr,
        warmup_ratio=0.1,
        scheduler_type="cosine",
        fp16=fp16,
        eval_strategy="epoch",
        save_best=True,
        metric_for_best="entity_f1",   # track entity F1 (not eval_loss)
        greater_is_better=True,         # higher F1 = better
        early_stopping=True,
        early_stopping_patience=early_stopping_patience,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        # MUST keep adapter-only during training.  save_adapter_only=False
        # triggers merge/unmerge in _save_checkpoint which breaks
        # requires_grad on LoRA params, killing training after first eval.
        save_adapter_only=True,
        report_to_wandb=report_to_wandb,
        wandb_project=wandb_project if report_to_wandb else None,
    )

    # --- Build compute_metrics callback ---
    # This is the trainer's designed injection point: it runs inside
    # _evaluate() BEFORE _log_metrics and best-metric comparison,
    # guaranteeing entity_f1 is available for early stopping.
    descriptions = {l: FIELD_DESCRIPTIONS.get(l, l) for l in labels}
    _labels_set = set(labels)

    def _compute_entity_metrics(model_ref, eval_dataset):
        """Compute entity F1/P/R on val_docs using live model."""
        model_ref.eval()
        preds_list = []
        gt_list = []
        with torch.no_grad():
            for doc in val_docs:
                ents: dict[str, list[str]] = {}
                for chunk in chunk_document(doc):
                    text = chunk["note_text"]
                    result = model_ref.extract_entities(
                        text, entity_types=descriptions, threshold=0.4,
                    )
                    if isinstance(result, dict):
                        entity_dict = result.get("entities", result)
                        for label_name, matches in entity_dict.items():
                            if isinstance(matches, list) and matches:
                                spans = [m["text"] if isinstance(m, dict) else str(m)
                                         for m in matches]
                                ents.setdefault(label_name, []).extend(spans)
                preds_list.append({"note_id": doc["note_id"], "entities": ents})
                gt_ents: dict[str, list[str]] = {}
                for e in doc["entities"]:
                    if e["label"] in _labels_set:
                        span = doc["note_text"][e["start"]:e["end"]]
                        gt_ents.setdefault(e["label"], []).append(span)
                gt_list.append({"note_id": doc["note_id"], "entities": gt_ents})
        macros = compute_metrics(preds_list, gt_list, labels)["macro"]
        return {
            "entity_f1": macros["f1"],
            "entity_precision": macros["precision"],
            "entity_recall": macros["recall"],
        }

    # --- Train ---
    print("\n" + "="*50)
    print("== STAGE 1: Natural Representation Flow ==")
    print("="*50 + "\n")
    trainer = GLiNER2Trainer(
        model, config, compute_metrics=_compute_entity_metrics,
    )

    # --- Monkey-patch _log_metrics for tqdm display only ---
    _orig_log_metrics = trainer._log_metrics

    def _patched_log_metrics(metrics, prefix=""):
        _orig_log_metrics(metrics, prefix)
        if prefix == "eval" and trainer.progress_bar is not None:
            f1 = metrics.get("entity_f1", 0.0)
            p = metrics.get("entity_precision", 0.0)
            r = metrics.get("entity_recall", 0.0)
            eval_loss = metrics.get("eval_loss", 0.0)
            trainer.progress_bar.set_postfix(
                f1=f"{f1:.2%}", p=f"{p:.2%}", r=f"{r:.2%}",
                eval_loss=f"{eval_loss:.4f}",
            )
            print(f"\n[GLiNER] epoch {trainer.epoch + 1}/{num_epochs}  "
                  f"F1={f1:.2%}  P={p:.2%}  R={r:.2%}  "
                  f"eval_loss={eval_loss:.4f}")

    trainer._log_metrics = _patched_log_metrics

    t0 = time.time()
    results = trainer.train(
        train_data=train_examples,
        eval_data=val_examples if val_examples else None,
    )
    elapsed = time.time() - t0
    print(f"[GLiNER] {group_name}: training done in {elapsed / 60:.1f} min")

    # --- Post-training: reload best adapter, merge LoRA, save full model ---
    # During training, only LoRA adapters were saved (save_adapter_only=True)
    # to avoid the merge/unmerge cycle that breaks gradients.
    # Now training is done: reload the BEST adapter (not last epoch),
    # merge into base weights, and save the self-contained model.
    best_adapter_path = model_dir / "best"
    merged_path = model_dir / "best_merged"
    try:
        from gliner2.training.lora import merge_lora_weights
        if best_adapter_path.exists():
            print(f"[GLiNER] {group_name}: reloading best adapter from {best_adapter_path}")
            model.load_adapter(str(best_adapter_path))
        merge_lora_weights(model)
        model.save_pretrained(str(merged_path))
        print(f"[GLiNER] {group_name}: merged LoRA → saved to {merged_path}")
    except Exception as e:
        print(f"[GLiNER] {group_name}: LoRA reload/merge failed: {e}")
        print(f"[GLiNER] {group_name}: using last-epoch model for final eval")

    # --- Final evaluation using in-memory (now merged) model ---
    model.eval()
    predictions = []
    with torch.no_grad():
        for doc in val_docs:
            ents: dict[str, list[str]] = {}
            for chunk in chunk_document(doc):
                text = chunk["note_text"]
                result = model.extract_entities(
                    text, entity_types=descriptions, threshold=0.4,
                )
                if isinstance(result, dict):
                    entity_dict = result.get("entities", result)
                    for label_name, matches in entity_dict.items():
                        if isinstance(matches, list) and matches:
                            spans = [m["text"] if isinstance(m, dict) else str(m)
                                     for m in matches]
                            ents.setdefault(label_name, []).extend(spans)
            predictions.append({"note_id": doc["note_id"], "entities": ents})

    ground_truth = _docs_to_entity_sets(val_docs, group_name)
    metrics_stage1 = compute_metrics(predictions, ground_truth, labels)

    print(f"[GLiNER] Stage 1 Macro F1 = {metrics_stage1['macro']['f1']:.2%}")

    if not use_crt:
        return ModelResult(
            group=group_name,
            method="gliner",
            per_label=metrics_stage1["per_label"],
            micro=metrics_stage1["micro"],
            macro=metrics_stage1["macro"],
            extra={
                "training_time_s": round(elapsed, 1),
                "num_train": len(train_examples),
                "num_val": len(val_examples),
            },
        )
        
    # --- STAGE 2: Classifier Re-Training (cRT) ---
    print("\n" + "="*50)
    print("== STAGE 2: Classifier Re-Training (cRT) ==")
    print("="*50 + "\n")

    from utils import create_balanced_dataset
    print("[GLiNER] cRT: Generating class-balanced dataloader subset...")
    balanced_docs = create_balanced_dataset(train_docs, group_name, target_count=150)
    balanced_examples = to_gliner_examples(balanced_docs, group_name)
    print(f"[GLiNER] cRT: Balanced examples count = {len(balanced_examples)}")
    
    print("[GLiNER] cRT: Freezing Transformer backbone and span representations...")
    model.encoder.requires_grad_(False)
    model.span_rep.requires_grad_(False)
    # Ensure classification heads are explicitly unfrozen
    model.classifier.requires_grad_(True)
    model.count_pred.requires_grad_(True)
    model.count_embed.requires_grad_(True)
    
    crt_epochs = 4
    config.num_epochs = crt_epochs
    config.use_lora = False # No need for LoRA to tune tiny linear layers
    config.save_adapter_only = False
    config.output_dir = str(model_dir / "best_crt")
    
    trainer_crt = GLiNER2Trainer(
        model, config, compute_metrics=_compute_entity_metrics,
    )
    trainer_crt._log_metrics = _patched_log_metrics
    
    print(f"[GLiNER] cRT: Initiating re-training for {crt_epochs} learning cycles...")
    t0_crt = time.time()
    trainer_crt.train(train_data=balanced_examples, eval_data=val_examples)
    elapsed_crt = time.time() - t0_crt
    
    final_model_path = model_dir / "best_merged_crt"
    model.save_pretrained(str(final_model_path))
    print(f"[GLiNER] cRT: Saved final balanced model to {final_model_path}")
    
    # Final Evaluation for Stage 2
    model.eval()
    predictions_crt = []
    with torch.no_grad():
        for doc in val_docs:
            ents: dict[str, list[str]] = {}
            for chunk in chunk_document(doc):
                text = chunk["note_text"]
                result = model.extract_entities(
                    text, entity_types=descriptions, threshold=0.4,
                )
                if isinstance(result, dict):
                    entity_dict = result.get("entities", result)
                    for label_name, matches in entity_dict.items():
                        if isinstance(matches, list) and matches:
                            spans = [m["text"] if isinstance(m, dict) else str(m)
                                     for m in matches]
                            ents.setdefault(label_name, []).extend(spans)
            predictions_crt.append({"note_id": doc["note_id"], "entities": ents})

    metrics_stage2 = compute_metrics(predictions_crt, ground_truth, labels)
    
    print(f"[GLiNER] cRT Pipeline finished. Stage 1 Macro F1 = {metrics_stage1['macro']['f1']:.2%} | Stage 2 Macro F1 = {metrics_stage2['macro']['f1']:.2%}")

    if metrics_stage2['macro']['f1'] > metrics_stage1['macro']['f1']:
        print("[GLiNER] cRT successfully improved model performance. Returning cRT model.")
        return ModelResult(
            group=group_name,
            method="gliner_crt",
            per_label=metrics_stage2["per_label"],
            micro=metrics_stage2["micro"],
            macro=metrics_stage2["macro"],
            extra={
                "training_time_s": round(elapsed + elapsed_crt, 1),
                "num_train_stage1": len(train_examples),
                "num_train_stage2": len(balanced_examples),
            },
        )
    else:
        print("[GLiNER] Stage 2 cRT failed to improve Macro F1. Falling back to Stage 1 model.")
        return ModelResult(
            group=group_name,
            method="gliner",
            per_label=metrics_stage1["per_label"],
            micro=metrics_stage1["micro"],
            macro=metrics_stage1["macro"],
            extra={
                "training_time_s": round(elapsed, 1),
                "num_train_stage1": len(train_examples),
                "num_train_stage2": len(balanced_examples),
            },
        )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_gliner(
    model_path: str | Path,
    docs: list[dict],
    group_name: str,
    *,
    threshold: float = 0.4,
) -> list[dict]:
    """Run inference on *docs* and return predictions in unified format.

    Returns list of {"note_id": str, "entities": {label: [span, ...]}}.
    """
    model = GLiNER2.from_pretrained(str(model_path), local_files_only=True)
    labels = GROUPS[group_name]
    descriptions = {l: FIELD_DESCRIPTIONS.get(l, l) for l in labels}

    print(f"[GLiNER] predict: model={model_path}, labels={labels}, threshold={threshold}")
    print(f"[GLiNER] predict: entity_types={descriptions}")

    predictions = []
    n_docs_with_ents = 0
    for i, doc in enumerate(docs):
        ents: dict[str, list[str]] = {}
        for chunk in chunk_document(doc):
            text = chunk["note_text"]
            result = model.extract_entities(
                text,
                entity_types=descriptions,  # {label: description} dict
                threshold=threshold,
            )

            # Debug: print raw result for first 3 docs
            if i < 3:
                print(f"[GLiNER] doc {i} raw result keys={list(result.keys()) if isinstance(result, dict) else type(result)}, "
                      f"result={result}")

            if isinstance(result, dict):
                # extract_entities returns {'entities': {label: [spans]}}
                entity_dict = result.get("entities", result)
                for label, matches in entity_dict.items():
                    if isinstance(matches, list) and matches:
                        spans = [m["text"] if isinstance(m, dict) else str(m)
                                for m in matches]
                        ents.setdefault(label, []).extend(spans)
        if ents:
            n_docs_with_ents += 1
        predictions.append({"note_id": doc["note_id"], "entities": ents})

    print(f"[GLiNER] predict: {n_docs_with_ents}/{len(docs)} docs have entities")
    return predictions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _docs_to_entity_sets(docs: list[dict], group_name: str) -> list[dict]:
    """Convert raw docs to the unified format for metric computation."""
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
