"""Unified training interface for BRIGHT NER models.

Orchestrates GLiNER and EDS-NLP training across all 10 semantic groups
with a single entry point.

Usage:
    from training import train_group, train_all_groups
    result = train_group("diagnosis", "gliner", data_path="...", output_dir="./output")
    df = train_all_groups(data_path="...", output_dir="./output")
"""

from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd
import torch

from utils import (
    GROUPS,
    GROUP_NAMES,
    DEFAULT_DATASET,
    ModelResult,
    load_dataset,
    filter_by_group,
    split_dataset,
    save_results,
    results_to_dataframe,
)


def train_group(
    group_name: str,
    method: str,
    data_path: str | Path | None = None,
    output_dir: str | Path = "./output",
    train_ratio: float = 0.8,
    seed: int = 42,
    use_crt: bool = False,
    **kwargs,
) -> ModelResult:
    """Train one method for one semantic group.

    Parameters
    ----------
    group_name : str
        One of the 10 semantic group names (e.g. "diagnosis", "ihc").
    method : str
        "gliner" or "eds".
    data_path : path, optional
        Path to the generated JSONL dataset.
    output_dir : path
        Root directory for saved models and results.
    train_ratio : float
        Fraction of patients used for training (rest is validation).
    seed : int
        Random seed for patient-level split.
    **kwargs
        Forwarded to the underlying training function.

    Returns
    -------
    ModelResult
    """
    if group_name not in GROUPS:
        raise ValueError(f"Unknown group: {group_name}. Choose from {GROUP_NAMES}")
    if method not in ("gliner", "eds"):
        raise ValueError(f"Unknown method: {method}. Choose 'gliner' or 'eds'")

    # Load and split data
    docs = load_dataset(data_path)
    group_docs = filter_by_group(docs, group_name)
    train_docs, val_docs = split_dataset(group_docs, train_ratio=train_ratio, seed=seed)

    print(f"\n{'='*60}")
    print(f"Training {method.upper()} for group '{group_name}'")
    print(f"  {len(train_docs)} train / {len(val_docs)} val documents")
    print(f"  Labels: {GROUPS[group_name]}")
    print(f"{'='*60}\n")

    if method == "gliner":
        from gliner_bright.training_gliner import train_gliner
        return train_gliner(group_name, train_docs, val_docs, output_dir, use_crt=use_crt, **kwargs)
    else:
        from eds_bright.training_eds import train_eds
        return train_eds(group_name, train_docs, val_docs, output_dir, use_crt=use_crt, **kwargs)


def train_all_groups(
    data_path: str | Path | None = None,
    output_dir: str | Path = "./output",
    methods: tuple[str, ...] = ("gliner", "eds"),
    groups: list[str] | None = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    use_crt: bool = False,
    gliner_kwargs: dict | None = None,
    eds_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Train all semantic groups for all methods.

    Parameters
    ----------
    data_path : path, optional
        Path to the generated JSONL dataset.
    output_dir : path
        Root directory for saved models and results.
    methods : tuple of str
        Which methods to train ("gliner", "eds", or both).
    groups : list of str, optional
        Subset of groups to train. Defaults to all 10.
    train_ratio : float
        Fraction of patients for training.
    seed : int
        Random seed.
    gliner_kwargs : dict, optional
        Extra kwargs forwarded to train_gliner.
    eds_kwargs : dict, optional
        Extra kwargs forwarded to train_eds.

    Returns
    -------
    pd.DataFrame
        Combined results with columns: group, method, label, precision, recall, f1, support.
    """
    groups = groups or GROUP_NAMES
    gliner_kwargs = gliner_kwargs or {}
    eds_kwargs = eds_kwargs or {}
    output_dir = Path(output_dir)

    # Load data once
    docs = load_dataset(data_path)

    all_results: list[ModelResult] = []

    for group_name in groups:
        group_docs = filter_by_group(docs, group_name)
        train_docs, val_docs = split_dataset(group_docs, train_ratio=train_ratio, seed=seed)

        print(f"\n{'#'*60}")
        print(f"# Group: {group_name} ({len(train_docs)} train / {len(val_docs)} val)")
        print(f"# Labels: {GROUPS[group_name]}")
        print(f"{'#'*60}")

        for method in methods:
            kwargs = gliner_kwargs if method == "gliner" else eds_kwargs
            try:
                if method == "gliner":
                    from gliner_bright.training_gliner import train_gliner
                    result = train_gliner(group_name, train_docs, val_docs,
                                          str(output_dir), use_crt=use_crt, **kwargs)
                else:
                    from eds_bright.training_eds import train_eds
                    result = train_eds(group_name, train_docs, val_docs,
                                       str(output_dir), use_crt=use_crt, **kwargs)

                all_results.append(result)
                print(f"  [{method.upper()}] micro F1={result.micro.get('f1', 'N/A')}")

            except Exception as e:
                print(f"  [{method.upper()}] FAILED: {e}")
                all_results.append(ModelResult(
                    group=group_name, method=method,
                    extra={"error": str(e)},
                ))
            finally:
                # Free GPU memory between methods/groups
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    except RuntimeError:
                        print(f"  [CLEANUP] CUDA error during cleanup, resetting device")
                        torch.cuda.reset_peak_memory_stats()

        # Save intermediate results after each group
        save_results(all_results, output_dir / "results.csv")

    # Final save
    results_path = output_dir / "results.csv"
    save_results(all_results, results_path)
    print(f"\nResults saved to {results_path}")

    return results_to_dataframe(all_results)


def evaluate_group(
    group_name: str,
    method: str,
    model_path: str | Path,
    data_path: str | Path | None = None,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> ModelResult:
    """Evaluate a previously trained model on the validation set.

    Parameters
    ----------
    group_name : str
        Semantic group name.
    method : str
        "gliner" or "eds".
    model_path : path
        Path to the saved model directory.
    data_path : path, optional
        Path to the JSONL dataset (uses default if None).
    train_ratio : float
        Must match the ratio used during training to get the same val split.
    seed : int
        Must match the seed used during training.

    Returns
    -------
    ModelResult
    """
    from utils import compute_metrics

    docs = load_dataset(data_path)
    group_docs = filter_by_group(docs, group_name)
    _, val_docs = split_dataset(group_docs, train_ratio=train_ratio, seed=seed)

    labels = GROUPS[group_name]

    if method == "gliner":
        from gliner_bright.training_gliner import predict_gliner, _docs_to_entity_sets
        predictions = predict_gliner(model_path, val_docs, group_name)
        ground_truth = _docs_to_entity_sets(val_docs, group_name)
    else:
        from eds_bright.training_eds import predict_eds, _docs_to_entity_sets
        predictions = predict_eds(model_path, val_docs, group_name)
        ground_truth = _docs_to_entity_sets(val_docs, group_name)

    metrics = compute_metrics(predictions, ground_truth, labels)
    return ModelResult(
        group=group_name,
        method=method,
        per_label=metrics["per_label"],
        micro=metrics["micro"],
        macro=metrics["macro"],
    )
