"""Visualization utilities for BRIGHT NER training results.

Provides comparison plots (GLiNER vs EDS) and per-model analysis plots.
Uses matplotlib + seaborn for a polished, professional look.

Usage:
    from viz import plot_overall_comparison, plot_model_per_group, ...
    plot_overall_comparison(results_df, save_path="comparison.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme("paper")

from matplotlib.colors import LinearSegmentedColormap

sns.set_theme(style="whitegrid", font_scale=1.1)

# Modern Pink and Purple theme for Models
METHOD_PALETTE = {
    "gliner": "#9C27B0", "gliner_crt": "#6A1B9A", # Purple
    "eds": "#E91E63", "eds_crt": "#AD1457",       # Pink
}

# Blue (Cold/Low) to Green (Positive/High)
SCORE_CMAP = LinearSegmentedColormap.from_list(
    "blue_to_green", ["#1565C0", "#00ACC1", "#43A047"]
)
METRIC_COLORS = {"precision": "#1565C0", "recall": "#00ACC1", "f1": "#43A047"}


def _save_or_show(fig: plt.Figure, save_path: Optional[str | Path]):
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _filter_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remove aggregate rows (micro, macro)."""
    return df[~df["label"].isin(["micro", "macro"])]


def _get_aggregates(df: pd.DataFrame, agg_name: str = "macro") -> pd.DataFrame:
    """Get only aggregate rows."""
    agg_name = agg_name.lower()
    return df[df["label"] == agg_name]


# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON PLOTS (GLiNER vs EDS)
# ═══════════════════════════════════════════════════════════════════════════


def plot_overall_comparison(
    results_df: pd.DataFrame,
    metric: str = "f1",
    agg_name: str = "macro",
    save_path: Optional[str | Path] = None,
):
    """Grouped bar chart: one metric per group, side-by-side GLiNER vs EDS."""
    agg = _get_aggregates(results_df, agg_name)
    if agg.empty:
        print("No aggregate metrics found.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=agg, x="group", y=metric, hue="method",
        palette=METHOD_PALETTE, ax=ax, edgecolor="white",
    )
    ax.set_title(f"Overall Comparison - {agg_name[0].upper()+agg_name[1:]} {metric.upper()} by Semantic Group", fontsize=14, pad=15)
    ax.set_xlabel("")
    ax.set_ylabel(metric.capitalize())
    ax.set_ylim(0, 1.05)
    
    # Safely rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Move legend outside to prevent overlap
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="Method")

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    _save_or_show(fig, save_path)


def plot_per_feature_comparison(
    results_df: pd.DataFrame,
    group_name: str,
    save_path: Optional[str | Path] = None,
):
    """Per-label P/R/F1 within a group, both methods side-by-side."""
    df = _filter_labels(results_df)
    df = df[df["group"] == group_name].copy()
    if df.empty:
        print(f"No per-label data for group '{group_name}'.")
        return

    # Sort by average F1
    avg_f1 = df.groupby("label")["f1"].mean().sort_values(ascending=False).index
    df["label"] = pd.Categorical(df["label"], categories=avg_f1, ordered=True)
    df = df.sort_values("label")

    # Melt to long format for metrics
    melted = df.melt(
        id_vars=["group", "method", "label"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric", value_name="score",
    )

    n_labels = df["label"].nunique()
    # Horizontal layouts are much more readable for numerous labels
    fig, axes = plt.subplots(1, 3, figsize=(12, max(4, n_labels * 0.4)), sharey=True)

    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        subset = melted[melted["metric"] == metric]
        sns.barplot(
            data=subset, y="label", x="score", hue="method",
            palette=METHOD_PALETTE, ax=ax, edgecolor="white",
        )
        ax.set_title(metric.capitalize(), fontsize=12)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Score")
        if ax != axes[0]:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Feature")
            
        if ax == axes[-1]:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="Method")
        else:
            if ax.get_legend() is not None:
                ax.legend().remove()

    fig.suptitle(f"Per-Feature Comparison - {group_name}", fontsize=14, y=1.02)
    _save_or_show(fig, save_path)


def plot_versus_heatmap(
    results_df: pd.DataFrame,
    group_name: Optional[str] = None,
    method_a: str = "gliner",
    method_b: str = "eds",
    agg_name: str = "macro",
    save_path: Optional[str | Path] = None,
):
    """Gap heatmap showing the absolute difference between two methods.
    If group_name is None, compares aggregate metrics across all groups.
    Otherwise, compares per-feature metrics within the specified group.
    Color indicates the winning method.
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    # Toggle between per-group (all groups) and per-label (specific group)
    if group_name is None:
        df = _get_aggregates(results_df, agg_name)
        index_col = "group"
        title_suffix = "All Groups"
        agg_addon = agg_name
    else:
        df = _filter_labels(results_df)
        df = df[df["group"] == group_name].copy()
        index_col = "label"
        title_suffix = group_name
        agg_addon = ""
    
    df_a_subset = df[df["method"].str.startswith(method_a)]
    df_b_subset = df[df["method"].str.startswith(method_b)]
    
    if df_a_subset.empty or df_b_subset.empty:
        print(f"Missing data for {method_a} or {method_b} in {title_suffix}")
        return

    actual_method_a = df_a_subset["method"].iloc[0]
    actual_method_b = df_b_subset["method"].iloc[0]

    df_a = df_a_subset.set_index(index_col)[["precision", "recall", "f1"]]
    df_b = df_b_subset.set_index(index_col)[["precision", "recall", "f1"]]

    common_labels = df_a.index.intersection(df_b.index)
    df_a = df_a.loc[common_labels]
    df_b = df_b.loc[common_labels]
    
    # Sort by maximum F1 to keep order relevant to performance
    max_f1 = np.maximum(df_a["f1"], df_b["f1"]).sort_values(ascending=False)
    df_a = df_a.loc[max_f1.index]
    df_b = df_b.loc[max_f1.index]

    # Calculate gap (A - B). Positive = A wins, Negative = B wins.
    gap_df = df_a - df_b
    
    # Custom Divergent Colormap: actual_method_b color -> white -> actual_method_a color
    color_b = METHOD_PALETTE.get(actual_method_b, "#FF9800")
    color_a = METHOD_PALETTE.get(actual_method_a, "#2196F3")
    cmap = LinearSegmentedColormap.from_list("gap_cmap", [color_b, "white", color_a])
    
    # Display absolute positive difference only
    if hasattr(gap_df, "map"):
        annot_df = gap_df.abs().map(lambda x: f"{x:.3f}")
    else:
        annot_df = gap_df.abs().applymap(lambda x: f"{x:.3f}")

    fig, ax = plt.subplots(figsize=(6, max(3, len(gap_df) * 0.4)))

    sns.heatmap(
        gap_df, annot=annot_df, fmt="", cmap=cmap, vmin=-1.0, vmax=1.0,
        linewidths=0.5, ax=ax,
    )

    ax.set_title(f"{actual_method_a.upper()} vs {actual_method_b.upper()} {agg_addon} Gap - {title_suffix}", fontsize=13, pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_xticklabels([c.get_text().capitalize() for c in ax.get_xticklabels()])

    # Adjust colorbar labels to absolute values and add directional text
    cbar = ax.collections[0].colorbar
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks) # Matplotlib >= 3.3 requirement
    cbar.set_ticklabels([f"{abs(t):.1f}" for t in ticks])
    
    cbar_label_text = f"← {actual_method_b.upper()} better     {actual_method_a.upper()} better →"
    cbar.ax.set_ylabel(cbar_label_text, fontsize=10, labelpad=15)

    _save_or_show(fig, save_path)


def plot_all_groups_heatmap(
    results_df: pd.DataFrame,
    metric: str = "f1",
    agg_name: str = "macro",
    save_path: Optional[str | Path] = None,
):
    """Heatmap of a metric across all groups × methods."""
    agg = _get_aggregates(results_df, agg_name)
    if agg.empty:
        print("No aggregate metrics found.")
        return

    pivot = agg.pivot_table(index="group", columns="method", values=metric)

    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap=SCORE_CMAP, vmin=0, vmax=1,
        linewidths=0.5, ax=ax, cbar_kws={"label": metric.capitalize()},
    )
    ax.set_title(f"{agg_name[0].upper()+agg_name[1:]} {metric.upper()} - All Groups × Methods", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _save_or_show(fig, save_path)


# ═══════════════════════════════════════════════════════════════════════════
# PER-MODEL ANALYSIS PLOTS
# ═══════════════════════════════════════════════════════════════════════════


def plot_model_per_group(
    results_df: pd.DataFrame,
    method: str,
    save_path: Optional[str | Path] = None,
):
    """Single-method: P/R/F1 across all 10 groups as grouped bars."""
    agg = _get_aggregates(results_df)
    agg = agg[agg["method"].str.startswith(method)].copy()
    if agg.empty:
        print(f"No aggregate data for method '{method}'.")
        return

    actual_method = agg["method"].iloc[0]
    melted = agg.melt(
        id_vars=["group", "method"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric", value_name="score",
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=melted, x="group", y="score", hue="metric",
        palette=METRIC_COLORS, ax=ax, edgecolor="white",
    )
    ax.set_title(f"{actual_method.upper()} - Micro P/R/F1 per Semantic Group", fontsize=14, pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="Metric")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    _save_or_show(fig, save_path)


def plot_model_per_feature(
    results_df: pd.DataFrame,
    method: str,
    group_name: str,
    save_path: Optional[str | Path] = None,
):
    """Single-method: per-label breakdown as a heatmap (Precision, Recall, F1)."""
    df = _filter_labels(results_df)
    df = df[(df["group"] == group_name) & (df["method"].str.startswith(method))].copy()
    if df.empty:
        print(f"No per-label data for {method}/{group_name}.")
        return

    actual_method = df["method"].iloc[0]
    df = df.sort_values("f1", ascending=False)
    
    heatmap_data = df.set_index("label")[["precision", "recall", "f1"]]

    fig, ax = plt.subplots(figsize=(6, max(3, len(heatmap_data) * 0.4)))

    sns.heatmap(
        heatmap_data, annot=True, fmt=".3f", cmap=SCORE_CMAP, vmin=0, vmax=1,
        linewidths=0.5, ax=ax, cbar_kws={"label": "Score"}
    )

    ax.set_title(f"{actual_method.upper()} - {group_name} (per label)", fontsize=13, pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("")
    
    ax.set_xticklabels([c.get_text().capitalize() for c in ax.get_xticklabels()])

    _save_or_show(fig, save_path)

def plot_all_labels_performance(
    results_df: pd.DataFrame,
    method: str,
    threshold: float = 0.6,
    save_path: Optional[str | Path] = None,
):
    """Bar plot of ALL labels across all groups for a model, sorted by F1 with a color gradient."""
    df = _filter_labels(results_df)
    df = df[df["method"].str.startswith(method)].copy()
    
    if df.empty:
        print(f"No labels found for method '{method}'.")
        return
        
    actual_method = df["method"].iloc[0].upper()
    
    # Sort by F1 descending
    df = df.sort_values("f1", ascending=False)
    
    # Prepend group name to label for clarity on the y-axis
    df["display_label"] = df["group"] + " : " + df["label"]
    
    # Determine passing vs poor
    passing_df = df[df["f1"] >= threshold]
    poor_df = df[df["f1"] < threshold]
    
    n_pass = len(passing_df)
    n_poor = len(poor_df)
    
    # Print the findings to console
    print(f"\n[{actual_method}] Passing labels (F1 >= {threshold}): {n_pass}")
    for _, row in passing_df.iterrows():
        print(f"  - {row['display_label']}: {row['f1']:.3f}")
        
    print(f"\n[{actual_method}] Poor labels (F1 < {threshold}): {n_poor}")
    for _, row in poor_df.iterrows():
        print(f"  - {row['display_label']}: {row['f1']:.3f}")
    
    # Dynamic figure height based on the number of labels
    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.3)))
    
    # Map F1 values (0 to 1) to colors using the existing colormap
    norm = plt.Normalize(0, 1)
    bar_colors = [SCORE_CMAP(norm(val)) for val in df["f1"]]
    
    sns.barplot(
        data=df, x="f1", y="display_label",
        palette=bar_colors, ax=ax, edgecolor="white"
    )
    
    # Add visual threshold line
    ax.axvline(x=threshold, color="red", linestyle="--", alpha=0.8, label=f"Threshold ({threshold})")
    
    ax.set_title(f"There are {n_pass} passing labels and {n_poor} poor labels", fontsize=14, pad=15)
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("")
    ax.set_xlim(0, 1.05)
    ax.legend(loc="lower right")
    
    # Add score annotations
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=3)
        
    _save_or_show(fig, save_path)

def plot_passing_comparison(
    results_df: pd.DataFrame,
    method_a: str = "gliner",
    method_b: str = "eds",
    threshold: float = 0.6,
    save_path: Optional[str | Path] = None,
):
    """Grouped bar chart comparing the total number of passing vs poorly performing 
    labels between two methods, including a 'Best of Both' union category.
    """
    df = _filter_labels(results_df)
    
    df_a = df[df["method"].str.startswith(method_a)]
    df_b = df[df["method"].str.startswith(method_b)]
    
    if df_a.empty or df_b.empty:
        print(f"Missing data for {method_a} or {method_b} to do a comparison.")
        return

    actual_name_a = df_a["method"].iloc[0].upper()
    actual_name_b = df_b["method"].iloc[0].upper()
    
    data = []
    
    # 1. Method A counts
    n_pass_a = (df_a["f1"] >= threshold).sum()
    n_poor_a = (df_a["f1"] < threshold).sum()
    data.append({"Method": actual_name_a, "Status": "Pass", "Count": n_pass_a})
    data.append({"Method": actual_name_a, "Status": "Poor", "Count": n_poor_a})
    
    # 2. Method B counts
    n_pass_b = (df_b["f1"] >= threshold).sum()
    n_poor_b = (df_b["f1"] < threshold).sum()
    data.append({"Method": actual_name_b, "Status": "Pass", "Count": n_pass_b})
    data.append({"Method": actual_name_b, "Status": "Poor", "Count": n_poor_b})
    
    # 3. Best of Both (Union of passes, Intersection of poors)
    # Use an outer join to handle cases where one method evaluated a label the other missed
    merged = pd.merge(
        df_a[["group", "label", "f1"]],
        df_b[["group", "label", "f1"]],
        on=["group", "label"],
        how="outer",
        suffixes=("_a", "_b")
    )
    
    # Calculate the max F1 score across both models for each label
    merged["best_f1"] = merged[["f1_a", "f1_b"]].max(axis=1)
    
    n_pass_best = (merged["best_f1"] >= threshold).sum()
    n_poor_best = (merged["best_f1"] < threshold).sum()
    
    data.append({"Method": "Best of Both", "Status": "Pass", "Count": n_pass_best})
    data.append({"Method": "Best of Both", "Status": "Poor", "Count": n_poor_best})

    # Prepare for plotting
    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 6))  # Slightly wider to accommodate 3 groups
    sns.barplot(
        data=plot_df, x="Method", y="Count", hue="Status",
        palette={"Pass": "#43A047", "Poor": "#E53935"}, ax=ax, edgecolor="white"
    )
    
    ax.set_title(f"Pass vs Poor Labels Comparison (F1 Threshold: {threshold})", fontsize=14, pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Number of Labels")
    
    # Add value labels on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", fontsize=10, padding=2)

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="Status")
    
    _save_or_show(fig, save_path)


def plot_group_pass_proportion(
    results_df: pd.DataFrame,
    method: str,
    threshold: float = 0.6,
    save_path: Optional[str | Path] = None,
):
    """100% stacked bar chart showing the proportion of passing vs poor labels per semantic group for a single method."""
    df = _filter_labels(results_df)
    df = df[df["method"].str.startswith(method)].copy()
    
    if df.empty:
        print(f"No per-label data for method '{method}'.")
        return

    actual_method = df["method"].iloc[0].upper()

    # Tag labels as Pass or Poor
    df["Status"] = np.where(df["f1"] >= threshold, "Pass", "Poor")
    
    # Count occurrences per group
    counts = df.groupby(["group", "Status"]).size().unstack(fill_value=0)
    
    # Ensure both columns exist even if a group is 100% pass or 100% poor
    for col in ["Pass", "Poor"]:
        if col not in counts:
            counts[col] = 0.0
            
    # Calculate proportions
    proportions = counts.div(counts.sum(axis=1), axis=0)
    
    # Sort groups by the proportion of passing labels (highest to lowest)
    proportions = proportions.sort_values("Pass", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Pandas built-in stacked bar chart works perfectly here
    proportions[["Pass", "Poor"]].plot(
        kind="bar", stacked=True, color=["#43A047", "#E53935"], ax=ax, edgecolor="white", width=0.8
    )

    ax.set_title(f"{actual_method} - Proportion of Passing Labels per Group (F1 >= {threshold})", fontsize=14, pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.legend(title="Status", loc="upper left", bbox_to_anchor=(1, 1))

    # Add percentage text inside the bars for readability
    for c in ax.containers:
        # Format as percentage, but hide 0% to prevent clutter
        labels = [f"{v.get_height():.0%}" if v.get_height() > 0.01 else "" for v in c]
        ax.bar_label(c, labels=labels, label_type='center', color='white', fontsize=9, weight='bold')

    _save_or_show(fig, save_path)

def plot_model_confusion(
    predictions: list[dict],
    ground_truth: list[dict],
    group_name: str,
    method: str,
    labels: list[str],
    save_path: Optional[str | Path] = None,
):
    """Per-label presence/absence confusion matrix for one model.

    ``predictions`` and ``ground_truth``: list of {"note_id", "entities": {label: [...]}}.
    """
    from sklearn.metrics import confusion_matrix

    gt_map = {d["note_id"]: set(d.get("entities", {}).keys()) for d in ground_truth}
    pred_map = {d["note_id"]: set(d.get("entities", {}).keys()) for d in predictions}
    all_ids = sorted(gt_map.keys() | pred_map.keys())

    # Build confusion per label
    n = len(labels)
    cm = np.zeros((n, 2, 2), dtype=int)
    for i, label in enumerate(labels):
        yt = [1 if label in gt_map.get(nid, set()) else 0 for nid in all_ids]
        yp = [1 if label in pred_map.get(nid, set()) else 0 for nid in all_ids]
        cm[i] = confusion_matrix(yt, yp, labels=[0, 1])

    # Aggregate into a single label × {TP, FP, FN, TN} heatmap
    summary = pd.DataFrame({
        "label": labels,
        "TP": cm[:, 1, 1],
        "FP": cm[:, 0, 1],
        "FN": cm[:, 1, 0],
        "TN": cm[:, 0, 0],
    }).set_index("label")

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.45)))
    sns.heatmap(
        summary[["TP", "FP", "FN"]], annot=True, fmt="d", cmap=SCORE_CMAP,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(f"{method.upper()} - {group_name} Confusion (TP/FP/FN)", fontsize=13)
    ax.set_ylabel("")
    _save_or_show(fig, save_path)


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════


def save_results_table(
    results_df: pd.DataFrame,
    save_path: str | Path,
    latex: bool = True,
):
    """Export results as CSV and optionally LaTeX."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = save_path.with_suffix(".csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    if latex:
        # LaTeX - aggregates only
        agg = _get_aggregates(results_df)
        if not agg.empty:
            pivot = agg.pivot_table(
                index="group", columns="method",
                values=["precision", "recall", "f1"],
            ).round(3)
            tex_path = save_path.with_suffix(".tex")
            pivot.to_latex(tex_path, multicolumn=True, multicolumn_format="c")
            print(f"Saved LaTeX: {tex_path}")

# ═══════════════════════════════════════════════════════════════════════════
# CLI EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate comprehensive visualization plots for BRIGHT NER results.")
    parser.add_argument("csv_path", type=str, help="Path to the results CSV file.")
    parser.add_argument("out_dir", type=str, help="Directory to save the generated plots.")
    parser.add_argument("--method_a", type=str, default="gliner", help="Base prefix for method A in comparison plots.")
    parser.add_argument("--method_b", type=str, default="eds", help="Base prefix for method B in comparison plots.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Threshold below which F1 is considered poor.")

    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    out_dir = Path(args.out_dir)
    threshold_val = args.threshold

    if not csv_path.exists():
        print(f"Error: Could not find '{csv_path}'.")
        exit(1)

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Uniformize method names globally (removes '_crt' ending if present)
    # This prevents splitting "gliner" and "gliner_crt" in plots and legends
    df["method"] = df["method"].astype(str).str.replace(r"_crt$", "", regex=True)

    # Auto-detect groups and methods from the dataset
    groups = [g for g in df["group"].unique() if pd.notna(g)]
    methods = [m for m in df["method"].unique() if pd.notna(m)]
    
    metrics = ["precision", "recall", "f1"]
    aggs = ["micro", "macro"]

    # Define subfolders for clean organization (split by barplots / heatmaps)
    overall_bars_dir = out_dir / "overall_comparisons" / "barplots"
    overall_heat_dir = out_dir / "overall_comparisons" / "heatmaps"
    
    versus_bars_dir = out_dir / "versus_comparisons" / "barplots"
    versus_heat_dir = out_dir / "versus_comparisons" / "heatmaps"
    
    per_model_bars_dir = out_dir / "per_model" / "barplots"
    per_model_heat_dir = out_dir / "per_model" / "heatmaps"
    
    # Create directories
    for d in [overall_bars_dir, overall_heat_dir, versus_bars_dir, versus_heat_dir, per_model_bars_dir, per_model_heat_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Overall Comparisons (All metrics and aggregation types)
    print("\n--- Generating Overall Comparisons ---")
    for agg in aggs:
        for metric in metrics:
            plot_overall_comparison(
                df, metric=metric, agg_name=agg, 
                save_path=overall_bars_dir / f"overall_comparison_{agg}_{metric}.png"
            )
            plot_all_groups_heatmap(
                df, metric=metric, agg_name=agg, 
                save_path=overall_heat_dir / f"overall_heatmap_{agg}_{metric}.png"
            )
        
        # Overall Versus Heatmap
        plot_versus_heatmap(
            df, group_name=None, method_a=args.method_a, method_b=args.method_b, agg_name=agg,
            save_path=overall_heat_dir / f"versus_all_groups_heatmap_{agg}.png"
        )

    # 2. Versus Comparisons (Per group)
    print("\n--- Generating Versus Comparisons (per group) ---")
    for group in groups:
        plot_per_feature_comparison(
            df, group_name=group, 
            save_path=versus_bars_dir / f"versus_{group}_barplots.png"
        )
        plot_versus_heatmap(
            df, group_name=group, method_a=args.method_a, method_b=args.method_b,
            save_path=versus_heat_dir / f"versus_{group}_heatmap.png"
        )

    # 3. Per-Model Breakdowns
    print("\n--- Generating Per-Model Breakdowns ---")
    for method in methods:
        plot_model_per_group(
            df, method=method, 
            save_path=per_model_bars_dir / f"{method}_all_groups_barplots.png"
        )
        for group in groups:
            plot_model_per_feature(
                df, method=method, group_name=group,
                save_path=per_model_heat_dir / f"{method}_{group}_heatmap.png"
            )

    # 4. Pass/Poor Threshold Comparisons
    plot_passing_comparison(
        df, method_a=args.method_a, method_b=args.method_b, threshold=threshold_val,
        save_path=out_dir / "overall_comparisons" / "pass_vs_poor_count_comparison.png"
    )

    for method in methods:
        plot_group_pass_proportion(
            df, method=method, threshold=threshold_val,
            save_path=out_dir / "per_model" / f"{method}_group_pass_proportions.png"
        )
        plot_all_labels_performance(df, method, threshold=threshold_val, save_path=out_dir / "per_model" / f"{method}_all_labels.png")

    print(f"\n✅ All visualizations successfully generated in '{out_dir.absolute()}'")