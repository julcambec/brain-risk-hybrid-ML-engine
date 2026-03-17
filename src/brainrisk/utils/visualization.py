"""
Shared plotting utilities for the brainrisk pipeline.

Provides matplotlib-based visualization functions for:
- HYDRA clustering model selection (ARI across k)
- Baseline model performance comparison
- Feature importance bar charts
- Longitudinal CBCL reliable-change proportions

All functions can read aggregate data from ``assets/data/*.json`` files
(storing real-results statistics) or accept data directly for use on
synthetic outputs.

Brain-atlas surface plots (e.g. regional Cohen's d maps) require the
optional ``ggseg`` package. Install with ``pip install ggseg`` if needed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------
# Helpers
# ---------


def _load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file and return its contents as a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _apply_style(ax: plt.Axes) -> None:
    """Apply a clean, publication-ready style to an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


# ------------
# Public API
# ------------


def plot_ari_across_k(
    k_values: list[int] | None = None,
    ari_values: list[float] | None = None,
    best_k: int | None = None,
    json_path: str | Path | None = None,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """
    Plot Adjusted Rand Index across HYDRA clustering solutions.

    Parameters
    ----------
    k_values, ari_values, best_k : optional
        Data to plot. If ``None``, loaded from *json_path*.
    json_path : str | Path | None
        Path to ``ari_across_k.json``. Defaults to
        ``assets/data/ari_across_k.json``.
    ax : plt.Axes | None
        Matplotlib axes to draw on. Created if ``None``.
    save_path : str | Path | None
        If provided, save the figure to this path.

    Returns
    -------
    plt.Axes
        The axes with the plot drawn.
    """
    if k_values is None or ari_values is None:
        data = _load_json(json_path or "assets/data/ari_across_k.json")
        k_vals: list[int] = data["k_values"]
        ari_vals: list[float] = data["ari_values"]
        if best_k is None:
            best_k = data.get("best_k")
    else:
        k_vals = k_values
        ari_vals = ari_values

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.plot(k_vals, ari_vals, "o-", color="#2c7fb8", linewidth=2, markersize=7)

    if best_k is not None and best_k in k_vals:
        idx = k_vals.index(best_k)
        ax.plot(
            best_k,
            ari_vals[idx],
            "o",
            color="#d95f02",
            markersize=12,
            zorder=5,
            label=f"Selected k={best_k}",
        )
        ax.legend(frameon=False)

    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_title("HYDRA Clustering: ARI Across k")
    ax.set_xticks(k_vals)
    _apply_style(ax)

    if save_path:
        ax.figure.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved ARI plot → %s", save_path)

    return ax


def plot_baseline_comparison(
    results_df: pd.DataFrame | None = None,
    json_path: str | Path | None = None,
    metric: str = "f1_macro",
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """
    Plot grouped bar chart comparing baseline model performance.

    Can accept a results DataFrame from the baselines module or load
    from the ``baseline_metrics.json`` file for real-results visualization.

    Parameters
    ----------
    results_df : pd.DataFrame | None
        DataFrame with columns ``task``, ``model_type``, and a metric
        column matching *metric*. If ``None``, a summary is built from
        the JSON file.
    json_path : str | Path | None
        Path to ``baseline_metrics.json``.
    metric : str
        Metric column to plot (default ``"f1_macro"``).
    ax : plt.Axes | None
        Matplotlib axes. Created if ``None``.
    save_path : str | Path | None
        If provided, save the figure.

    Returns
    -------
    plt.Axes
    """
    if results_df is None:
        data = _load_json(json_path or "assets/data/baseline_metrics.json")
        rows = []
        for _key, info in data.get("classification", {}).items():
            if info.get("best_f1_macro") is not None:
                rows.append(
                    {
                        "task": info["task"],
                        "model_type": "tuned",
                        "f1_macro": info["best_f1_macro"],
                    }
                )
            if info.get("dummy_f1_macro") is not None:
                rows.append(
                    {
                        "task": info["task"],
                        "model_type": "dummy",
                        "f1_macro": info["dummy_f1_macro"],
                    }
                )
        results_df = pd.DataFrame(rows)

    if results_df.empty:
        logger.warning("No data to plot for baseline comparison.")
        if ax is None:
            _, ax = plt.subplots()
        return ax

    tasks = results_df["task"].unique()
    model_types = results_df["model_type"].unique()
    x = np.arange(len(tasks))
    width = 0.8 / len(model_types)

    colors = {
        "dummy": "#bdbdbd",
        "tuned": "#2c7fb8",
        "hist_boosting": "#2c7fb8",
        "logistic": "#d95f02",
        "ridge": "#d95f02",
    }

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    for i, mt in enumerate(model_types):
        subset = results_df[results_df["model_type"] == mt]
        values = [
            float(subset.loc[subset["task"] == t, metric].iloc[0])
            if t in subset["task"].values
            else 0.0
            for t in tasks
        ]
        ax.bar(
            x + i * width,
            values,
            width,
            label=mt,
            color=colors.get(mt, "#7570b3"),
        )

    ax.set_xticks(x + width * (len(model_types) - 1) / 2)
    ax.set_xticklabels(tasks, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("ROI Baseline Model Comparison")
    ax.legend(frameon=False)
    _apply_style(ax)

    if save_path:
        ax.figure.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved baseline comparison → %s", save_path)

    return ax


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """
    Plot horizontal bar chart of top feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Output of :func:`brainrisk.baselines.interpretation.extract_feature_importance`.
        Must have ``feature`` and ``abs_importance`` columns.
    top_n : int
        Number of top features to display.
    ax : plt.Axes | None
        Matplotlib axes. Created if ``None``.
    save_path : str | Path | None
        If provided, save the figure.

    Returns
    -------
    plt.Axes
    """
    top = importance_df.head(top_n).iloc[::-1]  # reverse for horizontal bars

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))

    colors = np.where(top["importance"] >= 0, "#d95f02", "#2c7fb8")
    ax.barh(top["feature"], top["abs_importance"], color=colors)
    ax.set_xlabel("Absolute Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    _apply_style(ax)

    if save_path:
        ax.figure.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved feature importance plot → %s", save_path)

    return ax


def plot_rci_proportions(
    json_path: str | Path | None = None,
    measure: str = "internalizing",
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """
    Plot stacked bar chart of Reliable Change Index proportions.

    Parameters
    ----------
    json_path : str | Path | None
        Path to ``longitudinal_cbcl.json``.
    measure : str
        ``"internalizing"`` or ``"externalizing"``.
    ax : plt.Axes | None
        Matplotlib axes. Created if ``None``.
    save_path : str | Path | None
        If provided, save the figure.

    Returns
    -------
    plt.Axes
    """
    data = _load_json(json_path or "assets/data/longitudinal_cbcl.json")
    groups = data["groups"]
    m = data[measure]

    worsened = np.array(m["worsened"])
    no_change = np.array(m["no_change"])
    improved = np.array(m["improved"])

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(groups))
    width = 0.6

    ax.bar(x, improved, width, label="Improved", color="#2ca02c")
    ax.bar(x, no_change, width, bottom=improved, label="No Change", color="#bdbdbd")
    ax.bar(x, worsened, width, bottom=improved + no_change, label="Worsened", color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Proportion (%)")
    ax.set_title(f"Reliable Change in CBCL {measure.title()} Scores by Subtype")
    ax.legend(frameon=False, loc="upper right")
    _apply_style(ax)

    if save_path:
        ax.figure.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved RCI proportions plot → %s", save_path)

    return ax
