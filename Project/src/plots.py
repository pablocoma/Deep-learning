"""Simple plotting helpers for experiment outputs."""

import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
matplotlib_cache_dir = project_root / "outputs" / "matplotlib"
font_cache_dir = project_root / "outputs" / "cache"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
font_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(font_cache_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import ensure_dir


def plot_dropout_metric(rows, metric_name, ylabel, output_path):
    """Plot one metric against Hybrid CBM side-channel dropout."""
    dropouts = [row["side_dropout"] for row in rows]
    values = [row[metric_name] for row in rows]

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plt.figure(figsize=(6, 4))
    plt.plot(dropouts, values, marker="o")
    plt.xlabel("Side-channel dropout")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs side-channel dropout")
    plt.xticks(dropouts)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_concept_influence_ranking(
    rows,
    output_path,
    metric_name="avg_abs_change_original_class_prob",
    ylabel="Average absolute probability change",
):
    """Plot a horizontal bar chart for the concept influence ranking."""
    sorted_rows = sorted(rows, key=lambda row: row[metric_name])
    concept_names = [row["concept_name"] for row in sorted_rows]
    values = [row[metric_name] for row in sorted_rows]

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plt.figure(figsize=(7, 4.5))
    plt.barh(concept_names, values)
    plt.xlabel(ylabel)
    plt.ylabel("Concept")
    plt.title("Concept intervention influence")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
