"""Plotting: reliability diagrams and risk-coverage curves, theme-agnostic
(light background, works in a plain terminal-opened PNG). Kept deliberately
small -- this project's real output is numbers with confidence intervals,
not dashboards.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jev.eval import metrics as m

CATEGORY_COLOR = "#3B6FA0"
REFERENCE_COLOR = "#9AA5B1"
GAP_COLOR = "#C0392B"


def reliability_diagram(
    confidences: Sequence[float], correctness: Sequence[float],
    n_bins: int = 10, method: str = "equal_mass", title: str = "Reliability diagram",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    stats = m.compute_bin_stats(confidences, correctness, n_bins=n_bins, method=method)
    nonempty = stats.counts > 0

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
    else:
        fig = ax.figure

    ax.plot([0, 1], [0, 1], linestyle="--", color=REFERENCE_COLOR, linewidth=1, label="perfect calibration")
    ax.bar(
        stats.mean_confidence[nonempty], stats.mean_outcome[nonempty],
        width=0.04, color=CATEGORY_COLOR, alpha=0.85, edgecolor="white", label="observed",
    )
    for c, o in zip(stats.mean_confidence[nonempty], stats.mean_outcome[nonempty]):
        ax.plot([c, c], [c, o], color=GAP_COLOR, linewidth=1.5, alpha=0.7)

    ece = m.expected_calibration_error(confidences, correctness, n_bins=n_bins, method=method)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{title}\nECE={ece:.3f}")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def risk_coverage_plot(
    curves: Dict[str, tuple],  # name -> (coverage, risk)
    title: str = "Risk-coverage",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
    else:
        fig = ax.figure
    for name, (coverage, risk) in curves.items():
        aurc = m.area_under_risk_coverage(coverage, risk)
        ax.plot(coverage, risk, label=f"{name} (AURC={aurc:.3f})")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def save_fig(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
