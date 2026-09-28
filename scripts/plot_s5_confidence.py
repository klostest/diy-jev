"""S5 figure: Jev's `confidence` field against its own top probability, one
dot per email, with the diagonal where the two would be identical and the
rule that fits every `choice` answer recorded in this project (16,283 answers
with 2-6 options, all within about 0.02; see scripts/check_confidence_formula.py):

    confidence = (top probability - 1/K) / (1 - 1/K),   K = number of options

i.e. the top probability rescaled so a uniform guess maps to 0 and certainty
to 1. (`score`-type answers use some other, undetermined rule.)

    uv run python scripts/plot_s5_confidence.py

Reads outputs/probe_results.json; writes outputs/figures/s5_confidence.png.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("outputs/probe_results.json")
OUT = Path("outputs/figures/s5_confidence.png")

# Reference palette (dataviz skill, light mode).
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
DOT = "#2a78d6"   # categorical slot 1
REF = "#8c8b86"


def main():
    records = json.load(open(RESULTS))["S5"]["records"]
    conf = np.array([r["answer"]["confidence"] for r in records])
    top = np.array([max(r["answer"]["probabilities"].values()) for r in records])
    corr = float(np.corrcoef(conf, top)[0, 1])
    pairs = list(itertools.combinations(range(len(conf)), 2))
    discordant = sum((conf[i] - conf[j]) * (top[i] - top[j]) < 0 for i, j in pairs)
    print(f"n={len(conf)}  correlation {corr:.4f}  discordant pairs {discordant}/{len(pairs)}")
    print(f"mean (confidence - top probability): {np.mean(conf - top):+.3f}")

    fig, ax = plt.subplots(figsize=(5.6, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([0.25, 1], [0.25, 1], color=REF, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.37, 0.405, "confidence = top probability", rotation=45, rotation_mode="anchor",
            color=TEXT_SECONDARY, fontsize=8)
    xs = np.linspace(0.25, 1, 50)
    ax.plot(xs, (xs - 0.25) / 0.75, color=TEXT_PRIMARY, linewidth=1, zorder=2)
    ax.text(0.62, 0.40, "(top − ¼) / (1 − ¼)", color=TEXT_PRIMARY, fontsize=8.5, ha="left")
    ax.scatter(top, conf, s=24, color=DOT, alpha=0.6, edgecolor=SURFACE, linewidth=0.6, zorder=3)

    ax.set_xlim(0.25, 1.02)
    ax.set_ylim(0.25, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("Jev's top probability", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Jev's `confidence` field", color=TEXT_SECONDARY, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)

    fig.suptitle("Jev's confidence is just its top probability, rescaled",
                 x=0.03, ha="left", fontsize=11, color=TEXT_PRIMARY, y=0.985)
    fig.text(0.03, 0.93, "S5: 200 emails, one dot each. Every dot sits on the line (top − ¼) / (1 − ¼): the top\n"
             "probability rescaled so a four-way guess (0.25) reads 0. It carries no information the\n"
             f"probabilities don't already have (the two order the emails identically in all but {discordant}\n"
             f"of {len(pairs):,} pairs).",
             fontsize=8.5, color=TEXT_SECONDARY, ha="left", va="top", linespacing=1.4)
    fig.subplots_adjust(top=0.77, left=0.13, right=0.97, bottom=0.10)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
