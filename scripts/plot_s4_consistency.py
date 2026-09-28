"""S4 figure: the same priority question asked three ways (yes/no, choice,
score) in one request per email -- do the answers agree?

Each answer is converted to the same quantity, P(high or critical):
  yes/no  Jev's probability of "yes, expedite (high or critical)"
  choice  P(high) + P(critical) from its four-option distribution
  score   P(high) + P(critical) from its per-position distribution
and averaged over the emails containing each urgency phrase. The chart is
about whether the three formats agree with each other, so no "right answer"
line is drawn: priority here follows the generator's hidden rule, which Jev
had no way to know (see RESULTS_AND_DISCUSSION.md, S3).

    uv run python scripts/plot_s4_consistency.py

Reads outputs/probe_results.json; writes outputs/figures/s4_consistency.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jev.dgp.slots import INTENSITY_MARKER

RESULTS = Path("outputs/probe_results.json")
OUT = Path("outputs/figures/s4_consistency.png")

# Reference palette (dataviz skill, light mode): categorical slots 1-3, validated all-pairs.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
FORMATS = [("yes/no: \"Should this ticket be expedited?\"", "noul", "#2a78d6"),
           ("choice: pick low / medium / high / critical", "choice", "#eb6834"),
           ("score: rate on the low → critical scale", "score", "#1baf7a")]
ROWS = [("mild", '"no particular rush"'), ("none", "no urgency phrase"),
        ("strong", '"this is fairly urgent"'), ("extreme", '"URGENT -- please treat as a priority"')]


def urgency_phrase(text: str) -> str:
    for level in ["extreme", "strong", "mild"]:
        if any(p and p.lower() in text.lower() for p in INTENSITY_MARKER.phrases[level]):
            return level
    return "none"


def p_high_or_critical(record, fmt: str) -> float:
    if fmt == "noul":
        return float(record["noul"]["noul"])
    answer = record[fmt]
    probs = answer["probabilities"]
    if fmt == "score":  # positions keyed "0".."3"; map back to level names via the legend
        by_name = {answer["legend"][k]: v for k, v in probs.items()}
        return by_name.get("high", 0.0) + by_name.get("critical", 0.0)
    return probs.get("high", 0.0) + probs.get("critical", 0.0)


def main():
    records = json.load(open(RESULTS))["S4"]["records"]
    phrase = np.array([urgency_phrase(r["text"]) for r in records])
    values = {fmt: np.array([p_high_or_critical(r, fmt) for r in records]) for _, fmt, _ in FORMATS}

    flips = int(((values["noul"] >= 0.5) != (values["choice"] >= 0.5)).sum())
    gap = np.abs(values["noul"] - values["choice"])
    print(f"emails where an expedite-at-0.5 decision flips between yes/no and choice: {flips}/{len(records)}")
    print(f"mean |yes/no - choice| {gap.mean():.3f}; largest {gap.max():.2f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for y, (level, label) in enumerate(ROWS):
        mask = phrase == level
        means = [values[fmt][mask].mean() for _, fmt, _ in FORMATS]
        ax.plot([min(means), max(means)], [y, y], color=GRID, linewidth=3, zorder=2,
                solid_capstyle="round")
        for (name, fmt, color), mval in zip(FORMATS, means):
            ax.scatter([mval], [y], s=70, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        print(f"{level:8s} n={mask.sum():3d}  " + "  ".join(
            f"{fmt} {mv:.2f}" for (_, fmt, _), mv in zip(FORMATS, means)))

    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([f"{label}  ({int((phrase == level).sum())} emails)" for level, label in ROWS],
                       fontsize=8, color=TEXT_PRIMARY)
    ax.set_ylim(-0.6, len(ROWS) - 0.1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Jev's average P(high or critical priority)", color=TEXT_SECONDARY, fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=TEXT_SECONDARY, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)

    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=7, color=color, label=name)
               for name, _, color in FORMATS]
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.0), frameon=False,
                     fontsize=8, ncols=1, title="Question format (all three asked in one request per email)",
                     title_fontsize=8, alignment="left")
    leg.get_title().set_color(TEXT_PRIMARY)
    for t in leg.get_texts():
        t.set_color(TEXT_SECONDARY)

    fig.suptitle("Same email, same request, three question formats: answers split on urgent emails",
                 x=0.02, ha="left", fontsize=11, color=TEXT_PRIMARY, y=0.985)
    fig.text(0.02, 0.93, f"S4: 200 emails. On calm emails the formats agree; on urgent ones yes/no is far "
             f"more cautious.\nAn expedite-if-above-0.5 decision flips on {flips} of 200 emails depending "
             f"only on the question format.",
             fontsize=8.5, color=TEXT_SECONDARY, ha="left", va="top", linespacing=1.4)
    fig.subplots_adjust(top=0.78, left=0.40, right=0.97, bottom=0.33)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
