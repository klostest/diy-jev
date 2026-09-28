"""S2 figure: Jev's reported P(billing) against the exact P(billing), one dot
per email, for the four ambiguity levels of suite S2 (50 emails each).

A perfectly calibrated model would put every dot on the diagonal. Dots are
spread sideways within each level only so identical values don't hide each
other; the horizontal position within a column carries no meaning.

Two versions:
  s2_calibration.png             every dot the same color
  s2_calibration_by_wording.png  the 0.533 column colored by which of its three
                                 paraphrases the email used (the other columns'
                                 wording made no difference, so they stay gray)

    uv run python scripts/plot_s2_calibration.py

Reads outputs/probe_results.json; writes both PNGs to outputs/figures/.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

from jev.dgp.slots import COMPLAINT_PHRASE

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("outputs/probe_results.json")
OUT_DIR = Path("outputs/figures")

# Reference palette (dataviz skill, light mode).
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
DOT = "#2a78d6"     # categorical slot 1, blue: Jev's answers
REF = "#8c8b86"     # neutral: the perfect-calibration reference
NEUTRAL_DOT = "#a8a7a1"  # dots whose wording isn't being distinguished
WORDING_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)

# Where each level's example sentence is written: (x, y, horizontal alignment).
LEVEL_NOTES = {
    0.345: ("vague email,\nno topic clues", 0.345, 0.50, "center"),
    0.533: ('"might be a charge,\nmight be a feature"', 0.533, 0.81, "center"),
    0.714: ('"I think I\'m being charged\nincorrectly, though it\nmight be tied to a feature…"', 0.765, 0.875, "center"),
    1.0: ('"My invoice amount\nlooks wrong"', 1.04, 0.70, "right"),
}


MIXED_LEVEL = 0.533  # the column colored by wording


def _wording_index(text: str) -> int | None:
    for i, phrase in enumerate(COMPLAINT_PHRASE.phrases["mixed_lean"]):
        if phrase.rstrip(".") in text:
            return i
    return None


def draw(records, by_wording: bool, path: Path, blog: bool = False):
    """blog=True: no title/subtitle and no in-plot phrase notes (the post captions it)."""
    levels = sorted({round(r["achieved_p"], 3) for r in records})
    by_level = {lv: [r for r in records if round(r["achieved_p"], 3) == lv] for lv in levels}

    rng = np.random.default_rng(0)  # jitter only; fixed so the figure is reproducible
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot([0, 1], [0, 1], color=REF, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.20, 0.235, "perfectly calibrated", rotation=38, rotation_mode="anchor",
            color=TEXT_SECONDARY, fontsize=8)

    wording_handles = []
    for lv, rs in by_level.items():
        v = np.array([r["answer"]["probabilities"].get("billing", 0.0) for r in rs])
        x = lv + rng.uniform(-0.035, 0.035, size=len(v))
        if by_wording and lv == MIXED_LEVEL:
            idx = np.array([_wording_index(r["text"]) for r in rs])
            order = sorted(set(idx), key=lambda k: v[idx == k].mean())
            for color, k in zip(WORDING_COLORS, order):
                m = idx == k
                ax.scatter(x[m], v[m], s=22, color=color, alpha=0.8, edgecolor=SURFACE,
                           linewidth=0.6, zorder=3)
                phrase = COMPLAINT_PHRASE.phrases["mixed_lean"][k]
                wording_handles.append(plt.Line2D(
                    [], [], marker="o", linestyle="none", markersize=6, color=color,
                    label=textwrap.fill(f'mean {v[m].mean():.2f} ({m.sum()} emails): "{phrase}"',
                                        width=100, subsequent_indent="    ")))
        else:
            ax.scatter(x, v, s=22, color=NEUTRAL_DOT if by_wording else DOT, alpha=0.55,
                       edgecolor=SURFACE, linewidth=0.6, zorder=3)
        ax.plot([lv - 0.05, lv + 0.05], [v.mean()] * 2, color=TEXT_PRIMARY, linewidth=2, zorder=4)
        ax.plot([lv], [lv], marker="D", markersize=5, color=SURFACE, markeredgecolor=TEXT_PRIMARY,
                markeredgewidth=1.2, zorder=5)
        note = None if blog else LEVEL_NOTES.get(lv)
        if by_wording and lv == MIXED_LEVEL:
            note = None  # its wordings are named in the legend below the plot
        if by_wording and lv == 0.714:
            note = ("all three wordings,\nincluding \"…but it could also\nbe a glitch on your end\",\n"
                    "got 0.99–1.00", 0.765, 0.86, "center")
        if note:
            text, nx, ny, ha = note
            ax.text(nx, ny, text, ha=ha, va="center", fontsize=7.5, color=TEXT_SECONDARY,
                    linespacing=1.25, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none"))

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=6,
                   color=NEUTRAL_DOT if by_wording else DOT, alpha=0.7, label="Jev's answer, one per email"),
        plt.Line2D([], [], color=TEXT_PRIMARY, linewidth=2, label="Jev's average at that level"),
        plt.Line2D([], [], marker="D", linestyle="none", markersize=5, color=SURFACE,
                   markeredgecolor=TEXT_PRIMARY, markeredgewidth=1.2, label="exact answer"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(TEXT_SECONDARY)
    if wording_handles:
        ax.add_artist(leg)
        leg2 = fig.legend(handles=wording_handles, loc="lower left", bbox_to_anchor=(0.02, 0.005),
                          frameon=False, fontsize=7.5, handlelength=1.0,
                          title=f"Wordings at exact P(billing) = {MIXED_LEVEL}", title_fontsize=8,
                          alignment="left")
        leg2.get_title().set_color(TEXT_PRIMARY)
        for t in leg2.get_texts():
            t.set_color(TEXT_SECONDARY)

    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Exact P(billing), from the generator", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Jev's reported P(billing)", color=TEXT_SECONDARY, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)

    if by_wording:
        title = "Same meaning, different words: Jev's answer followed the wording"
        subtitle = ("S2: the three wordings of the balanced sentence got very different answers; "
                    "the hedged 0.714 sentence got\n0.99–1.00 in every wording. "
                    "Dots are spread sideways only to avoid overlap.")
    else:
        title = "Jev pushes in-between answers to 0 or 1"
        subtitle = ("S2: 50 emails at each of four exact values of P(billing). "
                    "Dots are spread sideways only to avoid overlap.")
    if blog:
        fig.set_size_inches(7.2, 5.0)
        fig.subplots_adjust(top=0.97, left=0.10, right=0.98, bottom=0.11)
    else:
        fig.suptitle(title, x=0.02, ha="left", fontsize=11.5, color=TEXT_PRIMARY, y=0.985)
        fig.text(0.02, 0.935, subtitle, fontsize=8.5, color=TEXT_SECONDARY, ha="left", va="top",
                 linespacing=1.4)
    if blog:
        pass
    elif by_wording:
        fig.set_size_inches(7.2, 6.6)
        fig.subplots_adjust(top=0.86, left=0.10, right=0.98, bottom=0.24)
    else:
        fig.subplots_adjust(top=0.88, left=0.10, right=0.98, bottom=0.10)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {path}")


def main():
    records = json.load(open(RESULTS))["S2"]["records"]
    for lv in sorted({round(r["achieved_p"], 3) for r in records}):
        v = np.array([r["answer"]["probabilities"].get("billing", 0.0) for r in records
                      if round(r["achieved_p"], 3) == lv])
        print(f"exact {lv:.3f}: n={len(v)}  Jev mean {v.mean():.3f}  "
              f"range {v.min():.2f}-{v.max():.2f}  at >=0.99: {(v >= 0.99).sum()}  at 0: {(v == 0).sum()}")
    draw(records, by_wording=False, path=OUT_DIR / "s2_calibration.png")
    draw(records, by_wording=True, path=OUT_DIR / "s2_calibration_by_wording.png")
    draw(records, by_wording=False, path=OUT_DIR / "s2_calibration_blog.png", blog=True)


if __name__ == "__main__":
    main()
