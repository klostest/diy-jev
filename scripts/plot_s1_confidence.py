"""S1 figures: on clear-cut emails, how confident Jev was, and when.

Two figures, each two small panels sharing one confidence axis (number of
emails at each confidence level). The confidence plotted is Jev's own
`confidence` field -- the number a user sees.

  s1_confidence.png          right answers (top) vs. wrong answers (bottom)
  s1_confidence_by_type.png  all other emails (top) vs. feature requests
                             (bottom), right/wrong stacked within each bar

    uv run python scripts/plot_s1_confidence.py

Reads outputs/probe_results.json; writes both PNGs to outputs/figures/.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("outputs/probe_results.json")
OUT_DIR = Path("outputs/figures")

# Reference palette (dataviz skill, light mode): categorical slots 1-2 plus text/surface roles.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
RIGHT = "#2a78d6"   # slot 1, blue
WRONG = "#eb6834"   # slot 2, orange

FEATURE_REQUEST_MARKERS = ["dark mode", "CSV export", "keyboard shortcuts", "bulk-edit"]


BINS = np.arange(0.0, 1.0001, 0.05)


def _is_feature_request(r) -> bool:
    return any(k in r["text"] for k in FEATURE_REQUEST_MARKERS)


def _is_right(r) -> bool:
    return r["answer"]["choice"] == r["true_label"]


def _conf(rs) -> np.ndarray:
    return np.array([r["answer"]["confidence"] for r in rs])


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    ax.set_ylabel("emails", color=TEXT_SECONDARY, fontsize=8)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=4))


def _median_marker(ax, values, top):
    med = float(np.median(values))
    ax.axvline(med, ymax=0.97, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (3, 3)))
    at_right_edge = med > 0.9
    ax.text(med - 0.01 if at_right_edge else med + 0.01, top * 0.97, f"median {med:.2f}",
            color=TEXT_SECONDARY, fontsize=8, va="top", ha="right" if at_right_edge else "left")


def _new_figure():
    fig, axes = plt.subplots(
        2, 1, figsize=(7.2, 4.6), sharex=True, dpi=200,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.35})
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _finish(fig, ax_bottom, title, subtitle, path):
    n_lines = subtitle.count("\n") + 1
    ax_bottom.set_xlim(0.25, 1.0)  # a four-label answer can't be below 0.25 confident
    ax_bottom.set_xlabel("Jev's reported confidence", color=TEXT_SECONDARY, fontsize=9)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11.5, color=TEXT_PRIMARY, y=0.99)
    fig.text(0.02, 0.935, subtitle, fontsize=8.5, color=TEXT_SECONDARY, ha="left", va="top",
             linespacing=1.4)
    fig.subplots_adjust(top=0.81 - 0.045 * (n_lines - 1), left=0.09, right=0.98, bottom=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {path}")


def plot_right_wrong(records):
    right = [r for r in records if _is_right(r)]
    wrong = [r for r in records if not _is_right(r)]
    all_wrong_fr = all(_is_feature_request(r) for r in wrong)
    fig, (ax_r, ax_w) = _new_figure()
    panels = [
        (ax_r, _conf(right), RIGHT, f"Right: {len(right)} of {len(records)} emails"),
        (ax_w, _conf(wrong), WRONG, f"Wrong: {len(wrong)} of {len(records)}"
                                    + (" — all feature requests" if all_wrong_fr else "")),
    ]
    for ax, values, color, label in panels:
        counts, edges = np.histogram(values, bins=BINS)
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=color,
               edgecolor=SURFACE, linewidth=1.5)
        ax.set_title(label, loc="left", fontsize=10, color=TEXT_PRIMARY, pad=4)
        top = counts.max() * 1.3  # headroom so the median label sits above the bars
        ax.set_ylim(0, top)
        _median_marker(ax, values, top)
        _style_axis(ax)
    _finish(fig, ax_w,
            "On clear-cut emails, Jev was usually right, and more confident when it was",
            "S1: 200 emails whose exact answer is at least 97% one category. "
            "Note the panels' different vertical scales.",
            OUT_DIR / "s1_confidence.png")


def plot_by_email_type(records):
    groups = [
        ("All other emails", [r for r in records if not _is_feature_request(r)]),
        ("Feature requests", [r for r in records if _is_feature_request(r)]),
    ]
    fig, axes = _new_figure()
    for (name, rs), ax in zip(groups, axes):
        right = _conf([r for r in rs if _is_right(r)])
        wrong = _conf([r for r in rs if not _is_right(r)])
        c_right, edges = np.histogram(right, bins=BINS)
        c_wrong, _ = np.histogram(wrong, bins=BINS)
        width = np.diff(edges)
        ax.bar(edges[:-1], c_right, width=width, align="edge", color=RIGHT,
               edgecolor=SURFACE, linewidth=1.5, label="right")
        ax.bar(edges[:-1], c_wrong, width=width, align="edge", bottom=c_right, color=WRONG,
               edgecolor=SURFACE, linewidth=1.5, label="wrong")
        ax.set_title(f"{name}: {len(rs)} emails — {len(right)} right, {len(wrong)} wrong",
                     loc="left", fontsize=10, color=TEXT_PRIMARY, pad=4)
        top = (c_right + c_wrong).max() * 1.3
        ax.set_ylim(0, top)
        _median_marker(ax, _conf(rs), top)
        _style_axis(ax)
    leg = axes[0].legend(loc="upper center", frameon=False, fontsize=8, handlelength=1.2,
                         ncols=2, bbox_to_anchor=(0.5, 0.97))
    for t in leg.get_texts():
        t.set_color(TEXT_SECONDARY)
    _finish(fig, axes[1],
            "Jev was unsure exactly where the label was ambiguous",
            "S1: 200 clear-cut emails. Our tables call feature requests sales; Jev leaned technical.\n"
            "Note the panels' different vertical scales.",
            OUT_DIR / "s1_confidence_by_type.png")


def main():
    records = json.load(open(RESULTS))["S1"]["records"]
    right = [r for r in records if _is_right(r)]
    wrong = [r for r in records if not _is_right(r)]
    gap = [abs(r["answer"]["confidence"] - max(r["answer"]["probabilities"].values())) for r in records]
    for label, rs in [("right", right), ("wrong", wrong)]:
        c = _conf(rs)
        print(f"{label}: {len(rs)}  median confidence {st.median(c):.2f}  range {c.min():.2f}-{c.max():.2f}")
    print(f"all wrong answers are feature requests: {all(_is_feature_request(r) for r in wrong)}")
    print(f"|confidence - top probability|: mean {st.mean(gap):.3f}, max {max(gap):.2f}")
    for label, rs in [("other emails", [r for r in records if not _is_feature_request(r)]),
                      ("feature requests", [r for r in records if _is_feature_request(r)])]:
        for kind, sub in [("right", [r for r in rs if _is_right(r)]), ("wrong", [r for r in rs if not _is_right(r)])]:
            if sub:
                c = _conf(sub)
                print(f"{label:16s} {kind}: {len(sub):3d}  confidence {c.min():.2f}-{c.max():.2f}, "
                      f"{int((c >= 0.95).sum())} at >= 0.95")
    plot_right_wrong(records)
    plot_by_email_type(records)


if __name__ == "__main__":
    main()
