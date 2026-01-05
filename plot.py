import json
import math
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def load_by_count(json_path: Path) -> dict:
    """Load by_count dict from JSON.
    Accepts either {"by_count": {...}} or {...} where {...} is already by_count.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "by_count" in data and isinstance(data["by_count"], dict):
        by_count = data["by_count"]
    elif isinstance(data, dict):
        by_count = data
    else:
        raise ValueError(f"Unexpected JSON structure in {json_path}")

    # Normalize keys to int where possible (JSON keys are often strings)
    norm = {}
    for k, v in by_count.items():
        try:
            kk = int(k)
        except (ValueError, TypeError):
            kk = k
        norm[kk] = v
    return norm


def compute_curve(by_count: dict):
    """Return sorted x (count) and y (accuracy) arrays."""
    x_axis = sorted([k for k in by_count.keys() if isinstance(k, int)])
    y_axis = []
    for k in x_axis:
        entry = by_count.get(k, {})
        denom = float(entry.get("reference_len_sum", 0.0) or 0.0)
        numer = float(entry.get("correct_sum", 0.0) or 0.0)
        y_axis.append((numer / denom) if denom > 0 else 0.0)
    return x_axis, y_axis


def paper_style_blue():
    """Paper-like styling with blue base."""
    mpl.rcParams.update({
        # Typography
        "font.family": "DejaVu Serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # Axes look
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,

        # Grid
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "grid.linewidth": 0.7,

        # Figure
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,

        # Lines
        "lines.linewidth": 2.2,
        "lines.markersize": 5.5,
    })


def nice_limits(x_list):
    if not x_list:
        return 0, 1
    xmin, xmax = min(x_list), max(x_list)
    if xmin == xmax:
        return xmin - 1, xmax + 1
    pad = max(1, int(math.ceil((xmax - xmin) * 0.03)))
    return xmin - pad, xmax + pad


def main():
    parser = argparse.ArgumentParser(
        description="Overlay Accuracy vs Count curves from a.json, b.json, c.json (paper-style, blue-based)."
    )
    parser.add_argument("--a", default="clevr_subset1_result_nano.json", help="Path to a.json")
    parser.add_argument("--b", default="clevr_subset1_result_iou_num.json", help="Path to b.json")
    parser.add_argument("--c", default="clevr_subset1_result_iou_num_dup.json", help="Path to c.json")
    parser.add_argument("--out_png", default="accuracy_vs_count_overlay.png", help="Output PNG path")
    parser.add_argument("--out_pdf", default="accuracy_vs_count_overlay.pdf", help="Output PDF path")
    parser.add_argument("--title", default="Accuracy vs Count", help="Figure title")
    args = parser.parse_args()

    paper_style_blue()

    paths = {
        "A": Path(args.a),
        "B": Path(args.b),
        "C": Path(args.c),
    }

    # Blue-based palette (distinct but coherent)
    colors = {
        "A": "#0B3C5D",  # deep blue
        "B": "#1D6FA3",  # medium blue
        "C": "#4FA3D1",  # light blue
    }

    curves = {}
    all_x = []
    for label, p in paths.items():
        by_count = load_by_count(p)
        x, y = compute_curve(by_count)
        curves[label] = (x, y)
        all_x.extend(x)

    fig = plt.figure(figsize=(7.2, 4.8))
    ax = plt.gca()

    # Plot curves
    LEGEND = {
    "A": "Baseline",
    "B": "RL w/ Unused-Ratio Penalty",
    "C": "RL w/ Unused-Ratio + Duplicate Penalty"
    }
    for label in ["A", "B", "C"]:
        x, y = curves[label]
        ax.plot(
            x, y,
            marker="o",
            linestyle="-",
            color=colors[label],
            label=f"{LEGEND[label]}",
            zorder=3
        )

    # Axes labels
    ax.set_xlabel("Difference Count")
    ax.set_ylabel("Accuracy")

    # Title
    ax.set_title(args.title)

    # Limits & ticks
    xmin, xmax = nice_limits(all_x)
    ax.set_xlim(2.5, 9.5)
    ax.set_ylim(0.0, 0.2)

    # Subtle baseline at y=0.5 (often useful for paper)
    ax.axhline(0.5, linewidth=0.9, alpha=0.25, color=colors["B"], zorder=1)

    # Legend (paper-ish: compact, no frame)
    leg = ax.legend(loc="best", frameon=False, handlelength=2.6)

    # Minor aesthetic: lighter left/bottom spines
    ax.spines["left"].set_alpha(0.7)
    ax.spines["bottom"].set_alpha(0.7)

    # Grid behind lines
    ax.set_axisbelow(True)

    plt.tight_layout()

    out_png = Path(args.out_png)
    out_pdf = Path(args.out_pdf)
    fig.savefig(out_png)
    #fig.savefig(out_pdf)

    print(f"Saved: {out_png.resolve()}")
    #print(f"Saved: {out_pdf.resolve()}")
    print("Done (overlay: a/b/c).")


if __name__ == "__main__":
    main()