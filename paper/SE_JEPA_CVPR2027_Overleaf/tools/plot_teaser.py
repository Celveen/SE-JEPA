"""Draw the lower panel of Figure 1 from its editable, planner-labeled success rates."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory


COLORS = {
    "Frozen": "#BB7429",
    "AdaJEPA": "#C1544A",
    "SE-JEPA": "#104B82",
    "DINO-WM": "#27829E",
    "LeWM": "#8866A9",
}
PANELS = {
    "pushobj_all": "PushObj\nAll shapes",
    "pushobj_unseen": "PushObj\nUnseen shapes",
    "pusht_shift": "PushT\nVisual shifts",
}
CSV_FIELDS = ["planner", "method", *PANELS, "pushobj_all_configuration"]
GROUPS = {
    "GD": ("Frozen", "AdaJEPA", "SE-JEPA"),
    "CEM": ("DINO-WM", "LeWM"),
}


def plot_teaser(data, output_dir):
    with Path(data).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(f"Expected CSV header: {','.join(CSV_FIELDS)}")
        rows = list(reader)
    expected = [(planner, method) for planner, methods in GROUPS.items() for method in methods]
    if [(row["planner"], row["method"]) for row in rows] != expected:
        raise ValueError(f"Supply rows in planner and method order: {expected}")

    style = {
        "font.family": "DejaVu Sans", "font.size": 7,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "se-jepa-figure-1", "hatch.linewidth": 0.45,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_centers = [0, 1, 2, 3.6, 4.6]
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.76), sharey=True)
        fig.subplots_adjust(left=.242, right=.965, bottom=.23, top=.775, wspace=.34)
        for i, (ax, (metric, title)) in enumerate(zip(axes, PANELS.items())):
            ax.set_xlim(0, 125)
            ax.set_ylim(row_centers[-1] + .51, -.51)
            ax.set_title(title, fontsize=7.3, color="#243446", pad=4, linespacing=1.13)
            ax.set_xticks([0, 50, 100])
            ax.tick_params(axis="x", length=0, pad=3, labelsize=6.3, colors="#46566A")
            ax.get_xticklabels()[0].set_ha("left")
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            for x in (0, 50, 100):
                ax.axvline(x, color="#BBC5CF" if x == 0 else "#DEE4EA",
                           linewidth=.6 if x == 0 else .4, zorder=0)
            for y, row in zip(row_centers, rows):
                method = row["method"]
                value = float(row[metric])
                if not 0 <= value <= 100:
                    raise ValueError(f"{metric} for {method} must be a percentage")
                ours = method == "SE-JEPA"
                bar = ax.barh(y, value, height=.69 if ours else .51, color=COLORS[method],
                              edgecolor="#AACBE2" if ours else "none",
                              linewidth=0, hatch="///" if ours else None)[0]
                bar.set_gid(f"{row['planner']}-{metric}-{method}")
                if i == 0:
                    ax.annotate(method, (0, y), (-4, 0), textcoords="offset points",
                                ha="right", va="center", fontsize=7.5,
                                weight="bold" if ours else "normal",
                                color=COLORS[method] if ours else "#213345")
                label = row[metric]
                if metric == "pushobj_all" and row["pushobj_all_configuration"] == "without familiarity read/write protection":
                    label += "*"
                ax.annotate(label, (value, y), (1.5, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=6.4,
                            weight="bold" if ours else "normal",
                            color=COLORS[method] if ours else "#213345",
                            bbox={"facecolor": "white", "edgecolor": "none",
                                  "boxstyle": "square,pad=0.03"})

        # Label the planner for each whole block, so CEM references are clearly
        # separated from the GD comparison even when the plot is printed in gray.
        group_transform = blended_transform_factory(fig.transFigure, axes[0].transData)
        for title, first, last in [("GD", 0, 2), ("CEM ref.", 3.6, 4.6)]:
            fig.text(.032, (first + last) / 2, title, transform=group_transform,
                     ha="center", va="center", rotation=90, fontsize=6.5, color="#344A5F")
            fig.add_artist(Line2D([.060, .060], [first - .43, last + .43],
                                  transform=group_transform, color="#95A5B3", linewidth=.7))
        fig.add_artist(Line2D([.018, .975], [2.8, 2.8], transform=group_transform,
                              color="#BAC4CD", linewidth=.55))
        fig.text(.602, .045, "Held-out success (%)", ha="center", fontsize=7.2, color="#253648")
        for extension in ("pdf", "svg", "png"):
            fig.savefig(output_dir / f"teaser_bars.{extension}", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("teaser_bars.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    plot_teaser(args.data, args.output_dir)
