"""Draw the lower panel of Figure 1 from its editable, planner-labeled success rates."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "Frozen": "#BB7429",
    "AdaJEPA": "#C1544A",
    "SE-JEPA": "#104B82",
    "DINO-WM": "#27829E",
    "LeWM": "#8866A9",
}
LABELS = {"AdaJEPA": "adaJEPA"}
PANELS = {
    "pushobj_all": "PushObj\nAll shapes",
    "pushobj_unseen": "PushObj\nUnseen shapes",
    "pusht_shift": "PushT\nVisual shifts",
}
CSV_FIELDS = ["planner", "method", *PANELS]


def plot_teaser(data, output_dir):
    with Path(data).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(f"Expected CSV header: {','.join(CSV_FIELDS)}")
        rows = list(reader)
    expected = [("GD", method) for method in COLORS]
    if sorted((row["planner"], row["method"]) for row in rows) != sorted(expected):
        raise ValueError(f"Supply one GD row per method: {expected}")
    rows.sort(key=lambda row: float(row["pushobj_all"]))

    style = {
        "font.family": "DejaVu Sans", "font.size": 7,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "se-jepa-figure-1", "hatch.linewidth": 0.45,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_centers = range(len(rows))
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.9), sharey=True)
        fig.subplots_adjust(left=.215, right=.975, bottom=.21, top=.78, wspace=.32)
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
                ours = method == "SE-JEPA"
                if i == 0:
                    ax.annotate(LABELS.get(method, method), (0, y), (-4, 0), textcoords="offset points",
                                ha="right", va="center", fontsize=7,
                                weight="bold" if ours else "normal",
                                color=COLORS[method] if ours else "#213345")
                label = row[metric]
                if not label:
                    continue
                value = float(label)
                if not 0 <= value <= 100:
                    raise ValueError(f"{metric} for {method} must be a percentage")
                bar = ax.barh(y, value, height=.69 if ours else .51, color=COLORS[method],
                              edgecolor="#AACBE2" if ours else "none",
                              linewidth=0, hatch="///" if ours else None)[0]
                bar.set_gid(f"{row['planner']}-{metric}-{method}")
                ax.annotate(label, (value, y), (1.5, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=6.4,
                            weight="bold" if ours else "normal",
                            color=COLORS[method] if ours else "#213345",
                            bbox={"facecolor": "white", "edgecolor": "none",
                                  "boxstyle": "square,pad=0.03"})

        fig.text(.595, .045, "Held-out GD success (%)", ha="center", fontsize=7.2, color="#253648")
        for extension in ("pdf", "svg", "png"):
            fig.savefig(output_dir / f"teaser_bars.{extension}", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("teaser_bars.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    plot_teaser(args.data, args.output_dir)
