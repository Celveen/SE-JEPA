"""Draw the paper's first figure from its editable success-rate CSV."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Row order matches the manuscript's first-page comparison. Colors can be
# changed independently of the data, and the same row labels apply to all tasks.
COLORS = {
    "Frozen": "#BB7429",
    "DINO-WM": "#27829E",
    "LeWM": "#8866A9",
    "AdaJEPA": "#C1544A",
    "SE-JEPA": "#104B82",
}
PANELS = {
    "pushobj_all": "PushObj\nAll shapes",
    "pushobj_unseen": "PushObj\nUnseen shapes",
    "pusht_shift": "PushT\nVisual shifts",
}


def plot_teaser(data, output_dir):
    with Path(data).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["method", *PANELS]:
            raise ValueError(f"Expected CSV header: method,{','.join(PANELS)}")
        rows = list(reader)
    methods = [row["method"] for row in rows]
    if sorted(methods) != sorted(COLORS):
        raise ValueError("Supply one CSV row for each of " + ", ".join(COLORS))

    style = {
        "font.family": "DejaVu Sans", "font.size": 7,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "se-jepa-figure-1", "hatch.linewidth": 0.45,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.76), sharey=True)
        fig.subplots_adjust(left=.19, right=.94, bottom=.23, top=.775, wspace=.30)
        for i, (ax, (metric, title)) in enumerate(zip(axes, PANELS.items())):
            # The success scale ends at 100; reserve space for the labels beyond it.
            ax.set_xlim(0, 120)
            ax.set_ylim(len(rows) - .49, -.49)
            ax.set_title(title, fontsize=7.3, color="#243446", pad=4, linespacing=1.13)
            ax.set_xticks([0, 50, 100])
            ax.tick_params(axis="x", length=0, pad=3, labelsize=6.3, colors="#46566A")
            ax.get_xticklabels()[0].set_ha("left")
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            for x in (0, 50, 100):
                ax.axvline(x, color="#BBC5CF" if x == 0 else "#DEE4EA",
                           linewidth=.6 if x == 0 else .4, zorder=0)
            for j, row in enumerate(rows):
                method = row["method"]
                value = float(row[metric])
                if not 0 <= value <= 100:
                    raise ValueError(f"{metric} for {method} must be a percentage")
                ours = method == "SE-JEPA"
                bar = ax.barh(j, value, height=.69 if ours else .51,
                              color=COLORS[method],
                              edgecolor="#AACBE2" if ours else "none",
                              linewidth=0, hatch="///" if ours else None)[0]
                bar.set_gid(f"{metric}-{method}")
                if i == 0:
                    ax.annotate(method, (0, j), (-4, 0), textcoords="offset points",
                                ha="right", va="center", fontsize=7.5,
                                weight="bold" if ours else "normal",
                                color=COLORS[method] if ours else "#213345")
                ax.annotate(f"{value:.1f}", (value, j), (1.5, 0),
                            textcoords="offset points", ha="left", va="center",
                            fontsize=6.6, weight="bold" if ours else "normal",
                            color=COLORS[method] if ours else "#213345",
                            bbox={"facecolor": "white", "edgecolor": "none",
                                  "boxstyle": "square,pad=0.03"})
        fig.text(.602, .045, "Held-out success (%)", ha="center", fontsize=7.2,
                 color="#253648")
        for extension in ("pdf", "svg", "png"):
            fig.savefig(output_dir / f"teaser_bars.{extension}", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("teaser_bars.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    plot_teaser(args.data, args.output_dir)
