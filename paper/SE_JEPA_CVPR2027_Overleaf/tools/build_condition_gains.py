"""Draw full SE-JEPA minus episodic AdaJEPA on matched held-out goals."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PANELS = {
    "PushObj": ("shape_full", [
        ("I", "I"), ("square", "Square"), ("small_tee", "Small tee"),
        ("L", "L"), ("Z", "Z"), ("plus", "Plus"), ("T", "T"),
    ]),
    "PushT": ("visual", [
        ("blur", "Blur"), ("snp1", "Noise"), ("dark", "Dark"),
        ("red_agent", "Red agent"), ("red_object", "Red object"), ("red_goal", "Red goal"),
    ]),
}
PLANNERS = {
    "GD": {"marker": "o", "color": "#124F85", "offset": -0.105},
    "CEM": {"marker": "s", "color": "#A64F1F", "offset": 0.105},
}


def main():
    source = json.loads((ROOT / "data/paper_snapshot.json").read_text())
    csv_path = ROOT / "data/condition_gains.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["task", "condition", "planner", "ours_percent", "adajepa_percent", "gain_pp"])
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                             "pdf.fonttype": 42, "axes.unicode_minus": True})
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.28), sharex=True)
        fig.subplots_adjust(left=.109, right=.982, top=.83, bottom=.21, wspace=.47)
        for ax, (task, (key, conditions)) in zip(axes, PANELS.items()):
            for row, (condition, _) in enumerate(conditions):
                values = []
                for planner, style in PLANNERS.items():
                    result = source[key]["by_shape"][planner][condition]
                    ours, ada = (result[m]["mean"] for m in ("ours", "adajepa"))
                    gain = ours - ada
                    writer.writerow([task, condition, planner, ours, ada, gain])
                    y = row + style["offset"]
                    values.append((gain, y))
                    ax.plot(gain, y, marker=style["marker"], markersize=5.2,
                            markerfacecolor=style["color"] if planner == "GD" else "white",
                            markeredgecolor=style["color"], markeredgewidth=1.25,
                            linestyle="none", label=planner if row == 0 else "", zorder=3)
                ax.plot(*zip(*values), color="#A4AFB8", linewidth=1, zorder=2)
            ax.set_yticks(range(len(conditions)), labels=[label for _, label in conditions])
            ax.set_ylim(6.55, -.55)
            ax.set_xlim(-4.0, 36)
            ax.set_xticks([0, 10, 20, 30])
            ax.axvline(0, color="#596977", lw=1.05)
            ax.grid(axis="x", linewidth=.55, color="#D8DEE3")
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", length=0, pad=3.2, colors="#24323F")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(task, loc="left", fontweight="bold", fontsize=9, pad=4)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
                   bbox_to_anchor=(.982, .998), frameon=False, ncol=2,
                   handletextpad=.25, columnspacing=.9, borderpad=.1)
        fig.supxlabel("Held-out success gain of SE-JEPA over AdaJEPA (percentage points)",
                      x=.52, y=.015, fontsize=8.2)
        fig.savefig(ROOT / "figures/condition_gains.pdf", metadata={"CreationDate": None})
        plt.close(fig)


if __name__ == "__main__":
    main()
