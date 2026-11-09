"""Plot the audited mixed-stream results and generate their numerical tables."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NAMES = {"frozen": "Frozen", "adajepa": "AdaJEPA", "v6a": r"\ours"}
DOMAINS = [
    ("default", "Original"), ("blur", "Blur"), ("snp1", "Noise"),
    ("dark", "Dark"), ("red_agent", "Red agent"),
    ("red_object", "Red object"), ("red_goal", "Red goal"),
    ("I", "I"), ("L", "L"), ("square", "Square"), ("Z", "Z"),
    ("small_tee", "Small tee"), ("plus", "Plus"), ("T", "T"),
]


def pm(mean, sd, digits=1):
    return f"${mean:.{digits}f}\\pm{sd:.{digits}f}$"


def main():
    data = json.loads((ROOT / "data/mixed14_audit.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "axes.unicode_minus": True})
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.05), sharey=True)
    fig.subplots_adjust(left=.075, right=.99, top=.75, bottom=.22, wspace=.2)
    styles = {"GD": {"color": "#B64E14", "linestyle": "--", "marker": "s"},
              "CEM": {"color": "#0072B2", "linestyle": "-", "marker": "o"}}
    for ax, (key, title) in zip(axes, [("all", "All 14 domains"),
                                      ("shape", "Seven shape domains"),
                                      ("visual", "Seven visual conditions")]):
        for planner, style in styles.items():
            points = data["round_curves"][planner][key]
            y = np.array([p["mean_pct"] for p in points])
            sd = np.array([p["sd_pct"] for p in points])
            for p in points:
                assert abs(p["successes"] / p["episodes"] * 100 - p["mean_pct"]) < 1e-10
            ax.fill_between(range(1, 6), y - sd, y + sd, color=style["color"], alpha=.12, lw=0)
            ax.plot(range(1, 6), y, label=planner, lw=1.8, ms=3.4,
                    markerfacecolor="white" if planner == "GD" else style["color"], **style)
        ax.set_title(title, fontsize=8, pad=7)
        ax.set(xlim=(.85, 5.15), ylim=(50, 90), xticks=range(1, 6), yticks=[50, 60, 70, 80, 90])
        ax.grid(axis="y", color="#E8EDF1", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#8A98A3")
        ax.tick_params(colors="#475563", length=3)
    axes[0].set_ylabel("Online success (%) ↑")
    fig.supxlabel("Continual-adaptation round", x=.54, y=.035, fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(.57, 1.05), title="SE-JEPA", frameon=False,
               title_fontproperties={"weight": "bold", "size": 8.5})
    fig.savefig(ROOT / "figures/mixed14_learning.pdf", metadata={"CreationDate": None})
    plt.close(fig)

    rows = []
    for row in data["final_comparison"]:
        if row["planner"] == "CEM" and row["method"] == "frozen":
            rows.append(r"\midrule")
        cells = [row["planner"], r"\textbf{\ours}" if row["method"] == "v6a" else NAMES[row["method"]]]
        for family in ["all", "shape", "visual"]:
            p = row[family]
            cells.append(pm(p["mean_pct"], p["sd_pct"]))
        p = row["all"]
        cells.append(pm(p["mean_replans"], p["sd_replans"], 2))
        if row["method"] == "v6a":
            cells[2:] = [r"\textbf{" + c[1:-1].replace(r"\pm", r"}$\,\pm\,$\textbf{") + "}" for c in cells[2:]]
        rows.append(" & ".join(cells) + r"\\")
    (ROOT / "tables/mixed14_summary.tex").write_text("\n".join(rows) + "\n")

    rows = []
    for i, (domain, label) in enumerate(DOMAINS):
        if i == 7:
            rows.append(r"\midrule")
        cells = ["Visual" if i < 7 else "Shape", label]
        for planner in ["GD", "CEM"]:
            for method in ["frozen", "adajepa", "v6a"]:
                p = data["final_by_domain"][planner][domain][method]
                value = f'{p["mean_pct"]:.1f}'
                best = max(v["mean_pct"] for v in data["final_by_domain"][planner][domain].values())
                cells.append(r"\best{" + value + "}" if abs(p["mean_pct"] - best) < 1e-9 else value)
        rows.append(" & ".join(cells) + r"\\")
    (ROOT / "tables/mixed14_domains.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
