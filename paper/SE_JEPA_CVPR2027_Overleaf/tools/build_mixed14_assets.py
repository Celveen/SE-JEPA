"""Plot the audited mixed-stream results and generate their numerical tables."""
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
from PIL import Image
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


def planner_average(gd, cem):
    """Average equally sized planner groups without pairing their runs."""
    assert gd["episodes"] == cem["episodes"]
    units = np.asarray(gd["unit_rates_pct"] + cem["unit_rates_pct"], dtype=float)
    assert len(gd["unit_rates_pct"]) == len(cem["unit_rates_pct"]) == 3
    mean = (gd["mean_pct"] + cem["mean_pct"]) / 2
    episodes = gd["episodes"] + cem["episodes"]
    successes = gd["successes"] + cem["successes"]
    assert abs(mean - units.mean()) < 1e-10
    assert abs(mean - successes / episodes * 100) < 1e-10
    return {"mean_pct": mean, "sd_pct": float(units.std(ddof=1)),
            "episodes": episodes, "successes": successes,
            "unit_rates_pct": units.tolist(),
            "planner_means_pct": {"GD": gd["mean_pct"], "CEM": cem["mean_pct"]}}


def chart_data(data):
    comparison = {(r["planner"], r["method"]): r for r in data["final_comparison"]}
    result = {"source": "data/mixed14_audit.json",
              "source_sha256": hashlib.sha256((ROOT / "data/mixed14_audit.json").read_bytes()).hexdigest(),
              "aggregation": "Equal-weight mean of GD and CEM; each planner has three equally sized runs.",
              "uncertainty": "Sample SD of the six actual planner-by-chain or planner-by-seed success rates; descriptive run dispersion, not a confidence interval or a paired-planner estimate.",
              "baseline_display": "Horizontal references from one evaluation per planner and baseline seed (100, 200, 300), not five-round baseline learning trajectories.",
              "families": {}}
    for family in ["all", "shape", "visual"]:
        rounds = [planner_average(gd, cem) for gd, cem in zip(
            data["round_curves"]["GD"][family], data["round_curves"]["CEM"][family])]
        assert len(rounds) == 5
        baseline = {method: planner_average(comparison["GD", method][family],
                                            comparison["CEM", method][family])
                    for method in ["frozen", "adajepa"]}
        result["families"][family] = {"v6a_rounds": rounds, "baseline_references": baseline}
    return result


def build_chart(data):
    plotted = chart_data(data)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "axes.unicode_minus": True})
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.65), sharey=True)
    fig.subplots_adjust(left=.075, right=.99, top=.59, bottom=.16, wspace=.2)
    observations = json.loads((ROOT / "data/revision_assets/mixed14_observations/manifest.json").read_text())
    for i, case in enumerate(observations["cases"]):
        x = .015 + i * .0695 + (.012 if i >= 7 else 0)
        ax_image = fig.add_axes([x, .80, .063, .162])
        path = ROOT / case["image"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["image_sha256"]
        ax_image.imshow(Image.open(path), interpolation="none")
        ax_image.set_xticks([]); ax_image.set_yticks([])
        for spine in ax_image.spines.values():
            spine.set_color("#D7DDE2"); spine.set_linewidth(.4)
        ax_image.set_xlabel(case["label"], fontsize=5.5, labelpad=2)
    fig.text(.255, .995, "PushObj shapes", ha="center", va="top", fontsize=6.8)
    fig.text(.754, .995, "PushT appearances", ha="center", va="top", fontsize=6.8)
    styles = {"frozen": {"color": "#553A76", "linestyle": (0, (4, 2.4))},
              "adajepa": {"color": "#BB4810", "linestyle": "-."},
              "v6a": {"color": "#006FC1", "linestyle": "-", "marker": "o"}}
    for ax, (key, title) in zip(axes, [("all", "All 14 domains"),
                                      ("shape", "Seven shape domains"),
                                      ("visual", "Seven visual conditions")]):
        family = plotted["families"][key]
        for method in ["frozen", "adajepa"]:
            point = family["baseline_references"][method]
            mean, sd = point["mean_pct"], point["sd_pct"]
            ax.axhspan(mean - sd, mean + sd, color=styles[method]["color"], alpha=.08, lw=0)
            ax.axhline(mean, label=NAMES[method], lw=1.15, **styles[method])
        points = family["v6a_rounds"]
        y = np.array([p["mean_pct"] for p in points])
        sd = np.array([p["sd_pct"] for p in points])
        ax.fill_between(range(1, 6), y - sd, y + sd, color=styles["v6a"]["color"], alpha=.14, lw=0)
        ax.plot(range(1, 6), y, label="SE-JEPA", lw=2.15, ms=3.4, **styles["v6a"])
        ax.set_title(title, fontsize=8, pad=7)
        ax.set(xlim=(.85, 5.15), ylim=(20, 100), xticks=range(1, 6), yticks=[20, 40, 60, 80, 100])
        ax.grid(axis="y", color="#E8EDF1", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#8A98A3")
        ax.tick_params(colors="#475563", length=3)
    axes[0].set_ylabel("Success (%) ↑")
    fig.supxlabel("Continual-adaptation round", x=.54, y=.014, fontsize=8)
    legend = fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
                        bbox_to_anchor=(.54, .77), frameon=False,
                        title_fontproperties={"size": 8})
    for label in legend.get_texts():
        label.set_fontweight("bold" if label.get_text() == "SE-JEPA" else "normal")
    fig.savefig(ROOT / "figures/mixed14_learning.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    (ROOT / "data/mixed14_planner_average.json").write_text(json.dumps(plotted, indent=2) + "\n")
    return plotted


def main():
    data = json.loads((ROOT / "data/mixed14_audit.json").read_text())
    build_chart(data)

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
