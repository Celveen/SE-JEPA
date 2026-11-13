"""Rebuild the November 13 routing and cost tables from the audited data bundle."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/revision_20261113"


def write(name, rows):
    (ROOT / "tables" / name).write_text("\n".join(rows) + "\n")


routing = json.loads((DATA / "routing_summary.json").read_text())["summaries"]
labels = [
    ("P0", "Pretrained predictor"),
    ("fixed_single", "Fixed domain + pretrained"),
    ("random_expectation", "Random eligible predictor"),
    ("selected", r"\textbf{Error-based selection (ours)}"),
    ("oracle", "Future-error oracle"),
]
rows = []
for key, label in labels:
    values = [routing["main_" + metric]["all_condition_macro_loss"][key] * 1000
              for metric in ("one_step", "five_step")]
    cells = [r"\textbf{" + f"{v:.2f}" + "}" if key == "selected" else f"{v:.2f}"
             for v in values]
    rows.append(label + " & " + " & ".join(cells) + r"\\")
write("mature_routing.tex", rows)

with (DATA / "cost_summary.csv").open(encoding="utf-8-sig") as source:
    costs = {(x["optimizer"], x["method"]): x for x in csv.DictReader(source)
             if x["condition"] == "all"}
labels = {"frozen": "Frozen", "adajepa": "AdaJEPA", "ours": r"\ours"}
rows, phases = [], []
for planner in ("GD", "CEM"):
    for method in ("frozen", "adajepa", "ours"):
        entry = costs[planner, method]
        values = [float(entry[key]) for key in
                  ("success_rate_pct", "mpc_mean", "latency_mean_s", "latency_p95_s")]
        values += [float(entry[key]) / 2**30 for key in
                   ("gpu_peak_allocated_bytes_max", "gpu_peak_reserved_bytes_max")]
        prefix = planner + " & " + labels[method] + " & "
        rows.append(prefix + " & ".join(f"{v:.2f}" for v in values) + r"\\")
        values = [float(entry[key]) for key in
                  ("planning_s", "adaptation_with_diagnostics_s", "retrieval_s", "environment_evaluation_s")]
        phases.append(prefix + " & ".join(f"{v:.3f}" for v in values) + r"\\")
    if planner == "GD":
        rows.append(r"\midrule")
        phases.append(r"\midrule")
write("deployment_cost.tex", rows)
write("deployment_cost_phases.tex", phases)

inventory = json.loads((DATA / "memory_inventory.json").read_text())
write("deployment_memory.tex", [
    f"{x['optimizer']} & {x['chain']} & {x['active_learners']} / {x['mature_learners']} & "
    f"{x['replay_count']} & {x['replay_tensor_bytes']/2**20:.1f} & "
    f"{x['checkpoint_bytes']/2**20:.1f} & {x['slow_steps']:,}" + r"\\"
    for x in inventory
])
print("Rebuilt four audited routing/cost tables.")
