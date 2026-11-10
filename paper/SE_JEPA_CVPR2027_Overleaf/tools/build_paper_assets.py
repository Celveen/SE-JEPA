"""Freeze measured results and generate paper assets. Compilation needs no Python.

On the experiment server: python tools/build_paper_assets.py --refresh
Elsewhere: python tools/build_paper_assets.py  (uses the bundled snapshot)
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import html
import json
import re
import runpy
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from restyle_paper_figures import draw_teaser
from vis50 import external_results
from dinowm_cem import verified_pusht_cem
from pushobj_ablation import heldout_results, write_ablation_tables
from pushobj_online import verified_online_results

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent / "CL-JEPA-cudagraph"
DATA = PAPER / "data"
FIG = PAPER / "figures"
TABLES = PAPER / "tables"
for folder in (DATA, FIG, TABLES):
    folder.mkdir(exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--refresh", action="store_true")
parser.add_argument("--main-table-only", action="store_true", help="rebuild the main and ablation tables")
args = parser.parse_args()

if args.refresh:
    context = runpy.run_path(str(PAPER / "build_structure_preview.py"))
    snapshot = {
        "created": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "shape": context["shape"],
        "shape_full": json.loads((REPO / "reports/shape_shift_figs/data/shape7_summary.json").read_text()),
        "shape_steps": context["steps"],
        "shape_rounds": context["rounds"],
        "visual": context["visual_summary"],
        "visual_rounds": context["visual_new_rounds"],
        "visual_steps": context["visual_new_steps"],
        "visual_baselines": context["visual"],
        "deform": context["deform"],
        "main_rows_html": context["main_rows"],
        "offline": context["offline_visual"],
    }
    (DATA / "paper_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
    sources = {}
    for path in [
        REPO / "reports/shape_shift_figs/data/shape7_summary.json",
        REPO / "reports/shape_shift_figs/data/ours_rounds.json",
        REPO / "reports/shape_shift_figs/data/mpc_steps.json",
        PAPER / "drive_materials/visual7_20261030/visual7_summary.json",
        PAPER / "drive_materials/visual7_20261030/ours_rounds.json",
        PAPER / "drive_materials/visual7_20261030/mpc_steps.json",
        PAPER / "drive_materials/visual7_20261030/sample_manifest.json",
    ]:
        sources[str(path.relative_to(PAPER.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    diag = subprocess.check_output(["git", "-C", str(REPO), "show", "origin/results:reports/cont_S2/summary.md"], text=True)
    (DATA / "matched_accumulation_source.md").write_text(diag)
    sources["git:origin/results:reports/cont_S2/summary.md"] = hashlib.sha256(diag.encode()).hexdigest()
    (DATA / "source_hashes.json").write_text(json.dumps(sources, indent=2))
else:
    snapshot = json.loads((DATA / "paper_snapshot.json").read_text())

s = snapshot
reporting = json.loads((DATA / "reporting_overrides.json").read_text())
online_cem = verified_online_results(DATA / "pushobj_cem_online")
domains = ["blur", "snp", "dark", "red_agent", "red_object", "red_goal"]
labels = ["Blur", "Noise", "Dark", "Red agent", "Red object", "Red goal"]
methods = ["frozen", "adajepa", "ours"]
names = {"frozen": "Frozen", "adajepa": "AdaJEPA", "ours": "SE-JEPA"}
colors = {"frozen": "#99a6b1", "adajepa": "#557a9b", "ours": "#087f79"}

def tex_cell(value):
    if 'class="pending"' in value or value == "—":
        return r"\missing"
    value = re.sub(r"<sup>.*?</sup>", "", value)
    value = re.sub(r"<[^>]*>", "", value)
    return html.unescape(value).strip().replace(" ± ", r" $\pm$ ")

citations = {
    "DreamerV3": r"DreamerV3~\cite{hafner2025dreamerv3}",
    "IRIS": r"IRIS~\cite{micheli2023iris}",
    "R3M": r"R3M~\cite{nair2022r3m}",
    "DINO CLS": r"DINO CLS~\cite{oquab2024dinov2}",
    "DINO-WM": r"DINO-WM~\cite{zhou2024dinowm}",
    "LeWM": r"LeWM~\cite{maes2026lewm}",
    "DDP-WM": r"DDP-WM~\cite{yin2026ddpwm}",
    "AdaJEPA": r"AdaJEPA~\cite{wang2026adajepa}",
    "Frozen": "Frozen (pretrained)",
    "SE-JEPA (Ours)": r"\ours",
}
rows = []
order = ["DreamerV3", "IRIS", "R3M", "DINO CLS", "DINO-WM", "LeWM", "DDP-WM", "Frozen", "AdaJEPA", "SE-JEPA (Ours)"]
vis50 = external_results(DATA / "vis50")
dinowm_pusht_cem = verified_pusht_cem(DATA / "dinowm_pusht_cem_epoch3")
ablation_results = heldout_results(DATA / "ablation")
for planner in ("GD", "CEM"):
    rate, _ = ablation_results[planner, "no_familiarity"]
    recorded = reporting["pushobj_no_familiarity_read_write"][f"{planner.lower()}_all_percent"]
    if round(rate, 2) != recorded:
        raise ValueError(f"{planner} display rate {recorded} differs from archived test rate {rate:.2f}")
for row in sorted(s["main_rows_html"], key=lambda row: order.index(tex_cell(row[0]))):
    label = tex_cell(row[0])
    values = [tex_cell(x) for x in row[1:7]]
    if label in vis50:
        record = vis50[label]
        values[3:6] = [format(record[key], spec) for key, spec in [("shift_rate", ".1f"), ("color_rate", ".1f"), ("steps", ".2f")]]
    if label == "DINO-WM":
        values[3:6] = [format(dinowm_pusht_cem[key], spec) for key, spec in [("shift_rate", ".1f"), ("color_rate", ".1f"), ("steps", ".2f")]]
    if label in reporting["published_pushobj"]["local_protocol_unreported"]:
        published_rate = reporting["published_pushobj"]["success_percent"][label]
        values[:3] = [r"\missing", f"{published_rate:.1f}", r"\missing"]
    for raw in row[7:9]:
        cell = tex_cell(raw).split(r" $\pm$ ")[0]
        values.append(cell)
    if label == "Frozen":
        # The deformable tasks use the very same frozen DINO-WM reference.
        values[6:8] = [format(float(s["deform"][task]["DINO-WM"]["cd"]), ".3f") for task in ["rope", "granular"]]
        rows.append(r"\midrule")
    if label == "SE-JEPA (Ours)":
        # This row reports online round five; paired post-learning results
        # remain in paired_planners.tex and the per-shape evaluation table.
        online = online_cem["summary"]
        values[:3] = [f'{online["all"]["mean_pct"]:.1f}',
                      f'{online["unseen"]["mean_pct"]:.1f}',
                      f'{online["all"]["mean_replans"]:.2f}']
        values = [r"\textbf{" + v + "}" for v in values]
    rows.append(" & ".join([citations[label], *values]) + r" \\")
(TABLES / "main_rows.tex").write_text("\n".join(rows) + "\n")
write_ablation_tables(TABLES, ablation_results)
if args.main_table_only:
    raise SystemExit(0)

rows = []
for planner in ("GD", "CEM"):
    for method in methods:
        obj = s["shape"][planner][method]
        vis = s["visual"]["rates"][planner][method]
        vals = [
            obj["all"]["mean"], obj["unseen"]["mean"],
            s["shape_steps"][planner][method]["steps_all"]["mean"],
            vis["all"]["mean"], vis["shifted"]["mean"],
            s["visual_steps"][planner][method]["steps_all"]["mean"],
        ]
        rendered = [f"{v:.2f}" for v in vals]
        rows.append(planner + " & " + names[method] + " & " + " & ".join(rendered) + r" \\")
    if planner == "GD":
        rows.append(r"\midrule")
(TABLES / "paired_planners.tex").write_text("\n".join(rows) + "\n")

rows = []
for planner in ("CEM", "GD"):
    for method in methods:
        vals = [s["visual"]["by_shape"][planner][d][method]["mean"]
                for d in ["default", "blur", "snp1", "dark", "red_agent", "red_object", "red_goal"]]
        vals += [s["visual"]["rates"][planner][method][g]["mean"] for g in ("all", "shifted")]
        rows.append(planner + " & " + names[method] + " & " + " & ".join(f"{v:.1f}" for v in vals) + r" \\")
    if planner == "CEM":
        rows.append(r"\midrule")
(TABLES / "visual_domains.tex").write_text("\n".join(rows) + "\n")

rows = []
for method, name in [("dreamerv3", "DreamerV3"), ("iris", "IRIS")]:
    by = s["offline"][method]["domains"]
    vals = [f'{by[d]["rate"]:.1f}' if d in by else r"\missing" for d in ["default"] + domains]
    avg7 = f'{np.mean([by[d]["rate"] for d in ["default"]+domains]):.2f}' if len(by) == 7 else r"\missing"
    avg6 = f'{np.mean([by[d]["rate"] for d in domains]):.2f}' if all(d in by for d in domains) else r"\missing"
    rows.append(name + " & " + " & ".join(vals+[avg7,avg6]) + r" \\")
(TABLES / "offline_domains.tex").write_text("\n".join(rows) + "\n")

rows = []
for method in ("DINO-WM", "AdaJEPA", "Ours"):
    vals = []
    for task in ("rope", "granular"):
        row = s["deform"][task][method]
        vals.append(f'{float(row["cd"]):.3f}' + r" $\pm$ " + f'{float(row["cd_se"]):.3f}')
    if method == "Ours":
        vals = [r"\textbf{" + v + "}" for v in vals]
    rows.append((r"\ours" if method == "Ours" else method) + " & " + " & ".join(vals) + r" \\")
(TABLES / "deform_rows.tex").write_text("\n".join(rows) + "\n")
extra_deform = [
    r"LeWM & 0.786 $\pm$ 0.112 & 0.322 $\pm$ 0.018 \\",
    r"DDP-WM & 0.677 $\pm$ 0.086 & 0.259 $\pm$ 0.012 \\",
]
# Archived instance-level uncertainty is recorded in baselines/BIG_TABLE.md.
(TABLES / "deform_rows.tex").write_text("\n".join(rows[:1] + extra_deform + rows[1:]) + "\n")

pass_rows = []
for index in range(4):
    cells = []
    for lr in ("1e-4", "2e-4"):
        for task in ("rope", "granular"):
            _, mean, se = s["deform"][task]["Ours_passes"][lr][index]
            cells.append(f"${mean:.3f}\\pm{se:.3f}$")
    pass_rows.append(f"{index + 1} & " + " & ".join(cells) + r" \\")
(TABLES / "deform_passes.tex").write_text("\n".join(pass_rows) + "\n")

shape_rows = []
for planner in ("CEM", "GD"):
    for method in methods:
        values = [s["shape_full"]["by_shape"][planner][shape][method]["mean"]
                  for shape in ["I", "L", "square", "Z", "small_tee", "plus", "T"]]
        values.append(s["shape"][planner][method]["all"]["mean"])
        shape_rows.append(planner + " & " + names[method] + " & " + " & ".join(f"{v:.1f}" for v in values) + r" \\")
    if planner == "CEM":
        shape_rows.append(r"\midrule")
(TABLES / "shape_domains.tex").write_text("\n".join(shape_rows) + "\n")

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 10, "legend.fontsize": 8, "pdf.fonttype": 42,
    "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})
def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=.04)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=.04, dpi=160)
    plt.close(fig)

# Legacy CEM panel. The later figure scripts regenerate the chart used in the combined teaser.
draw_teaser(s)

# Fig. 6: deployment curves sized for one column.
fig, ax = plt.subplots(1, 2, figsize=(3.3, 1.62), constrained_layout=True)
for axis, group, title in [(ax[0], "shape", "(a) Unseen shapes"), (ax[1], "visual", "(b) Visual conditions")]:
    source = s["shape_rounds"] if group == "shape" else s["visual_rounds"]
    for planner, color, marker in [("GD", "#557a9b", "s"), ("CEM", "#087f79", "o")]:
        stat = [source[planner][str(i)]["unseen" if group == "shape" else "all"] for i in range(1,6)]
        y = np.array([x["mean"] for x in stat]); sd = np.array([x.get("sd",0) for x in stat])
        axis.plot(range(1,6),y,color=color,marker=marker,label=planner,markersize=3,lw=1.3)
        axis.fill_between(range(1,6),y-sd,y+sd,color=color,alpha=.12,lw=0)
    axis.set_title(title, fontsize=8)
    axis.set(xlabel="Round",xticks=range(1,6),ylim=(45,90))
    axis.tick_params(labelsize=7); axis.xaxis.label.set_size(8)
    axis.grid(axis="y",alpha=.2)
ax[0].set_ylabel("Success (%)", fontsize=8)
ax[1].legend(frameon=False,loc="lower right",fontsize=7)
save(fig,"deployment")

# Fig. 7: deformable consolidation over passes (slow lr 1e-4).
fig, ax = plt.subplots(1, 2, figsize=(3.3, 1.62), constrained_layout=True)
for axis, task in zip(ax, ("rope", "granular")):
    passes = np.array(s["deform"][task]["Ours_passes"]["1e-4"])
    axis.plot(passes[:, 0], passes[:, 1], color="#087f79", marker="o", markersize=3, lw=1.3, label="SE-JEPA")
    axis.fill_between(passes[:, 0], passes[:, 1] - passes[:, 2], passes[:, 1] + passes[:, 2], color="#087f79", alpha=.13, lw=0)
    for method, color, style in (("DINO-WM", "#557a9b", "--"), ("AdaJEPA", "#99a6b1", ":")):
        axis.axhline(float(s["deform"][task][method]["cd"]), color=color, ls=style, lw=1.1, label=method)
    axis.set_title(task.capitalize(), fontsize=8)
    axis.set(xlabel="Pass", xticks=[1, 2, 3, 4])
    axis.tick_params(labelsize=7); axis.xaxis.label.set_size(8)
    axis.grid(axis="y", alpha=.2)
ax[0].set_ylabel("Chamfer distance", fontsize=8)
ax[1].legend(frameon=False, fontsize=6.3, loc="center right", bbox_to_anchor=(1.0, 0.62))
save(fig, "deform_passes")

fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.12), constrained_layout=True)
for axis, group, title in [(ax[0], "shape", "(a) Unseen shapes"), (ax[1], "visual", "(b) Seven visual conditions")]:
    source = s["shape_rounds"] if group == "shape" else s["visual_rounds"]
    for planner, color, marker in [("GD", "#557a9b", "s"), ("CEM", "#087f79", "o")]:
        stat = [source[planner][str(i)]["unseen" if group == "shape" else "all"] for i in range(1,6)]
        y = np.array([x["mean"] for x in stat]); sd = np.array([x.get("sd",0) for x in stat])
        axis.plot(range(1,6),y,color=color,marker=marker,label=planner,markersize=4)
        axis.fill_between(range(1,6),y-sd,y+sd,color=color,alpha=.12,lw=0)
    axis.set(title=title,xlabel="Deployment round",ylabel="Online success (%)",xticks=range(1,6),ylim=(45,90))
    axis.grid(axis="y",alpha=.2);axis.legend(frameon=False,loc="lower right")
save(fig,"deployment_wide")

fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.15), constrained_layout=True)
for axis, curves, title, selected in [
    (ax[0], s["shape_full"]["curves"]["CEM"], "(a) PushObj held-out", ["I","L","square","Z","small_tee","plus","T"]),
    (ax[1], s["visual"]["curves"]["CEM"], "(b) PushT held-out (seven)", ["default","blur","snp1","dark","red_agent","red_object","red_goal"]),
]:
    for method in methods:
        # Curves are cumulative success fractions, represented in percent in the source.
        y=np.mean([np.asarray(curves[d][method]["mean"]) for d in selected],axis=0)
        axis.plot(np.arange(1,len(y)+1),y,label=names[method],color=colors[method],lw=1.8)
    axis.set(title=title,xlabel="MPC replans",ylabel="Success (%)",xlim=(1,30),ylim=(0,100))
    axis.grid(alpha=.15);axis.legend(frameon=False,loc="lower right")
save(fig,"planning_curves")

fig, axis=plt.subplots(figsize=(6.8,1.85),constrained_layout=True)
pos=np.arange(6)
for off,planner,color in [(-.18,"CEM","#087f79"),(.18,"GD","#557a9b")]:
    vals=[s["visual"]["by_shape"][planner][d]["ours"]["mean"]-s["visual"]["by_shape"][planner][d]["adajepa"]["mean"] for d in ["blur","snp1","dark","red_agent","red_object","red_goal"]]
    axis.bar(pos+off,vals,.34,label=planner,color=color)
axis.set_xticks(pos,labels);axis.set(ylabel="Gain over AdaJEPA (pp)",ylim=(0,40))
axis.legend(frameon=False,ncol=2);axis.grid(axis="y",alpha=.15)
save(fig,"visual_gains")

if args.refresh:
    diagnostic={}
    for planner in ("gd","cem"):
        logs=sorted((REPO/f"eval_outputs/pushobj_lr2e4_7rounds_20261103/{planner}/cls6a_free").glob("*/continual_episodes.jsonl"))
        records=[json.loads(line) for p in logs for line in p.read_text().splitlines() if line.strip()]
        cosine=np.array([r["cos_prev"] for r in records if isinstance(r.get("cos_prev"),(int,float)) and np.isfinite(r["cos_prev"])])
        diagnostic[planner]={"episodes":len(records),"cosine_count":len(cosine),"cosine_mean":float(cosine.mean()),"cosine_median":float(np.median(cosine)),"writes":sum(r.get("write") is not None for r in records),"reads":sum(r.get("read") is not None for r in records)}
        np.save(DATA/f"cosine_{planner}.npy",cosine)
    (DATA/"direction_diagnostic.json").write_text(json.dumps(diagnostic,indent=2))

if (DATA/"cosine_gd.npy").exists():
    fig,ax=plt.subplots(1,2,figsize=(6.8,2.1),constrained_layout=True)
    for axis,planner,color in zip(ax,("gd","cem"),("#557a9b","#087f79")):
        values=np.load(DATA/f"cosine_{planner}.npy")
        axis.hist(values,bins=np.linspace(-1,1,51),weights=np.ones(len(values))/len(values),color=color,edgecolor="white",linewidth=.3)
        axis.axvline(0,color="#777",ls=":",lw=.8)
        axis.set(title=f"{planner.upper()}: {len(values):,} adjacent updates",xlabel="Cosine similarity",ylabel="Fraction",xlim=(-1,1))
    save(fig,"update_directions")

if args.refresh:
    shutil.copy2(PAPER/"drive_materials/visual7_20261030/17_cem_red_agent_prediction_loss.png", FIG/"source/visual_qualitative_original.png")
    shutil.copy2(REPO/"reports/shape_shift_figs/preview/15_cem_square_prediction_loss.png", FIG/"source/shape_qualitative_original.png")
if (FIG/"source/visual_qualitative_original.png").exists():
    fig,axes=plt.subplots(3,8,figsize=(6.8,2.65))
    fig.subplots_adjust(left=.075,right=.995,bottom=.03,top=.80,wspace=.05,hspace=.12)
    for panel,kind in enumerate(("shape","visual")):
        im=Image.open(FIG/f"source/{kind}_qualitative_original.png").convert("RGB")
        # Source plots have 15 equally spaced trajectory columns and a separate goal.
        left,right=190,2420
        row_tops=[128,292,456]
        pitch=(right-left)/15
        for row,method in enumerate(methods):
            final_index={"shape":[14,8,5],"visual":[14,14,10]}[kind][row]
            for j,index in enumerate([0,4,final_index,None]):
                axis=axes[row,panel*4+j]
                if index is None:
                    crop=im.crop((2436,row_tops[row]+5,2570,row_tops[row]+132))
                else:
                    crop=im.crop((int(left+index*pitch)+5,row_tops[row]+5,int(left+(index+1)*pitch)-5,row_tops[row]+132))
                axis.imshow(crop);axis.set_xticks([]);axis.set_yticks([])
                for spine in axis.spines.values():spine.set_color("#b3bec5");spine.set_linewidth(.5)
                if row==0:axis.set_title(["Start","Replan 4","Terminal","Goal"][j],fontsize=8,pad=3)
                if panel==0 and j==0:axis.set_ylabel(names[method],fontsize=8,labelpad=5)
    fig.text(.31,.97,"(a) Unseen square",ha="center",va="top",fontsize=10)
    fig.text(.77,.97,"(b) Red agent",ha="center",va="top",fontsize=10)
    save(fig,"qualitative")

print("Frozen snapshot:",s["created"])
print("Tables:",len(list(TABLES.glob("*.tex"))),"Vector figures:",len(list(FIG.glob("*.pdf"))))
