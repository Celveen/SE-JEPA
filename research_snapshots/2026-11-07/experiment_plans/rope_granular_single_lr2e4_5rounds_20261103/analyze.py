"""Summarize complete passes and paired marginal gains without dropping instances."""
import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PLAN = Path(__file__).resolve().parent


def paired_gain(previous, current, threshold):
    delta = np.asarray(previous) - np.asarray(current)
    rng = np.random.default_rng(20261103)
    means = delta[rng.integers(len(delta), size=(10000,len(delta)))].mean(axis=1)
    low, high = np.quantile(means,[.025,.975])
    before, after = np.asarray(previous)<threshold, np.asarray(current)<threshold
    gained, lost = int((~before & after).sum()), int((before & ~after).sum())
    return dict(cd_improvement=float(delta.mean()), cd_improvement_ci95=[float(low),float(high)],
        improved=int((delta>0).sum()), worsened=int((delta<0).sum()),
        wilcoxon_p=float(stats.wilcoxon(delta).pvalue) if np.any(delta) else 1.,
        success_change_pp=float(100*(after.mean()-before.mean())), gained_success=gained,lost_success=lost,
        mcnemar_p=float(stats.binomtest(min(gained,lost),gained+lost,.5).pvalue) if gained+lost else 1.)


def extract(out, console, obj, index):
    text=console.read_text(errors="replace")
    if "Traceback (most recent call last)" in text or "simulator exploded" in text:
        raise ValueError(f"Runtime/simulation error: {console}")
    values=np.asarray([float(x) for x in re.findall(r"^CD:\s+(\S+)",text,re.MULTILINE)])
    if len(values)!=30 or not np.isfinite(values).all():
        raise ValueError(f"Expected 30 finite CD values: {console}")
    cd=values.reshape(10,3)[:,-1]
    logs=[json.loads(s) for s in (out/"logs.json").read_text().splitlines() if s.strip()]
    mean=[s["final_eval/mean_chamfer_distance"] for s in logs if "final_eval/mean_chamfer_distance" in s][-1]
    assert np.isclose(cd.mean(),mean,atol=1e-6,rtol=0)
    records=[json.loads(s) for s in (out/"continual_episodes.jsonl").read_text().splitlines() if s.strip()]
    assert len(records)==10 and [r["episode"] for r in records]==list(range(index*10,index*10+10))
    assert all(r["tag"]==obj and r["write"]==obj and r["write_rule"]=="oracle" for r in records)
    with np.load(out/"final_eval_results.npz") as data:
        assert len(data["actions"])==10 and np.isfinite(data["actions"]).all()
    return cd,records


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("object",choices=["rope","granular"])
    args=ap.parse_args(); m=json.loads((PLAN/"manifest.json").read_text()); obj=m["objects"][args.object]
    threshold=obj["success_cd"]; rounds=[]; per_episode=[]; arrays={}
    for r in range(1,6):
        jobs=[j for j in obj["jobs"] if j["round"]==r]
        if not all((Path(j["output"])/"final_eval_results.npz").exists() for j in jobs): break
        cd_parts=[]; records=[]
        for job in jobs:
            out=Path(job["output"]); cd,rows=extract(out,out/"console.log",args.object,job["index"])
            cd_parts.append(cd); records.extend(rows)
            for i,row in enumerate(rows):
                per_episode.append(dict(round=r,sample_index=job["chunk"]*10+i,sample_id=job["sample_ids"][i],
                    cd=float(cd[i]),success=int(cd[i]<threshold),read=int(row.get("read") is not None),
                    val_ratio=row.get("val_ratio"),slow_loss=row.get("slow_loss")))
        cd=np.concatenate(cd_parts); arrays[r]=cd
        ratios=np.asarray([x.get("val_ratio",np.nan) for x in records]); ratios=ratios[np.isfinite(ratios)&(ratios>0)]
        loss=np.asarray([x.get("slow_loss",np.nan) for x in records]); loss=loss[np.isfinite(loss)]
        summary=dict(round=r,n=50,mean_cd=float(cd.mean()),se_cd=float(cd.std(ddof=1)/np.sqrt(50)),
            median_cd=float(np.median(cd)),successes=int((cd<threshold).sum()),success_rate=float((cd<threshold).mean()),
            learner_reads=sum(x.get("read") is not None for x in records),
            val_ratio_geomean=float(np.exp(np.log(ratios).mean())) if len(ratios) else None,
            slow_loss_median=float(np.median(loss)) if len(loss) else None,
            domain=records[-1]["domains"],large_cd_samples=int((cd>10).sum()))
        if r>1: summary["gain_from_previous"]=paired_gain(arrays[r-1],cd,threshold)
        rounds.append(summary)
    if not rounds: raise ValueError("No complete rounds yet")
    report=dict(object=args.object,protocol=m["protocol"],slow_lr=m["slow_lr"],rounds=rounds,
                final_vs_first=paired_gain(arrays[1],arrays[max(arrays)],threshold) if len(rounds)>1 else None,
                note="Target-level paired bootstrap intervals; repeated targets measure revisit performance, not unseen-target generalization. No samples excluded.")
    folder=Path(obj["output"])/"analysis"; folder.mkdir(exist_ok=True)
    (folder/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    with (folder/"per_episode.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(per_episode[0])); writer.writeheader(); writer.writerows(per_episode)
    lines=[f"# {args.object}: 2e-4 单领域五轮", "", "同一批 50 个目标，固定次序和种子，cl_noada；未剔除任何实例。", "",
        "| 轮次 | 平均 CD ± SE | 成功率 | 读学习器 | 验证误差比 | 回放损失中位数 |",
        "|---|---:|---:|---:|---:|---:|"]
    for s in rounds:
        lines.append(f"| {s['round']} | {s['mean_cd']:.4f} ± {s['se_cd']:.4f} | {s['successes']}/50 | {s['learner_reads']}/50 | {s['val_ratio_geomean']:.4f} | {s['slow_loss_median']:.5f} |")
    lines += ["", "| 相邻轮 | CD 改善（正值更好） | 配对 bootstrap 95% CI | 成功率变化 | Wilcoxon p |", "|---|---:|---|---:|---:|"]
    for s in rounds[1:]:
        g=s["gain_from_previous"]; lo,hi=g["cd_improvement_ci95"]
        lines.append(f"| {s['round']-1} → {s['round']} | {g['cd_improvement']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {g['success_change_pp']:+.1f} pp | {g['wilcoxon_p']:.4g} |")
    lines += ["", "边际效应需结合第 3→4、4→5 轮的增益、误差比和回放损失走势。置信区间包含 0 不证明完全饱和；五轮结果仅说明本目标集和预算下的表现。", ""]
    (folder/"SUMMARY.md").write_text("\n".join(lines))
    x=[r["round"] for r in rounds]; fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout="constrained")
    for ax,y,label in ((axes[0],[r['mean_cd'] for r in rounds],'Final Chamfer distance'),
                       (axes[1],[100*r['success_rate'] for r in rounds],'Success rate (%)'),
                       (axes[2],[r['val_ratio_geomean'] for r in rounds],'Prediction error ratio')):
        ax.plot(x,y,marker='o'); ax.set(xlabel='Round',ylabel=label,xticks=range(1,6),xlim=(.8,5.2)); ax.grid(alpha=.2)
    fig.suptitle(f'{args.object}: slow learner only, lr=2e-4, same 50 targets')
    fig.savefig(folder/'rounds.png',dpi=180); fig.savefig(folder/'rounds.pdf'); plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__": main()
