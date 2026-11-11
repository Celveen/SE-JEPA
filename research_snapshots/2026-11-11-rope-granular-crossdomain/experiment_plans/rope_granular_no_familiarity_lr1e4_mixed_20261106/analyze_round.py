"""Summarize a completed round; performance diagnostics never truncate the budget."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

PLAN = Path(__file__).resolve().parent
THRESHOLDS = {"rope": 0.5, "granular": 0.2}


def chamfer(a, b):
    a = np.asarray(a).reshape(-1, 4)[:, :3]
    b = np.asarray(b).reshape(-1, 4)[:, :3]
    return float(cKDTree(a).query(b, workers=1)[0].mean() + cKDTree(b).query(a, workers=1)[0].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", type=int, choices=[1, 2, 3], default=1)
    args = ap.parse_args()
    m = json.loads((PLAN / "manifest.json").read_text())
    blocks = [b for b in m["blocks"] if b["round"] == args.round]
    outcomes, records, errors = defaultdict(list), [], []
    for b in blocks:
        out = Path(m["output"]) / "free_abs" / b["chunk"]
        cfg = OmegaConf.load(out / ".hydra/config.yaml")
        assert cfg.planner.continual.slow_lr == m["slow_lr"]
        assert cfg.planner.continual.write == "delta"
        assert cfg.planner.continual.familiar_thr == 0 and not cfg.planner.continual.familiar_read
        text = (out / "console.log").read_text(errors="replace")
        if "Traceback (most recent call last)" in text or "simulator exploded" in text:
            errors.append(f"{b['chunk']}: simulator/runtime error in console")
        values = [float(x) for x in re.findall(r"^CD:\s+(\S+)", text, flags=re.MULTILINE)]
        if len(values) != b["n"] * 3:
            raise ValueError(f"{b['chunk']}: {len(values)} CD evaluations, expected {b['n']*3}")
        all_cd = np.asarray(values).reshape(b["n"], 3)
        final = all_cd[:, -1]
        log_rows = [json.loads(s) for s in (out / "logs.json").read_text().splitlines() if s.strip()]
        logged = [s["final_eval/mean_chamfer_distance"] for s in log_rows if "final_eval/mean_chamfer_distance" in s][-1]
        if not np.isfinite(all_cd).all() or not np.isfinite(logged):
            errors.append(f"{b['chunk']}: nonfinite Chamfer distance")
        elif not np.isclose(final.mean(), logged, atol=1e-6, rtol=0):
            raise ValueError(f"{b['chunk']}: final CD mean does not match logs.json")
        rows = [json.loads(s) for s in (out / "continual_episodes.jsonl").read_text().splitlines() if s.strip()]
        assert len(rows) == b["n"] and all(s["tag"] == b["object"] for s in rows)
        assert all(s.get("write_rule") != "familiar" for s in rows), "Familiarity-based write skip must be disabled"
        assert [s["episode"] for s in rows] == list(range(b["index"]*10, (b["index"]+1)*10))
        records.extend(rows)
        with np.load(out / "final_eval_results.npz") as data:
            assert len(data["actions"]) == b["n"]
            if not np.isfinite(data["actions"]).all():
                errors.append(f"{b['chunk']}: nonfinite actions")
            for i, row in enumerate(rows):
                initial = chamfer(data["state_0"][i], data["state_g"][i])
                outcomes[b["object"]].append(dict(chunk=b["chunk"], sample=i,
                    trajectory=b["samples"][i]["trajectory"], offset=b["samples"][i]["offset"],
                    initial_cd=initial, final_cd=float(final[i]),
                    success=bool(final[i] < THRESHOLDS[b["object"]]),
                    read=row.get("read"), write=row.get("write"), write_rule=row.get("write_rule")))
    domains = {k: v for k, v in records[-1]["domains"].items() if v["n"] > 0}
    stats = {}
    for obj in ("rope", "granular"):
        data = outcomes[obj]
        initial = np.asarray([x["initial_cd"] for x in data])
        final = np.asarray([x["final_cd"] for x in data])
        rs = [r for r in records if r["tag"] == obj]
        stats[obj] = dict(n=len(data), initial_cd_mean=float(initial.mean()),
                         final_cd_mean=float(final.mean()), final_cd_median=float(np.median(final)),
                         final_cd_max=float(final.max()), mean_cd_improvement=float((initial-final).mean()),
                         improved_samples=int((final < initial).sum()),
                         successes=sum(x["success"] for x in data), threshold=THRESHOLDS[obj],
                         read_counts=dict(Counter(str(r.get("read")) for r in rs)),
                         write_counts=dict(Counter(str(r.get("write")) for r in rs)),
                         write_rules=dict(Counter(r.get("write_rule") for r in rs)))
    identities = []
    for d in domains.values():
        tags = [tag for tag, count in d["tags"].items() if count > 0]
        identities.append(tags[0] if len(tags) == 1 else "mixed")
    checks = dict(
        complete_100_samples=len(records) == 100 and all(s["n"] == 50 for s in stats.values()),
        no_runtime_or_numerical_error=not errors,
        exactly_two_active_domains=len(domains) == 2,
        domains_are_pure_rope_and_granular=sorted(identities) == ["granular", "rope"],
        both_domains_have_mature_learners=len(domains) == 2 and all(d["learner"] and d["mature"] for d in domains.values()),
        both_objects_improve_mean_cd=all(np.isfinite(s["mean_cd_improvement"]) and s["mean_cd_improvement"] > 0 for s in stats.values()),
        both_objects_have_successes=all(s["successes"] > 0 for s in stats.values()),
    )
    total_members = sum(d["n"] for d in domains.values())
    purity = sum(max(d["tags"].values()) for d in domains.values()) / total_members if total_members else 0
    integrity_ok = checks["complete_100_samples"] and checks["no_runtime_or_numerical_error"]
    report = dict(round=args.round, checked_at=datetime.now(timezone.utc).isoformat(),
                  continuation_policy=m["continuation_policy"], integrity_ok=integrity_ok,
                  diagnostic_checks=checks,
                  errors=errors, performance=stats, domains=domains, member_weighted_purity=purity,
                  merges=[x for r in records for x in r.get("merges", [])],
                  absorbs=[x for r in records for x in r.get("absorbs", [])], samples=dict(outcomes))
    report_path = PLAN / f"round_{args.round}_review.json"
    tmp = report_path.with_suffix(".tmp")
    def finite_json(value):
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    tmp.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False) + "\n"); tmp.replace(report_path)
    lines = [f"# 第 {args.round} 轮检查", "",
             f"结果完整性：{'通过' if integrity_ok else '异常，停查'}。领域数量和成功样本数只作诊断，不触发性能早停。", "",
             "| 对象 | 样本 | 初始 CD | 最终 CD | 成功数 | CD 改善样本 |",
             "|---|---:|---:|---:|---:|---:|"]
    for obj, s in stats.items():
        lines.append(f"| {obj} | {s['n']} | {s['initial_cd_mean']:.4f} | {s['final_cd_mean']:.4f} | {s['successes']}/50 | {s['improved_samples']}/50 |")
    lines += ["", "领域成员：", "", "```json", json.dumps(domains, ensure_ascii=False, indent=2), "```", "", "检查项：", ""]
    lines += [f"- {k}: {v}" for k, v in checks.items()]
    lines += ["", "纯度只针对已写入领域的成员；familiar / pending / 拒写样本另见 JSON。初始到最终 CD 的改善不是与 Frozen 基线配对的算法收益。", ""]
    (PLAN / f"round_{args.round}_review.md").write_text("\n".join(lines))
    print(json.dumps({k: report[k] for k in ("round", "integrity_ok", "diagnostic_checks", "performance", "domains")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
