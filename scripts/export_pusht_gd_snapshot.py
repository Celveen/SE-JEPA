"""Archive the completed DINO-WM and LeWM seven-condition PushT GD results."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CONDITIONS = ["default", "blur", "snp", "dark", "red_agent", "red_object", "red_goal"]
PRIVATE = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    files, runs, results = [], [], {}
    stored_paths = set()

    def store(source, relative, role="result"):
        if relative.as_posix() in stored_paths:
            return
        before = source.stat()
        payload = source.read_bytes()
        after = source.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), source
        if PRIVATE.search(payload):
            raise ValueError(f"Credential-like content in {source}")
        compressed = source.name in {"logs.json", "per_iter_logs.json"} or source.suffix == ".jsonl"
        if compressed:
            relative = relative.with_name(relative.name + ".gz")
        stored = gzip.compress(payload, compresslevel=6, mtime=0) if compressed else payload
        assert len(stored) < 50 * 2**20, source
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(stored)
        stored_paths.add(relative.as_posix())
        files.append({"path": relative.as_posix(), "source": str(source), "bytes": len(stored),
                      "source_bytes": len(payload), "sha256": digest(stored),
                      "source_sha256": digest(payload), "gzip": compressed, "role": role})

    specs = [
        ("dinowm", "dinowm_pusht_random16k_e10_20261106", "gd", 3,
         "dinowm_random16k_original_e10_20261106", "last", 25, 6),
        ("lewm", "lewm_pusht_GD_T16k_e10_20261108", "", 10,
         "lewm_pusht_GD_T16k_e10_20261108", "staged", 5, 30),
    ]
    target_hashes = set()
    for method, name, subdir, epoch, plan_name, objective, executed, cap in specs:
        base = ROOT / "eval_outputs/baselines" / name
        summary = json.loads((base / "summary.json").read_text())
        expected = summary["by_planner"]["gd"] if method == "dinowm" else summary
        status = json.loads((base / "pipeline_status.json").read_text())
        assert status["stage"] == "complete"
        values = {}
        for condition in CONDITIONS:
            folder = base / subdir / condition
            relative = Path("completed_runs") / method / condition
            receipt = json.loads((folder / "runtime_protocol.json").read_text())
            frozen = json.loads((folder / "frozen_weights_check.json").read_text())
            config = yaml.safe_load((folder / ".hydra/config.yaml").read_text())
            assert receipt["planner"] == "gd" and receipt["model_epoch"] == epoch
            assert receipt["condition"] == condition and not receipt["diagnostic_only"]
            assert receipt["objective"] == objective and receipt["max_replans"] == cap
            assert receipt["executed_physical_steps_per_replan"] == executed
            assert receipt["max_physical_steps"] == 150 and receipt["optimizer_iterations"] == 100
            assert receipt["model_parameter_gradients_disabled"] and frozen["unchanged"]
            assert frozen["before_sha256"] == frozen["after_sha256"]
            assert config["seed"] == 100 and config["n_evals"] == 50
            rows = [json.loads(line) for line in (folder / "logs.json").read_text().splitlines() if line.strip()]
            log_rates = [r["final_eval/success_rate"] for r in rows if "final_eval/success_rate" in r]
            with np.load(folder / "final_eval_results.npz", allow_pickle=False) as data:
                success = data["successes"].astype(bool)
                replans = np.where(success, data["action_len"] / receipt["executed_model_steps_per_replan"], cap)
            assert len(success) == 50 and log_rates and np.isclose(success.mean(), log_rates[-1])
            assert np.isfinite(replans).all() and ((replans >= 0) & (replans <= cap)).all()
            values[condition] = {"episodes": 50, "successes": int(success.sum()),
                                 "success_rate": float(success.mean()), "mean_mpc_replans": float(replans.mean()),
                                 "mean_physical_steps": float(replans.mean() * executed)}
            for key in ["success_rate", "mean_mpc_replans", "mean_physical_steps"]:
                assert np.isclose(values[condition][key], expected["conditions"][condition][key])
            for source in sorted(folder.iterdir()):
                if source.is_file() and source.suffix in {".npz", ".json", ".jsonl", ".csv"}:
                    store(source, relative / source.name)
            for source in sorted((folder / ".hydra").glob("*.yaml")):
                store(source, relative / ".hydra" / source.name, "run_config")
            target = Path(config["eval_data_path"])
            target_hashes.add(digest(target.read_bytes()))
            store(target, Path("references") / method / "targets_T.pkl", "target")
            store(Path(config["ckpt_base_path"]) / "hydra.yaml",
                  Path("training_metadata") / method / "hydra.yaml", "model_config")
            runs.append({"directory": relative.as_posix(), "source": str(folder), "method": method,
                         "condition": condition, "episodes": 50, "recorded_success_rate": float(success.mean()),
                         "final_log_matches_results": True, "frozen_weights_unchanged": True})
        results[method] = {"task": "pusht", "planner": "gd", "epoch": epoch, "seed": 100, "episodes": 350,
                           "conditions": values, "training_shapes": ["T"],
                           "objective": objective, "physical_steps_per_replan": executed, "max_replans": cap,
                           "max_physical_steps": 150,
                           "seven_condition_success_rate": float(np.mean([v["success_rate"] for v in values.values()])),
                           "six_shift_success_rate": float(np.mean([values[c]["success_rate"] for c in CONDITIONS[1:]])),
                           "color_success_rate": float(np.mean([values[c]["success_rate"] for c in CONDITIONS[-3:]])),
                           "mean_mpc_replans": float(np.mean([v["mean_mpc_replans"] for v in values.values()])),
                           "mean_physical_steps": float(np.mean([v["mean_physical_steps"] for v in values.values()]))}
        for source in sorted(base.iterdir()):
            if source.is_file() and source.suffix in {".json", ".yaml", ".yml"}:
                store(source, Path("source_pipeline_metadata") / method / source.name, "pipeline_metadata")
        plan = WORKSPACE / "experiment_plans" / plan_name
        for source in sorted(plan.rglob("*")):
            relative = source.relative_to(plan)
            if not source.is_file() or "__pycache__" in relative.parts:
                continue
            if method == "dinowm" and len(relative.parts) > 1 and relative.parts[0] not in {"code", "evaluation_dual"}:
                continue
            if source.suffix in {".py", ".json", ".yaml", ".yml", ".md", ".patch"}:
                store(source, Path("experiment_plans") / plan_name / relative, "experiment_recipe")
    assert len(target_hashes) == 1 and len(runs) == 14
    for name in ["AGENTS.md", "EXPERIMENT_POLICY.md"]:
        store(WORKSPACE / name, Path("protocols") / name, "experiment_policy")
    manifest = {"format_version": 1, "created_utc": created,
                "source_git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "scope": "DINO-WM and LeWM PushT GD only; 7 conditions x 50 episodes x 2 models",
                "completed_directories": len(runs), "stored_bytes": sum(f["bytes"] for f in files),
                "files": sorted(files, key=lambda f: f["path"]), "runs": runs,
                "shared_target_sha256": next(iter(target_hashes)),
                "excluded": ["model weights", "training datasets", "raw console logs", "rendered observation arrays", "CEM result payloads"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "gd_results.json").write_text(json.dumps(results, indent=2))
    with (output / "run_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(runs)
    table = "\n".join(f"| {c} | {results['dinowm']['conditions'][c]['success_rate'] * 100:.0f}% | {results['lewm']['conditions'][c]['success_rate'] * 100:.0f}% |" for c in CONDITIONS)
    readme = f"""# Completed PushT GD evaluations: DINO-WM and LeWM

Created: {created}

Both methods completed all seven conditions, 50 episodes each, seed 100: **350 per method, 700 total**. All final success rates match numerical logs and source summaries; all model parameter/buffer hashes are unchanged.

| Condition | DINO-WM GD | LeWM GD |
|---|---:|---:|
{table}
| Seven-condition mean | 39.71% | 42.86% |
| Six-shift mean | 38.00% | 39.33% |
| Three-color mean | 42.00% | 18.00% |
| Mean physical steps | 102.50 | 100.44 |

## Protocols and interpretation

- DINO-WM: 16,000 random T training trajectories, fixed epoch 3; `last` objective, execute 25 physical steps per replan, at most 6 replans. Historical e10 directory names do not change the effective three-epoch budget.
- LeWM: reused T-only T16k checkpoint, fixed epoch 10, tensor-identical to the previous VIS50 CEM latest checkpoint; `staged` objective, execute 5 physical steps per replan, at most 30 replans. No new training.
- The initial/goal target files are identical. Both methods use GD100, lr 0.1, Adam with cosine decay, frozen model parameters and a 150-physical-step cap. Their training data, training budgets, and controllers differ; this table does not represent a controlled model-only comparison.
- `red_goal` recolors the fixed reference marker. Six-shift means exclude original T. Mean steps include failures at the configured cap. Feedback corruption seeds are recorded; prior CEM noise realizations were not saved.

## Files and verification

`gd_results.json` contains GD-only metrics recomputed from the final NPZ files. `completed_runs/` contains the 14 complete result sets, compressed numerical logs, Hydra configurations, launch commands and frozen-weight checks. `references/` contains the small shared target files. Code snapshots, controller recipes and checkpoint identities are preserved under `experiment_plans/` and `source_pipeline_metadata/`.

DINO-WM source pipeline metadata also describes its completed CEM branch; only GD result payloads are included in this release. Original absolute paths in configurations must be remapped when rerunning elsewhere. Model weights and training datasets remain external.

From the repository root:

```bash
python scripts/verify_research_snapshot.py research_snapshots/{output.name}
```

`manifest.json` records source and stored SHA256 values for every payload. Gzip files use deterministic compression and can be opened with Python's `gzip` module. This focused release supplements the November 7 archive without rewriting it.
"""
    (output / "README.md").write_text(readme)
    print(json.dumps({"snapshot": str(output), "completed_directories": len(runs), "episodes": 700,
                      "payload_files": len(files), "MiB": round(manifest["stored_bytes"] / 2**20, 2)}))


if __name__ == "__main__":
    main()
