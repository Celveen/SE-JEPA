"""Evaluate the existing LeWM and DINO-WM checkpoints on seven H7t3 shapes."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
ROOT = WORKSPACE / "CL-JEPA-cudagraph"
PYTHON = "/data/zjc/envs/cljepa/bin/python"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def gpu_free(index):
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    ids = {r.split(",")[0].strip(): r.split(",")[1].strip() for r in devices.splitlines()}
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
    return ids[index] not in [r.strip() for r in active.splitlines()]


def completed(folder, epoch):
    names = ["final_eval_results.npz", "logs.json", "runtime_protocol.json", "frozen_weights_check.json"]
    if not all((folder / n).exists() for n in names):
        return False
    receipt = json.loads((folder / "runtime_protocol.json").read_text())
    if receipt["planner"] != "gd" or receipt["model_epoch"] != epoch or receipt["diagnostic_only"]:
        return False
    if not json.loads((folder / "frozen_weights_check.json").read_text())["unchanged"]:
        return False
    rows = [json.loads(line) for line in (folder / "logs.json").read_text().splitlines() if line.strip()]
    rates = [r["final_eval/success_rate"] for r in rows if "final_eval/success_rate" in r]
    with np.load(folder / "final_eval_results.npz", allow_pickle=False) as data:
        return len(data["successes"]) == 50 and bool(rates) and np.isclose(data["successes"].mean(), rates[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    assert args.gpu.isdigit()
    models = [
        {"method": "lewm", "checkpoint": str(WORKSPACE / "CL-JEPA/checkpoints/lewm_pushobj_bias16k"),
         "epoch": 10, "microbatch": 50, "output": str(ROOT / "eval_outputs/baselines/lewm_H7t3_GD_bias16k_e10_20261106")},
        {"method": "dinowm", "checkpoint": str(WORKSPACE / "CL-JEPA/checkpoints/dinowm_pushobj_bias16k_reference_e3"),
         "epoch": 3, "microbatch": 8, "output": str(ROOT / "eval_outputs/baselines/dinowm_H7t3_GD_bias16k_reference_e3_20261106")},
    ]
    targets = json.loads((HERE / "target_manifest.json").read_text())
    assert len(targets) == 7 and all(r["seed"] == 2000 and r["n"] == 50 for r in targets)
    jobs = []
    for model in models:
        ck = Path(model["checkpoint"])
        model["checkpoint_file"] = str(ck / "checkpoints" / f"model_{model['epoch']}.pth")
        model["checkpoint_sha256"] = sha(model["checkpoint_file"])
        model["config_sha256"] = sha(ck / "hydra.yaml")
        for block in targets:
            folder = Path(model["output"]) / block["chunk"]
            command = [
                PYTHON, "-u", str(HERE / "evaluation/evaluate.py"),
                f"ckpt_base_path={model['checkpoint']}", f"model_epoch={model['epoch']}",
                "n_evals=50", "seed=2000", f"eval_data_path={block['target']}",
                f"hydra.run.dir={folder}", f"planner.continual.tag={block['tag']}",
                "audit.planner_kind=gd", f"audit.condition={block['shape']}",
                f"audit.pretraining_shape_status={block['pretraining_shape_status']}",
                f"audit.expected_model_epoch={model['epoch']}",
                f"planner.sub_planner.microbatch_size={model['microbatch']}",
                f"audit.expected_microbatch_size={model['microbatch']}",
            ]
            jobs.append({"method": model["method"], "chunk": block["chunk"], "shape": block["shape"],
                         "output": str(folder), "target": block["target"], "target_sha256": block["target_sha256"],
                         "epoch": model["epoch"], "command": command})
    protocol = {"scope": "Existing checkpoint GD counterpart to the archived seven-shape CEM evaluations",
                "models": models, "seed": 2000, "episodes_per_shape": 50, "episodes_per_model": 350,
                "total_episodes": 700, "objective": "staged", "physical_steps_per_replan": 5,
                "max_replans": 30, "max_physical_steps": 150, "horizon_model_steps": 5,
                "gd": {"steps": 100, "lr": .1, "optimizer": "adam", "cosine_scheduler": True,
                       "initialization": "zero physical action", "action_noise": 0, "cuda_graph": False},
                "training_shapes": ["T","L","Z","+"], "unseen_shapes": ["I","square","small_tee"],
                "new_training": False, "gpu": args.gpu}
    path = HERE / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Existing protocol differs; preserve it instead of mixing results")
    save(path, protocol)
    save(HERE / "commands.json", jobs)
    if args.prepare_only:
        print(json.dumps({"jobs": len(jobs), "episodes": 700, "gpu": args.gpu}))
        return
    check = json.loads((HERE / "gradient_checks.json").read_text())
    for model in models:
        test = check[model["method"]]
        assert test["finite_actions"] and test["parameters_unchanged"] and test["action_gradient_norm"] > 0
        assert test["microbatch_size"] == model["microbatch"] and test["peak_reserved_GiB"] < 30
    for name, expected in json.loads((HERE / "source_receipt.json").read_text())["files"].items():
        assert sha(HERE / name) == expected, name
    lock = (HERE / ".queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_lock = (ROOT / f"eval_outputs/baselines/.legacy_gd_gpu_{args.gpu}.lock").open("a")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"pid": os.getpid(), "gpu": args.gpu, "stage": "queued", "started": now(),
             "total_jobs": 14, "planned_episodes": 700, "jobs": {}}
    states = {m["method"]: {"method": m["method"], "pid": os.getpid(), "gpu": args.gpu,
                           "stage": "queued", "planned_episodes": 350, "checkpoint": m["checkpoint"],
                           "epoch": m["epoch"], "planner": "gd", "jobs": {}} for m in models}

    def persist():
        state["updated"] = now()
        save(HERE / "pipeline_status.json", state)
        for m in models:
            states[m["method"]]["updated"] = state["updated"]
            save(Path(m["output"]) / "pipeline_status.json", states[m["method"]])
    persist()
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu,
                   "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "SDL_VIDEODRIVER": "dummy",
                   "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "WANDB_MODE": "disabled"}
    try:
        for model in models:
            method = model["method"]
            assert sha(model["checkpoint_file"]) == model["checkpoint_sha256"]
            assert sha(Path(model["checkpoint"]) / "hydra.yaml") == model["config_sha256"]
            for job in [j for j in jobs if j["method"] == method]:
                folder = Path(job["output"])
                key = method + "/" + job["chunk"]
                if completed(folder, model["epoch"]):
                    state["jobs"][key] = states[method]["jobs"][job["chunk"]] = {"status": "complete", "resumed": True}
                    persist()
                    continue
                assert sha(job["target"]) == job["target_sha256"]
                state.update(stage="waiting_for_gpu", active_job=key)
                states[method]["stage"] = "waiting_for_gpu"
                persist()
                while not gpu_free(args.gpu):
                    time.sleep(10)
                folder.mkdir(parents=True, exist_ok=True)
                save(folder / "launch_command.json", job["command"])
                save(folder / "checkpoint_identity.json", model)
                with (folder / "console.log").open("a") as log:
                    process = subprocess.Popen(job["command"], cwd=HERE / "code", env=environment,
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    entry = {"status": "running", "pid": process.pid, "started": now()}
                    state["jobs"][key] = states[method]["jobs"][job["chunk"]] = entry
                    state.update(stage="evaluating", active_child_pid=process.pid)
                    states[method].update(stage="evaluating", active_child_pid=process.pid, current_shape=job["shape"])
                    persist()
                    print(now(), "START", key, "GPU", args.gpu, flush=True)
                    code = process.wait()
                ok = code == 0 and completed(folder, model["epoch"])
                entry.update(status="complete" if ok else "failed", exit_code=code, ended=now())
                persist()
                if not ok:
                    raise RuntimeError(f"GD evaluation failed: {key}; inspect its console.log")
            summary = {"method": method, "epoch": model["epoch"], "checkpoint_sha256": model["checkpoint_sha256"],
                       "planner": "gd", "episodes": 350, "seed": 2000, "by_shape": {}}
            for job in [j for j in jobs if j["method"] == method]:
                with np.load(Path(job["output"]) / "final_eval_results.npz", allow_pickle=False) as data:
                    success = data["successes"].astype(bool)
                    steps = np.where(success, data["action_len"], 30)
                assert len(success) == 50 and np.isfinite(steps).all()
                summary["by_shape"][job["shape"]] = {"success_rate": float(success.mean()),
                    "successes": int(success.sum()), "episodes": 50, "mean_mpc_steps": float(steps.mean())}
            values = summary["by_shape"]
            summary["all_success_rate"] = float(np.mean([v["success_rate"] for v in values.values()]))
            summary["unseen_success_rate"] = float(np.mean([values[s]["success_rate"] for s in protocol["unseen_shapes"]]))
            summary["seen_success_rate"] = float(np.mean([values[s]["success_rate"] for s in protocol["training_shapes"]]))
            summary["mean_mpc_steps"] = float(np.mean([v["mean_mpc_steps"] for v in values.values()]))
            summary["mean_physical_steps"] = summary["mean_mpc_steps"] * 5
            save(Path(model["output"]) / "summary.json", summary)
            states[method].update(stage="complete", summary=summary, ended=now())
            persist()
        state.update(stage="complete", ended=now())
        persist()
    except Exception as error:
        state.update(stage="failed", error=str(error), ended=now())
        if "method" in locals() and states[method]["stage"] != "complete":
            states[method].update(stage="failed", error=str(error))
        persist()
        raise


if __name__ == "__main__":
    main()
