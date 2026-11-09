"""Run the frozen LeWM PushT GD jobs on two idle GPUs; preserve partial results."""
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
ROOT = HERE.parents[1] / "CL-JEPA-cudagraph"
OUTPUT = ROOT / "eval_outputs/baselines/lewm_pusht_GD_T16k_e10_20261108"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def completed(job):
    folder = Path(job["output"])
    names = ["final_eval_results.npz", "logs.json", "runtime_protocol.json", "frozen_weights_check.json"]
    if not all((folder / name).exists() for name in names):
        return False
    receipt = json.loads((folder / "runtime_protocol.json").read_text())
    if (receipt["planner"] != "gd" or receipt["model_epoch"] != 10 or receipt["diagnostic_only"]
            or receipt["condition"] != job["condition"] or receipt["max_replans"] != 30
            or receipt["executed_physical_steps_per_replan"] != 5):
        return False
    if not json.loads((folder / "frozen_weights_check.json").read_text())["unchanged"]:
        return False
    rows = [json.loads(line) for line in (folder / "logs.json").read_text().splitlines() if line.strip()]
    rates = [r["final_eval/success_rate"] for r in rows if "final_eval/success_rate" in r]
    with np.load(folder / "final_eval_results.npz", allow_pickle=False) as data:
        return len(data["successes"]) == 50 and bool(rates) and np.isclose(data["successes"].mean(), rates[-1])


def free_gpus():
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
    used = set(active.splitlines())
    return {row.split(",")[0].strip() for row in devices.splitlines() if row.split(",")[1].strip() not in used}


def main():
    locks = []
    for path in [HERE / ".queue.lock", *[ROOT / f"eval_outputs/baselines/.legacy_gd_gpu_{gpu}.lock" for gpu in [2, 3]]]:
        handle = path.open("a")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    protocol = json.loads((HERE / "protocol.json").read_text())
    identity = json.loads((HERE / "checkpoint_identity.json").read_text())
    target = json.loads((HERE / "target_manifest.json").read_text())
    jobs = json.loads((HERE / "commands.json").read_text())
    assert len(jobs) == 7 and {j["condition"] for j in jobs} == set(protocol["conditions"])
    assert sha(identity["checkpoint_file"]) == identity["checkpoint_sha256"]
    assert sha(Path(identity["checkpoint"]) / "hydra.yaml") == identity["config_sha256"]
    assert sha(target["path"]) == target["sha256"]
    for name, expected in json.loads((HERE / "source_receipt.json").read_text())["files"].items():
        assert sha(HERE / name) == expected, name
    check = json.loads((HERE / "gradient_checks.json").read_text())["lewm"]
    assert check["parameters_unchanged"] and check["finite_actions"] and check["action_gradient_norm"] > 0
    assert check["microbatch_size"] == 50 and check["peak_reserved_GiB"] < 30
    state = {"method": "lewm", "task": "pusht", "planner": "gd", "pid": os.getpid(),
             "checkpoint": identity["checkpoint"], "epoch": 10, "seed": 100,
             "gpus": ["2", "3"], "stage": "evaluating", "started": now(),
             "planned_episodes": 350, "jobs": {j["condition"]: {"status": "queued", "gpu": j["gpu"]} for j in jobs}}

    def persist():
        state["updated"] = now()
        state["episodes_completed"] = 50 * sum(v["status"] == "complete" for v in state["jobs"].values())
        save(HERE / "pipeline_status.json", state)
        save(OUTPUT / "pipeline_status.json", state)

    pending, running = [], {}
    for job in jobs:
        if completed(job):
            state["jobs"][job["condition"]].update(status="complete", resumed=True)
        elif (Path(job["output"]) / "logs.json").exists():
            raise RuntimeError(f"Partial result exists; inspect before resuming: {job['output']}")
        else:
            pending.append(job)
    persist()
    failed = False
    try:
        while pending or running:
            for gpu, (job, process, handle) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                ok = code == 0 and completed(job)
                state["jobs"][job["condition"]].update(status="complete" if ok else "failed", exit_code=code, ended=now())
                failed |= not ok
                del running[gpu]
                persist()
            if failed:
                if not running:
                    raise RuntimeError("A condition failed; completed and partial files were preserved")
            else:
                available = free_gpus()
                for gpu in ["2", "3"]:
                    candidates = [j for j in pending if j["gpu"] == gpu]
                    if gpu in running or gpu not in available or not candidates:
                        continue
                    job = candidates[0]
                    folder = Path(job["output"])
                    folder.mkdir(parents=True, exist_ok=True)
                    save(folder / "launch_command.json", job["command"])
                    save(folder / "checkpoint_identity.json", identity)
                    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
                                   "SDL_VIDEODRIVER": "dummy", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
                                   "MKL_NUM_THREADS": "4", "WANDB_MODE": "disabled"}
                    handle = (folder / "console.log").open("a")
                    process = subprocess.Popen(job["command"], cwd=HERE / "code", env=environment,
                                               stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
                    running[gpu] = (job, process, handle)
                    pending.remove(job)
                    state["jobs"][job["condition"]].update(status="running", pid=process.pid, started=now())
                    persist()
                    print(now(), "START", job["condition"], "GPU", gpu, flush=True)
            if pending or running:
                time.sleep(5)
        values = {}
        for job in jobs:
            with np.load(Path(job["output"]) / "final_eval_results.npz", allow_pickle=False) as data:
                success = data["successes"].astype(bool)
                steps = np.where(success, data["action_len"], 30)
            assert len(success) == 50 and np.isfinite(steps).all() and ((steps >= 0) & (steps <= 30)).all()
            values[job["condition"]] = {"episodes": 50, "successes": int(success.sum()),
                                      "success_rate": float(success.mean()), "mean_mpc_replans": float(steps.mean()),
                                      "mean_physical_steps": float(steps.mean() * 5)}
        summary = {"method": "lewm", "task": "pusht", "planner": "gd", "epoch": 10, "seed": 100,
                   "episodes": 350, "checkpoint_sha256": identity["checkpoint_sha256"], "conditions": values,
                   "seven_condition_success_rate": float(np.mean([v["success_rate"] for v in values.values()])),
                   "six_shift_success_rate": float(np.mean([values[c]["success_rate"] for c in protocol["conditions"][1:]])),
                   "color_success_rate": float(np.mean([values[c]["success_rate"] for c in ["red_agent", "red_object", "red_goal"]])),
                   "mean_mpc_replans": float(np.mean([v["mean_mpc_replans"] for v in values.values()])),
                   "mean_physical_steps": float(np.mean([v["mean_physical_steps"] for v in values.values()])),
                   "controller": {k: protocol[k] for k in ["objective", "physical_steps_per_replan", "max_replans", "max_physical_steps"]},
                   "mean_steps_definition": "All episodes; failures count as 30 replans / 150 physical steps."}
        save(OUTPUT / "summary.json", summary)
        state.update(stage="complete", ended=now())
        persist()
    except Exception as error:
        state.update(stage="failed", error=str(error), ended=now())
        persist()
        raise


if __name__ == "__main__":
    main()
