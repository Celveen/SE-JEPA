"""Resume-safe data -> 10-epoch training -> frozen original-controller evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PLAN = WORKSPACE / "experiment_plans/dinowm_random16k_original_e10_20261106"
PYTHON = "/data/zjc/envs/cljepa/bin/python"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def process_alive(pid, expected):
    path = Path(f"/proc/{pid}/cmdline")
    return path.exists() and expected.encode() in path.read_bytes()


def gpus_free(gpus):
    names = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    ids = {row.split(",")[0].strip(): row.split(",")[1].strip() for row in names.splitlines()}
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
    return not any(ids[gpu] in active.splitlines() for gpu in gpus)


def training_finished(checkpoint):
    if not (checkpoint / "checkpoints/model_10.pth").exists():
        return False
    path = checkpoint / "metrics.jsonl"
    if not path.exists():
        return False
    return any(row.get("epoch") == 10 and math.isfinite(row.get("val_loss", float("nan")))
               for row in [json.loads(line) for line in path.read_text().splitlines() if line.strip()])


def finished(folder, episodes=50):
    path = folder / "final_eval_results.npz"
    log = folder / "logs.json"
    receipt = folder / "frozen_weights_check.json"
    if not path.exists() or not log.exists() or not receipt.exists():
        return False
    if not json.loads(receipt.read_text()).get("unchanged"):
        return False
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    rates = [r["final_eval/success_rate"] for r in rows if "final_eval/success_rate" in r]
    with np.load(path, allow_pickle=False) as data:
        return len(data["successes"]) == episodes and bool(rates) and np.isclose(data["successes"].mean(), rates[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["pushobj", "pusht"])
    parser.add_argument("--gpus", required=True)
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    assert len(gpus) == 2 and len(set(gpus)) == 2 and all(g.isdigit() for g in gpus)
    code = PLAN / "code"
    data = WORKSPACE / "CL-JEPA/data" / f"dinowm_{args.task}_random16k_20261106"
    checkpoint = WORKSPACE / "CL-JEPA/checkpoints" / f"dinowm_{args.task}_random16k_orig_e10_20261106"
    output = ROOT / "eval_outputs/baselines" / f"dinowm_{args.task}_random16k_e10_20261106"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    lock = (output / ".pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_lock = threading.Lock()
    state = {"task": args.task, "pid": os.getpid(), "gpus": gpus, "stage": "waiting_for_dataset",
             "dataset": str(data), "checkpoint": str(checkpoint), "code_snapshot": str(code),
             "epochs": 10, "training_trajectories": 16000, "validation_trajectories": 1600,
             "max_physical_steps": 150, "physical_steps_per_replan": 25, "max_replans": 6,
             "planned_episodes": 100 if args.task == "pushobj" else 350,
             "started": now(), "jobs": {}}

    def update(**values):
        with state_lock:
            state.update(values, updated=now())
            write_json(output / "pipeline_status.json", state)
    update()
    try:
        while not (data / "generation_complete.json").exists():
            launch = json.loads((PLAN / args.task / "generation_launch.json").read_text())
            if not process_alive(launch["pid"], "gen_dinowm_random_dataset.py"):
                raise RuntimeError("Dataset generator stopped before completion; inspect generation_console.log")
            time.sleep(10)
        audit = json.loads((data / "dataset_audit.json").read_text())
        assert audit["complete"] and audit["splits"]["train"]["episodes"] == 16000
        assert audit["splits"]["val"]["episodes"] == 1600
        assert audit["train_validation_duplicate_trajectories"] == 0
        for name, expected in json.loads((PLAN / "source_receipt.json").read_text())["files"].items():
            assert file_hash(code / name) == expected, name

        # Test files are only inspected after collection; they never guide sampling.
        target_names = ["pushobj_L.pkl", "pushobj_plus.pkl"] if args.task == "pushobj" else ["pusht_T.pkl"]
        initial = []
        for name in target_names:
            with (PLAN / "targets" / name).open("rb") as handle:
                targets = pickle.load(handle)
            initial.extend(np.asarray(s["states"][0], np.float32)[:5] for s in targets["segments"][:50])
        training = torch.load(data / "train/states.pth", map_location="cpu", weights_only=True).numpy()
        rows = np.ascontiguousarray(training, dtype=np.float32).reshape(-1, 5).view("V20").reshape(-1)
        probes = np.ascontiguousarray(initial, dtype=np.float32).view("V20").reshape(-1)
        matching = np.isin(probes, rows)
        write_json(output / "heldout_overlap_check.json", {
            "training_state_rows": len(rows), "test_initial_states": len(probes),
            "exact_matches": int(matching.sum()), "target_files": target_names,
            "generation_used_test_targets": False,
        })
        if matching.any():
            raise RuntimeError("Exact training/test initial-state matches need review; data are not silently resampled")
        del rows, training, probes
        update(stage="waiting_for_gpus")
        gpu_locks = []
        while True:
            try:
                for gpu in sorted(gpus):
                    handle = (PLAN / f".gpu_{gpu}.lock").open("a")
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    gpu_locks.append(handle)
                if gpus_free(gpus):
                    break
            except BlockingIOError:
                pass
            for handle in gpu_locks:
                handle.close()
            gpu_locks.clear()
            time.sleep(10)
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus),
                       "DATASET_DIR": str(data.parent), "WANDB_MODE": "disabled",
                       "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "SDL_VIDEODRIVER": "dummy",
                       "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "2",
                       "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}
        training_command = [
            PYTHON, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
            str(code / "train.py"), "--config-name", "train_dinowm_random16k_e10",
            f"env.dataset.data_path={data}", "training.epochs=10", "training.batch_size=32",
            "env.num_workers=4", f"hydra.run.dir={checkpoint}",
        ]
        write_json(output / "training_command.json", training_command)
        if not training_finished(checkpoint):
            update(stage="training")
            with (output / "training_console.log").open("a") as log:
                child = subprocess.Popen(training_command, cwd=code, env=environment,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                update(training_pid=child.pid)
                result = child.wait()
            if result or not training_finished(checkpoint):
                raise RuntimeError(f"Training did not finish epoch 10 (exit {result})")
        update(stage="evaluating")
        model_hash = file_hash(checkpoint / "checkpoints/model_10.pth")
        write_json(output / "evaluation_checkpoint.json", {"epoch": 10, "sha256": model_hash})
        conditions = [("L", "pushobj_L.pkl", [], 2000), ("plus", "pushobj_plus.pkl", [], 2000)] if args.task == "pushobj" else [
            ("default", "pusht_T.pkl", [], 100),
            ("blur", "pusht_T.pkl", ["ood_corruption=blur", "ood_level=2.0"], 100),
            ("snp", "pusht_T.pkl", ["ood_corruption=snp1", "ood_level=0.01"], 100),
            ("dark", "pusht_T.pkl", ["ood_corruption=dark", "ood_level=0.9"], 100),
            ("red_agent", "pusht_T.pkl", ["++env_kwargs_override.agent_color=Red"], 100),
            ("red_object", "pusht_T.pkl", ["++env_kwargs_override.color=Red"], 100),
            ("red_goal", "pusht_T.pkl", ["++env_kwargs_override.goal_color=Red"], 100),
        ]

        def evaluate_lane(gpu, assigned):
            for condition, target, overrides, seed in assigned:
                folder = output / condition
                if finished(folder):
                    with state_lock:
                        state["jobs"][condition] = {"status": "complete", "gpu": gpu, "resumed": True}
                    update()
                    continue
                folder.mkdir(exist_ok=True)
                command = [PYTHON, "-u", str(code / "scripts/plan_dinowm_random.py"),
                           "--config-name", "plan_dinowm_original_budget",
                           f"ckpt_base_path={checkpoint}", "model_epoch=10", "n_evals=50",
                           f"eval_data_path={PLAN / 'targets' / target}", f"seed={seed}",
                           "+wandb_logging=false", f"planner.continual.tag={condition}",
                           f"hydra.run.dir={folder}", *overrides]
                write_json(folder / "launch_command.json", command)
                with (folder / "console.log").open("a") as log:
                    child = subprocess.Popen(command, cwd=code,
                                             env={**environment, "CUDA_VISIBLE_DEVICES": gpu},
                                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    with state_lock:
                        state["jobs"][condition] = {"status": "running", "pid": child.pid, "gpu": gpu, "started": now()}
                    update()
                    result = child.wait()
                success = result == 0 and finished(folder)
                with state_lock:
                    state["jobs"][condition].update(status="complete" if success else "failed",
                                                    exit_code=result, ended=now())
                update()
                if not success:
                    raise RuntimeError(f"Evaluation failed: {condition}")

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(evaluate_lane, gpu, conditions[index::2]) for index, gpu in enumerate(gpus)]
            for future in as_completed(futures):
                future.result()
        summary = {"task": args.task, "episodes": 50 * len(conditions), "epoch": 10,
                   "checkpoint_sha256": model_hash, "conditions": {},
                   "mean_steps_definition": "All episodes; failures use six MPC replans / 150 physical steps."}
        for condition, *_ in conditions:
            with np.load(output / condition / "final_eval_results.npz", allow_pickle=False) as data_row:
                success = data_row["successes"].astype(bool)
                mpc = np.where(success, data_row["action_len"] / 5, 6)
            summary["conditions"][condition] = {
                "episodes": 50, "successes": int(success.sum()), "success_rate": float(success.mean()),
                "mean_mpc_replans": float(mpc.mean()), "mean_physical_steps": float(mpc.mean() * 25),
            }
        summary["all_success_rate"] = float(np.mean([r["success_rate"] for r in summary["conditions"].values()]))
        if args.task == "pusht":
            summary["color_success_rate"] = float(np.mean([summary["conditions"][d]["success_rate"]
                                                          for d in ("red_agent", "red_object", "red_goal")]))
            summary["six_shift_success_rate"] = float(np.mean([v["success_rate"] for k,v in summary["conditions"].items() if k != "default"]))
        write_json(output / "summary.json", summary)
        update(stage="complete", summary=summary, ended=now())
        return 0
    except Exception as error:
        update(stage="failed", error=str(error), ended=now())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
