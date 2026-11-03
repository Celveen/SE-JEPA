import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT.parent / "CL-JEPA/checkpoints"
DATASET = ROOT.parent / "CL-JEPA/data/pushobj_bias_16k"
UNSEEN = {"I", "square", "small_tee"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def finished(folder):
    final = folder / "final_eval_results.npz"
    log = folder / "logs.json"
    if not final.exists() or not log.exists():
        return False
    try:
        records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        rates = [record["final_eval/success_rate"] for record in records if "final_eval/success_rate" in record]
        with np.load(final, allow_pickle=False) as data:
            successes = data["successes"]
        return len(successes) == 50 and bool(rates) and bool(np.isclose(successes.mean(), rates[-1]))
    except (OSError, ValueError, KeyError, EOFError):
        return False


def environment(gpus):
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(gpus),
        "DATASET_DIR": str(DATASET.parent),
        "WANDB_MODE": "offline",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "2",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_SHM_DISABLE": "1",
        "SDL_VIDEODRIVER": "dummy",
    }


def training_command(method, checkpoint, gpus, epochs, batch_size):
    encoder = "dino" if method == "dinowm" else method
    if method == "dinowm":
        return [
            sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node", str(len(gpus)), str(ROOT / "train.py"),
            "--config-name", "train_dinowm_reference",
            f"env.dataset.data_path={DATASET}", "env.num_workers=8",
            f"training.epochs={epochs}", f"training.batch_size={batch_size}",
            f"hydra.run.dir={checkpoint}",
        ]
    return [
        sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node", str(len(gpus)), str(ROOT / "train.py"),
        "env=pusht", "env.name=pushobj", f"env.dataset.data_path={DATASET}",
        "env.num_workers=8", f"encoder={encoder}", "model.train_encoder=false",
        "model.train_decoder=false", "has_decoder=false", "training.straighten=False",
        "+training.eval_lpips=false", f"training.epochs={epochs}", f"training.batch_size={batch_size}",
        "training.mixed_precision=bf16",
        "training.save_every_x_epoch=1", "training.save_every_x_iterations=0",
        f"hydra.run.dir={checkpoint}",
    ]


def evaluation_command(checkpoint, folder, block, epoch="latest"):
    return [
        sys.executable, "-u", str(ROOT / "plan.py"), "--config-name", "cont_plan_gd_pushobj.yaml",
        f"ckpt_base_path={checkpoint}", f"model_epoch={epoch}",
        f"eval_data_path={ROOT / 'data/cont_streams/H7t3' / (block['chunk'] + '.pkl')}",
        "n_evals=50", "+wandb_logging=false", "planner.continual.w=0",
        f"planner.continual.tag={block['shape']}", "planner.continual.state_in=null",
        "planner._target_=planning.continual.SlowFrozenMPCPlanner",
        "planner.sub_planner.target=planning.cem.CEMPlanner",
        "+planner.sub_planner.topk=30", "+planner.sub_planner.num_samples=200",
        "planner.sub_planner.opt_steps=10", f"seed={block['seed']}", f"hydra.run.dir={folder}",
    ]


def training_finished(checkpoint, epochs):
    metrics = checkpoint / "metrics.jsonl"
    if not (checkpoint / "checkpoints" / f"model_{epochs}.pth").exists() or not metrics.exists():
        return False
    records = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    return any(record.get("epoch") == epochs and math.isfinite(record.get("val_loss", float("nan"))) for record in records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["dino_cls", "r3m", "ddpwm", "dinowm"])
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval-epoch", default="latest")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2000])
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.epochs is None:
        if arguments.method == "dinowm" and arguments.train:
            parser.error("DINO-WM training needs an explicit --epochs budget; the reference config uses 100")
        arguments.epochs = 3
    gpus = arguments.gpus.split(",")
    if not gpus or any(not gpu.isdigit() for gpu in gpus) or len(set(gpus)) != len(gpus):
        parser.error("--gpus must contain distinct numeric GPU indices")
    if arguments.epochs < 1:
        parser.error("--epochs must be positive")
    if arguments.eval_epoch != "latest" and (not arguments.eval_epoch.isdigit() or int(arguments.eval_epoch) < 1):
        parser.error("--eval-epoch must be latest or a positive epoch number")
    if arguments.method == "ddpwm" and arguments.train:
        parser.error("DDP-WM uses its existing checkpoint")
    batch_size = arguments.batch_size if arguments.batch_size is not None else (32 if arguments.method == "dinowm" else 64)
    if batch_size < len(gpus) or batch_size % len(gpus):
        parser.error("--batch-size must be positive and divisible by the GPU count")
    if arguments.method == "ddpwm":
        checkpoint_name = "ddpwm_bias16k"
    elif arguments.method == "dinowm" and not arguments.train:
        checkpoint_name = "dinowm_pusht_official"
    elif arguments.method == "dinowm" and arguments.train:
        checkpoint_name = f"dinowm_pushobj_bias16k_reference_e{arguments.epochs}"
    elif arguments.train:
        checkpoint_name = f"{arguments.method}_pushobj_bias16k_e{arguments.epochs}"
    else:
        checkpoint_name = f"{arguments.method}_T16k_e10"
    checkpoint = CHECKPOINTS / checkpoint_name
    transfer = arguments.method != "ddpwm" and not arguments.train
    suffix = "_official_pusht" if arguments.method == "dinowm" and transfer else ("_T16k" if transfer else "")
    output = ROOT / "eval_outputs/baselines" / f"{arguments.method}_H7t3{suffix}"
    training_dataset = DATASET.parent / "pusht_noise" if arguments.method == "dinowm" and transfer else (DATASET.parent / "pushobj_T16k_bias" if transfer else DATASET)
    blocks = json.loads((ROOT / "data/cont_streams/H7t3/manifest.json").read_text())["blocks"]
    assert len(blocks) == 21 and {block["seed"] for block in blocks} == {2000, 2100, 2200}
    if len(set(arguments.seeds)) != len(arguments.seeds) or not set(arguments.seeds).issubset({2000, 2100, 2200}):
        parser.error("--seeds must contain distinct H7t3 seeds from 2000, 2100, 2200")
    blocks = [block for block in blocks if block["seed"] in arguments.seeds]
    missing = [block for block in blocks if not finished(output / block["chunk"])]
    train = training_command(arguments.method, checkpoint, gpus, arguments.epochs, batch_size) if arguments.train else None
    plan = {
        "method": arguments.method, "gpus": gpus, "checkpoint": str(checkpoint),
        "dataset": str(training_dataset),
        "evaluation_scope": "T-only checkpoint transferred to H7t3" if transfer else "PushObj checkpoint evaluated on H7t3",
        "output": str(output), "train_command": train, "training_batch_size": batch_size if train else None,
        "evaluation_epoch": arguments.eval_epoch,
        "training_profile": "train_dinowm_reference" if arguments.method == "dinowm" and arguments.train else None,
        "test_seeds": arguments.seeds, "episodes_per_shape": 50,
        "planned_episodes": len(blocks) * 50,
        "total_chunks": len(blocks), "already_complete": len(blocks) - len(missing),
        "missing_chunks": [block["chunk"] for block in missing],
        "first_evaluation": evaluation_command(checkpoint, output / missing[0]["chunk"], missing[0], arguments.eval_epoch) if missing else None,
    }
    if arguments.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not DATASET.is_dir():
        raise FileNotFoundError(DATASET)
    if train is None and not (checkpoint / "checkpoints/model_latest.pth").exists():
        raise FileNotFoundError(checkpoint)
    needs_training = train is not None and not training_finished(checkpoint, arguments.epochs)
    if missing or needs_training:
        gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
        gpu_uuids = {row.split(",")[0].strip(): row.split(",")[1].strip() for row in gpu_rows.splitlines()}
        selected_uuids = {gpu_uuids[gpu] for gpu in gpus}
        applications = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        busy = [row for row in applications.splitlines() if row.split(",")[0].strip() in selected_uuids]
        if busy:
            raise RuntimeError(f"Requested GPUs already have active processes: {busy}")
    output.mkdir(parents=True, exist_ok=True)
    lock_file = (output / ".pipeline.lock").open("a")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_lock = threading.Lock()
    state = {**plan, "pid": os.getpid(), "started": now(), "stage": "starting", "jobs": {}}

    def save_state(job=None, **updates):
        with state_lock:
            if job is not None:
                name, details = job
                state["jobs"].setdefault(name, {}).update(details)
            state.update(updates)
            state["updated"] = now()
            temporary = output / "pipeline_status.json.tmp"
            temporary.write_text(json.dumps(state, indent=2))
            temporary.replace(output / "pipeline_status.json")

    def run(command, log_path, assigned_gpus):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(now(), "START", shlex.join(command), "GPUS", ",".join(assigned_gpus), flush=True)
        with log_path.open("a") as log:
            log.write(f"\n{now()} COMMAND {shlex.join(command)}\n")
            log.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=environment(assigned_gpus), stdout=log, stderr=subprocess.STDOUT)
            result = process.wait()
            log.write(f"\n{now()} EXIT={result}\n")
        return result

    save_state()
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True)
    (output / "source_revision.txt").write_text(revision)
    patch = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--", "models/encoder/r3m/models/models_r3m.py"], text=True)
    (output / "encoder_local_changes.patch").write_text(patch)
    source_changes = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--", "train.py", "conf/train.yaml", "models/visual_world_model.py", "models/dino.py", "models/vit.py"], text=True)
    (output / "training_source_changes.patch").write_text(source_changes)
    if arguments.method == "dinowm" and arguments.train:
        (output / "training_reference_config.yaml").write_text((ROOT / "conf/train_dinowm_reference.yaml").read_text())
    if needs_training:
        save_state(stage="training")
        result = run(train, output / "training_console.log", gpus)
        if result != 0 or not training_finished(checkpoint, arguments.epochs):
            save_state(stage="training_failed", train_exit_code=result)
            return 1
    save_state(stage="evaluating")

    def worker(gpu, assigned_blocks):
        failures = []
        for block in assigned_blocks:
            name = block["chunk"]
            folder = output / name
            save_state(job=(name, {"gpu": gpu, "status": "running", "started": now()}))
            result = run(evaluation_command(checkpoint, folder, block, arguments.eval_epoch), folder / "console.log", [gpu])
            complete = result == 0 and finished(folder)
            save_state(job=(name, {"status": "complete" if complete else "failed", "exit_code": result, "ended": now()}))
            if not complete:
                failures.append(name)
        return failures

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu, missing[index::len(gpus)]) for index, gpu in enumerate(gpus)]
        failures = [name for future in futures for name in future.result()]
    all_successes, unseen_successes = [], []
    for block in blocks:
        folder = output / block["chunk"]
        if not finished(folder):
            continue
        with np.load(folder / "final_eval_results.npz", allow_pickle=False) as data:
            values = data["successes"].astype(bool).tolist()
        all_successes.extend(values)
        if block["shape"] in UNSEEN:
            unseen_successes.extend(values)
    completed = sum(finished(output / block["chunk"]) for block in blocks)
    summary = {
        "complete_chunks": completed, "episodes": len(all_successes),
        "success_rate": float(np.mean(all_successes)) if all_successes else None,
        "unseen_success_rate": float(np.mean(unseen_successes)) if unseen_successes else None,
    }
    save_state(stage="complete" if not failures and completed == len(blocks) else "evaluation_failed", failures=failures, summary=summary, ended=now())
    print(now(), "SUMMARY", json.dumps(summary), flush=True)
    return 0 if not failures and completed == len(blocks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
