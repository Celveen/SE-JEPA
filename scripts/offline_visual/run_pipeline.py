import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

from common import CACHE, DOMAINS, ROOT, WORKSPACE, save_json


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def gpu_is_free(gpu):
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    identifiers = {row.split(",")[0].strip(): row.split(",")[1].strip() for row in rows.splitlines()}
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    return not any(row.split(",")[0].strip() == identifiers[str(gpu)] for row in active.splitlines())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["dreamerv3", "iris"])
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=3)
    arguments = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    output = ROOT / "eval_outputs/baselines" / f"{arguments.method}_VIS350"
    checkpoint = WORKSPACE / "CL-JEPA/checkpoints" / f"{arguments.method}_T16k_offline_e{arguments.epochs}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    pipeline_lock = (output / ".pipeline.lock").open("a")
    fcntl.flock(pipeline_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"method": arguments.method, "pid": os.getpid(), "gpu": arguments.gpu, "epochs": arguments.epochs, "epochs_per_stage": arguments.epochs, "conditions": DOMAINS, "episodes_per_condition": 50, "total_evaluation_episodes": 350, "planning_seed": 100, "checkpoint": str(checkpoint), "stage": "waiting_for_gpu", "started": now()}

    def update(**values):
        state.update(values, updated=now())
        save_json(output / "pipeline_status.json", state)

    update()
    shared_lock = (ROOT / "eval_outputs/baselines" / f".offline_visual_gpu_{arguments.gpu}.lock").open("a")
    while True:
        try:
            fcntl.flock(shared_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if gpu_is_free(arguments.gpu):
                break
            fcntl.flock(shared_lock, fcntl.LOCK_UN)
        except BlockingIOError:
            pass
        time.sleep(10)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(arguments.gpu), "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "JAX_COMPILATION_CACHE_DIR": str(CACHE / "jax_compilation"), "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "SDL_VIDEODRIVER": "dummy", "TORCH_HOME": str(CACHE / "torch")}
    dreamer_python = WORKSPACE / "baselines/dreamerv3-env/bin/python"
    torch_python = Path('/data/zjc/envs/cljepa/bin/python')
    training_python = dreamer_python if arguments.method == "dreamerv3" else torch_python
    training_script = scripts / ("train_dreamer.py" if arguments.method == "dreamerv3" else "train_iris.py")
    relevant = ["common.py", "evaluate.py", training_script.name, "dreamer_model.py" if arguments.method == "dreamerv3" else "iris_model.py"]
    source_hashes = {name: file_hash(scripts / name) for name in relevant}
    receipt = checkpoint / "adapter_sources.json"
    if receipt.exists() and json.loads(receipt.read_text()) != source_hashes:
        raise RuntimeError("Adapter code changed since this training directory was created")
    save_json(receipt, source_hashes)
    (checkpoint / "hydra.yaml").write_text(yaml.safe_dump({"method": arguments.method, "frameskip": 5, "num_hist": 3, "num_pred": 1, "img_size": 64, "training": {"epochs": arguments.epochs, "seed": 0}, "env": {"name": "pushobj", "dataset": {"data_path": str(WORKSPACE / "CL-JEPA/data/pushobj_T16k_bias")}}, "planning_backend": "scripts/offline_visual/evaluate.py", "parameter_source": "official upstream model modules; IRIS continuous-control adapter disclosed in protocol.json"}, sort_keys=False))

    def run(command, name):
        print(now(), "RUN", " ".join(map(str, command)), flush=True)
        with (output / name).open("a") as log:
            process = subprocess.Popen(list(map(str, command)), env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            update(active_child_pid=process.pid)
            return process.wait()

    try:
        if not (checkpoint / "training_complete.json").exists():
            update(stage="training")
            code = run([training_python, "-u", training_script, "--output", checkpoint, "--epochs", str(arguments.epochs)], "training_console.log")
            if code != 0 or not (checkpoint / "training_complete.json").exists():
                update(stage="training_failed", exit_code=code)
                return 1
        update(stage="starting_evaluation_server")
        model_file = checkpoint / ("model_latest.pkl" if arguments.method == "dreamerv3" else "world_model.pt")
        checksum = file_hash(model_file)
        if arguments.method == "iris":
            checksum = hashlib.sha256((checksum + file_hash(checkpoint / "tokenizer_model.pt")).encode()).hexdigest()
        socket = Path(f"/tmp/sejepa_{arguments.method}_{os.getpid()}.sock")
        server_command = [training_python, "-u", scripts / ("serve_dreamer.py" if arguments.method == "dreamerv3" else "serve_iris.py")]
        server_command += ["--checkpoint", model_file] if arguments.method == "dreamerv3" else ["--checkpoint-dir", checkpoint]
        server_command += ["--socket", socket]
        with (output / "inference_server.log").open("a") as log:
            server = subprocess.Popen(list(map(str, server_command)), env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 240
                while not socket.exists():
                    if server.poll() is not None:
                        raise RuntimeError("Inference server exited before it became ready")
                    if time.monotonic() > deadline:
                        raise TimeoutError("Inference server startup timed out")
                    time.sleep(1)
                update(stage="evaluating", server_pid=server.pid)
                code = run([torch_python, "-u", scripts / "evaluate.py", "--method", arguments.method, "--checkpoint-dir", checkpoint, "--checkpoint-sha256", checksum, "--socket", socket, "--output", output], "evaluation_console.log")
            finally:
                server.terminate()
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                if socket.exists():
                    socket.unlink()
        if code != 0:
            update(stage="evaluation_failed", exit_code=code)
            return 1
        rates, steps = {}, {}
        for domain in DOMAINS:
            with np.load(output / domain / "final_eval_results.npz") as result:
                assert len(result["successes"]) == 50
                rates[domain] = float(np.mean(result["successes"]))
                steps[domain] = float(np.mean(np.where(np.isfinite(result["action_len"]), result["action_len"], 30)))
        summary = {"episodes": 350, "domain_success_rates": rates, "domain_mean_mpc_steps": steps, "all_seven_success_rate": float(np.mean(list(rates.values()))), "six_shifts_success_rate": float(np.mean([rates[domain] for domain in DOMAINS[1:]])), "six_shifts_mean_mpc_steps": float(np.mean([steps[domain] for domain in DOMAINS[1:]]))}
        save_json(output / "summary.json", summary)
        update(stage="complete", summary=summary, ended=now())
        return 0
    except Exception as error:
        update(stage="failed", error=str(error), ended=now())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
