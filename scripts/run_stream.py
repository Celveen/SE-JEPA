"""Run a deployment stream (scripts/build_streams.py) for one method on PushObj or PushT.

    sejepa   SE-JEPA. The chunks run in stream order as one chain: each chunk starts from the memory state
             (continual_state.pt) the previous chunk saved. With --state-in, every chunk instead evaluates that
             saved state read-only (fast adaptation and learner reads as usual; no writes, training or merges),
             so chunks are independent.
    adajepa  AdaJEPA: within-episode fast adaptation, reset after every episode. Chunks are independent.
    frozen   No adaptation, batched MPC. Chunks are independent.

Independent chunks run in parallel, one at a time per entry of --gpus (an id may repeat). Finished chunks are
skipped, so re-running the same command resumes. Output: <root>/<method>/<chunk>/ (logs.json holds
final_eval/success_rate).

    python scripts/run_stream.py --stream shape_s100 --method sejepa --gpus 0 --ckpt /abs/ckpt --epoch 3
    python scripts/run_stream.py --stream shape_test --method sejepa --gpus 0,1 --ckpt /abs/ckpt --epoch 3 \
        --state-in eval_outputs/shape_s100/sejepa/chunk_34_T/continual_state.pt --root eval_outputs/test_s100
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "sejepa": [],  # the default planner of the configs
    "adajepa": ["planner._target_=planning.continual.ContinualMPCPlanner"],  # w = 0: reset every episode
    "frozen": ["planner._target_=planning.mpc.MPCPlanner", "~planner.adapt", "~planner.probe", "~planner.continual"],
}
print_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(msg, flush=True)


def finished(out, need_state=False):
    logs = out / "logs.json"
    ok = (out / "final_eval_results.npz").exists() and logs.exists() and "final_eval/success_rate" in logs.read_text()
    return ok and (not need_state or (out / "continual_state.pt").exists())


def command(b, out, a, extra):
    n = min(b["n"], a.max_episodes) if a.max_episodes else b["n"]
    cmd = [sys.executable, str(ROOT / "plan.py"), "--config-name", a.config,
           f"ckpt_base_path={Path(a.ckpt).resolve()}", f"model_epoch={a.epoch}",
           f"eval_data_path={ROOT / a.streams / a.stream / (b['chunk'] + '.pkl')}", f"n_evals={n}",
           f"seed={b['seed']}", f"hydra.run.dir={out}"] + METHODS[a.method] + b.get("overrides", []) + extra
    if a.method != "frozen":
        cmd.append(f"planner.continual.tag={b['tag']}")
    return cmd + a.extra


def run(cmd, out, gpu):
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "WANDB_MODE": "offline", "OMP_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "2", "SDL_VIDEODRIVER": "dummy"}
    log(f"START {out} GPU={gpu}")
    t0 = time.monotonic()
    with (out / "console.log").open("w") as fh:
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    log(f"{'DONE' if rc == 0 else 'FAIL'} {out} seconds={time.monotonic() - t0:.0f}" + ("" if rc == 0 else f" exit={rc}"))
    return rc


def chain(blocks, root, a):
    """SE-JEPA learning: one chain through the memory state; resumes after the last chunk with a saved state."""
    start = 0
    for k, b in enumerate(blocks):
        if not finished(root / b["chunk"]):
            break
        if (root / b["chunk"] / "continual_state.pt").exists():
            start = k + 1
    prev = root / blocks[start - 1]["chunk"] / "continual_state.pt" if start else None
    for k in range(start, len(blocks)):
        out = root / blocks[k]["chunk"]
        cmd = command(blocks[k], out, a, [f"planner.continual.state_in={prev if prev else 'null'}"])
        if run(cmd, out, a.gpus[0]) != 0 or not finished(out, need_state=True):
            sys.exit(f"chunk {blocks[k]['chunk']} failed (see {out}/console.log); later chunks need its state")
        if a.keep_states is not None and prev is not None and (k - 1) not in a.keep_states:
            prev.unlink()  # only this chunk needed it
        prev = out / "continual_state.pt"


def parallel(blocks, root, a):
    extra = []
    if a.method == "sejepa":
        extra = [f"planner.continual.state_in={Path(a.state_in).resolve()}", "planner.continual.read_only=true"]
    elif a.method == "adajepa":
        extra = ["planner.continual.state_in=null"]
    jobs = queue.Queue()
    for b in blocks:
        if not finished(root / b["chunk"]):
            jobs.put(b)
    failed = []

    def worker(gpu):
        while True:
            try:
                b = jobs.get_nowait()
            except queue.Empty:
                return
            out = root / b["chunk"]
            if run(command(b, out, a, extra), out, gpu) != 0 or not finished(out):
                failed.append(b["chunk"])

    threads = [threading.Thread(target=worker, args=(g,)) for g in a.gpus]
    for t in threads:
        t.start()
        time.sleep(a.stagger)
    for t in threads:
        t.join()
    if failed:
        sys.exit(f"failed chunks: {failed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stream", required=True, help="stream name under --streams")
    ap.add_argument("--method", choices=list(METHODS), default="sejepa")
    ap.add_argument("--config", default="sejepa_gd_pushobj", help="sejepa_gd_pushobj or sejepa_cem_pushobj")
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (hydra.yaml + checkpoints/model_<epoch>.pth)")
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--gpus", type=lambda s: s.split(","), default=["0"], help="comma-separated GPU ids")
    ap.add_argument("--root", default=None, help="output root (default eval_outputs/<stream>)")
    ap.add_argument("--streams", default="data/streams")
    ap.add_argument("--state-in", default=None, help="sejepa: evaluate this saved memory state read-only")
    ap.add_argument("--keep-states", type=int, nargs="*", default=None,
                    help="sejepa chain: delete each chunk's state once the next chunk has used it, except these "
                         "chunk indices and the last one (default: keep all)")
    ap.add_argument("--max-episodes", type=int, default=0, help="cap episodes per chunk (smoke tests)")
    ap.add_argument("--stagger", type=float, default=15.0, help="seconds between parallel starts")
    ap.add_argument("--extra", nargs="*", default=[], help="extra Hydra overrides for every chunk")
    a = ap.parse_args()

    blocks = json.loads((ROOT / a.streams / a.stream / "manifest.json").read_text())["blocks"]
    root = Path(a.root or ROOT / "eval_outputs" / a.stream) / a.method
    if a.method == "sejepa" and not a.state_in:
        chain(blocks, root, a)
    else:
        parallel(blocks, root, a)
    log(f"all chunks finished: {root}")


if __name__ == "__main__":
    main()
