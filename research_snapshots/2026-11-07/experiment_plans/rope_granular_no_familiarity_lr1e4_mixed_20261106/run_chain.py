"""Run one shared-state CEM chain after targets are generated and the user confirms."""
import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PLAN = Path(__file__).resolve().parent


def finished(out):
    logs = out / "logs.json"
    records = out / "continual_episodes.jsonl"
    return (logs.exists() and "final_eval/mean_chamfer_distance" in logs.read_text()
            and (out / "final_eval_results.npz").exists() and records.exists()
            and len(records.read_text().splitlines()) == 10)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--through-round", type=int, choices=[1, 2, 3], default=1)
    args = ap.parse_args()
    m = json.loads((PLAN / "manifest.json").read_text())
    jobs = json.loads((PLAN / "commands.json").read_text())
    blocks = [b for b in m["blocks"] if b["round"] <= args.through_round]
    jobs = jobs[:len(blocks)]
    if not args.run:
        print(f"Dry run only: through round {args.through_round}, {len(jobs)} sequential blocks, {len(jobs)*10} episodes, GPU {m['chain_gpu']}, slow_lr={m['slow_lr']}")
        return
    for b in blocks:
        target = Path(b["target"])
        if not target.exists() or not target.with_suffix(".json").exists():
            raise FileNotFoundError(f"Generate and validate all targets before planning: {target}")
        saved = json.loads(target.with_suffix(".json").read_text())
        if (saved["block_sha256"] != hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest()
                or saved["target_sha256"] != hashlib.sha256(target.read_bytes()).hexdigest()):
            raise ValueError(f"Target does not match the approved manifest: {target}")
    root = Path(m["root"])
    flex = root.parent / "PyFleX"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": m["chain_gpu"], "EGL_GPU": m["chain_gpu"],
           "PYFLEXROOT": str(flex), "DATASET_DIR": str(Path(m["source_data"]).parent),
           "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "WANDB_MODE": "offline", "SDL_VIDEODRIVER": "dummy",
           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    env["PYTHONPATH"] = str(flex / "bindings/build") + ":" + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = str(flex / "external/SDL2-2.0.4/lib/x64") + ":" + env.get("LD_LIBRARY_PATH", "")
    status = dict(status="running", pid=os.getpid(), through_round=args.through_round,
                  gpu=m["chain_gpu"], started_at=datetime.now(timezone.utc).isoformat())
    status_path = PLAN / "execution.json"

    def save():
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2) + "\n"); tmp.replace(status_path)

    save()
    try:
        start = 0
        for i, job in enumerate(jobs):
            out = Path(job["output"])
            if not finished(out):
                break
            if (out / "continual_state.pt").exists():
                start = i + 1
        for i in range(start, len(jobs)):
            job = jobs[i]
            out = Path(job["output"])
            out.mkdir(parents=True, exist_ok=True)
            status.update(current_chunk=job["chunk"], current_round=m["blocks"][i]["round"]); save()
            print(f"START {job['chunk']}", flush=True)
            with (out / "console.log").open("w") as f:
                rc = subprocess.call(job["argv"], cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT)
            if rc != 0 or not finished(out) or not (out / "continual_state.pt").exists():
                raise RuntimeError(f"{job['chunk']} failed, rc={rc}; chain stopped")
            status.update(completed_chunks=i+1, completed_episodes=(i+1)*10); save()
            print(f"DONE {job['chunk']}", flush=True)
            if i > 0 and i-1 not in m["keep_state_indices"]:
                previous = Path(jobs[i-1]["output"]) / "continual_state.pt"
                if previous.exists():
                    previous.unlink()
        status.update(status="round_completed" if args.through_round < 3 else "completed",
                      completed_chunks=len(jobs), completed_episodes=len(jobs)*10)
    except Exception as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        status["ended_at"] = datetime.now(timezone.utc).isoformat(); save()


if __name__ == "__main__":
    main()
