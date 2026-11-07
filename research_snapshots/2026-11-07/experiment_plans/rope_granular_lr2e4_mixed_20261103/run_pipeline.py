"""Generate targets, run round 1, review it, and continue only on a clean review."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PLAN = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if not args.run:
        print("Dry run: round 1 targets on GPUs 4/5/6/7, CEM chain on GPU 4; review before later rounds.")
        return
    with (PLAN / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute()


def execute():
    m = json.loads((PLAN / "manifest.json").read_text())
    output = Path(m["output"])
    logs = output / "pipeline_logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = PLAN / "pipeline_status.json"
    status = dict(status="running", pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat())

    def save(**updates):
        status.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2) + "\n"); tmp.replace(status_path)

    def call(script, args, name):
        with (logs / name).open("a") as f:
            rc = subprocess.call([sys.executable, str(PLAN / script), *args], cwd=m["root"],
                                 env={**os.environ, "PYTHONUNBUFFERED": "1"}, stdout=f, stderr=subprocess.STDOUT)
        if rc:
            raise RuntimeError(f"{script} failed rc={rc}; see {logs/name}")

    def wait_for_devices(gpus):
        mapping = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
        uuids = {row.split(",")[1].strip() for row in mapping.splitlines() if row.split(",")[0].strip() in gpus}
        if len(uuids) != len(gpus):
            raise RuntimeError(f"GPU mapping incomplete: {gpus}")
        while True:
            active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
            if not uuids.intersection(active.splitlines()):
                return
            save(stage="waiting_for_free_devices", requested_gpus=gpus)
            time.sleep(30)

    def generate(round_number):
        wait_for_devices(["4", "5", "6", "7"])
        save(stage="generating_targets", round=round_number)
        jobs = []
        for obj, shard, gpu in (("rope", 0, "4"), ("rope", 1, "5"), ("granular", 0, "6"), ("granular", 1, "7")):
            path = logs / f"targets_r{round_number}_{obj}_s{shard}_gpu{gpu}.log"
            fh = path.open("a")
            proc = subprocess.Popen([sys.executable, str(PLAN / "generate_targets.py"), obj, "--run",
                                     "--round", str(round_number), "--gpu", gpu,
                                     "--shard-index", str(shard), "--num-shards", "2"],
                                    cwd=m["root"], env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                    stdout=fh, stderr=subprocess.STDOUT)
            jobs.append((proc, fh, path, gpu))
        save(generation_workers=[dict(pid=p.pid, gpu=g, log=str(f)) for p, _, f, g in jobs])
        failures = []
        for proc, fh, path, _ in jobs:
            rc = proc.wait(); fh.close()
            if rc:
                failures.append(dict(log=str(path), returncode=rc))
        if failures:
            raise RuntimeError(f"Target generation failed: {failures}")

    save(stage="starting")
    try:
        generate(1)
        save(stage="running_round_1", gpu="4")
        call("run_chain.py", ["--run", "--through-round", "1"], "chain_round_1.log")
        save(stage="reviewing_round_1")
        call("analyze_round.py", ["--round", "1"], "review_round_1.log")
        review = json.loads((PLAN / "round_1_review.json").read_text())
        save(round_1_review=str(PLAN / "round_1_review.md"), round_1_checks=review["checks"],
             round_1_performance=review["performance"], round_1_domains=review["domains"])
        if not review["continue_authorized"]:
            save(status="needs_review", stage="stopped_after_round_1",
                 reason="First-round checks did not all pass; later rounds were not started")
            return
        for r in (2, 3):
            generate(r)
            save(stage=f"running_round_{r}", gpu="4")
            call("run_chain.py", ["--run", "--through-round", str(r)], f"chain_round_{r}.log")
            call("analyze_round.py", ["--round", str(r)], f"review_round_{r}.log")
        save(status="completed", stage="completed_3_rounds")
    except Exception as exc:
        save(status="failed", error=repr(exc))
        raise
    finally:
        save(ended_at=datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    main()
