"""Print a prepared command; --run is used only after the user confirms the run."""
import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("planner", choices=["gd", "cem"])
parser.add_argument("--run", action="store_true", help="execute after user confirmation")
args = parser.parse_args()
plan = Path(__file__).resolve().parent
job = json.loads((plan / "commands.json").read_text())[args.planner]
print(shlex.join(job["argv"]), flush=True)
if args.run:
    status_path = plan / f"execution_{args.planner}.json"
    status = dict(status="running", planner=args.planner, pid=os.getpid(), gpu=job["gpu"],
                  started_at=datetime.now(timezone.utc).isoformat(), command=job["argv"])

    def save_status():
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2) + "\n")
        tmp.replace(status_path)

    save_status()
    try:
        rc = subprocess.call(job["argv"], cwd=job["cwd"], env={**os.environ, **job["env"]})
        stream = job["argv"][job["argv"].index("--stream") + 1]
        manifest = json.loads((Path(job["cwd"]) / "data/cont_streams" / stream / "manifest.json").read_text())
        completed = []
        for block in manifest["blocks"]:
            chunk = Path(job["output_root"]) / "cls6a_free" / block["chunk"]
            logs = chunk / "logs.json"
            if ((chunk / "final_eval_results.npz").exists() and logs.exists()
                    and "final_eval/success_rate" in logs.read_text()):
                completed.append(block["chunk"])
        status.update(returncode=rc, completed_chunks=len(completed), expected_chunks=len(manifest["blocks"]))
        ok = rc == 0 and len(completed) == len(manifest["blocks"])
        status["status"] = "completed" if ok else "failed"
    except Exception as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        status["ended_at"] = datetime.now(timezone.utc).isoformat()
        save_status()
    print(json.dumps(status), flush=True)
    raise SystemExit(0 if ok else 1)
print("Dry run only: no experiment started.")
