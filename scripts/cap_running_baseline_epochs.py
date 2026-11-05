import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def identity(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


def live_group(group):
    members = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == group:
                members.append(int(directory.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return members


def write_json(path, record):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    arguments = parser.parse_args()
    output = arguments.run_root.resolve()
    lock = (output / ".epoch_budget.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = output / "pipeline_status.json"
    state = json.loads(status_path.read_text())
    if state.get("stage") != "training" or state.get("method") != "dinowm":
        raise RuntimeError("This handoff requires an active DINO-WM training pipeline")
    original_epochs = next(int(item.split("=", 1)[1]) for item in state["train_command"] if item.startswith("training.epochs="))
    if not 1 <= arguments.epochs <= original_epochs:
        raise ValueError("The epoch cap must be positive and no larger than the active training budget")
    controller = state["pid"]
    initial = identity(controller)
    if initial is None or initial[0] == "Z":
        raise RuntimeError("The training controller is no longer running")
    command = (Path("/proc") / str(controller) / "cmdline").read_bytes().split(b"\0")
    assert str(ROOT / "scripts/run_pushobj_encoder_baseline.py").encode() in command
    assert b"dinowm" in command and b"--train" in command and os.getpgid(controller) == controller
    checkpoint = Path(state["checkpoint"])
    target = checkpoint / "checkpoints" / f"model_{arguments.epochs}.pth"
    alias = checkpoint.parent / f"dinowm_pushobj_bias16k_reference_e{arguments.epochs}"
    if (alias.exists() or alias.is_symlink()) and alias.resolve() != checkpoint.resolve():
        raise RuntimeError(f"Conflicting checkpoint path: {alias}")
    guard_path = output / "epoch_budget_guard.json"
    guard = {"pid": os.getpid(), "controller_pid": controller, "controller_start_ticks": initial[1], "original_epochs": original_epochs, "effective_epochs": arguments.epochs, "checkpoint": str(target), "checkpoint_alias": str(alias), "stage": "waiting_for_checkpoint", "started": now()}
    write_json(guard_path, guard)
    state.update(requested_training_epochs=arguments.epochs, evaluation_epoch=str(arguments.epochs), epoch_budget_guard_pid=os.getpid(), epoch_budget_guard=str(guard_path), updated=now())
    write_json(status_path, state)
    print(now(), f"Keeping current progress; stop after model_{arguments.epochs}.pth is saved.", flush=True)
    try:
        while not target.exists():
            current = identity(controller)
            if current is None or current[0] == "Z" or current[1] != initial[1]:
                guard.update(stage="controller_ended_before_checkpoint", updated=now())
                write_json(guard_path, guard)
                return 1
            time.sleep(0.5)
        current = identity(controller)
        if current is not None and current[1] == initial[1]:
            os.killpg(controller, signal.SIGTERM)
        deadline = time.monotonic() + 20
        while live_group(controller) and time.monotonic() < deadline:
            time.sleep(0.25)
        if live_group(controller):
            os.killpg(controller, signal.SIGKILL)
        records = [json.loads(line) for line in (checkpoint / "metrics.jsonl").read_text().splitlines() if line.strip()]
        final = next((record for record in records if record.get("epoch") == arguments.epochs), None)
        if final is None or not math.isfinite(final.get("val_loss", float("nan"))):
            guard.update(stage="checkpoint_validation_invalid", updated=now())
            write_json(guard_path, guard)
            return 1
        if not alias.exists():
            alias.symlink_to(checkpoint, target_is_directory=True)
        override = {"original_planned_epochs": original_epochs, "effective_epochs": arguments.epochs, "evaluation_epoch": arguments.epochs, "reason": "User requested the same three-epoch training budget as the main method", "original_directory_preserved": str(checkpoint), "evaluation_path": str(alias), "updated": now()}
        write_json(checkpoint / "training_budget_override.json", override)
        guard.update(stage="evaluation_handoff", updated=now())
        write_json(guard_path, guard)
        command = [sys.executable, "-u", str(ROOT / "scripts/run_pushobj_encoder_baseline.py"), "dinowm", "--gpus", ",".join(state["gpus"]), "--train", "--epochs", str(arguments.epochs), "--batch-size", str(state["training_batch_size"]), "--seeds", *map(str, state["test_seeds"]), "--eval-epoch", str(arguments.epochs)]
        print(now(), "Checkpoint ready; training stopped. Handing off to pinned-epoch evaluation.", flush=True)
        os.execv(sys.executable, command)
    except Exception as error:
        current = identity(controller)
        if current is not None and current[0] != "Z" and current[1] == initial[1]:
            os.killpg(controller, signal.SIGTERM)
        guard.update(stage="failed_training_stopped", error=str(error), updated=now())
        write_json(guard_path, guard)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
