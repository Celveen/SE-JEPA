import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
LOG_NAMES = ["logs.json", "continual_episodes.jsonl", "adajepa_losses.csv", "per_iter_logs.json"]
PRIVATE_SIGNATURES = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"Snapshot directory already exists: {output}")
    output.mkdir(parents=True)
    cutoff = time.time_ns()
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    candidates = []
    for base in [WORKSPACE / "CL-JEPA/eval_outputs", ROOT / "eval_outputs"]:
        for final in base.rglob("final_eval_results.npz"):
            log = final.parent / "logs.json"
            if not log.exists() or max(final.stat().st_mtime_ns, log.stat().st_mtime_ns) > cutoff:
                continue
            text = log.read_text()
            if "final_eval/" in text:
                candidates.append((base, final.parent))
    candidates.sort(key=lambda item: str(item[1]))
    print(f"Exporting {len(candidates)} completed evaluation directories", flush=True)

    def store_file(source, relative, compressed=False, role="result"):
        before = source.stat()
        payload = source.read_bytes()
        after = source.stat()
        if before.st_mtime_ns != after.st_mtime_ns or before.st_size != after.st_size:
            raise RuntimeError(f"File changed while reading: {source}")
        if PRIVATE_SIGNATURES.search(payload):
            raise RuntimeError(f"Credential-like content found in selected file: {source}")
        stored = gzip.compress(payload, compresslevel=6, mtime=0) if compressed else payload
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(stored)
        return {"path": relative.as_posix(), "source": str(source), "bytes": len(stored), "source_bytes": len(payload), "sha256": digest(stored), "source_sha256": digest(payload), "gzip": compressed, "role": role}

    def export_run(candidate):
        base, directory = candidate
        relative = Path("completed_runs") / base.parent.name / directory.relative_to(base)
        final = directory / "final_eval_results.npz"
        records = [json.loads(line) for line in (directory / "logs.json").read_text().splitlines() if line.strip()]
        finals = [record for record in records if "final_eval/success_rate" in record]
        with np.load(final, allow_pickle=False) as data:
            successes = data["successes"].astype(bool)
        consistency = bool(finals and np.isclose(successes.mean(), finals[-1]["final_eval/success_rate"]))
        entries = [store_file(final, relative / final.name)]
        for name in LOG_NAMES:
            path = directory / name
            if path.exists():
                entries.append(store_file(path, relative / (name + ".gz"), compressed=True))
        for name in ["protocol.json", "run_metadata.json"]:
            path = directory / name
            if path.exists():
                entries.append(store_file(path, relative / name, role="run_protocol"))
        references = []
        for name in ["config.yaml", "overrides.yaml"]:
            config_path = directory / ".hydra" / name
            if config_path.exists():
                entries.append(store_file(config_path, relative / ".hydra" / name, role="run_config"))
                if name == "config.yaml":
                    config = yaml.safe_load(config_path.read_text())
                    target = config.get("eval_data_path")
                    if target and isinstance(target, str):
                        target = Path(target)
                        references.append((target if target.is_absolute() else base.parent / target, "target"))
                    checkpoint = config.get("ckpt_base_path")
                    if checkpoint and isinstance(checkpoint, str):
                        checkpoint = Path(checkpoint)
                        if not checkpoint.is_absolute():
                            checkpoint = base.parent / checkpoint
                        if config.get("model_name"):
                            checkpoint = checkpoint / config["model_name"]
                        references.append((checkpoint / "hydra.yaml", "model_config"))
        diagnostic = any("smoke" in part or "startup_failure" in part for part in directory.parts)
        protocol = directory / "protocol.json"
        if protocol.exists():
            diagnostic |= bool(json.loads(protocol.read_text()).get("diagnostic_only", False))
        record = {"directory": relative.as_posix(), "source": str(directory), "episodes": len(successes), "recorded_success_rate": float(successes.mean()), "final_log_matches_results": consistency, "group": f"{base.parent.name}/{directory.relative_to(base).parts[0]}", "diagnostic_only": diagnostic}
        return record, entries, references

    runs, files, references = [], [], []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, (record, entries, linked) in enumerate(executor.map(export_run, candidates), 1):
            runs.append(record)
            files.extend(entries)
            references.extend(linked)
            if index % 200 == 0:
                print(f"Completed {index}/{len(candidates)} directories", flush=True)
    reference_status = []
    seen = set()
    for path, role in references:
        identity = str(path.resolve())
        if identity in seen:
            continue
        seen.add(identity)
        if not path.resolve().is_relative_to(WORKSPACE):
            reference_status.append({"path": str(path), "status": "outside_workspace_not_copied", "role": role})
            continue
        if not path.is_file():
            reference_status.append({"path": str(path), "status": "not_available_on_this_host", "role": role})
            continue
        if path.stat().st_size > 5 * 2**20:
            reference_status.append({"path": str(path), "status": "large_reference_not_copied", "role": role, "bytes": path.stat().st_size})
            continue
        prefix = digest(identity.encode())[:12]
        relative = Path("references") / role / f"{prefix}_{path.name}"
        files.append(store_file(path, relative, role=role))
        reference_status.append({"path": str(path), "status": "copied", "snapshot_path": relative.as_posix(), "role": role})

    paper = WORKSPACE / "paper_cvpr2027"
    unavailable = []
    for name in ["baseline_metrics_snapshot.json", "structure_data_snapshot.json", "dinowm_parameter_audit.json"]:
        path = paper / name
        if path.exists():
            files.append(store_file(path, Path("paper_data") / name, role="summary"))
        else:
            unavailable.append(str(path))
    collections = [
        (ROOT / "reports/shape_shift_figs/data", Path("paper_data/shape"), {".json", ".csv"}),
        (ROOT / "reports/figs", Path("paper_data/rounds"), {".json", ".csv", ".md"}),
        (ROOT / "eval_outputs/rope_cl_single/paper/data", Path("paper_data/deformable"), {".csv"}),
        (paper / "drive_materials", Path("paper_data/drive_materials"), {".json", ".csv", ".txt", ".md", ".png"}),
        (paper / "structure_assets", Path("plots"), {".png"}),
    ]
    for source, destination, suffixes in collections:
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix in suffixes:
                files.append(store_file(path, destination / path.relative_to(source), role="paper_data"))
    files.append(store_file(ROOT / "eval_outputs/baselines/BIG_TABLE.md", Path("paper_data/historical_baseline_tables.md"), role="historical_summary"))
    for name in ["AGENTS.md", "EXPERIMENT_POLICY.md"]:
        files.append(store_file(WORKSPACE / name, Path("protocols") / name, role="experiment_policy"))
    plan_root = WORKSPACE / "experiment_plans"
    suffixes = {".py", ".json", ".yaml", ".yml", ".csv", ".md", ".patch", ".png", ".pdf"}
    for path in sorted(plan_root.rglob("*")):
        if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts:
            files.append(store_file(path, Path("experiment_plans") / path.relative_to(plan_root), role="experiment_recipe_or_analysis"))
    for family in sorted((ROOT / "eval_outputs").iterdir()):
        if not family.is_dir() or "202611" not in family.name:
            continue
        for path in sorted(family.rglob("*")):
            if path.is_file() and "analysis" in path.parts and path.suffix in suffixes:
                files.append(store_file(path, Path("analyses") / path.relative_to(ROOT / "eval_outputs"), role="analysis"))
    latest = {}
    domains = ["default", "blur", "snp", "dark", "red_agent", "red_object", "red_goal"]
    shapes = ["I", "L", "square", "Z", "small_tee", "plus", "T"]
    for method in ["dreamerv3", "iris", "dinowm"]:
        name = "dinowm_H7t3" if method == "dinowm" else f"{method}_VIS350"
        folder = ROOT / "eval_outputs/baselines" / name
        status_path = folder / "pipeline_status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        for filename in ["pipeline_status.json", "summary.json", "epoch_budget_guard.json", "training_reference_config.yaml", "source_revision.txt", "training_source_changes.patch", "encoder_local_changes.patch"]:
            path = folder / filename
            if path.exists():
                files.append(store_file(path, Path("baseline_metadata") / name / filename, role="pipeline_metadata"))
        checkpoint = Path(status["checkpoint"])
        if not checkpoint.is_dir():
            unavailable.append(str(checkpoint))
            continue
        for path in sorted(checkpoint.iterdir()):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".yaml"}:
                compressed = path.suffix == ".jsonl"
                files.append(store_file(path, Path("training_metadata") / method / (path.name + (".gz" if compressed else "")), compressed, "training_metadata"))
        receipt = checkpoint / "adapter_sources.json"
        if receipt.exists():
            for filename, expected in json.loads(receipt.read_text()).items():
                actual = digest((ROOT / "scripts/offline_visual" / filename).read_bytes())
                if actual != expected:
                    raise RuntimeError(f"Source differs from training receipt: {method}/{filename}")
        model = yaml.safe_load((checkpoint / "hydra.yaml").read_text())
        rates, steps, counts = {}, {}, {}
        for index, condition in enumerate(shapes if method == "dinowm" else domains):
            directory = folder / (f"chunk_{index:02d}_{condition}" if method == "dinowm" else condition)
            final = directory / "final_eval_results.npz"
            log = directory / "logs.json"
            if not final.exists() or not log.exists() or "final_eval/success_rate" not in log.read_text():
                continue
            config = yaml.safe_load((directory / ".hydra/config.yaml").read_text())
            actions_per_replan = int(config["planner"]["n_taken_actions"]) // int(model["frameskip"])
            limit = int(config["planner"]["max_iter"])
            with np.load(final, allow_pickle=False) as data:
                successes = data["successes"].astype(bool)
                lengths = np.asarray(data["action_len"], dtype=float)
            assert len(successes) == 50 and actions_per_replan > 0
            replans = np.where(successes, lengths / actions_per_replan, limit)
            assert np.isfinite(replans).all() and ((replans >= 0) & (replans <= limit)).all()
            rates[condition] = float(successes.mean() * 100)
            counts[condition] = int(successes.sum())
            steps[condition] = float(replans.mean())
        complete = len(rates) == 7
        item = {"task": "PushObj H7t3" if method == "dinowm" else "PushT VIS350", "training_epochs_per_stage": 3, "complete": complete, "pipeline_stage": status["stage"], "conditions_completed": len(rates), "episodes_completed": len(rates) * 50, "planned_episodes": 350, "successes": counts, "success_rates_percent": rates, "mean_mpc_steps": steps, "all_seven_success_rate_percent": float(np.mean(list(rates.values()))) if complete else None, "all_seven_mean_mpc_steps": float(np.mean(list(steps.values()))) if complete else None}
        if complete and method != "dinowm":
            item["six_shift_success_rate_percent"] = float(np.mean([rates[d] for d in domains[1:]]))
            item["six_shift_mean_mpc_steps"] = float(np.mean([steps[d] for d in domains[1:]]))
        if complete and method == "dinowm":
            item["unseen_success_rate_percent"] = float(np.mean([rates[d] for d in ["I", "square", "small_tee"]]))
        latest[method] = item
    (output / "latest_baseline_results.json").write_text(json.dumps(latest, indent=2))
    (output / "unavailable_sources.json").write_text(json.dumps(unavailable, indent=2))
    sizes = sum(entry["bytes"] for entry in files)
    manifest = {"format_version": 1, "created_utc": created, "source_git_commit": revision, "completed_directories": len(runs), "stored_bytes": sizes, "groups": dict(sorted(Counter(record["group"] for record in runs).items())), "files": sorted(files, key=lambda entry: entry["path"]), "runs": runs, "references": reference_status, "unavailable_sources": unavailable, "excluded": ["training datasets", "model checkpoints", "wandb directories", "raw console logs", "unfinished evaluation directories", "full manuscript drafts"], "selection_policy": "All evaluation directories with final result files and final-evaluation markers before the cutoff; extra baseline seeds and diagnostic runs are archived, not selected as main-table results."}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    with (output / "run_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    mismatches = [record for record in runs if not record["final_log_matches_results"]]
    (output / "consistency_notes.json").write_text(json.dumps(mismatches, ensure_ascii=False, indent=2))
    readme = f"""# Completed experiment data snapshot

Created: {created}

This snapshot archives {len(runs)} completed evaluation directories ({sizes / 2**20:.1f} MiB). It includes original final NPZ results, gzip-compressed numerical logs, per-run configurations, available small target references, paper tables/plot data, and selected plots. Model checkpoints and training datasets remain external. Full manuscript drafts are not part of this data release.

`manifest.json` records every payload's source, original SHA256, stored SHA256, and compression status. `run_index.csv` indexes completed chunks. Use `python scripts/verify_research_snapshot.py research_snapshots/{output.name}` from the repository root to verify checksums.

## Statistical scope

- Newly added external PushObj baselines use seed 2000, 50 episodes per shape: 350 per method. Extra seeds already computed are retained as archives, not folded into those single-seed rows. Older official Frozen/AdaJEPA comparisons retain their recorded multi-seed scope in the paper data.
- Ours retains its prescribed multiple-seed/chained evaluations. Shape CEM has both original and replacement-chain results; source selection is recorded in the paper data and must not be silently mixed.
- Newly running experiments may contribute completed chunks only. A completed chunk does not imply that its entire experiment has finished.
- New visual experiments use resampled learning targets and an independent held-out set. Existing VIS50 baselines use their recorded targets/noise protocol. Their sources remain distinct.
- Existing Rope/Granular paper data are the earlier version, pending the author's newer data. Legacy environment `successes` for deformable tasks are not the paper's CD-threshold success metric; use CD summaries rather than treating those raw flags as paper SR.
- The new DreamerV3 and IRIS scope is original T plus six visual shifts, seven conditions x 50 episodes. New DINO-WM evaluation uses seven PushObj shapes x 50 episodes, seed 2000.
- The effective training budget is three epochs per stage. IRIS has separate tokenizer/dynamics stages. DINO-WM uses the full 52.32M model and fixed epoch 3; historical e100 paths/configs do not change that effective budget.
- Pipeline state and training records are preserved under baseline_metadata/ and training_metadata/. The current complete baseline aggregates are in latest_baseline_results.json; historical_baseline_tables.md is source history, not the latest aggregate.
- experiment_plans/ and analyses/ preserve new seven-round PushObj and Rope/Granular experiments, including stopped/partial mixed streams. A stopped run is not silently relabeled complete.

## Reuse

NPZ files are unchanged. Numerical logs use deterministic gzip (`mtime=0`); decompress with Python's `gzip` module or `gunzip -k`. Original absolute paths in configs may need remapping for a different machine. References unavailable on this host or larger than 5 MiB are listed in the manifest rather than silently invented.

There are {len(mismatches)} final-log/result consistency flags; inspect `consistency_notes.json` before aggregating flagged runs. No failed or historical results were replaced by a different seed during this export.
"""
    (output / "README.md").write_text(readme)
    print(json.dumps({"snapshot": str(output), "completed_directories": len(runs), "payload_files": len(files), "MiB": round(sizes / 2**20, 2), "consistency_flags": len(mismatches)}), flush=True)


if __name__ == "__main__":
    main()
