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
        record = {"directory": relative.as_posix(), "source": str(directory), "episodes": len(successes), "recorded_success_rate": float(successes.mean()), "final_log_matches_results": consistency, "group": f"{base.parent.name}/{directory.relative_to(base).parts[0]}"}
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
    for name in ["baseline_metrics_snapshot.json", "structure_data_snapshot.json", "dinowm_parameter_audit.json"]:
        files.append(store_file(paper / name, Path("paper_data") / name, role="summary"))
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
    files.append(store_file(ROOT / "eval_outputs/baselines/BIG_TABLE.md", Path("paper_data/baseline_tables.md"), role="summary"))
    for name in ["pipeline_status.json", "training_reference_config.yaml", "source_revision.txt", "training_source_changes.patch", "encoder_local_changes.patch"]:
        path = ROOT / "eval_outputs/baselines/dinowm_H7t3" / name
        if path.exists():
            files.append(store_file(path, Path("running_dinowm_reference100") / name, role="running_metadata_not_final_result"))
    sizes = sum(entry["bytes"] for entry in files)
    manifest = {"format_version": 1, "created_utc": created, "source_git_commit": revision, "completed_directories": len(runs), "stored_bytes": sizes, "groups": dict(sorted(Counter(record["group"] for record in runs).items())), "files": sorted(files, key=lambda entry: entry["path"]), "runs": runs, "references": reference_status, "excluded": ["training datasets", "model checkpoints", "wandb directories", "raw console logs", "unfinished evaluation directories", "full manuscript drafts"], "selection_policy": "All evaluation directories with final result files and final-evaluation markers before the cutoff; extra baseline seeds and diagnostic runs are archived, not selected as main-table results."}
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
- `running_dinowm_reference100/` is metadata only: the complete 52.32M-parameter DINO-WM has started 100-epoch training. It has no final PushObj result in this snapshot.

## Reuse

NPZ files are unchanged. Numerical logs use deterministic gzip (`mtime=0`); decompress with Python's `gzip` module or `gunzip -k`. Original absolute paths in configs may need remapping for a different machine. References unavailable on this host or larger than 5 MiB are listed in the manifest rather than silently invented.

There are {len(mismatches)} final-log/result consistency flags; inspect `consistency_notes.json` before aggregating flagged runs. No failed or historical results were replaced by a different seed during this export.
"""
    (output / "README.md").write_text(readme)
    print(json.dumps({"snapshot": str(output), "completed_directories": len(runs), "payload_files": len(files), "MiB": round(sizes / 2**20, 2), "consistency_flags": len(mismatches)}), flush=True)


if __name__ == "__main__":
    main()
