"""Freeze the existing LeWM VIS50 setup and prepare its seven-condition GD run."""
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import torch
import yaml

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
ROOT = WORKSPACE / "CL-JEPA-cudagraph"
OLD = HERE.parent / "legacy_dinowm_lewm_GD_H7t3_20261106"
CHECKPOINT = WORKSPACE / "CL-JEPA/checkpoints/lewm_pushobj_T16k"
OUTPUT = ROOT / "eval_outputs/baselines/lewm_pusht_GD_T16k_e10_20261108"
CONDITIONS = ["default", "blur", "snp", "dark", "red_agent", "red_object", "red_goal"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def main():
    if (HERE / "protocol.json").exists():
        raise RuntimeError("Already prepared; use run.py instead of changing the protocol")
    old_receipt = json.loads((OLD / "source_receipt.json").read_text())["files"]
    for name, expected in old_receipt.items():
        assert sha(OLD / name) == expected, name
    for folder in ["code", "evaluation"]:
        shutil.copytree(OLD / folder, HERE / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Parameter values, not pickle container bytes, establish checkpoint identity.
    fixed = torch.load(CHECKPOINT / "checkpoints/model_10.pth", map_location="cpu", weights_only=False)
    latest = torch.load(CHECKPOINT / "checkpoints/model_latest.pth", map_location="cpu", weights_only=False)
    assert fixed["epoch"] == latest["epoch"] == 10
    assert fixed["model"].keys() == latest["model"].keys()
    assert all(torch.equal(v, latest["model"][k]) for k, v in fixed["model"].items())
    save(HERE / "checkpoint_identity.json", {
        "checkpoint": str(CHECKPOINT), "epoch": 10,
        "checkpoint_file": str(CHECKPOINT / "checkpoints/model_10.pth"),
        "checkpoint_sha256": sha(CHECKPOINT / "checkpoints/model_10.pth"),
        "config_sha256": sha(CHECKPOINT / "hydra.yaml"),
        "latest_file_sha256": sha(CHECKPOINT / "checkpoints/model_latest.pth"),
        "latest_tensor_identical": True, "training_shapes": ["T"],
    })
    del fixed, latest
    target_source = ROOT / "data/cont_streams/VIS50/targets_T.pkl"
    target = HERE / "targets_T.pkl"
    shutil.copy2(target_source, target)
    with target.open("rb") as handle:
        targets = pickle.load(handle)
    assert targets["n_samples"] == 50 and targets["shape"] == "T"
    save(HERE / "target_manifest.json", {
        "source": str(target_source), "path": str(target), "sha256": sha(target),
        "episodes": 50, "shape": "T", "evaluation_seed": 100,
        "identical_to_prior_VIS50": True,
    })
    configs = {}
    for condition in CONDITIONS[1:]:
        source = ROOT / "eval_outputs/baselines/lewm_VIS50" / condition / ".hydra/config.yaml"
        cfg = yaml.safe_load(source.read_text())
        assert cfg["seed"] == 100 and cfg["n_evals"] == 50
        assert cfg["ckpt_base_path"] == str(CHECKPOINT)
        assert sha(cfg["eval_data_path"]) == sha(target)
        assert cfg["objective"]["mode"] == "staged"
        assert cfg["planner"]["n_taken_actions"] == 5 and cfg["planner"]["max_iter"] == 30
        destination = HERE / "cem_reference_configs" / f"{condition}.yaml"
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
        configs[condition] = cfg
    jobs = []
    for i, condition in enumerate(CONDITIONS):
        cfg = configs.get(condition, {})
        folder = OUTPUT / condition
        command = [
            "/data/zjc/envs/cljepa/bin/python", "-u", str(HERE / "evaluation/evaluate.py"),
            f"ckpt_base_path={CHECKPOINT}", "model_epoch=10", "n_evals=50", "seed=100",
            f"eval_data_path={target}", f"hydra.run.dir={folder}",
            f"planner.continual.tag={condition}", "audit.planner_kind=gd",
            f"audit.condition={condition}", "audit.pretraining_shape_status=seen",
            "audit.expected_model_epoch=10", "planner.sub_planner.microbatch_size=50",
            "audit.expected_microbatch_size=50",
        ]
        for key in ["ood_corruption", "ood_level"]:
            if cfg.get(key) is not None:
                command.append(f"{key}={cfg[key]}")
        for key, value in (cfg.get("env_kwargs_override") or {}).items():
            command.append(f"++env_kwargs_override.{key}={value}")
        jobs.append({"condition": condition, "gpu": str([2, 3][i % 2]),
                     "output": str(folder), "command": command})
    save(HERE / "commands.json", jobs)
    protocol = {
        "method": "lewm", "task": "pusht", "planner": "gd", "checkpoint": str(CHECKPOINT),
        "epoch": 10, "training_shapes": ["T"], "new_training": False,
        "conditions": CONDITIONS, "seed": 100, "episodes_per_condition": 50,
        "planned_episodes": 350, "gpus": ["2", "3"],
        "objective": "staged", "horizon_model_steps": 5, "frameskip": 5,
        "physical_steps_per_replan": 5, "max_replans": 30, "max_physical_steps": 150,
        "gd": {"iterations": 100, "lr": 0.1, "optimizer": "adam", "cosine_scheduler": True,
               "initialization": "zero physical action", "reuse_actions": True,
               "action_noise": 0, "microbatch_size": 50, "cuda_graph": False},
        "model_frozen": True, "online_learning": False,
        "noise_protocol": "Fixed goal seed; deterministic distinct seed per feedback call, recorded in JSONL",
        "noise_comparison_note": "Prior CEM feedback noise seeds were not recorded; identical noisy pixels across planners are not claimed.",
        "reference": "Existing LeWM VIS50 CEM weights/targets/controller; original T additionally evaluated.",
        "comparison_note": "New random16k DINO-WM uses a different last/25/6 controller; report separately.",
    }
    save(HERE / "protocol.json", protocol)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ["protocol.json", "checkpoint_identity.json", "target_manifest.json"]:
        shutil.copy2(HERE / name, OUTPUT / name)
    # Adapt the already validated gradient diagnostic to the T-only checkpoint.
    text = (OLD / "check_gradients.py").read_text()
    text = text.replace('("lewm", "lewm_pushobj_bias16k", 10, 50),\n        ("dinowm", "dinowm_pushobj_bias16k_reference_e3", 3, 8),',
                        '("lewm", "lewm_pushobj_T16k", 10, 50),')
    text = text.replace('f"eval_outputs/baselines/{name}_H7t3/chunk_00_I/plan_targets.pkl"',
                        '"eval_outputs/baselines/lewm_VIS50/blur/plan_targets.pkl"')
    assert 'lewm_pushobj_bias16k' not in text and '"dinowm",' not in text
    (HERE / "check_gradients.py").write_text(text)
    files = [p for p in HERE.rglob("*") if p.is_file() and
             "__pycache__" not in p.parts and p.suffix in {".py", ".yaml", ".pkl"}]
    save(HERE / "source_receipt.json", {"files": {str(p.relative_to(HERE)): sha(p) for p in files},
         "prepared_from": str(OLD), "legacy_source_receipt_sha256": sha(OLD / "source_receipt.json")})
    print(json.dumps({"jobs": len(jobs), "episodes": 350, "checkpoint": str(CHECKPOINT), "gpus": [2, 3]}))


if __name__ == "__main__":
    main()
