"""Prepare two fixed-target, slow-only, five-round CEM chains on CPU."""
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[1] / "CL-JEPA-cudagraph"
DATA = ROOT.parent / "CL-JEPA/data/deformable"
OUTPUT = ROOT / "eval_outputs/rope_granular_single_lr2e4_5rounds_20261103"
PYTHON = "/data/zjc/envs/cljepa/bin/python"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for piece in iter(lambda: f.read(1024*1024), b""):
            h.update(piece)
    return h.hexdigest()


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        out.update(flatten(v, key) if isinstance(v, dict) else {key: v})
    return out


def main():
    if (PLAN / "manifest.json").exists():
        raise FileExistsError("Prepared manifest already exists")
    manifests, diffs = {}, {}
    for obj, gpu, old_model in (("rope", "5", "dino_ours"), ("granular", "6", "granular_ref")):
        ckpt = ROOT.parent / "CL-JEPA/checkpoints" / f"{obj}_dino_h1_fs1_e100_seed0"
        targets = []
        sample_ids = []
        for k in range(5):
            p = (ROOT / "eval_outputs/rope_cl_single/stream" / f"chunk_{k}.pkl" if obj == "rope" else
                 ROOT / "eval_outputs/rope_granular/targets" / f"granular_seed{301+k}.pkl")
            data = pickle.loads(p.read_bytes())
            assert data["goal_H"] == 5 and len(data["state_0"]) == len(data["state_g"]) == 10
            ids = [hashlib.sha256(data["state_0"][i].tobytes()+data["state_g"][i].tobytes()).hexdigest() for i in range(10)]
            sample_ids.extend(ids)
            targets.append(dict(path=str(p), sha256=sha256(p), sample_ids=ids,
                                seed=(99 if obj == "rope" else 301)+k, n=10))
        assert len(set(sample_ids)) == 50
        jobs, previous = [], "null"
        for r in range(1, 6):
            for k, target in enumerate(targets):
                index = len(jobs)
                chunk = f"round_{r:02d}/chunk_{k:02d}"
                out = OUTPUT / obj / "cl_noada" / chunk
                overrides = [f"ckpt_base_path={ckpt}", "model_epoch=latest", f"eval_data_path={DATA}",
                    "goal_source=file", f"+goal_file_path={target['path']}", "n_evals=10", f"seed={target['seed']}",
                    "+wandb_logging=false", "planner.resim_from_start=false", f"planner.continual.tag={obj}",
                    "planner.continual.save_deltas=false", "planner._target_=planning.continual.CLSMPCPlanner",
                    "planner.continual.write=oracle", "planner.continual.mature_k=3", "planner.continual.mature_ratio=0.9",
                    "planner.adapt.steps=0", "planner.continual.slow_lr=2e-4",
                    f"planner.continual.state_in={previous}", f"hydra.run.dir={out}"]
                if obj == "granular":
                    overrides.append("++env_kwargs_override.object_name=granular")
                with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
                    cfg = compose(config_name="cont_plan_cem_rope", overrides=overrides)
                assert cfg.planner.adapt.steps == 0 and cfg.planner.continual.slow_lr == 2e-4
                if index == 0:
                    OmegaConf.save(cfg, PLAN / f"resolved_{obj}_first.yaml", resolve=True)
                    old = OmegaConf.load(ROOT / "eval_outputs/rope_cl_single" / old_model / "cl_noada/chunk_0/.hydra/config.yaml")
                    old_dict = flatten(OmegaConf.to_container(old, resolve=True))
                    new_dict = flatten(OmegaConf.to_container(cfg, resolve=True))
                    diff = {key: [old_dict[key], new_dict[key]] for key in new_dict.keys() & old_dict.keys()
                            if new_dict[key] != old_dict[key]}
                    assert set(diff) <= {"planner.continual.slow_lr", "goal_file_path"}, diff
                    diffs[obj] = diff
                jobs.append(dict(index=index, round=r, chunk=k, n=10, seed=target["seed"], output=str(out),
                                 target_index=k, sample_ids=target["sample_ids"],
                                 argv=[PYTHON, str(ROOT / "plan.py"), "--config-name", "cont_plan_cem_rope", *overrides]))
                previous = str(out / "continual_state.pt")
        manifests[obj] = dict(object=obj, gpu=gpu, historical_model=old_model,
            checkpoint=str(ckpt / "checkpoints/model_latest.pth"), checkpoint_sha256=sha256(ckpt / "checkpoints/model_latest.pth"),
            output=str(OUTPUT / obj), targets=targets, jobs=jobs, success_cd=0.5 if obj=="rope" else 0.2,
            keep_state_indices=[4,9,14,19,24])
    manifest = dict(root=str(ROOT), output=str(OUTPUT), status="authorized_prepared", rounds=5,
        samples_per_round=50, episodes_per_chain=250, total_episode_executions=500,
        protocol="same 50 targets/order/seeds repeated each round; slow-only cl_noada; one oracle domain per chain",
        slow_lr=2e-4, fast_steps=0, objects=manifests,
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    (PLAN / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (PLAN / "validation.json").write_text(json.dumps(dict(status="passed_cpu_only",unique_targets_per_object=50,
        rounds=5,episodes_per_object=250,all_50_configs_composed=True,
        differences_from_historical_single_domain_configs=diffs),indent=2)+"\n")
    print(json.dumps({k:manifest[k] for k in ("status","rounds","total_episode_executions","slow_lr","fast_steps")},indent=2))


if __name__ == "__main__":
    main()
