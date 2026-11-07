"""Continue historical 1e-4 slow learners from round 3 through rounds 4 and 5."""
import copy
import hashlib
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PLAN = Path(__file__).resolve().parent
REFERENCE = PLAN.parent / "rope_granular_single_lr2e4_5rounds_20261103/manifest.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        out.update(flat(v, key) if isinstance(v, dict) else {key: v})
    return out


def main():
    if (PLAN / "manifest.json").exists():
        raise FileExistsError("Continuation manifest already exists")
    ref = json.loads(REFERENCE.read_text())
    root = Path(ref["root"])
    output = root / "eval_outputs/rope_granular_single_lr1e4_extend5_20261104"
    objects, validation = {}, {}
    for obj in ("rope", "granular"):
        source = ref["objects"][obj]
        old_root = root / "eval_outputs/rope_cl_single" / source["historical_model"]
        state_path = old_root / "cl_noada_r3/chunk_4/continual_state.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=False, mmap=True)
        cfg = state["cls_cfg"]; domain = state["domains"][obj]
        assert state["episode"] == domain["n"] == len(domain["replay"]) == 150
        assert cfg["slow_lr"] == 1e-4 and cfg["write"] == "oracle"
        assert domain["learner"] is not None and set(state["domains"]) == {obj}
        state_info = dict(episode=150, slow_lr=cfg["slow_lr"], replay_episodes=len(domain["replay"]),
                          validation_history=len(domain["val"]), domain=obj)
        del state
        assert sha256(source["checkpoint"]) == source["checkpoint_sha256"]
        for target in source["targets"]:
            assert sha256(target["path"]) == target["sha256"]
        history = []
        for j in source["jobs"][:15]:
            arm = "cl_noada" if j["round"] == 1 else f"cl_noada_r{j['round']}"
            out = old_root / arm / f"chunk_{j['chunk']}"
            assert (out / "final_eval_results.npz").exists()
            history.append(dict(index=j["index"], round=j["round"], chunk=j["chunk"], n=10,
                seed=j["seed"], sample_ids=j["sample_ids"], output=str(out),
                console_path=str(root / "eval_outputs/rope_cl_single/logs" /
                                 f"{source['historical_model']}_{arm}_chunk_{j['chunk']}.log")))
        previous = str(state_path)
        jobs = []
        for old_job in source["jobs"][15:]:
            j = copy.deepcopy(old_job)
            out = output / obj / "cl_noada" / f"round_{j['round']:02d}" / f"chunk_{j['chunk']:02d}"
            if out.exists():
                raise FileExistsError(out)
            replacements = {"planner.continual.slow_lr": "1e-4", "planner.continual.state_in": previous,
                            "hydra.run.dir": str(out)}
            argv = [f"{arg.split('=',1)[0]}={replacements[arg.split('=',1)[0]]}"
                    if arg.split('=',1)[0] in replacements else arg for arg in j["argv"]]
            j.update(output=str(out), argv=argv)
            with initialize_config_dir(config_dir=str(root / "conf"), version_base=None):
                resolved = compose(config_name="cont_plan_cem_rope", overrides=argv[4:])
            assert resolved.planner.continual.slow_lr == 1e-4 and resolved.planner.adapt.steps == 0
            assert resolved.planner.continual.state_in == previous
            if not jobs:
                OmegaConf.save(resolved, PLAN / f"resolved_{obj}_round4.yaml", resolve=True)
                before = flat(OmegaConf.to_container(OmegaConf.load(old_root / "cl_noada/chunk_0/.hydra/config.yaml"), resolve=True))
                after = flat(OmegaConf.to_container(resolved, resolve=True))
                differences = {k: [before[k], after[k]] for k in before.keys() & after.keys() if before[k] != after[k]}
                assert set(differences) <= {"planner.continual.state_in", "goal_file_path"}, differences
                validation[obj] = dict(source_state=state_info, config_differences=differences)
            jobs.append(j)
            previous = str(out / "continual_state.pt")
        objects[obj] = {**source, "output": str(output / obj), "jobs": jobs, "history_jobs": history,
            "initial_state": str(state_path), "initial_state_sha256": sha256(state_path),
            "initial_episodes": 150, "keep_state_indices": [4, 9]}
    m = dict(root=str(root), output=str(output), status="authorized_prepared", slow_lr=1e-4, fast_steps=0,
        starting_round=4, final_round=5, additional_rounds=2, additional_episodes_per_object=100,
        additional_episodes_total=200, final_episodes_per_object=250,
        protocol="Resume original 1e-4 cl_noada round-3 state; reuse the same 50 targets, order and seeds for rounds 4 and 5",
        comparison_manifest=str(REFERENCE), objects=objects)
    (PLAN / "manifest.json").write_text(json.dumps(m, indent=2)+"\n")
    (PLAN / "validation.json").write_text(json.dumps(dict(status="passed_cpu_only",new_jobs=20,
        historical_jobs_not_rerun=30,source_states_and_configs=validation),indent=2)+"\n")
    print(json.dumps({k:m[k] for k in ("status","slow_lr","starting_round","final_round","additional_episodes_total")},indent=2))


if __name__ == "__main__": main()
