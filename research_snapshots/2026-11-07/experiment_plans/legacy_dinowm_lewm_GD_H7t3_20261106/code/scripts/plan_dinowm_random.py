"""Frozen DINO-WM evaluation with explicit physical-step and noise receipts."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import hydra
from omegaconf import OmegaConf, open_dict
import plan
from planning.evaluator import PlanEvaluator

original_corrupt = PlanEvaluator._corrupt


def deterministic_corrupt(self, obs, seed_offset=0, random_per_call=False):
    if random_per_call and self.ood_corruption is not None:
        counter = getattr(self, "_dinowm_noise_call", 0) + 1
        self._dinowm_noise_call = counter
        seed_offset = 100003 * counter
        with open("observation_noise_seeds.jsonl", "a") as handle:
            first_seed = int(self.seed[0]) if hasattr(self.seed, "__len__") else int(self.seed)
            handle.write(json.dumps({"call": counter, "seed": first_seed + seed_offset,
                                     "corruption": self.ood_corruption}) + "\n")
    return original_corrupt(self, obs, seed_offset=seed_offset, random_per_call=False)


def weights_hash(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


class AuditedWorkspace(plan.PlanWorkspace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.planner.n_taken_actions == 5
        assert self.frameskip == 5
        assert self.planner.max_iter == 6
        assert self.planner.sub_planner.horizon == 5
        assert self.cfg_dict["objective"]["mode"] == "last"
        assert self.cfg_dict["debug_dset_init"] is False
        assert self.cfg_dict["planner"]["continual"]["state_in"] is None
        assert self.planner.sub_planner.eval_every == -1
        receipt = {
            "planner": type(self.planner).__name__,
            "frozen_model": True, "ground_truth_action_initialization": False,
            "objective": "last", "model_frameskip": self.frameskip,
            "horizon_model_steps": self.planner.sub_planner.horizon,
            "executed_model_steps_per_replan": self.planner.n_taken_actions,
            "executed_physical_steps_per_replan": 25, "max_replans": 6,
            "max_physical_steps": 150,
            "candidates": self.planner.sub_planner.num_samples,
            "elite_count": self.planner.sub_planner.topk,
            "cem_iterations": self.planner.sub_planner.opt_steps,
            "oracle_optimizer_probes": False,
            "noise_protocol": "goal seed unchanged; deterministic distinct seed per online observation call",
            "episodes": self.n_evals,
        }
        Path("runtime_protocol.json").write_text(json.dumps(receipt, indent=2))

    def perform_planning(self):
        before = weights_hash(self.wm)
        result = super().perform_planning()
        after = weights_hash(self.wm)
        assert before == after, "Frozen baseline parameters/buffers changed during evaluation"
        Path("frozen_weights_check.json").write_text(json.dumps({
            "before_sha256": before, "after_sha256": after, "unchanged": True,
        }, indent=2))
        return result


PlanEvaluator._corrupt = deterministic_corrupt
plan.PlanWorkspace = AuditedWorkspace

@hydra.main(config_path="../conf", config_name="plan_dinowm_original_budget", version_base="1.1")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = str(Path.cwd())
    config = plan.cfg_to_dict(cfg)
    config["wandb_logging"] = bool(config.get("wandb_logging", False))
    plan.planning_main(config)

if __name__ == "__main__":
    main()
