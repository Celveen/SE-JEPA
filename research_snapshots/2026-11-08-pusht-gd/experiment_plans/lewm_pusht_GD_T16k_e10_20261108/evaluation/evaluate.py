"""Versioned GD/CEM evaluation against the unchanged training code snapshot."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))
import hydra
from omegaconf import OmegaConf, open_dict
import plan
from planning.evaluator import PlanEvaluator

original_corrupt = PlanEvaluator._corrupt


def deterministic_corrupt(self, obs, seed_offset=0, random_per_call=False):
    if random_per_call and self.ood_corruption is not None:
        call = getattr(self, "_paired_noise_call", 0) + 1
        self._paired_noise_call = call
        seed_offset = 100003 * call
        first = int(self.seed[0]) if hasattr(self.seed, "__len__") else int(self.seed)
        with open("observation_noise_seeds.jsonl", "a") as handle:
            handle.write(json.dumps({"call": call, "seed": first + seed_offset,
                                     "corruption": self.ood_corruption}) + "\n")
    return original_corrupt(self, obs, seed_offset=seed_offset, random_per_call=False)


def weights_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class AuditedWorkspace(plan.PlanWorkspace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        audit = self.cfg_dict["audit"]
        assert self.frameskip == 5 and self.planner.sub_planner.horizon == 5
        assert self.planner.n_taken_actions * self.frameskip == audit["expected_physical_steps"]
        assert self.planner.max_iter == audit["expected_replans"]
        assert audit["expected_physical_steps"] * audit["expected_replans"] == 150
        assert self.cfg_dict["objective"]["mode"] == audit["expected_objective"]
        assert not self.cfg_dict["debug_dset_init"]
        assert self.cfg_dict["planner"]["continual"]["state_in"] is None
        assert self.planner.sub_planner.eval_every == -1
        assert self.planner.sub_planner.opt_steps == audit["expected_opt_steps"]
        # GD updates actions only. Disabling parameter gradients preserves the
        # action-gradient path while avoiding unused world-model gradients.
        self.wm.requires_grad_(False)
        self.wm.eval()
        if not audit["diagnostic_only"]:
            assert self.n_evals == 50 and int(self.cfg_dict["model_epoch"]) == audit["expected_model_epoch"]
        receipt = {
            "planner": audit["planner_kind"], "condition": audit["condition"],
            "pretraining_shape_status": audit["pretraining_shape_status"],
            "model_epoch": self.cfg_dict["model_epoch"], "frozen_model": True,
            "model_parameter_gradients_disabled": True, "ground_truth_action_initialization": False,
            "objective": self.cfg_dict["objective"]["mode"], "model_frameskip": self.frameskip,
            "horizon_model_steps": self.planner.sub_planner.horizon,
            "executed_model_steps_per_replan": self.planner.n_taken_actions,
            "executed_physical_steps_per_replan": audit["expected_physical_steps"],
            "max_replans": self.planner.max_iter, "max_physical_steps": 150,
            "optimizer_iterations": self.planner.sub_planner.opt_steps,
            "episodes": self.n_evals, "diagnostic_only": audit["diagnostic_only"],
            "oracle_optimizer_probes": False,
            "noise_protocol": "identical goal seed; distinct deterministic seed per feedback call",
        }
        if audit["planner_kind"] == "cem":
            assert self.planner.sub_planner.num_samples == audit["expected_candidates"]
            assert self.planner.sub_planner.topk == audit["expected_elites"]
            receipt.update(candidates=self.planner.sub_planner.num_samples,
                           elites=self.planner.sub_planner.topk)
        else:
            assert audit["planner_kind"] == "gd"
            assert self.planner.sub_planner.microbatch_size == audit["expected_microbatch_size"]
            receipt.update(lr=self.planner.sub_planner.lr,
                           action_optimizer=self.planner.sub_planner.optimizer_name,
                           cosine_scheduler=self.planner.sub_planner.use_cosine_scheduler,
                           action_initialization=self.planner.sub_planner.sample_type,
                           action_noise=self.planner.sub_planner.action_noise,
                           microbatch_size=self.planner.sub_planner.microbatch_size)
        Path("runtime_protocol.json").write_text(json.dumps(receipt, indent=2))
        Path("protocol.json").write_text(json.dumps(receipt, indent=2))

    def perform_planning(self):
        before = weights_hash(self.wm)
        result = super().perform_planning()
        after = weights_hash(self.wm)
        assert before == after, "Model weights or buffers changed during frozen evaluation"
        Path("frozen_weights_check.json").write_text(json.dumps({
            "before_sha256": before, "after_sha256": after, "unchanged": True,
        }, indent=2))
        return result


PlanEvaluator._corrupt = deterministic_corrupt
plan.PlanWorkspace = AuditedWorkspace


@hydra.main(config_path="conf", config_name="plan", version_base="1.1")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = str(Path.cwd())
    plan.planning_main(plan.cfg_to_dict(cfg))


if __name__ == "__main__":
    main()
