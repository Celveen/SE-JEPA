"""Standard-library-only semantic checks of the published ablation.

The checks execute the actual stock routing/writing methods and actual wrapper
methods with small substitutes for tensors, model losses, and persistence.
They are NOT a model run, CUDA test, or replacement for trajectory equivalence.
"""
import ast
from collections import Counter
from contextlib import nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def read_tree(relative):
    return ast.parse((ROOT / relative).read_text(encoding="utf-8-sig"))


def class_node(tree, name):
    return next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == name)


def method_node(cls, name):
    return next(node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


STOCK = class_node(read_tree("planning/continual_v7.py"), "CLSv7MPCPlanner")
BASE = class_node(read_tree("planning/continual.py"), "CLSMPCPlanner")
GD = class_node(read_tree("planning/familiarity_ablation.py"), "NoFamiliarityPlanner")
SPLIT_TREE = read_tree("planning/familiarity_split_ablation.py")
CEM = class_node(SPLIT_TREE, "SplitFamiliarityPlanner")


def check_source_invariants():
    # The independent write-disabled reference must differ in one gate only.
    expected = copy.deepcopy(method_node(STOCK, "_end_episode"))
    changed = 0
    for node in ast.walk(expected):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "familiar"):
            assert ast.unparse(node.value) == "fam <= self.familiar_thr"
            node.value = ast.Constant(value=False)
            changed += 1
    reference = method_node(class_node(SPLIT_TREE, "WriteDisabledReferencePlanner"), "_end_episode")
    assert changed == 1
    assert ast.dump(expected, include_attributes=False) == ast.dump(reference, include_attributes=False)
    absorption = ast.unparse(method_node(STOCK, "_absorb"))
    assert "familiar_thr" not in absorption and "_familiarity" not in absorption
    # No gate-removal wrapper replaces the domain allocation/maturity rules.
    forbidden = {"_learners", "_mature", "_rejects", "_pend", "_try_merge", "_absorb"}
    for cls in (GD, CEM):
        names = {node.name for node in cls.body if isinstance(node, ast.FunctionDef)}
        assert not (names & forbidden), names & forbidden
    latent_probe = ast.unparse(method_node(GD, "_theta0_latents"))
    assert "torch.random.fork_rng" in latent_probe
    digest = ast.unparse(method_node(GD, "_persistent_digest"))
    for required in ("domains", "pending", "slow", "slow_sampler_rng", "bit_generator.state"):
        assert required in digest


class FakeTorch:
    random = SimpleNamespace(fork_rng=lambda **kwargs: nullcontext())

    @staticmethod
    def no_grad():
        return lambda function: function


class FakeNumpy:
    @staticmethod
    def argmin(values):
        return min(range(len(values)), key=values.__getitem__)

    @staticmethod
    def mean(values):
        return sum(values) / len(values)


class Latents:
    device = SimpleNamespace(type="cpu", index=None)

    def detach(self):
        return self


class Parameter:
    device = "cpu"

    def detach(self):
        return 1.0


class Predictor:
    def __init__(self, loss):
        self.loss = loss
        self.weight = 0

    def state_dict(self):
        return {"weight": self.weight}


class Optimizer:
    param_groups = [{"lr": 1e-4}]

    def state_dict(self):
        return {"lr": 1e-4}


class FakeContinual:
    def _end_episode(self):
        self._ep_records.append({"episode": self._episode})
        self._episode += 1


class FakeStock(FakeContinual):
    """Only expensive numerical helpers are replaced; gate branches are real."""

    def _extract_adajepa_data(self, *args):
        return "observations", "actions"

    def _theta0_latents(self, *args):
        return Latents()

    def _episode_latents(self):
        return self._theta0_latents(None, None)

    def _familiarity(self, z):
        return self.probe_score

    def _latent_loss(self, module, z):
        return module.loss

    _rollout_err = _latent_loss

    def _load_base(self, key, carry_fast):
        assert carry_fast
        self._base = key

    def _rejects(self, key, z):
        return self.reject_candidate

    def _val_ratio(self, key, z):
        return 0.5

    def _add(self, domain, tag, z, fingerprint):
        domain["n"] += 1
        domain["replay"].append("latent-window")

    def _learn(self, domain):
        return self._train(domain, 15)

    def _train(self, domain, steps):
        domain["learner"].weight += steps
        return 0.001

    def _try_merge(self):
        self.merge_checks += 1
        return []

    def _absorb(self):
        self.absorption_checks += 1
        self._train(self.domains["d0"], 2)
        return []

    def _save_state(self, path):
        self.observed_save_threshold = self.familiar_thr

    def plan(self, *args, **kwargs):
        if self.inject_persistent_mutation:
            self.domains["d0"]["n"] += 1
        return "stand-in plan result"


def fake_digest(value):
    # Tests change detection only; production uses its tensor-aware tree_hash.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


SCOPE = {"torch": FakeTorch, "np": FakeNumpy, "Counter": Counter, "json": json,
         "math": math, "Path": Path, "tree_hash": fake_digest,
         "ContinualMPCPlanner": FakeContinual, "CLSv7MPCPlanner": FakeStock}


def compile_method(cls, name):
    node = copy.deepcopy(method_node(cls, name))
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = dict(SCOPE)
    exec(compile(module, "<actual method: " + name + ">", "exec"), scope)
    return scope[name]


# Execute the repository's actual read, write, and maturity branches.
for source, names in ((STOCK, ("_route", "_end_episode")), (BASE, ("_learners", "_mature"))):
    for name in names:
        setattr(FakeStock, name, compile_method(source, name))
for cls in (GD, CEM):
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(cls)], type_ignores=[]))
    exec(compile(module, "<actual wrapper: " + cls.name + ">", "exec"), SCOPE)
GDPlanner, CEMPlanner = SCOPE["NoFamiliarityPlanner"], SCOPE["SplitFamiliarityPlanner"]


def instance(cls, readonly=False, score=0.005, mature=True, reject=False, mode="no_familiarity"):
    # Constructor/config integration is a separate check; no model is loaded.
    obj = object.__new__(cls)
    obj.ablation = mode
    read_gate, write_gate = CEMPlanner.MODES[mode]
    obj.familiarity_read_gate_enabled, obj.familiarity_write_gate_enabled = read_gate, write_gate
    obj._configured_familiar_thr = 0.0 if mode in ("no_familiarity", "bothoff") else 0.0146
    obj.familiar_thr, obj.familiar_read = obj._configured_familiar_thr, read_gate
    obj._write_threshold_override_active = False
    obj.read_only, obj.probe_score, obj.reject_candidate = readonly, score, reject
    obj._diagnostic_scope = obj._diagnostic_score = None
    obj._diagnostic_counts = Counter()
    obj._ep_records, obj._route_log = [], []
    obj._audit_seen = obj._audit_initial_episode = 0
    obj._episode, obj.iter = 1, 1
    obj._obs_buffer, obj._act_buffer, obj._slow, obj.pending = [], [], [], []
    obj._params, obj._start = [Parameter()], [0.0]
    obj._base, obj.tag = None, "I"
    obj._pred0_module = Predictor(1.0)
    obj.domains = {"d0": {"n": 4, "tags": {}, "replay": [], "fp_n": 0,
                          "learner": Predictor(0.5), "opt": Optimizer(),
                          "val": [0.5] * (3 if mature else 2)}}
    obj._rng = SimpleNamespace(bit_generator=SimpleNamespace(state={"state": 123}))
    obj.slow_lr, obj.mature_k, obj.mature_ratio = 1e-4, 3, 0.9
    obj.join_ratio, obj.abs_ratio, obj.learner_win = 0, 0, False
    obj.absorb_every, obj.merge_checks, obj.absorption_checks = 25, 0, 0
    obj.inject_persistent_mutation = False
    obj.state_out = "must_not_exist.pt"
    obj._audit_initial_digest = obj._persistent_digest()
    return obj


def raises_assertion(call):
    try:
        call()
    except AssertionError:
        return
    raise AssertionError("Expected the production audit to reject this operation")


def check_runtime_semantics():
    route_cases = write_cases = readonly_cases = 0
    for cls in (GDPlanner, CEMPlanner):
        for score in (0.005, 0.03):
            for mature in (False, True):
                obj = instance(cls, score=score, mature=mature)
                obj._route(None, None)
                assert obj._base == ("d0" if mature else None)
                assert len(obj._route_log[-1]["errs"]) == (2 if mature else 1)
                route_cases += 1
            for reject in (False, True):
                obj = instance(cls, score=score, reject=reject)
                obj._base = "d0"
                obj._end_episode()
                record = obj._ep_records[-1]
                assert record["write_rule"] == ("rejected" if reject else "learner")
                assert record["write"] == (None if reject else "d0")
                assert record["slow_train_steps"] == (0 if reject else 15)
                assert obj.merge_checks == (0 if reject else 1)
                write_cases += 1
            obj = instance(cls, readonly=True, score=score)
            obj._base = "d0"
            obj._end_episode()
            assert obj._persistent_digest() == obj._audit_initial_digest
            assert obj._ep_records[-1]["write_rule"] == "read_only"
            assert obj.merge_checks == obj.absorption_checks == 0
            obj.plan()
            assert not Path(obj.state_out).exists()
            for name, args in (("_add", (obj.domains["d0"], "I", Latents(), None)),
                               ("_learn", (obj.domains["d0"],)),
                               ("_train", (obj.domains["d0"], 1))):
                raises_assertion(lambda name=name, args=args: getattr(obj, name)(*args))
            obj.inject_persistent_mutation = True
            raises_assertion(obj.plan)
            readonly_cases += 1

    full = instance(CEMPlanner, mode="full")
    full._route(None, None)
    assert full._base is None and len(full._route_log[-1]["errs"]) == 1
    full._episode = 24
    full._end_episode()
    assert full._ep_records[-1]["write_rule"] == "familiar"
    assert full._ep_records[-1]["write"] is None
    assert full.absorption_checks == 1
    assert full._ep_records[-1]["slow_train_steps"] == 2

    write_off = instance(CEMPlanner, mode="no_familiar_write")
    write_off._base = "d0"
    write_off._end_episode()
    assert write_off._ep_records[-1]["write"] == "d0"
    assert write_off.familiar_thr == 0.0146
    write_off._save_state("unused")
    assert write_off.observed_save_threshold == 0.0146

    saved = FakeStock._end_episode
    try:
        def fail(self):
            assert self.familiar_thr == 0.0
            raise RuntimeError("injected write failure")
        FakeStock._end_episode = fail
        try:
            write_off._end_episode()
        except RuntimeError as error:
            assert str(error) == "injected write failure"
        else:
            raise AssertionError("Expected injected failure")
        assert write_off.familiar_thr == 0.0146
        assert not write_off._write_threshold_override_active
        assert write_off._diagnostic_scope is None and write_off._diagnostic_score is None
    finally:
        FakeStock._end_episode = saved
    return route_cases, write_cases, readonly_cases


def main():
    check_source_invariants()
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="no_familiarity_validation_") as temporary:
        try:
            os.chdir(temporary)
            route_cases, write_cases, readonly_cases = check_runtime_semantics()
        finally:
            os.chdir(previous)
    print(json.dumps({
        "passed": True,
        "validation_kind": "actual stock/wrapper methods with numerical stand-ins; standard library only",
        "real_model_rerun": False,
        "constructor_config_and_CUDA_validation_included": False,
        "actual_routing_and_maturity_cases": route_cases,
        "actual_write_and_reject_cases": write_cases,
        "readonly_persistence_and_illegal_training_cases": readonly_cases,
        "familiar_inputs_can_read_and_write_when_both_gates_disabled": True,
        "immature_readers_and_rejected_writes_still_blocked": True,
        "full_protection_still_blocks_direct_write_but_keeps_absorption": True,
        "readonly_audit_detects_persistent_mutation": True,
        "threshold_restored_before_save_and_after_failure": True,
        "independent_write_reference_has_only_one_gate_AST_change": True,
        "real_model_equivalence_requires_separate_smoke": True,
    }, indent=2))


if __name__ == "__main__":
    main()
