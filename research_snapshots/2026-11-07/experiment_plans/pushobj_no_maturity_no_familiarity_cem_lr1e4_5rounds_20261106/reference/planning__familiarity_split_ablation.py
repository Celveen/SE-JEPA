"""Split only the CLSv7 familiarity read/write gates, with passive diagnostics.

The production no-write arm temporarily disables the threshold only while the
original end-of-episode write path runs. Routing and state serialization always
see the configured threshold. Periodic absorption is the unmodified v7 code.
"""
from collections import Counter
import json
import math
from pathlib import Path

import torch

from .adajepa_probe import _cos
from .continual import ContinualMPCPlanner
from .continual_v7 import CLSv7MPCPlanner
from .familiarity_ablation import NoFamiliarityPlanner


class SplitFamiliarityPlanner(NoFamiliarityPlanner):
    """Reuse passive hashes/counters, bypass the both-off parent's gate methods."""

    MODES = {
        'no_familiar_read': (False, True),
        'no_familiar_write': (True, False),
        'no_familiarity': (False, False),
        'full': (True, True),
        'bothoff': (False, False),
    }

    def __init__(self, ablation='no_familiar_read', continual=None, **kwargs):
        assert ablation in self.MODES, ablation
        self.ablation = ablation
        self.familiarity_read_gate_enabled, self.familiarity_write_gate_enabled = self.MODES[ablation]
        self._configured_familiar_thr = 0.0 if ablation in ('no_familiarity', 'bothoff') else self.DEFAULT_FAMILIAR_THRESHOLD
        c = dict(continual or {})
        c.setdefault('familiar_thr', self._configured_familiar_thr)
        c.setdefault('familiar_read', self.familiarity_read_gate_enabled)
        assert float(c['familiar_thr']) == self._configured_familiar_thr, (ablation, c['familiar_thr'])
        assert bool(c['familiar_read']) == self.familiarity_read_gate_enabled, (ablation, c['familiar_read'])
        self._diagnostic_scope = None
        self._diagnostic_score = None
        self._diagnostic_counts = Counter()
        self._write_threshold_override_active = False
        # NoFamiliarityPlanner's constructor intentionally only permits both-off.
        # Its persistent digest, RNG-neutral latent probe and update counters are
        # reused; the actual model initialization is the unchanged stock v7 one.
        CLSv7MPCPlanner.__init__(self, continual=c, **kwargs)
        self._assert_configured_gates()
        assert not self._pred0_module.training, 'Familiarity diagnostics must use the stock eval-mode theta0 predictor'
        assert self.slow_lr == 1e-4 and self.slow_steps == 15 and self.slow_batch == 8
        assert self.mature_k == 3 and self.mature_ratio == .9 and self.reject_ratio == 1.2
        assert self.min_n == 3 and self.write_w == 0 and self.write_mode == 'delta'
        assert self.pending_m == 3 and self.tau_new == .05 and self.fp_unit
        assert self.tau == .1 and self.merge_k == 2 and self.merge_eps == .5
        assert self.absorb_every == 25 and self.absorb_frag == 6 and not self.absorb_pending
        assert self.join_ratio == 0 and self.absorb_ratio == .8 and not self.learner_win
        assert self.abs_ratio == 0
        self._audit_initial_episode = self._episode
        self._audit_seen = 0
        self._audit_initial_digest = self._persistent_digest()
        self._check_optimizer_lrs()

    def _assert_configured_gates(self):
        assert not self._write_threshold_override_active
        assert self.familiar_thr == self._configured_familiar_thr
        assert self.familiar_read == self.familiarity_read_gate_enabled

    def _route(self, e_obses, taken_actions):
        self._assert_configured_gates()
        available = len(self._learners())  # Pure eligibility query; no updates or RNG.
        previous = self._diagnostic_scope, self._diagnostic_score
        self._diagnostic_scope, self._diagnostic_score = 'route', None
        try:
            result = CLSv7MPCPlanner._route(self, e_obses, taken_actions)
            score = self._diagnostic_score
        finally:
            self._diagnostic_scope, self._diagnostic_score = previous
        assert score is not None, 'Missing route familiarity diagnostic'
        familiar = math.isfinite(score) and score <= self.DEFAULT_FAMILIAR_THRESHOLD
        record = self._route_log[-1]
        blocked = bool(self.familiarity_read_gate_enabled and familiar and available > 0)
        if self.familiarity_read_gate_enabled and familiar:
            assert record['best'] is None and len(record['errs']) == 1
        else:
            assert len(record['errs']) == 1 + available
        record.update(
            diagnostic_familiarity=score, diagnostic_default_familiar=familiar,
            diagnostic_reference_threshold=self.DEFAULT_FAMILIAR_THRESHOLD,
            familiarity_read_gate_enabled=self.familiarity_read_gate_enabled,
            familiarity_write_gate_enabled=self.familiarity_write_gate_enabled,
            available_slow_candidates=available,
            default_familiar_read_gate_blocked=blocked,
        )
        self._diagnostic_counts['route_calls'] += 1
        self._diagnostic_counts['default_familiar_route_calls'] += int(familiar)
        self._diagnostic_counts['routes_with_slow_candidates'] += int(len(record['errs']) > 1)
        self._diagnostic_counts['default_familiar_candidate_routes'] += int(familiar and available > 0)
        self._diagnostic_counts['default_familiar_read_gate_blocks'] += int(blocked)
        self._diagnostic_counts['slow_read_route_calls'] += int(record['best'] is not None)
        self._diagnostic_counts['default_familiar_slow_read_route_calls'] += int(familiar and record['best'] is not None)
        return result

    def _end_episode(self):
        self._assert_configured_gates()
        before = self._diagnostic_counts.copy()
        previous = self._diagnostic_scope, self._diagnostic_score
        self._diagnostic_scope, self._diagnostic_score = 'episode', None
        saved_threshold = self.familiar_thr
        # Read-only tests never enter the write path. Keep the normal score and
        # read threshold there. The temporary value is not a persistent config.
        disable_write = not self.familiarity_write_gate_enabled and not self.read_only
        try:
            if disable_write:
                self._write_threshold_override_active = True
                self.familiar_thr = 0.0
            result = CLSv7MPCPlanner._end_episode(self)
            score = self._diagnostic_score
        finally:
            self.familiar_thr = saved_threshold
            self._write_threshold_override_active = False
            self._diagnostic_scope, self._diagnostic_score = previous
        self._assert_configured_gates()
        assert score is not None, 'Missing episode familiarity diagnostic'
        record = self._ep_records[-1]
        familiar = math.isfinite(score) and score <= self.DEFAULT_FAMILIAR_THRESHOLD
        written = record['write'] is not None
        if not self.read_only:
            if self.familiarity_write_gate_enabled and familiar:
                assert record['write_rule'] == 'familiar' and not written
            elif not self.familiarity_write_gate_enabled:
                assert record['write_rule'] != 'familiar'
        self._diagnostic_counts['episodes'] += 1
        self._diagnostic_counts['default_familiar_episodes'] += int(familiar)
        self._diagnostic_counts['direct_write_episodes'] += int(written)
        self._diagnostic_counts['default_familiar_direct_write_episodes'] += int(familiar and written)
        self._diagnostic_counts['default_familiar_write_gate_blocks'] += int(
            not self.read_only and familiar and record['write_rule'] == 'familiar')
        self._diagnostic_counts['slow_read_episodes'] += int(record['read'] is not None)
        record.update(
            ablation=self.ablation, diagnostic_familiarity=score,
            diagnostic_default_familiar=familiar,
            diagnostic_reference_threshold=self.DEFAULT_FAMILIAR_THRESHOLD,
            familiarity_read_gate_enabled=self.familiarity_read_gate_enabled,
            familiarity_write_gate_enabled=self.familiarity_write_gate_enabled,
            configured_familiar_threshold=self._configured_familiar_thr,
            effective_write_familiar_threshold=0.0 if disable_write else saved_threshold,
            replay_add_calls=self._diagnostic_counts['replay_add_calls'] - before['replay_add_calls'],
            slow_learn_calls=self._diagnostic_counts['learn_calls'] - before['learn_calls'],
            slow_train_calls=self._diagnostic_counts['slow_train_calls'] - before['slow_train_calls'],
            slow_train_steps=self._diagnostic_counts['slow_train_steps'] - before['slow_train_steps'],
        )
        if self.read_only:
            assert not written and record['write_rule'] == 'read_only'
            assert record['slow_train_steps'] == record['replay_add_calls'] == record['slow_learn_calls'] == 0
        # A familiar episode can still coincide with stock periodic absorption;
        # do not incorrectly forbid that unrelated maintenance training.
        self._audit_seen += 1
        with open('episodes.jsonl', 'a') as stream:
            stream.write(json.dumps(record) + '\n')
        return result

    def _save_state(self, path):
        # Stock save records cls_cfg.familiar_thr. It must never serialize the
        # temporary zero used inside an end-of-episode write operation.
        self._assert_configured_gates()
        return CLSv7MPCPlanner._save_state(self, path)

    def plan(self, *args, **kwargs):
        self._assert_configured_gates()
        result = CLSv7MPCPlanner.plan(self, *args, **kwargs)
        self._assert_configured_gates()
        final = self._persistent_digest()
        lrs = self._check_optimizer_lrs()
        if self.read_only:
            assert final == self._audit_initial_digest, 'Test changed persistent slow state'
            assert not Path(self.state_out).exists()
            assert all(self._diagnostic_counts[key] == 0 for key in
                       ['replay_add_calls', 'learn_calls', 'slow_train_calls', 'slow_train_steps'])
        counts = {key: int(self._diagnostic_counts[key]) for key in [
            'episodes', 'default_familiar_episodes', 'direct_write_episodes',
            'default_familiar_direct_write_episodes', 'default_familiar_write_gate_blocks',
            'slow_read_episodes', 'route_calls', 'default_familiar_route_calls',
            'routes_with_slow_candidates', 'default_familiar_candidate_routes',
            'default_familiar_read_gate_blocks', 'slow_read_route_calls',
            'default_familiar_slow_read_route_calls', 'replay_add_calls', 'learn_calls',
            'slow_train_calls', 'slow_train_steps',
        ]}
        Path('ablation_audit.json').write_text(json.dumps({
            'ablation': self.ablation, 'read_only': self.read_only,
            'initial_episode': self._audit_initial_episode, 'final_episode': self._episode,
            'episodes': self._audit_seen, 'initial_persistent_sha256': self._audit_initial_digest,
            'final_persistent_sha256': final, 'persistent_state_unchanged': final == self._audit_initial_digest,
            'persistent_hash_scope': 'domains, slow optimizers, replay, fingerprints, pending, slow parameters, slow RNG',
            'transient_diagnostics_excluded_from_hash': 'episode counters, transfer memory, previous fast delta',
            'active_learners': sum(d['learner'] is not None for d in self.domains.values()),
            'domain_keys': list(self.domains), 'slow_lr': self.slow_lr, 'slow_optimizer_lrs': lrs,
            'familiarity_read_gate_enabled': self.familiarity_read_gate_enabled,
            'familiarity_write_gate_enabled': self.familiarity_write_gate_enabled,
            'familiar_thr': self.familiar_thr, 'familiar_read': self.familiar_read,
            'reference_familiar_threshold': self.DEFAULT_FAMILIAR_THRESHOLD,
            'write_gate_scope': 'end_episode_only; normal configuration restored before routing and save',
            'threshold_restored_before_state_save': True,
            'mature_k_for_write_and_absorption': self.mature_k, 'read_maturity_gate_removed': False,
            'periodic_absorption_unchanged': True, 'state_written': not self.read_only,
            'diagnostics': counts,
            'default_familiar_episode_rate': counts['default_familiar_episodes'] / max(1, counts['episodes']),
            'default_familiar_route_rate': counts['default_familiar_route_calls'] / max(1, counts['route_calls']),
            'diagnostics_only_no_familiarity_gate_decisions': True,
        }, indent=2))
        return result


class WriteDisabledReferencePlanner(CLSv7MPCPlanner):
    """Smoke-only independent reference: stock v7 with one Boolean gate removed.

    Its copied end-of-episode method keeps the actual familiarity calculation and
    changes only ``familiar = fam <= self.familiar_thr`` to ``familiar = False``.
    It never changes the threshold. The validation script checks this AST delta.
    """

    def __init__(self, continual=None, **kwargs):
        super().__init__(continual=continual, **kwargs)
        assert self.familiar_thr == .0146 and self.familiar_read
        assert not self._pred0_module.training

    def _end_episode(self):
        if self.read_only:  # frozen evaluation: record what was read, change no domain, learner or pending episode
            z = self._episode_latents()
            fam = self._familiarity(z) if self.familiar_thr > 0 else float("nan")
            ContinualMPCPlanner._end_episode(self)  # w = 0 here, so this only records the episode
            self._ep_records[-1].update({"read": self._base, "route": self._route_log, "write": None,
                                         "write_rule": "read_only", "familiar_score": fam})
            return
        delta = [p.detach() - s for p, s in zip(self._params, self._start)]
        z = self._episode_latents()
        fam = self._familiarity(z) if self.familiar_thr > 0 else float("nan")
        familiar = False  # Reference-only: disable this write gate; retain actual score.
        base = self._base
        ratios = {}
        if not familiar:  # every learner validated on this (unseen) episode: merge evidence, write gate, join
            e0 = max(self._rollout_err(self._pred0_module, z), 1e-12)
            ratios = {dk: self._rollout_err(d["learner"], z) / e0 for dk, d in self.domains.items()
                      if d["learner"] is not None and not d.get("merged_into")}
        self._val_cache = dict(ratios)
        self._abs_cache = {}
        rule, key, rejected, new_members = ("familiar" if familiar else "none"), None, [], []
        if not familiar:
            lost = False  # learner_win: the read learner did not beat theta0 on the whole episode
            if base is not None:
                if self._rejects(base, z):
                    rejected.append(base)
                elif self.learner_win and not self._wins(base, z):
                    lost = True
                else:
                    rule, key = "learner", base
            if key is None:  # the mature domain whose learner predicts the episode best
                fit = sorted((r, dk) for dk, r in ratios.items() if dk not in rejected and self._mature(self.domains[dk]))
                if fit and fit[0][0] <= self.join_ratio:
                    if self._abs_rejects(fit[0][1], z):
                        rejected.append(fit[0][1])
                    else:
                        rule, key = "join", fit[0][1]
            if key is None and base is None:  # planned with theta0: v5's fingerprint join, then the pending pool
                cands = sorted(((_cos(delta, [s / d["fp_n"] for s in d["fp_sum"]]), dk)
                                for dk, d in self.domains.items() if d["fp_n"] > 0 and not d.get("merged_into")),
                               reverse=True)
                for c, dk in cands:
                    if c < self.tau:
                        break
                    if dk in rejected or self._rejects(dk, z):
                        if dk not in rejected:
                            rejected.append(dk)
                        continue
                    rule, key = "direction", dk
                    break
                if key is None:
                    key = self._abs_join(z, rejected)
                    rule = "abs_join"
                if key is None:
                    rule, key, new_members = self._pend(delta, z)
            elif key is None and any(dk in self._abs_cache and self._abs_cache[dk] > self.abs_ratio for dk in rejected):
                # far from the domain it read: the closest other domain within abs_ratio, or the pending pool
                rule, key = "abs_join", self._abs_join(z, rejected)
                if key is None:
                    rule, key, new_members = self._pend(delta, z)
            elif key is None:
                rule = "none" if lost else "rejected"
        slow_loss, val_ratio = float("nan"), float("nan")
        if key is not None:
            d = self.domains.setdefault(key, {"n": 0, "tags": {}, "replay": [], "fp_sum": None, "fp_n": 0,
                                              "learner": None, "opt": None, "val": []})
            if d["learner"] is not None:  # unseen by this learner: validate before training on it
                val_ratio = self._val_ratio(key, z)
                if rule != "join":
                    d.setdefault("val", []).append(float(val_ratio))
            dev = self._params[0].device
            members = [(m["tag"], m["z"].to(dev), self._unflat(m["fp"].float().to(dev))) for m in new_members]
            for tag, mz, mfp in members + [(self.tag, z.detach(), self._fp(delta) if base is None else None)]:
                self._add(d, tag, mz, mfp)
            slow_loss = self._learn(d)
        ContinualMPCPlanner._end_episode(self)
        rec = self._ep_records[-1]
        merges, absorbs = [], []
        if key is not None and ratios:
            self.domains[key].setdefault("vals", []).append((rec["episode"], ratios))
            merges = self._try_merge()
        if self.absorb_every > 0 and self._episode % self.absorb_every == 0:
            absorbs = self._absorb()
        rec.update({
            "read": base, "route": self._route_log, "write": key, "write_rule": rule, "slow_loss": slow_loss,
            "val_ratio": val_ratio, "rejected_by": rejected,
            "rejected_ratios": {dk: self._val_cache[dk] for dk in rejected if dk in self._val_cache},
            "abs_ratios": dict(self._abs_cache),
            "familiar_score": fam, "pending_n": len(self.pending), "new_members": [m["tag"] for m in new_members],
            "val_all": ratios, "merges": merges, "absorbs": absorbs,
            "domains": {dk: {"n": d["n"], "tags": dict(d["tags"]), "learner": d["learner"] is not None,
                             "mature": d["learner"] is not None and self._mature(d), "val_last": d.get("val", [])[-3:]}
                        for dk, d in self.domains.items() if d["n"] >= 2},
        })
