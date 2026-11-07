"""Remove both familiarity gates; diagnostics never influence the original planner."""
from collections import Counter
import json
import math
from pathlib import Path

import torch

from .continual_v7 import CLSv7MPCPlanner
from .ablation_audit import tree_hash


class NoFamiliarityPlanner(CLSv7MPCPlanner):
    DEFAULT_FAMILIAR_THRESHOLD = .0146

    def __init__(self, ablation='no_familiarity', continual=None, **kwargs):
        assert ablation == 'no_familiarity'
        self.ablation = ablation
        self._diagnostic_scope = None
        self._diagnostic_score = None
        self._diagnostic_counts = Counter()
        super().__init__(continual=continual, **kwargs)
        assert self.familiar_thr == 0 and not self.familiar_read
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

    def _persistent_digest(self):
        domains = {}
        for key, domain in self.domains.items():
            domains[key] = {name: value for name, value in domain.items() if name not in ['learner', 'opt']}
            domains[key]['learner'] = domain['learner'].state_dict() if domain['learner'] is not None else None
            domains[key]['opt'] = domain['opt'].state_dict() if domain['opt'] is not None else None
        return tree_hash({'domains': domains, 'pending': self.pending, 'slow': self._slow,
                          'slow_sampler_rng': self._rng.bit_generator.state})

    def _check_optimizer_lrs(self):
        values = [float(group['lr']) for domain in self.domains.values() if domain['opt'] is not None
                  for group in domain['opt'].param_groups]
        assert all(value == 1e-4 for value in values), values
        return values

    @torch.no_grad()
    def _theta0_latents(self, obs, act):
        # Use the exact latents that the original route/end-of-episode path computes.
        # Do not encode the observations twice or alter route/write decisions.
        z = super()._theta0_latents(obs, act)
        if self._diagnostic_scope is not None and self._diagnostic_score is None:
            devices = [z.device.index] if z.device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                self._diagnostic_score = float(self._familiarity(z))
        return z

    def _route(self, e_obses, taken_actions):
        previous = self._diagnostic_scope, self._diagnostic_score
        self._diagnostic_scope, self._diagnostic_score = 'route', None
        try:
            result = super()._route(e_obses, taken_actions)
            score = self._diagnostic_score
        finally:
            self._diagnostic_scope, self._diagnostic_score = previous
        assert score is not None, 'Missing diagnostic familiarity for routing'
        familiar = math.isfinite(score) and score <= self.DEFAULT_FAMILIAR_THRESHOLD
        record = self._route_log[-1]
        record.update(diagnostic_familiarity=score, diagnostic_default_familiar=familiar,
                      familiarity_read_gate_enabled=False)
        self._diagnostic_counts['route_calls'] += 1
        self._diagnostic_counts['default_familiar_route_calls'] += int(familiar)
        self._diagnostic_counts['routes_with_slow_candidates'] += int(len(record['errs']) > 1)
        self._diagnostic_counts['slow_read_route_calls'] += int(record['best'] is not None)
        self._diagnostic_counts['default_familiar_slow_read_route_calls'] += int(familiar and record['best'] is not None)
        return result

    def _add(self, *args, **kwargs):
        assert not self.read_only
        result = super()._add(*args, **kwargs)
        self._diagnostic_counts['replay_add_calls'] += 1
        return result

    def _learn(self, domain):
        assert not self.read_only
        self._check_optimizer_lrs()
        result = super()._learn(domain)
        self._check_optimizer_lrs()
        self._diagnostic_counts['learn_calls'] += 1
        return result

    def _train(self, domain, steps):
        assert not self.read_only
        self._check_optimizer_lrs()
        result = super()._train(domain, steps)
        self._diagnostic_counts['slow_train_calls'] += 1
        self._diagnostic_counts['slow_train_steps'] += int(steps)
        return result

    def _end_episode(self):
        before = self._diagnostic_counts.copy()
        previous = self._diagnostic_scope, self._diagnostic_score
        self._diagnostic_scope, self._diagnostic_score = 'episode', None
        try:
            result = super()._end_episode()
            score = self._diagnostic_score
        finally:
            self._diagnostic_scope, self._diagnostic_score = previous
        assert score is not None, 'Missing diagnostic end-of-episode familiarity'
        record = self._ep_records[-1]
        familiar = math.isfinite(score) and score <= self.DEFAULT_FAMILIAR_THRESHOLD
        written = record['write'] is not None
        assert record['write_rule'] != 'familiar', 'Disabled familiarity gate still blocked writing'
        self._diagnostic_counts['episodes'] += 1
        self._diagnostic_counts['default_familiar_episodes'] += int(familiar)
        self._diagnostic_counts['direct_write_episodes'] += int(written)
        self._diagnostic_counts['default_familiar_direct_write_episodes'] += int(familiar and written)
        self._diagnostic_counts['slow_read_episodes'] += int(record['read'] is not None)
        record.update(
            ablation=self.ablation, diagnostic_familiarity=score,
            diagnostic_default_familiar=familiar,
            diagnostic_reference_threshold=self.DEFAULT_FAMILIAR_THRESHOLD,
            familiarity_read_gate_enabled=False, familiarity_write_gate_enabled=False,
            replay_add_calls=self._diagnostic_counts['replay_add_calls'] - before['replay_add_calls'],
            slow_learn_calls=self._diagnostic_counts['learn_calls'] - before['learn_calls'],
            slow_train_calls=self._diagnostic_counts['slow_train_calls'] - before['slow_train_calls'],
            slow_train_steps=self._diagnostic_counts['slow_train_steps'] - before['slow_train_steps'],
        )
        if self.read_only:
            assert not written and record['write_rule'] == 'read_only'
            assert record['slow_train_steps'] == record['replay_add_calls'] == 0
        self._audit_seen += 1
        with open('episodes.jsonl', 'a') as stream:
            stream.write(json.dumps(record) + '\n')
        return result

    def plan(self, *args, **kwargs):
        result = super().plan(*args, **kwargs)
        final = self._persistent_digest()
        lrs = self._check_optimizer_lrs()
        if self.read_only:
            assert final == self._audit_initial_digest, 'Test changed persistent slow learning state'
            assert not Path(self.state_out).exists()
            assert self._diagnostic_counts['slow_train_steps'] == 0
            assert self._diagnostic_counts['replay_add_calls'] == 0
        counts = {key: int(self._diagnostic_counts[key]) for key in [
            'episodes', 'default_familiar_episodes', 'direct_write_episodes',
            'default_familiar_direct_write_episodes', 'slow_read_episodes', 'route_calls',
            'default_familiar_route_calls', 'routes_with_slow_candidates', 'slow_read_route_calls',
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
            'active_learners': sum(domain['learner'] is not None for domain in self.domains.values()),
            'domain_keys': list(self.domains), 'slow_lr': self.slow_lr, 'slow_optimizer_lrs': lrs,
            'familiarity_protection_kept': False, 'familiarity_read_gate_enabled': False,
            'familiarity_write_gate_enabled': False, 'familiar_thr': self.familiar_thr,
            'familiar_read': self.familiar_read, 'reference_familiar_threshold': self.DEFAULT_FAMILIAR_THRESHOLD,
            'mature_k_for_write_and_absorption': self.mature_k, 'read_maturity_gate_removed': False,
            'state_written': not self.read_only, 'diagnostics': counts,
            'default_familiar_episode_rate': counts['default_familiar_episodes'] / max(1, counts['episodes']),
            'default_familiar_route_rate': counts['default_familiar_route_calls'] / max(1, counts['route_calls']),
            'diagnostics_only_no_familiarity_gate_decisions': True,
        }, indent=2))
        return result
