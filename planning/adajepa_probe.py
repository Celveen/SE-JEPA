"""Per-sample AdaJEPA MPC planner with diagnostics; the base class of the continual (SE-JEPA) planners.

ProbeMPCPlanner runs the AdaJEPA loop one evaluation sample at a time: after every replan, one Adam step on
the recent buffer of executed transitions adapts the predictor's last block and the encoder head. Each replan
also logs task progress (distance to the goal state) and, around the update, eval-mode prediction errors
under the current and the pre-adaptation weights: on the whole buffer, and on the newest frame alone, which
no update has seen yet. After the update it logs the step and displacement norms per module and the induced
change in predictions. All of it lands in per_iter_logs.json next to the usual fields.
"""

import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .adajepa_mpc import AdaJEPAMPCPlanner
from .mpc import _cuda_sync

class ProbeMPCPlanner(AdaJEPAMPCPlanner):
    def __init__(self, probe=None, **kwargs):
        super().__init__(**kwargs)
        if self.wm.concat_dim != 1:
            raise ValueError("ProbeMPCPlanner assumes concat_dim=1 (visual | proprio | action channels)")
        probe = dict(probe) if probe else {}
        if self.replay_mode == "hard":
            raise ValueError("ProbeMPCPlanner merges the buffer into one sequence; use recent<N> or false")
        self.probe_seed = int(probe.get("seed", 0))
        tr = self.adajepa_trainer
        self._groups = {"pred": len(tr._ada_predictor_params), "enc": len(tr._ada_encoder_params)}
        self._params = list(tr._ada_predictor_params) + list(tr._ada_encoder_params)
        self._init = [p.detach().clone() for p in self._params]
        self._drop_act = self.wm.action_dim
        self._drop_obs = self.wm.proprio_dim + self.wm.action_dim
        self._last_eval = None
        meta = {
            "seed": self.probe_seed,
            "n_tensors": self._groups,
            "n_params": {"pred": sum(p.numel() for p in self._params[: self._groups["pred"]]),
                         "enc": sum(p.numel() for p in self._params[self._groups["pred"]:])},
            "lr": tr.lr, "encoder_lr": tr.encoder_lr, "steps": tr.steps, "replay": [self.replay_mode, self.replay_size],
        }
        Path("probe_meta.json").write_text(json.dumps(meta, indent=2))

    # ------------------------------------------------------------------ episode setup
    def _plan_single(self, **kw):
        """Keep the env feedback of every replan."""
        evaluator = self.evaluator
        inner = evaluator.eval_actions

        def eval_and_keep(*args, **kwargs):
            out = inner(*args, **kwargs)
            self._last_eval = out
            return out

        evaluator.eval_actions = eval_and_keep
        try:
            return super()._plan_single(**kw)
        finally:
            del evaluator.eval_actions

    def _record_eval(self, out):
        # resim_from_start=False: eval_and_keep saw only the last iteration's actions; keep the whole episode
        self._last_eval = out

    # ------------------------------------------------------------------ MPC hook
    def _post_env_feedback(self, taken_actions, e_obses):
        obs_seq, act_seq = self._extract_adajepa_data(e_obses, taken_actions, self.iter)
        self._obs_buffer.append(obs_seq)
        self._act_buffer.append(act_seq)
        self._segment_scores = self._trim_buffer(self._obs_buffer, self._act_buffer, self._segment_scores)
        rec = {
            "sample_idx": self._sample_idx,
            "step": self.iter + 1,
            "success": bool(self.is_success[0]),
            "t_plan_s": self.t_plan_s,
            "buf_n": len(self._obs_buffer),
            "act": [round(float(x), 4) for x in taken_actions[0].flatten().tolist()],
        }
        rec.update(self._task_progress())
        extra_logs = None
        if (self.iter + 1) % self.adajepa_finetune_every == 0:
            _cuda_sync()
            t0 = time.perf_counter()
            obs, act = self._merged_buffer()
            rec.update(self._losses_now_and_init(obs, act))
            step_losses, applied, d_own = self._update(obs, act)
            rec.update(self._update_stats(obs, act, applied, d_own))
            _cuda_sync()
            rec["t_adajepa_s"] = time.perf_counter() - t0
            pred_loss = step_losses[-1] if step_losses else float("nan")
            rec["adajepa/pred_loss"] = pred_loss
            self._record_adajepa_losses(self._sample_idx, self.iter + 1, step_losses)
            extra_logs = {"adajepa/pred_loss": pred_loss}
        self._records.append(rec)
        return extra_logs

    def _merged_buffer(self):
        if len(self._obs_buffer) == 1:
            return self._obs_buffer[0], self._act_buffer[0]
        obs, act = self.adajepa_trainer._merge_segments(self._obs_buffer, self._act_buffer)
        return obs[0], act[0]

    # ------------------------------------------------------------------ updates
    def _update(self, obs, act):
        """One AdaJEPA step on the merged buffer. Returns (step losses, applied change, applied change)."""
        before = [p.detach().clone() for p in self._params]
        step_losses = self.adajepa_trainer.finetune([obs], [act], merge=False)
        applied = [p.detach() - b for p, b in zip(self._params, before)]
        return step_losses, applied, applied

    @torch.no_grad()
    def _set(self, values):
        for p, v in zip(self._params, values):
            p.copy_(v)

    @contextlib.contextmanager
    def _initial_weights(self):
        """Temporarily put the pre-adaptation values back into the adapted tensors."""
        current = [p.detach().clone() for p in self._params]
        self._set(self._init)
        try:
            yield
        finally:
            self._set(current)

    # ------------------------------------------------------------------ diagnostics
    def _task_progress(self):
        e_states = self._last_eval[3]
        s = np.asarray(e_states[0, -1], dtype=np.float64)
        g = np.asarray(self.evaluator.state_g[0], dtype=np.float64)
        if g.ndim != 1:  # not a PushT/PushObj state vector (rope: particles); the evaluator logs its own distance
            return {}
        return {
            "dist4": float(np.linalg.norm(g[:4] - s[:4])),  # the success test uses < 20
            "dist_obj": float(np.linalg.norm(g[2:4] - s[2:4])),
            "dist_agent": float(np.linalg.norm(g[:2] - s[:2])),
            "ang": float(np.abs((g[4] - s[4] + np.pi) % (2 * np.pi) - np.pi)),  # success: < pi/9
            "state": [round(float(x), 3) for x in s],
        }

    @torch.no_grad()
    def _predictions(self, obs, act):
        """Teacher-forced predictions for every sliding window, plus the newest-frame one."""
        tr = self.adajepa_trainer
        o, a = tr._prepare_segment(obs, act)
        z = self.wm.encode(o, a)
        T = z.shape[1] - 1
        w = min(self.wm.num_hist, T)
        windows = [(self.wm.predict(z[:, t : t + w]), z[:, t + 1 : t + 1 + w]) for t in range(T - w + 1)]
        return windows

    def _mse(self, pred, tgt, drop):
        return float(torch.mean((pred[..., :-drop] - tgt[..., :-drop]) ** 2))

    def _losses(self, obs, act):
        windows = self._predictions(obs, act)
        pred_last, tgt_last = windows[-1][0][:, -1:], windows[-1][1][:, -1:]
        return {
            "buf": float(np.mean([self._mse(p, t, self._drop_act) for p, t in windows])),
            "new": self._mse(pred_last, tgt_last, self._drop_act),  # newest frame: not trained on yet
            "new_vis": self._mse(pred_last, tgt_last, self._drop_obs),
        }

    def _losses_now_and_init(self, obs, act):
        now = self._losses(obs, act)
        with self._initial_weights():
            init = self._losses(obs, act)
        out = {}
        for k in now:
            out[f"loss_{k}_cur"] = now[k]
            out[f"loss_{k}_init"] = init[k]
        return out

    def _update_stats(self, obs, act, applied, d_own):
        out = {}
        split = self._groups["pred"]
        groups = {"pred": slice(0, split), "enc": slice(split, None)}
        with torch.no_grad():
            for name, sl in groups.items():
                cur = self._params[sl]
                ini = self._init[sl]
                out[f"disp_{name}"] = _norm([p - q for p, q in zip(cur, ini)])
                out[f"step_{name}"] = _norm(applied[sl]) if applied is not None else 0.0
                out[f"cos_own_{name}"] = _cos(applied[sl], d_own[sl]) if applied is not None else float("nan")
            out["cos_own"] = _cos(applied, d_own) if applied is not None else float("nan")
        if applied is None:
            out["func_change"] = 0.0
            return out
        now = self._predictions(obs, act)
        with self._initial_weights():
            init = self._predictions(obs, act)
        out["func_change"] = float(np.mean([self._mse(p1, p0, self._drop_act) for (p1, _), (p0, _) in zip(now, init)]))
        return out


def _norm(tensors):
    return float(torch.sqrt(sum((t.double() ** 2).sum() for t in tensors))) if tensors else 0.0


def _cos(a, b):
    if not a:
        return float("nan")
    dot = sum((x.double() * y.double()).sum() for x, y in zip(a, b))
    denom = _norm(a) * _norm(b)
    return float(dot / denom) if denom > 0 else float("nan")
