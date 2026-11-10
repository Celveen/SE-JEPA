"""Memory-bounded action optimization; model weights remain frozen."""
import numpy as np
import torch
from planning.gd import GDPlanner


class GDMicrobatchPlanner(GDPlanner):
    def __init__(self, microbatch_size=1, **kwargs):
        super().__init__(**kwargs)
        self.microbatch_size = int(microbatch_size)
        if self.microbatch_size < 1:
            raise ValueError("microbatch_size must be positive")

    def plan(self, obs_0, obs_g, actions=None, step=None):
        predictions, lengths = [], []
        prefix = self.logging_prefix
        count = obs_0["visual"].shape[0]
        try:
            for start in range(0, count, self.microbatch_size):
                stop = min(start + self.microbatch_size, count)
                self.logging_prefix = f"{prefix}_samples_{start}_{stop}"
                current = {key: value[start:stop] for key, value in obs_0.items()}
                goal = {key: value[start:stop] for key, value in obs_g.items()}
                warm = None if actions is None else actions[start:stop]
                predicted, valid = super().plan(current, goal, actions=warm, step=step)
                predictions.append(predicted.detach())
                lengths.append(valid)
        finally:
            self.logging_prefix = prefix
        return torch.cat(predictions, dim=0), np.concatenate(lengths)
