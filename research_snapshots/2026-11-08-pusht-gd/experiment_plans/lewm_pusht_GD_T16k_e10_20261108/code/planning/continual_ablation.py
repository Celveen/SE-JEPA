"""Ablations that keep the remaining cls6a/v7 mechanisms unchanged."""

from .continual_v7 import CLSv7MPCPlanner


class CLSv7NoReadMaturityMPCPlanner(CLSv7MPCPlanner):
    """Allow every existing slow learner to compete for reads, regardless of maturity.

    Keep mature_k/mature_ratio and _mature() unchanged: the write/absorption logic
    and diagnostic records still use the original definition of maturity.
    Setting mature_k=0 would also change those mechanisms and is a broader ablation.
    Familiar episodes still use theta0, and a learner must still beat theta0's
    prediction error to be selected. Learner creation (min_n) is unchanged.
    """

    def _learners(self):
        return [key for key, domain in self.domains.items() if domain["learner"] is not None]
