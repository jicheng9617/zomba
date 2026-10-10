"""Environments that serve many independent runs in lock-step."""
import numpy as np


class Env:
    """Base class for bandit environments.

    A round is ``observe()`` (contexts, or ``None``), then ``regret(arms)`` and
    ``pull(arms)`` for the arms the policy chose; ``pull`` ends the round.

    Subclasses implement ``_reset``, ``mean_rewards`` and ``_draw``, and may override
    ``observe`` and ``info``.
    """

    name = None
    n_arms = None
    dim = None

    @property
    def label(self):
        return self.name or type(self).__name__

    def reset(self, n_runs=1, seed=None):
        self.n_runs = int(n_runs)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self._rows = np.arange(self.n_runs)
        self._reset()
        return self

    def _reset(self):
        raise NotImplementedError

    def observe(self):
        """Contexts for this round, ``(n_runs, n_arms, dim)``, or ``None`` for K-armed bandits."""
        return None

    def mean_rewards(self):
        """Expected reward of every arm in the current round, ``(n_runs, n_arms)``."""
        raise NotImplementedError

    def regret(self, arms):
        """Pseudo-regret of playing ``arms`` in the current round, ``(n_runs,)``."""
        m = self.mean_rewards()
        return m.max(axis=1) - m[self._rows, arms]

    def pull(self, arms):
        """Rewards for ``arms`` in the current round, ``(n_runs,)``; ends the round."""
        r = self._draw(self.mean_rewards()[self._rows, arms])
        self.t += 1
        return r

    def _draw(self, means):
        raise NotImplementedError

    def info(self):
        """Problem quantities that published bounds are stated in (see zomba.theory)."""
        return {"n_arms": self.n_arms}
