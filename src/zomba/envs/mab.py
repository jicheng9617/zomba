"""Stochastic K-armed bandits."""
import numpy as np

from ..core.env import Env


class KArmedBandit(Env):
    """K-armed bandit with fixed (or piecewise-constant) mean rewards.

    Parameters
    ----------
    means : array of shape (K,) or (n_runs, K)
        Mean reward of every arm, shared by all runs or one instance per run
        (see ``repeat_instances``).
    changes : sequence of (t, means), optional
        Switch to new means from round ``t`` on (rounds are counted from 0), which
        makes the problem piecewise stationary.
    """

    def __init__(self, means, changes=()):
        self.means = np.asarray(means, dtype=float)
        if self.means.ndim not in (1, 2):
            raise ValueError("means must have shape (K,) or (n_runs, K)")
        self.n_arms = self.means.shape[-1]
        self.changes = sorted(((int(t), np.asarray(m, dtype=float)) for t, m in changes), key=lambda c: c[0])
        for _, m in [(0, self.means), *self.changes]:
            self._check_means(m)

    def _check_means(self, m):
        if m.shape[-1] != self.n_arms:
            raise ValueError("every set of means needs the same number of arms")

    def _broadcast(self, m):
        if m.ndim == 2 and m.shape[0] != self.n_runs:
            raise ValueError(f"means are given for {m.shape[0]} runs but n_runs={self.n_runs}")
        return np.broadcast_to(m, (self.n_runs, self.n_arms))

    def _reset(self):
        self._segments = [(0, self._broadcast(self.means))] + [(t, self._broadcast(m)) for t, m in self.changes]
        self._seg = 0

    def mean_rewards(self):
        while self._seg + 1 < len(self._segments) and self.t >= self._segments[self._seg + 1][0]:
            self._seg += 1
        return self._segments[self._seg][1]

    def info(self):
        return {"n_arms": self.n_arms, "means": self._broadcast(self.means)}


class BernoulliBandit(KArmedBandit):
    """Arms pay 1 with probability equal to their mean, else 0."""

    def _check_means(self, m):
        super()._check_means(m)
        if np.any((m < 0) | (m > 1)):
            raise ValueError("Bernoulli means must lie in [0, 1]")

    def _draw(self, means):
        # one uniform per run and round, whatever arm is played: common random numbers
        return (self.rng.random(self.n_runs) < means).astype(float)


class GaussianBandit(KArmedBandit):
    """Rewards are mean + sigma * N(0, 1)."""

    def __init__(self, means, sigma=1.0, changes=()):
        super().__init__(means, changes)
        self.sigma = float(sigma)

    def _draw(self, means):
        return means + self.sigma * self.rng.standard_normal(self.n_runs)

    def info(self):
        return {**super().info(), "noise_sigma": self.sigma}


class StudentTBandit(KArmedBandit):
    """Heavy-tailed rewards: mean + sigma * t_df rescaled to unit variance (needs df > 2).

    The noise has finite variance sigma^2 but is not subgaussian, so the assumptions
    of most published bounds fail on purpose.
    """

    def __init__(self, means, df=3.0, sigma=1.0, changes=()):
        if df <= 2:
            raise ValueError("df must exceed 2 for the variance to be finite")
        super().__init__(means, changes)
        self.df, self.sigma = float(df), float(sigma)

    def _draw(self, means):
        scale = self.sigma * np.sqrt((self.df - 2.0) / self.df)
        return means + scale * self.rng.standard_t(self.df, self.n_runs)


def repeat_instances(means, runs_per_instance):
    """Stack instances ``(M, K)`` into per-run means ``(M * runs_per_instance, K)``."""
    return np.repeat(np.asarray(means, dtype=float), runs_per_instance, axis=0)
