"""Stochastic linear bandits with a finite arm set."""
import numpy as np

from ..core.env import Env


def _on_sphere(rng, shape, radius=1.0):
    x = rng.standard_normal(shape)
    return radius * x / np.linalg.norm(x, axis=-1, keepdims=True)


class LinearBandit(Env):
    """Reward of arm x is <x, theta> + noise * N(0, 1).

    Parameters
    ----------
    n_arms, dim : int
    theta : array (dim,) or (n_runs, dim), optional
        Unknown parameter; drawn uniformly on the sphere of radius ``theta_norm`` for
        every run when omitted.
    arms : array (n_arms, dim) or (n_runs, n_arms, dim), optional
        Fixed arm features; drawn uniformly on the unit sphere for every run when
        omitted. Ignored when ``fresh_contexts`` is set.
    noise : float
        Standard deviation of the Gaussian noise (R-subgaussian with R = noise).
    fresh_contexts : bool
        Draw new arms on the unit sphere every round (contextual setting).
    """

    def __init__(self, n_arms, dim, theta=None, arms=None, noise=1.0, fresh_contexts=False, theta_norm=1.0):
        self.n_arms, self.dim = int(n_arms), int(dim)
        self.theta = None if theta is None else np.asarray(theta, dtype=float)
        self.arms = None if arms is None else np.asarray(arms, dtype=float)
        self.noise = float(noise)
        self.fresh_contexts = bool(fresh_contexts)
        self.theta_norm = float(theta_norm)

    def _reset(self):
        B, K, d = self.n_runs, self.n_arms, self.dim
        if self.theta is None:
            self._theta = _on_sphere(self.rng, (B, d), self.theta_norm)
        else:
            self._theta = np.broadcast_to(self.theta, (B, d))
        if self.fresh_contexts:
            self._X = None
        elif self.arms is None:
            self._X = _on_sphere(self.rng, (B, K, d))
        else:
            self._X = np.broadcast_to(self.arms, (B, K, d))
        self._means = None if self._X is None else np.einsum("bkd,bd->bk", self._X, self._theta)

    def observe(self):
        if self.fresh_contexts:
            self._X = _on_sphere(self.rng, (self.n_runs, self.n_arms, self.dim))
            self._means = np.einsum("bkd,bd->bk", self._X, self._theta)
        return self._X

    def mean_rewards(self):
        return self._means

    def _draw(self, means):
        return means + self.noise * self.rng.standard_normal(self.n_runs)

    def info(self):
        S = float(np.linalg.norm(self._theta, axis=-1).max())
        L = 1.0 if self.fresh_contexts or self.arms is None else float(np.linalg.norm(self._X, axis=-1).max())
        return {"n_arms": self.n_arms, "dim": self.dim, "noise_sigma": self.noise, "theta_norm_bound": S,
                "arm_norm_bound": L}
