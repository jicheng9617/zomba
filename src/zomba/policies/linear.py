"""Linear bandit policies on finite arm sets (contexts of shape (n_runs, n_arms, dim))."""
import numpy as np

from ..core.policy import Policy, rand_argmax
from ..theory import bounds as B


class _Ridge(Policy):
    """Regularised least squares, batched over runs.

    Keeps V = lam I + sum x x^T, its inverse (Sherman-Morrison, re-inverted every
    1000 rounds), b = sum r x, theta_hat = V^-1 b and log det V - d log lam.
    """

    param_names = ("lam",)

    def __init__(self, lam=1.0):
        self.lam = float(lam)

    def _reset(self):
        if self.dim is None:
            raise ValueError(f"{self!r} needs the context dimension; pass dim= to reset()")
        B_, d = self.n_runs, int(self.dim)
        eye = np.broadcast_to(np.eye(d), (B_, d, d))
        self.V = self.lam * eye.copy()
        self.Vinv = eye.copy() / self.lam
        self.b = np.zeros((B_, d))
        self.theta = np.zeros((B_, d))
        self.logdet = np.zeros(B_)

    def width(self, context):
        """||x||_{V^-1} for every arm, (n_runs, n_arms)."""
        XV = np.matmul(context, self.Vinv)
        return np.sqrt(np.maximum((XV * context).sum(axis=-1), 0.0))

    def update(self, arms, rewards, context=None):
        rewards = np.broadcast_to(np.asarray(rewards, dtype=float), (self.n_runs,))
        x = context[self._rows, arms]
        Vx = np.einsum("bij,bj->bi", self.Vinv, x)
        q = np.einsum("bi,bi->b", x, Vx)
        self.Vinv -= Vx[:, :, None] * Vx[:, None, :] / (1.0 + q)[:, None, None]
        self.logdet += np.log1p(q)
        self.V += x[:, :, None] * x[:, None, :]
        self.b += rewards[:, None] * x
        self.t += 1
        if self.t % 1000 == 0:
            self.Vinv = np.linalg.inv(self.V)
        self.theta = np.einsum("bij,bj->bi", self.Vinv, self.b)


class OFUL(_Ridge):
    """OFUL (Abbasi-Yadkori, Pal & Szepesvari 2011) on a finite arm set.

    Index x.theta_hat + radius_scale * sqrt(beta_t) * ||x||_{V^-1} with the
    Theorem 2 radius sqrt(beta_t) = R sqrt(log det V_t - d log lam + 2 log(1/delta))
    + sqrt(lam) S. ``radius_scale = 1`` is the analysed algorithm; ``0`` is greedy.
    ``noise`` (R) and ``S`` are the policy's assumptions about the problem.
    """

    param_names = ("noise", "S", "lam", "delta", "radius_scale")
    theory = (B.OFUL_REGRET,)

    def __init__(self, noise=1.0, S=1.0, lam=1.0, delta=0.01, radius_scale=1.0):
        super().__init__(lam)
        self.noise, self.S, self.delta, self.radius_scale = float(noise), float(S), float(delta), float(radius_scale)

    def radius(self):
        """sqrt(beta_t) of Theorem 2 for every run."""
        return B.oful_radius(self.logdet, self.noise, self.lam, self.S, self.delta)

    def index(self, context):
        mu = np.einsum("bkd,bd->bk", context, self.theta)
        if self.radius_scale == 0.0:
            return mu
        return mu + self.radius_scale * self.radius()[:, None] * self.width(context)

    def select(self, context=None):
        return rand_argmax(self.index(context), self.rng)


class LinUCB(OFUL):
    """Constant-width optimism: x.theta_hat + alpha ||x||_{V^-1} (the usual practical LinUCB)."""

    param_names = ("alpha", "lam")
    theory = ()

    def __init__(self, alpha=1.0, lam=1.0):
        super().__init__(lam=lam)
        self.alpha = float(alpha)

    def index(self, context):
        return np.einsum("bkd,bd->bk", context, self.theta) + self.alpha * self.width(context)


class LinGreedy(OFUL):
    """Play the arm with the largest x.theta_hat (OFUL with radius_scale = 0)."""

    param_names = ("lam",)
    theory = ()

    def __init__(self, lam=1.0):
        super().__init__(lam=lam, radius_scale=0.0)


class LinTS(_Ridge):
    """Linear Thompson sampling: play argmax x.theta_tilde with theta_tilde ~ N(theta_hat, v^2 V^-1).

    ``inflation`` sets v:
      "icml2013"  (default) v = R sqrt(24 / eps * d * ln(1/delta)), eps = 1 / ln n (Agrawal & Goyal 2013;
                  the setting their Theorem 1 covers)
      "arxiv-v4"  v = R sqrt(9 d ln(n / delta)) (the 2014 arXiv version)
      "bayes"     v = R (the posterior under a N(0, R^2/lam I) prior; no inflation, the practical choice)
      a number    used as v.
    """

    param_names = ("noise", "lam", "delta", "inflation")
    theory = (B.LINTS_REGRET,)

    def __init__(self, noise=1.0, lam=1.0, delta=0.01, inflation="icml2013"):
        super().__init__(lam)
        if not (inflation in ("icml2013", "arxiv-v4", "bayes") or isinstance(inflation, (int, float))):
            raise ValueError("inflation must be 'icml2013', 'arxiv-v4', 'bayes' or a number")
        self.noise, self.delta, self.inflation = float(noise), float(delta), inflation

    @property
    def needs_horizon(self):
        return self.inflation in ("icml2013", "arxiv-v4")

    def _reset(self):
        super()._reset()
        R, d, n, delta = self.noise, int(self.dim), self.horizon, self.delta
        if self.inflation == "icml2013":
            self.v = R * np.sqrt(24.0 * np.log(n) * d * np.log(1.0 / delta))
        elif self.inflation == "arxiv-v4":
            self.v = R * np.sqrt(9.0 * d * np.log(n / delta))
        elif self.inflation == "bayes":
            self.v = R
        else:
            self.v = float(self.inflation)

    def select(self, context=None):
        L = np.linalg.cholesky(0.5 * (self.Vinv + np.swapaxes(self.Vinv, 1, 2)))
        z = self.rng.standard_normal((self.n_runs, int(self.dim)))
        theta = self.theta + self.v * np.einsum("bij,bj->bi", L, z)
        return rand_argmax(np.einsum("bkd,bd->bk", context, theta), self.rng)
