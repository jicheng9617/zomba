"""K-armed bandit policies.

Defaults are the analysed versions (the parameters the cited theorems assume), so
``Policy.bounds`` applies to them; practical variants are a parameter away, e.g.
``KLUCB(c=0)`` or ``UCB1(c=0.5)``.
"""
import numpy as np

from ..core.policy import IndexPolicy, Policy, rand_argmax
from ..theory import bounds as B


class UCB1(IndexPolicy):
    """UCB1 (Auer, Cesa-Bianchi & Fischer 2002): mean_i + sqrt(c ln t / n_i), t = plays so far.

    ``c = 2`` is the analysed value, for rewards in [0, 1].
    """

    param_names = ("c",)
    theory = (B.UCB1_GAP,)

    def __init__(self, c=2.0):
        self.c = float(c)

    def index(self):
        return self.means + np.sqrt(self.c * np.log(self.t) / self.counts)


class UCB(IndexPolicy):
    """UCB(delta) (L&S Alg. 3): mean_i + sigma sqrt(2 log(1/delta) / n_i).

    ``delta=None`` is the analysed choice delta = 1/n^2, which needs the horizon.
    """

    param_names = ("delta", "sigma")
    theory = (B.UCB_DELTA_GAP, B.UCB_DELTA_MINIMAX)

    def __init__(self, delta=None, sigma=1.0):
        self.delta = delta
        self.sigma = float(sigma)

    @property
    def needs_horizon(self):
        return self.delta is None

    def index(self):
        delta = 1.0 / self.horizon ** 2 if self.delta is None else self.delta
        return self.means + self.sigma * np.sqrt(2.0 * np.log(1.0 / delta) / self.counts)


class AsymptoticUCB(IndexPolicy):
    """Asymptotically optimal UCB (L&S Alg. 6): mean_i + sigma sqrt(2 log f(t) / n_i),
    f(t) = 1 + t log^2 t, t = current round. Anytime."""

    param_names = ("sigma",)
    theory = (B.UCB_AO_GAP, B.UCB_AO_ASYMPTOTIC)

    def __init__(self, sigma=1.0):
        self.sigma = float(sigma)

    def index(self):
        s = self.t + 1.0
        return self.means + self.sigma * np.sqrt(2.0 * np.log(1.0 + s * np.log(s) ** 2) / self.counts)


class MOSS(IndexPolicy):
    """MOSS (Audibert & Bubeck 2009; L&S Alg. 7): mean_i + sigma sqrt(4 / n_i log+(n / (K n_i)))."""

    param_names = ("sigma",)
    theory = (B.MOSS_MINIMAX,)

    def __init__(self, sigma=1.0):
        self.sigma = float(sigma)

    @property
    def needs_horizon(self):
        return True

    def index(self):
        logplus = np.log(np.maximum(1.0, self.horizon / (self.n_arms * self.counts)))
        return self.means + self.sigma * np.sqrt(4.0 / self.counts * logplus)


def klucb_bernoulli(p, n, level, iters=8):
    """max{q in [p, 1] : n kl(p, q) <= level}, vectorised.

    Newton's method in w = -log(1 - q), started from Pinsker's upper bound
    p + sqrt(level / (2 n)). As a function of w, kl(p, q) is convex and increasing
    for q > p, so the iterates decrease monotonically to the root; unlike Newton in
    q it does not crawl when the root is close to 1. Eight steps reach ~1e-12.
    """
    p = np.clip(p, 0.0, 1.0)
    target = np.maximum(level, 0.0) / n
    pc = np.clip(p, 1e-15, 1.0 - 1e-15)
    w_min = -np.log1p(-pc)
    w = np.maximum(-np.log1p(-np.minimum(p + np.sqrt(target / 2.0), 1.0 - 1e-15)), w_min)
    for _ in range(iters):
        f = B.kl_bernoulli(p, -np.expm1(-w)) - target
        slope = (1.0 - pc) - pc / np.expm1(w)
        step = np.divide(f, slope, out=np.zeros_like(w), where=(slope > 0) & (f > 0))
        w = np.maximum(w - step, w_min)
    return -np.expm1(-w)


class KLUCB(IndexPolicy):
    """KL-UCB (Garivier & Cappe 2011): max{q : n_i kl(mean_i, q) <= log t + c log log t}.

    ``c = 3`` is the analysed value; the authors recommend ``c = 0`` in practice.
    ``family="bernoulli"`` is for rewards in [0, 1]; ``"gaussian"`` uses the Gaussian
    KL with variance ``sigma**2`` (closed form). log log t is taken as 0 for t < e.
    """

    param_names = ("c", "family", "sigma")
    theory = (B.KLUCB_ASYMPTOTIC, B.KLUCB_FINITE)

    def __init__(self, c=3.0, family="bernoulli", sigma=1.0):
        if family not in ("bernoulli", "gaussian"):
            raise ValueError("family must be 'bernoulli' or 'gaussian'")
        self.c, self.family, self.sigma = float(c), family, float(sigma)

    def index(self):
        lt = np.log(self.t)
        level = lt + self.c * np.log(max(lt, 1.0))
        if self.family == "gaussian":
            return self.means + self.sigma * np.sqrt(2.0 * level / self.counts)
        return klucb_bernoulli(np.clip(self.means, 0.0, 1.0), self.counts, level)


class ThompsonSampling(Policy):
    """Beta-Bernoulli Thompson sampling with a uniform prior (Agrawal & Goyal 2017).

    Rewards in [0, 1] that are not 0/1 are turned into a Bernoulli trial with that
    success probability, as in the paper.
    """

    theory = (B.TS_BETA_GAP, B.TS_BETA_MINIMAX)

    def _reset(self):
        self.successes = np.zeros((self.n_runs, self.n_arms))
        self.failures = np.zeros((self.n_runs, self.n_arms))

    def select(self, context=None):
        return rand_argmax(self.rng.beta(1.0 + self.successes, 1.0 + self.failures), self.rng)

    def update(self, arms, rewards, context=None):
        hit = self.rng.random(self.n_runs) < rewards
        self.successes[self._rows, arms] += hit
        self.failures[self._rows, arms] += ~hit
        self.t += 1


class GaussianThompsonSampling(Policy):
    """Thompson sampling with Gaussian priors (Agrawal & Goyal 2017):
    sample N(S_i / (n_i + 1), sigma^2 / (n_i + 1)) for every arm."""

    param_names = ("sigma",)
    theory = (B.TS_GAUSS_MINIMAX,)

    def __init__(self, sigma=1.0):
        self.sigma = float(sigma)

    def _reset(self):
        self.counts = np.zeros((self.n_runs, self.n_arms))
        self.sums = np.zeros((self.n_runs, self.n_arms))

    def select(self, context=None):
        n1 = self.counts + 1.0
        draws = self.sums / n1 + self.sigma / np.sqrt(n1) * self.rng.standard_normal(self.counts.shape)
        return rand_argmax(draws, self.rng)

    def update(self, arms, rewards, context=None):
        self.counts[self._rows, arms] += 1.0
        self.sums[self._rows, arms] += rewards
        self.t += 1


class ExploreThenCommit(IndexPolicy):
    """Play every arm ``m`` times round-robin, then commit to the best empirical mean.

    ``m="minimax"`` uses ceil((n / K)^(2/3)), the gap-free tuning (regret O(n^(2/3))).
    """

    param_names = ("m",)
    theory = (B.ETC_GAP,)

    def __init__(self, m="minimax"):
        if not (m == "minimax" or (isinstance(m, (int, np.integer)) and m >= 1)):
            raise ValueError("m must be a positive int or 'minimax'")
        self.m = m

    @property
    def needs_horizon(self):
        return self.m == "minimax"

    def _reset(self):
        super()._reset()
        self._m = B.minimax_etc_m(self.horizon, self.n_arms) if self.m == "minimax" else int(self.m)
        self._commit = None

    def select(self, context=None):
        if self.t < self._m * self.n_arms:
            return self._order[:, self.t % self.n_arms].copy()
        if self._commit is None:
            self._commit = rand_argmax(self.means, self.rng)
        return self._commit.copy()


class EpsilonGreedy(IndexPolicy):
    """With probability eps play a uniformly random arm, otherwise the best empirical mean.

    Fixed ``epsilon`` (linear regret in the worst case), or a decaying schedule
    eps_t = min(1, decay * K / t) when ``decay`` is given.
    """

    param_names = ("epsilon", "decay")

    def __init__(self, epsilon=0.1, decay=None):
        self.epsilon, self.decay = float(epsilon), decay

    def select(self, context=None):
        if self.t < self.n_arms:
            return super().select(context)
        eps = self.epsilon if self.decay is None else min(1.0, self.decay * self.n_arms / self.t)
        greedy = rand_argmax(self.means, self.rng)
        explore = self.rng.random(self.n_runs) < eps
        return np.where(explore, self.rng.integers(self.n_arms, size=self.n_runs), greedy)


class Greedy(IndexPolicy):
    """Play every arm once, then always the best empirical mean."""

    def index(self):
        return self.means


class Uniform(Policy):
    """Uniformly random arm every round (a baseline with linear regret)."""

    def _reset(self):
        pass

    def select(self, context=None):
        return self.rng.integers(self.n_arms, size=self.n_runs)

    def update(self, arms, rewards, context=None):
        self.t += 1
