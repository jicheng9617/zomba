"""Policies that run many independent copies in lock-step.

Every piece of policy state carries a leading run axis of length ``n_runs``.
``n_runs=1`` is ordinary online use; ``n_runs=1000`` is a Monte-Carlo batch, and
both go through the same code.
"""
import numpy as np


def rand_argmax(values, rng):
    """Row-wise argmax of a ``(n_runs, n_arms)`` array, breaking ties uniformly at random.

    ``np.argmax`` alone would always favour the lowest index, which biases any
    experiment whose best arm happens to sit at index 0.
    """
    best = values.max(axis=1, keepdims=True)
    tol = np.where(np.isfinite(best), 1e-12 * (1.0 + np.abs(best)), 0.0)
    ties = values >= best - tol
    if not (ties.sum(axis=1) > 1).any():
        return values.argmax(axis=1)
    return np.where(ties, rng.random(values.shape), -1.0).argmax(axis=1)


class Policy:
    """Base class for bandit policies.

    Subclasses set hyperparameters in ``__init__`` (listed in ``param_names``) and
    implement ``_reset``, ``select`` and ``update``.

    Class attributes
    ----------------
    theory : tuple of zomba.theory.Bound
        Published results about this policy. ``bounds()`` evaluates the ones whose
        conditions the configured policy meets (usually: the analysed parameters).
    """

    name = None
    param_names = ()
    theory = ()

    @property
    def needs_horizon(self):
        """True if the policy uses the horizon (MOSS, ETC, UCB with delta = 1/n^2...)."""
        return False

    @property
    def label(self):
        return self.name or type(self).__name__

    def params(self):
        return {k: getattr(self, k) for k in self.param_names}

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{type(self).__name__}({args})"

    # ------------------------------------------------------------------ interface
    def reset(self, n_arms, n_runs=1, *, horizon=None, dim=None, seed=None):
        """Start ``n_runs`` fresh copies on a problem with ``n_arms`` arms.

        ``seed`` is anything ``np.random.default_rng`` accepts. ``dim`` is the context
        dimension (linear policies) and ``horizon`` the number of rounds, which only
        horizon-dependent policies need.
        """
        if self.needs_horizon and horizon is None:
            raise ValueError(f"{self!r} needs the horizon; pass horizon= to reset()")
        self.n_arms = int(n_arms)
        self.n_runs = int(n_runs)
        self.horizon = None if horizon is None else int(horizon)
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self._rows = np.arange(self.n_runs)
        self._reset()
        return self

    def _reset(self):
        raise NotImplementedError

    def select(self, context=None):
        """Arm to play in every run, an int array of shape ``(n_runs,)``."""
        raise NotImplementedError

    def update(self, arms, rewards, context=None):
        """Feed back the played ``arms`` and observed ``rewards`` (both ``(n_runs,)``)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ theory
    def bounds(self, env, horizon):
        """Evaluate the published bounds that apply to this configuration.

        Returns a list of ``(bound, value)`` pairs; ``value`` has one entry per run
        when the instance differs across runs, and is ``None`` for results stated
        only up to constants. ``env`` must have been reset.
        """
        problem = {**env.info(), **self.params(), "horizon": horizon}
        return [(b, None if b.fn is None else b.evaluate(**problem)) for b in self.theory if b.applies_to(self)]


class IndexPolicy(Policy):
    """K-armed policy that plays every arm once, in a random order per run, and then
    the arm with the largest ``index()``."""

    def _reset(self):
        B, K = self.n_runs, self.n_arms
        self.counts = np.zeros((B, K))
        self.sums = np.zeros((B, K))
        self._order = self.rng.permuted(np.tile(np.arange(K), (B, 1)), axis=1)

    @property
    def means(self):
        """Empirical means, 0 for arms not played yet."""
        return np.divide(self.sums, self.counts, out=np.zeros_like(self.sums), where=self.counts > 0)

    def index(self):
        """Index of every arm, ``(n_runs, n_arms)``; called once every arm has been played."""
        raise NotImplementedError

    def select(self, context=None):
        if self.t < self.n_arms:
            return self._order[:, self.t].copy()
        return rand_argmax(self.index(), self.rng)

    def update(self, arms, rewards, context=None):
        self.counts[self._rows, arms] += 1.0
        self.sums[self._rows, arms] += rewards
        self.t += 1
