"""Run a policy against an environment for many runs at once."""
from dataclasses import dataclass, field

import numpy as np


def default_checkpoints(horizon, num=100):
    """About ``num`` rounds, log-spaced from 1 to ``horizon`` (always included)."""
    return np.unique(np.round(np.geomspace(1, horizon, num)).astype(np.int64))


@dataclass
class Result:
    """Cumulative pseudo-regret of every run at every checkpoint.

    The mean over runs estimates the expected regret R_n that regret bounds are about.
    ``seed`` is the entropy of the root ``SeedSequence``: passing it back to
    ``simulate`` reproduces the result.
    """

    policy: str
    env: str
    horizon: int
    n_runs: int
    seed: int
    checkpoints: np.ndarray
    regret: np.ndarray
    pulls: np.ndarray = field(default=None, repr=False)

    @property
    def final(self):
        """Regret of every run at the horizon."""
        return self.regret[:, -1]

    def mean(self):
        return self.regret.mean(axis=0)

    def stderr(self):
        if self.n_runs < 2:
            return np.full(self.regret.shape[1], np.nan)
        return self.regret.std(axis=0, ddof=1) / np.sqrt(self.n_runs)

    def quantile(self, q):
        return np.quantile(self.regret, q, axis=0)

    def summary(self):
        m, se = self.mean()[-1], self.stderr()[-1]
        return {"policy": self.policy, "env": self.env, "horizon": self.horizon, "n_runs": self.n_runs,
                "regret": float(m), "ci95": float(1.96 * se)}

    def __repr__(self):
        s = self.summary()
        return (f"Result({s['policy']} on {s['env']}: regret at n={s['horizon']} is "
                f"{s['regret']:.4g} +- {s['ci95']:.2g} (95% CI, {s['n_runs']} runs))")


def simulate(policy, env, horizon, n_runs=1, *, seed=None, checkpoints=None):
    """Run ``n_runs`` independent copies of ``policy`` on ``env`` for ``horizon`` rounds.

    The environment and the policy get independent random streams spawned from
    ``seed``. Environments draw their noise per round rather than per arm, so two
    policies simulated with the same seed see the same noise sequence (common random
    numbers), which makes paired comparisons less noisy.
    """
    horizon = int(horizon)
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    env_seed, policy_seed = root.spawn(2)
    env.reset(n_runs, seed=env_seed)
    policy.reset(env.n_arms, n_runs, horizon=horizon, dim=env.dim, seed=policy_seed)

    cps = default_checkpoints(horizon) if checkpoints is None else np.unique(np.asarray(checkpoints, np.int64))
    if cps[0] < 1 or cps[-1] > horizon:
        raise ValueError("checkpoints must lie in [1, horizon]")
    out = np.empty((n_runs, len(cps)))
    cum = np.zeros(n_runs)
    pulls = np.zeros((n_runs, env.n_arms), dtype=np.int64)
    rows = np.arange(n_runs)
    c = 0
    for t in range(1, horizon + 1):
        context = env.observe()
        arms = policy.select(context)
        cum += env.regret(arms)
        rewards = env.pull(arms)
        policy.update(arms, rewards, context)
        pulls[rows, arms] += 1
        if c < len(cps) and t == cps[c]:
            out[:, c] = cum
            c += 1
    return Result(policy=policy.label, env=env.label, horizon=horizon, n_runs=n_runs, seed=root.entropy,
                  checkpoints=cps, regret=out, pulls=pulls)
