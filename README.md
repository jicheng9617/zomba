# zomba

Bandit algorithms with their theory attached.

Every policy ships in the configuration its paper analyses, and knows which
published bounds apply to it: the formula with its constants, the theorem it comes
from and the assumptions it needs. Policies and environments run many independent
runs in lock-step, so comparing what an algorithm does with what the theory says
takes a few lines and seconds rather than an afternoon.

## Install

```bash
pip install -e .              # core: numpy only
pip install -e ".[dev]"       # + pytest
pip install -e ".[mo]"        # zomba.multiObjective (torch, scipy, scikit-learn, networkx, ...)
```

Python 3.10 or newer.

## Quick start

```python
import numpy as np
from zomba import envs, policies, simulate, theory

env = envs.BernoulliBandit([0.9, 0.8])
n = 10_000
for pol in [policies.UCB1(), policies.KLUCB(), policies.KLUCB(c=0), policies.ThompsonSampling()]:
    res = simulate(pol, env, horizon=n, n_runs=200, seed=0)
    print(res)
    for bound, value in pol.bounds(env, n):   # only the analysed configuration has bounds
        if value is not None:
            print(f"    {bound.kind} bound, {bound.regime} ({bound.cite()}): {np.mean(value):.1f}")
for bound, value in theory.lower_bounds(env, n):
    print(f"{bound.name}: {np.mean(value):.1f}")
```

```text
Result(UCB1 on BernoulliBandit: regret at n=10000 is 88.06 +- 2.3 (95% CI, 200 runs))
    upper bound, gap-dependent (Auer, Cesa-Bianchi & Fischer (2002), Theorem 1): 737.3
Result(KLUCB on BernoulliBandit: regret at n=10000 is 24.15 +- 1.4 (95% CI, 200 runs))
    upper bound, asymptotic (Garivier & Cappe (2011), Theorem 1): 20.7
Result(KLUCB on BernoulliBandit: regret at n=10000 is 15.18 +- 1.2 (95% CI, 200 runs))
Result(ThompsonSampling on BernoulliBandit: regret at n=10000 is 10.18 +- 1.4 (95% CI, 200 runs))
Lai-Robbins asymptotic lower bound (Bernoulli): 20.7
```

`KLUCB(c=0)` is the setting Garivier & Cappe recommend in practice; their theorems
need `c >= 3`, so no bound is attached to it. "Asymptotic" values are C log n, not
finite-time bounds.

The same objects work online, one decision at a time:

```python
pol = policies.ThompsonSampling().reset(n_arms=3, seed=0)
arm = pol.select()[0]
pol.update(arm, 1.0)
```

## Design

* **Batched runs.** `Policy` and `Env` state carries a leading run axis; `n_runs=1`
  is ordinary online use. `simulate` returns the cumulative pseudo-regret of every run
  at log-spaced checkpoints.
* **Reproducible.** The environment and the policy draw from independent streams
  spawned from one seed (`Result.seed` reproduces a run made with `seed=None`).
  Environments draw noise per round rather than per arm, so policies compared under
  the same seed see the same noise.
* **Theory as data.** `zomba.theory` holds `Bound` records (statement, constants,
  reference, theorem number, assumptions, whether the formula was checked against the
  source). `Policy.bounds(env, n)` evaluates the ones whose conditions the configured
  policy meets; `theory.lower_bounds(env, n)` gives the problem-level ones.
* **Faithful defaults.** Defaults are the analysed versions; practical variants are a
  parameter away (`KLUCB(c=0)`, `OFUL(radius_scale=0.3)`, `LinTS(inflation="bayes")`).

## What is included

| Policy | Default | Attached results |
|---|---|---|
| `UCB1(c=2)` | Auer, Cesa-Bianchi & Fischer (2002) | gap-dependent bound (Thm 1) |
| `UCB(delta=None, sigma=1)` | UCB(delta) with delta = 1/n^2 (L&S Alg. 3) | L&S Thm 7.1, Thm 7.2 |
| `AsymptoticUCB(sigma=1)` | f(t) = 1 + t log^2 t (L&S Alg. 6) | L&S Thm 8.1, Eq. (8.2) |
| `MOSS(sigma=1)` | Audibert & Bubeck (2009); L&S Alg. 7 | L&S Thm 9.1 |
| `KLUCB(c=3)` | Garivier & Cappe (2011); Bernoulli or Gaussian KL | Thm 1 (asymptotic), Thm 2 |
| `ThompsonSampling()` | Beta(1, 1) prior (Agrawal & Goyal 2017) | Thm 1.1, 1.2 |
| `GaussianThompsonSampling(sigma=1)` | Agrawal & Goyal (2017) | Thm 1.3 |
| `ExploreThenCommit(m="minimax")` | m = ceil((n/K)^(2/3)) | L&S Thm 6.1 |
| `EpsilonGreedy`, `Greedy`, `Uniform` | baselines | |
| `OFUL(radius_scale=1)` | Abbasi-Yadkori, Pal & Szepesvari (2011) | Thm 3 |
| `LinTS(inflation="icml2013")` | Agrawal & Goyal (2013) | Thm 1 |
| `LinUCB(alpha)`, `LinGreedy` | baselines | |

L&S is Lattimore & Szepesvari, *Bandit Algorithms* (2020). Environments:
`BernoulliBandit`, `GaussianBandit`, `StudentTBandit` (heavy tails), each optionally
piecewise stationary, and `LinearBandit` (fixed arms or fresh contexts). Lower
bounds: Lai-Robbins (Bernoulli, Gaussian) and the minimax sqrt((K-1)n)/27.

## Theory vs. practice

[`experiments/theory_gap`](experiments/theory_gap/README.md) is a first measurement
of how far published bounds are from what the algorithms do. In short: growth rates
match the theory, constants do not (the MOSS bound is 28x above the empirical worst
case at K = 10), and in linear bandits OFUL's confidence ellipsoid is nearly tight,
yet using its full radius costs 3.5-8x more regret than a radius shrunk to 0.2-0.5.

## Legacy API and multi-objective bandits

The pre-0.2 classes (`MABAgent`, `contextMABAgent`, ..., and the
`singleObjective` algorithms and simulators) are in `zomba.legacy`, unchanged apart
from import paths. `zomba.multiObjective` still builds on them and imports as
before; it will be ported to the new core.

## License

MIT
