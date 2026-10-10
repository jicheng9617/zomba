"""Published regret bounds, with their constants, sources and assumptions.

Every ``Bound`` records where it comes from, what it assumes, whether the formula
here has been compared with the source (``checked``), and, when the source gives
explicit constants, a function that evaluates it. Functions take keyword
arguments named after problem and policy quantities (``means``, ``horizon``,
``sigma``, ...) and broadcast over leading run axes of ``means``.

L&S below is Lattimore & Szepesvari, "Bandit Algorithms" (2020).
"""
import inspect
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .references import REFERENCES, Reference

UPPER, LOWER = "upper", "lower"
GAP, MINIMAX, ASYMPTOTIC, HIGH_PROB, ORDER = "gap-dependent", "minimax", "asymptotic", "high-probability", "order"


@dataclass(frozen=True)
class Bound:
    """One published result about the regret.

    ``regime`` is one of: gap-dependent (expected regret, explicit in the instance),
    minimax (worst case over a class), asymptotic (a constant C with R_n ~ C log n;
    ``evaluate`` returns C log n, which is not a finite-time bound), high-probability,
    or order (no explicit constants; ``evaluate`` is unavailable).
    """

    name: str
    kind: str
    regime: str
    statement: str
    reference: Reference
    location: str
    assumptions: tuple = ()
    fn: Optional[Callable] = field(default=None, repr=False)
    condition: Optional[Callable] = field(default=None, repr=False)
    checked: bool = True

    def evaluate(self, **problem):
        if self.fn is None:
            raise ValueError(f"{self.name}: the source gives no explicit constants")
        names = inspect.signature(self.fn).parameters
        return self.fn(**{k: v for k, v in problem.items() if k in names})

    def applies_to(self, policy):
        return self.condition is None or bool(self.condition(policy))

    def cite(self):
        return f"{self.reference.authors} ({self.reference.year}), {self.location}"


# ------------------------------------------------------------------ helpers
def gaps(means):
    m = np.asarray(means, dtype=float)
    return m.max(axis=-1, keepdims=True) - m


def _sum_inv(numer, D):
    """sum over arms with Delta > 0 of numer / Delta."""
    pos = D > 0
    return np.where(pos, numer / np.where(pos, D, 1.0), 0.0).sum(axis=-1)


def kl_bernoulli(p, q):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    q = np.clip(q, 1e-15, 1 - 1e-15)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def minimax_etc_m(horizon, n_arms):
    """Exploration length ceil((n / K)^(2/3)): gap-free tuning of ETC (regret O(n^(2/3)))."""
    return int(np.ceil((horizon / n_arms) ** (2.0 / 3.0)))


# ------------------------------------------------------------------ K-armed: upper bounds
def _ucb1_gap(means, horizon):
    D = gaps(means)
    return 8.0 * _sum_inv(np.log(horizon), D) + (1.0 + np.pi ** 2 / 3.0) * D.sum(axis=-1)


def _ucb_delta_gap(means, horizon, sigma=1.0):
    D = gaps(means)
    return 3.0 * D.sum(axis=-1) + _sum_inv(16.0 * sigma ** 2 * np.log(horizon), D)


def _ucb_delta_minimax(means, horizon, sigma=1.0):
    m = np.asarray(means, dtype=float)
    K = m.shape[-1]
    return 8.0 * sigma * np.sqrt(horizon * K * np.log(horizon)) + 3.0 * gaps(m).sum(axis=-1)


def _ucb_ao_gap(means, horizon, sigma=1.0):
    # the infimum over eps in Thm 8.1 has a closed form: the objective is convex in eps
    D = gaps(means)
    lf = np.log(1.0 + horizon * np.log(horizon) ** 2)
    c = 2.0 * sigma ** 2 * (lf + np.sqrt(np.pi * lf) + 1.0)
    eps = D / (1.0 + (c / (5.0 * sigma ** 2)) ** (1.0 / 3.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = D * (1.0 + 5.0 * sigma ** 2 / eps ** 2 + c / (D - eps) ** 2)
    return np.where(D > 0, v, 0.0).sum(axis=-1)


def _gauss_asymptotic(means, horizon, sigma=1.0):
    return _sum_inv(2.0 * sigma ** 2, gaps(means)) * np.log(horizon)


def _moss_minimax(means, horizon, sigma=1.0):
    m = np.asarray(means, dtype=float)
    return 39.0 * sigma * np.sqrt(m.shape[-1] * horizon) + gaps(m).sum(axis=-1)


def _bern_asymptotic(means, horizon):
    m = np.asarray(means, dtype=float)
    D = gaps(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = kl_bernoulli(m, m.max(axis=-1, keepdims=True))
    return np.where(D > 0, D / np.where(D > 0, kl, 1.0), 0.0).sum(axis=-1) * np.log(horizon)


def _etc_gap(means, horizon, m, n_arms):
    if isinstance(m, str):
        m = minimax_etc_m(horizon, n_arms)
    D = gaps(means)
    return m * D.sum(axis=-1) + (horizon - m * n_arms) * (D * np.exp(-m * D ** 2 / 4.0)).sum(axis=-1)


SUBG = "rewards sigma-subgaussian with sigma the policy's `sigma` (stated for sigma = 1; general sigma by rescaling)"

UCB1_GAP = Bound(
    "UCB1 expected regret", UPPER, GAP,
    "8 sum_{i: Delta_i > 0} ln n / Delta_i + (1 + pi^2/3) sum_j Delta_j",
    REFERENCES["ACF2002"], "Theorem 1", ("rewards in [0, 1]",),
    fn=_ucb1_gap, condition=lambda p: p.c == 2.0)

UCB_DELTA_GAP = Bound(
    "UCB(delta = 1/n^2) expected regret", UPPER, GAP,
    "3 sum_i Delta_i + sum_{i: Delta_i > 0} 16 sigma^2 log n / Delta_i",
    REFERENCES["LS2020"], "Theorem 7.1", (SUBG,),
    fn=_ucb_delta_gap, condition=lambda p: p.delta is None)

UCB_DELTA_MINIMAX = Bound(
    "UCB(delta = 1/n^2) worst-case regret", UPPER, MINIMAX,
    "8 sigma sqrt(n K log n) + 3 sum_i Delta_i",
    REFERENCES["LS2020"], "Theorem 7.2", (SUBG,),
    fn=_ucb_delta_minimax, condition=lambda p: p.delta is None)

UCB_AO_GAP = Bound(
    "Asymptotically optimal UCB expected regret", UPPER, GAP,
    "sum_{i: Delta_i > 0} inf_{eps in (0, Delta_i)} Delta_i (1 + 5 sigma^2/eps^2 "
    "+ 2 sigma^2 (log f(n) + sqrt(pi log f(n)) + 1) / (Delta_i - eps)^2), f(n) = 1 + n log^2 n",
    REFERENCES["LS2020"], "Theorem 8.1", (SUBG,), fn=_ucb_ao_gap)

UCB_AO_ASYMPTOTIC = Bound(
    "Asymptotically optimal UCB, limsup R_n / log n", UPPER, ASYMPTOTIC,
    "limsup R_n / log n <= sum_{i: Delta_i > 0} 2 sigma^2 / Delta_i",
    REFERENCES["LS2020"], "Eq. (8.2)", (SUBG,), fn=_gauss_asymptotic)

MOSS_MINIMAX = Bound(
    "MOSS worst-case regret", UPPER, MINIMAX,
    "39 sigma sqrt(K n) + sum_i Delta_i",
    REFERENCES["LS2020"], "Theorem 9.1", (SUBG,), fn=_moss_minimax)

KLUCB_ASYMPTOTIC = Bound(
    "KL-UCB, limsup R_n / log n", UPPER, ASYMPTOTIC,
    "limsup R_n / log n <= sum_{a: Delta_a > 0} Delta_a / kl(mu_a, mu*)",
    REFERENCES["GC2011"], "Theorem 1", ("rewards in [0, 1]", "c >= 3 (Remark 5)"),
    fn=_bern_asymptotic, condition=lambda p: p.family == "bernoulli" and p.c >= 3)

KLUCB_FINITE = Bound(
    "KL-UCB pulls of a suboptimal arm", UPPER, ORDER,
    "E[N_n(a)] <= (1 + eps) log n / kl(mu_a, mu*) + C_1 log log n + C_2(eps) / n^beta(eps)",
    REFERENCES["GC2011"], "Theorem 2", ("rewards in [0, 1]", "c = 3"),
    condition=lambda p: p.family == "bernoulli" and p.c >= 3)

TS_BETA_GAP = Bound(
    "Beta-prior Thompson sampling expected regret", UPPER, ORDER,
    "(1 + eps) sum_{i >= 2} ln T Delta_i / kl(mu_i, mu_1) + O(N / eps^2)",
    REFERENCES["AG2017"], "Theorem 1.1", ("rewards in [0, 1]", "unique optimal arm"))

TS_BETA_MINIMAX = Bound(
    "Beta-prior Thompson sampling worst-case regret", UPPER, ORDER, "O(sqrt(N T ln T))",
    REFERENCES["AG2017"], "Theorem 1.2", ("rewards in [0, 1]",))

TS_GAUSS_MINIMAX = Bound(
    "Gaussian-prior Thompson sampling worst-case regret", UPPER, ORDER, "O(sqrt(N T ln N)) for T >= N",
    REFERENCES["AG2017"], "Theorem 1.3", ("rewards in [0, 1]",))

ETC_GAP = Bound(
    "Explore-then-commit expected regret", UPPER, GAP,
    "m sum_i Delta_i + (n - m K) sum_i Delta_i exp(-m Delta_i^2 / 4)",
    REFERENCES["LS2020"], "Theorem 6.1", ("1-subgaussian rewards", "1 <= m <= n / K"),
    fn=_etc_gap)


# ------------------------------------------------------------------ K-armed: lower bounds
LAI_ROBBINS_BERNOULLI = Bound(
    "Lai-Robbins asymptotic lower bound (Bernoulli)", LOWER, ASYMPTOTIC,
    "liminf R_n / log n >= sum_{i: Delta_i > 0} Delta_i / kl(mu_i, mu*) for consistent policies",
    REFERENCES["LS2020"], "Theorem 16.2", ("Bernoulli arms", "consistent policy"), fn=_bern_asymptotic)


def _lr_gauss(means, horizon, noise_sigma=1.0):
    return _gauss_asymptotic(means, horizon, noise_sigma)


LAI_ROBBINS_GAUSSIAN = Bound(
    "Lai-Robbins asymptotic lower bound (Gaussian, known variance)", LOWER, ASYMPTOTIC,
    "liminf R_n / log n >= sum_{i: Delta_i > 0} 2 sigma^2 / Delta_i for consistent policies",
    REFERENCES["LS2020"], "Theorem 16.2 and Table 16.1", ("Gaussian arms, known variance", "consistent policy"),
    fn=_lr_gauss)


def _minimax_lower(n_arms, horizon):
    return np.sqrt((n_arms - 1) * horizon) / 27.0


MINIMAX_LOWER = Bound(
    "Minimax lower bound", LOWER, MINIMAX,
    "for every policy some mean vector in [0, 1]^K gives R_n >= sqrt((K - 1) n) / 27",
    REFERENCES["LS2020"], "Theorem 15.2", ("unit-variance Gaussian arms", "n >= K - 1"), fn=_minimax_lower)


# ------------------------------------------------------------------ linear bandits
def oful_radius(logdet_ratio, noise, lam, S, delta):
    """sqrt(beta_t) of Abbasi-Yadkori et al. (2011), Thm 2, with logdet_ratio = log det V_t - d log lam."""
    return noise * np.sqrt(logdet_ratio + 2.0 * np.log(1.0 / delta)) + np.sqrt(lam) * S


def _oful_regret(horizon, dim, noise, lam, S, delta, arm_norm_bound=1.0):
    n, d, L = float(horizon), float(dim), float(arm_norm_bound)
    return 4.0 * np.sqrt(n * d * np.log(lam + n * L / d)) * (
        np.sqrt(lam) * S + noise * np.sqrt(2.0 * np.log(1.0 / delta) + d * np.log(1.0 + n * L / (lam * d))))


OFUL_REGRET = Bound(
    "OFUL regret (high probability)", UPPER, HIGH_PROB,
    "w.p. >= 1 - delta: 4 sqrt(n d log(lam + n L / d)) (sqrt(lam) S "
    "+ R sqrt(2 log(1/delta) + d log(1 + n L / (lam d))))",
    REFERENCES["AYPS2011"], "Theorem 3",
    ("noise R-subgaussian with R the policy's `noise`", "||theta*|| <= S", "||x|| <= L",
     "<x, theta*> in [-1, 1]"),
    fn=_oful_regret, condition=lambda p: p.radius_scale == 1.0)

LINTS_REGRET = Bound(
    "Linear Thompson sampling regret (high probability)", UPPER, ORDER,
    "O(d^2 / eps sqrt(T^(1 + eps)) ln(T d) ln(1/delta))",
    REFERENCES["AG2013"], "Theorem 1", ("noise R-subgaussian", "inflation v = R sqrt(24 / eps d ln(1/delta))"),
    condition=lambda p: p.inflation == "icml2013")

K_ARMED_LOWER = (LAI_ROBBINS_BERNOULLI, LAI_ROBBINS_GAUSSIAN, MINIMAX_LOWER)
