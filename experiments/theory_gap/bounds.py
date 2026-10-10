"""Published regret bounds, written exactly as stated (constants included).

L&S = Lattimore & Szepesvari, "Bandit Algorithms" (2020).
Every function returns a bound on the expected regret R_n (or its coefficient).
"""
import numpy as np


def gaps(mu):
    mu = np.asarray(mu, float)
    return mu.max() - mu


def kl_bern(p, q):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    q = np.clip(q, 1e-15, 1 - 1e-15)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


# ------------------------------------------------------------- gap-dependent
def ucb1_bound(mu, n):
    """Auer, Cesa-Bianchi & Fischer (2002), Thm 1; rewards in [0, 1]."""
    D = gaps(mu)
    Dp = D[D > 0]
    return 8.0 * np.sum(np.log(n) / Dp) + (1.0 + np.pi ** 2 / 3.0) * D.sum()


def lai_robbins_bern(mu):
    """Coefficient C with liminf R_n / log n >= C (Lai & Robbins 1985; L&S Thm 16.2), Bernoulli arms.

    In general the denominator is d_inf(P_i, mu*, M_i); it equals kl(mu_i, mu*) for
    one-parameter families such as Bernoulli.
    """
    mu = np.asarray(mu, float)
    D = gaps(mu)
    m = D > 0
    return np.sum(D[m] / kl_bern(mu[m], mu.max()))


def lai_robbins_gauss(mu, sigma=1.0):
    """Gaussian specialization: sum_i 2 sigma^2 / Delta_i."""
    D = gaps(mu)
    return np.sum(2.0 * sigma ** 2 / D[D > 0])


def ucb_delta_gap_bound(mu, n):
    """L&S Thm 7.1, UCB(delta = 1/n^2), 1-subgaussian."""
    D = gaps(mu)
    return 3.0 * D.sum() + np.sum(16.0 * np.log(n) / D[D > 0])


def ucb_ao_bound(mu, n):
    """L&S Thm 8.1 (f(t) = 1 + t log^2 t), inf over eps taken on a fine grid."""
    D = gaps(mu)
    lf = np.log(1.0 + n * np.log(n) ** 2)
    total = 0.0
    for d in D[D > 0]:
        eps = d * np.linspace(1e-3, 1 - 1e-3, 4000)
        total += np.min(d * (1 + 5 / eps ** 2 + 2 * (lf + np.sqrt(np.pi * lf) + 1) / (d - eps) ** 2))
    return total


# ------------------------------------------------------------- minimax (worst case)
def ucb_delta_minimax(n, K, sum_gaps=0.0):
    """L&S Thm 7.2: 8 sqrt(n K log n) + 3 sum Delta."""
    return 8.0 * np.sqrt(n * K * np.log(n)) + 3.0 * sum_gaps


def moss_minimax(n, K, sum_gaps=0.0):
    """L&S Thm 9.1: 39 sqrt(K n) + sum Delta."""
    return 39.0 * np.sqrt(K * n) + sum_gaps


def minimax_lower(n, K):
    """L&S Thm 15.2: some instance with means in [0,1]^K (unit-variance Gaussian) has
    R_n >= sqrt((K - 1) n) / 27, for every policy."""
    return np.sqrt((K - 1) * n) / 27.0


# ------------------------------------------------------------- linear bandits
def oful_radius(logdet_ratio, R, lam, S, delta):
    """Abbasi-Yadkori, Pal & Szepesvari (2011) Thm 2 radius sqrt(beta_t)."""
    return R * np.sqrt(logdet_ratio + 2.0 * np.log(1.0 / delta)) + np.sqrt(lam) * S


def oful_regret_bound(n, d, R, lam, S, L, delta):
    """Abbasi-Yadkori et al. (2011) Thm 3 (holds w.p. >= 1 - delta)."""
    n = np.asarray(n, float)
    return 4.0 * np.sqrt(n * d * np.log(lam + n * L / d)) * (
        np.sqrt(lam) * S + R * np.sqrt(2.0 * np.log(1.0 / delta) + d * np.log(1.0 + n * L / (lam * d))))


def lints_inflation(R, d, eps, delta):
    """Agrawal & Goyal (ICML 2013) posterior inflation v = R sqrt(24 / eps * d * ln(1/delta));
    Remark 2 suggests eps = 1 / ln T when T is known."""
    return R * np.sqrt(24.0 / eps * d * np.log(1.0 / delta))


def lints_inflation_v4(R, d, T, delta):
    """Agrawal & Goyal, arXiv:1209.3352v4 (2014): v = R sqrt(9 d ln(T / delta)), no eps."""
    return R * np.sqrt(9.0 * d * np.log(T / delta))
