"""Numba simulation kernels for the theory-vs-practice pilot.

Every run is independent and seeded on its own (numba keeps one RNG per thread,
and each prange iteration re-seeds it), so results do not depend on how runs are
scheduled across threads.

The quantity recorded is the pseudo-regret sum_t Delta_{A_t}; its mean over runs
is an unbiased estimate of the expected regret R_n that the theorems bound.
"""
import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------- reward families
BERNOULLI, GAUSSIAN, STUDENT_T = 0, 1, 2

# ---------------------------------------------------------------- K-armed policies
UCB1 = 0         # Auer et al. (2002): mu + sqrt(prm0 * ln t / n_i), theory prm0 = 2
UCB_DELTA = 1    # L&S Alg. 3 with delta = 1/n^2: mu + prm0 * sqrt(4 ln n / n_i), prm0 = sigma
UCB_AO = 2       # L&S Alg. 6: mu + prm0 * sqrt(2 ln f(t) / n_i), f(t) = 1 + t ln^2 t
MOSS = 3         # mu + prm0 * sqrt(4 / n_i * ln+(n / (K n_i)))
KLUCB_BERN = 4   # Garivier & Cappe (2011), level ln t + prm0 * ln ln t
KLUCB_GAUSS = 5  # same with Gaussian KL: mu + prm0 * sqrt(2 (ln t + prm1 ln ln t) / n_i)
TS_BETA = 6      # Beta(1, 1) prior; rewards in [0, 1] are binarized (Agrawal & Goyal)
TS_GAUSS = 7     # Agrawal & Goyal Gaussian TS: N(S_i / (n_i + 1), prm0^2 / (n_i + 1))
ETC = 8          # explore each arm prm0 times round-robin, then commit
EPS_GREEDY = 9   # eps = prm0 (fixed) or, if prm1 > 0, eps_t = min(1, prm1 * K / t)
GREEDY = 10      # one pull per arm, then argmax of empirical means

ALG_NAMES = {
    UCB1: "UCB1", UCB_DELTA: "UCB(delta=1/n^2)", UCB_AO: "UCB-AO", MOSS: "MOSS",
    KLUCB_BERN: "KL-UCB", KLUCB_GAUSS: "KL-UCB-G", TS_BETA: "TS-Beta",
    TS_GAUSS: "TS-Gauss", ETC: "ETC", EPS_GREEDY: "eps-greedy", GREEDY: "Greedy",
}


@njit(cache=True)
def argmax_rand(v):
    """Argmax with uniform tie-breaking (np.argmax would favour low indices)."""
    K = v.shape[0]
    best = v[0]
    for i in range(1, K):
        if v[i] > best:
            best = v[i]
    if np.isinf(best):
        thr = best
    else:
        thr = best - 1e-12 * (1.0 + abs(best))
    cnt = 0
    for i in range(K):
        if v[i] >= thr:
            cnt += 1
    r = np.random.randint(cnt)
    for i in range(K):
        if v[i] >= thr:
            if r == 0:
                return i
            r -= 1
    return K - 1


@njit(cache=True)
def draw_reward(mu, family, sigma, df):
    if family == BERNOULLI:
        return 1.0 if np.random.random() < mu else 0.0
    elif family == GAUSSIAN:
        return mu + sigma * np.random.standard_normal()
    else:  # Student-t rescaled to variance sigma^2 (finite variance, not subgaussian)
        return mu + sigma * np.random.standard_t(df) * np.sqrt((df - 2.0) / df)


@njit(cache=True)
def kl_bern(p, q):
    eps = 1e-15
    p = min(max(p, eps), 1.0 - eps)
    q = min(max(q, eps), 1.0 - eps)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


@njit(cache=True)
def klucb_bern_index(p, n, level):
    """max{q in [p, 1] : n * kl(p, q) <= level} by bisection (precision ~1e-9)."""
    target = level / n
    lo = p
    hi = 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if kl_bern(p, mid) > target:
            hi = mid
        else:
            lo = mid
    return lo


@njit(cache=True)
def run_one(alg, prm, mu, mu2, tau, family, sigma, df, n, checkpoints):
    """One run of a K-armed policy; returns pseudo-regret at each checkpoint.

    mu is used for rounds t < tau and mu2 afterwards (tau >= n: stationary).
    """
    K = mu.shape[0]
    cnt = np.zeros(K)
    sums = np.zeros(K)
    succ = np.zeros(K)
    fail = np.zeros(K)
    idx = np.empty(K)
    out = np.zeros(checkpoints.shape[0])
    c = 0
    regret = 0.0
    star1 = mu.max()
    star2 = mu2.max()
    committed = -1
    for t in range(n):  # t = number of plays so far; current round is t + 1
        s = t + 1.0
        if alg == TS_BETA:
            for i in range(K):
                idx[i] = np.random.beta(1.0 + succ[i], 1.0 + fail[i])
            a = argmax_rand(idx)
        elif alg == TS_GAUSS:
            for i in range(K):
                idx[i] = sums[i] / (cnt[i] + 1.0) + prm[0] / np.sqrt(cnt[i] + 1.0) * np.random.standard_normal()
            a = argmax_rand(idx)
        elif alg == ETC:
            m = int(prm[0])
            if t < m * K:
                a = t % K
            else:
                if committed < 0:
                    for i in range(K):
                        idx[i] = sums[i] / cnt[i]
                    committed = argmax_rand(idx)
                a = committed
        elif t < K:  # index policies and greedy variants: one pull of each arm first
            a = t
        elif alg == GREEDY:
            for i in range(K):
                idx[i] = sums[i] / cnt[i]
            a = argmax_rand(idx)
        elif alg == EPS_GREEDY:
            eps = prm[0]
            if prm[1] > 0.0:
                eps = min(1.0, prm[1] * K / t)
            if np.random.random() < eps:
                a = np.random.randint(K)
            else:
                for i in range(K):
                    idx[i] = sums[i] / cnt[i]
                a = argmax_rand(idx)
        else:
            lt = np.log(t)
            if alg == UCB1:
                for i in range(K):
                    idx[i] = sums[i] / cnt[i] + np.sqrt(prm[0] * lt / cnt[i])
            elif alg == UCB_DELTA:
                for i in range(K):
                    idx[i] = sums[i] / cnt[i] + prm[0] * np.sqrt(4.0 * np.log(n) / cnt[i])
            elif alg == UCB_AO:
                ls = np.log(s)
                lf = np.log(1.0 + s * ls * ls)
                for i in range(K):
                    idx[i] = sums[i] / cnt[i] + prm[0] * np.sqrt(2.0 * lf / cnt[i])
            elif alg == MOSS:
                for i in range(K):
                    lp = max(0.0, np.log(n / (K * cnt[i])))
                    idx[i] = sums[i] / cnt[i] + prm[0] * np.sqrt(4.0 / cnt[i] * lp)
            elif alg == KLUCB_BERN:
                ls = np.log(s)
                level = ls + prm[0] * max(0.0, np.log(ls))
                for i in range(K):
                    idx[i] = klucb_bern_index(sums[i] / cnt[i], cnt[i], level)
            elif alg == KLUCB_GAUSS:
                ls = np.log(s)
                level = ls + prm[1] * max(0.0, np.log(ls))
                for i in range(K):
                    idx[i] = sums[i] / cnt[i] + prm[0] * np.sqrt(2.0 * level / cnt[i])
            a = argmax_rand(idx)

        if t < tau:
            mean_a = mu[a]
            regret += star1 - mean_a
        else:
            mean_a = mu2[a]
            regret += star2 - mean_a
        r = draw_reward(mean_a, family, sigma, df)
        cnt[a] += 1.0
        sums[a] += r
        if alg == TS_BETA:
            rb = r
            if family != BERNOULLI:  # binarization trick for rewards in [0, 1]
                rb = 1.0 if np.random.random() < r else 0.0
            if rb > 0.5:
                succ[a] += 1.0
            else:
                fail[a] += 1.0
        while c < checkpoints.shape[0] and checkpoints[c] == t + 1:
            out[c] = regret
            c += 1
    return out


@njit(parallel=True, cache=True)
def run_batch(alg, prm, MU, MU2, tau, family, sigma, df, n, checkpoints, seeds):
    """Run B independent runs; row b uses instance MU[b] (MU2[b] after tau)."""
    B = MU.shape[0]
    out = np.zeros((B, checkpoints.shape[0]))
    for b in prange(B):
        np.random.seed(seeds[b])
        out[b, :] = run_one(alg, prm, MU[b], MU2[b], tau, family, sigma, df, n, checkpoints)
    return out


# ---------------------------------------------------------------- linear bandits
OFUL = 0     # optimism with radius c * sqrt(beta_t), beta_t from Abbasi-Yadkori et al. (2011) Thm 2
LINTS = 1    # theta_tilde ~ N(theta_hat, v^2 V^{-1})


@njit(cache=True)
def _unit_rows(K, d):
    X = np.empty((K, d))
    for i in range(K):
        nrm = 0.0
        for j in range(d):
            X[i, j] = np.random.standard_normal()
            nrm += X[i, j] * X[i, j]
        nrm = np.sqrt(nrm)
        for j in range(d):
            X[i, j] /= nrm
    return X


@njit(cache=True)
def run_linear_one(policy, scale, X0, theta, R, lam, S, delta, n, fresh, checkpoints):
    """One linear-bandit run.

    policy OFUL: index x.theta_hat + scale * sqrt(beta_t) * ||x||_{V^-1}
    policy LINTS: sample with inflation v = scale
    Also returns max_t ||theta_hat_t - theta*||_{V_t} / sqrt(beta_t), where sqrt(beta_t)
    is the Thm 2 radius R sqrt(log det V_t - d log lam + 2 log(1/delta)) + sqrt(lam) S.
    Thm 2 says this exceeds 1 with probability at most delta, for any policy.
    Third output: the same for the noise term alone, max_t ||sum_s x_s eta_s||_{V_t^-1} /
    (R sqrt(log det V_t - d log lam + 2 log(1/delta))), i.e. the self-normalized bound (Thm 1),
    which separates concentration slack from the deterministic sqrt(lam) S bias term.
    """
    K = X0.shape[0]
    d = X0.shape[1]
    V = lam * np.eye(d)
    Vinv = np.eye(d) / lam
    bvec = np.zeros(d)
    th = np.zeros(d)
    logdet = 0.0  # log det V_t - log det(lam I)
    l1d = np.log(1.0 / delta)
    sql = np.sqrt(lam) * S
    regret = 0.0
    worst = 0.0
    worst_noise = 0.0
    svec = np.zeros(d)
    out = np.zeros(checkpoints.shape[0])
    c = 0
    X = X0.copy()
    idx = np.empty(K)
    for t in range(n):
        if fresh:
            X = _unit_rows(K, d)
        means = X @ theta
        best = means.max()
        if policy == OFUL:
            mu_hat = X @ th
            if scale > 0.0:
                rad = scale * (R * np.sqrt(logdet + 2.0 * l1d) + sql)
                XV = X @ Vinv
                for i in range(K):
                    w = 0.0
                    for j in range(d):
                        w += XV[i, j] * X[i, j]
                    idx[i] = mu_hat[i] + rad * np.sqrt(max(0.0, w))
            else:
                for i in range(K):
                    idx[i] = mu_hat[i]
        else:
            L = np.linalg.cholesky(0.5 * (Vinv + Vinv.T))
            z = np.random.standard_normal(d)
            tt = th + scale * (L @ z)
            for i in range(K):
                idx[i] = X[i] @ tt
        a = argmax_rand(idx)
        x = X[a].copy()
        regret += best - means[a]
        eta = R * np.random.standard_normal()
        r = means[a] + eta
        svec += eta * x
        # Sherman-Morrison update of V^{-1}; log det via the matrix determinant lemma
        Vx = Vinv @ x
        q = x @ Vx
        Vinv -= np.outer(Vx, Vx) / (1.0 + q)
        logdet += np.log1p(q)
        V += np.outer(x, x)
        bvec += r * x
        if (t + 1) % 1000 == 0:
            Vinv = np.ascontiguousarray(np.linalg.inv(V))
        th = Vinv @ bvec
        diff = th - theta
        err = np.sqrt(max(0.0, diff @ (V @ diff)))
        ratio = err / (R * np.sqrt(logdet + 2.0 * l1d) + sql)
        if ratio > worst:
            worst = ratio
        nr = np.sqrt(max(0.0, svec @ (Vinv @ svec))) / (R * np.sqrt(logdet + 2.0 * l1d))
        if nr > worst_noise:
            worst_noise = nr
        while c < checkpoints.shape[0] and checkpoints[c] == t + 1:
            out[c] = regret
            c += 1
    return out, worst, worst_noise


@njit(parallel=True, cache=True)
def run_linear_batch(policy, scale, X0s, thetas, R, lam, S, delta, n, fresh, checkpoints, seeds):
    B = thetas.shape[0]
    out = np.zeros((B, checkpoints.shape[0]))
    worst = np.zeros(B)
    worst_noise = np.zeros(B)
    for b in prange(B):
        np.random.seed(seeds[b])
        o, w, wn = run_linear_one(policy, scale, X0s[b], thetas[b], R, lam, S, delta, n, fresh, checkpoints)
        out[b, :] = o
        worst[b] = w
        worst_noise[b] = wn
    return out, worst, worst_noise
