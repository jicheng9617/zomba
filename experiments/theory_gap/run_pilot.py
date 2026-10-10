"""Pilot study: how do empirical regrets compare with what bandit theory predicts?

    python run_pilot.py e1 e2 e3 e4      # any subset; results go to ./results

E1  gap-dependent bounds and asymptotic optimality (Bernoulli arms)
E2  empirical worst-case (minimax) regret vs. sqrt(Kn) theory (unit-variance Gaussian arms)
E3  linear bandits: slack of the OFUL confidence set, theory-tuned vs practical exploration (E3b: across d and noise)
E4  generality: one fixed configuration per algorithm across environment families
"""
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

import bounds as bd
import kernels as k

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def seeds(tag, B):
    ss = np.random.SeedSequence(zlib.crc32(tag.encode()))
    return (ss.generate_state(B, dtype=np.uint32) % (2 ** 31 - 1)).astype(np.int64)


def run(alg, prm, MU, n, cps, tag, family=k.GAUSSIAN, sigma=1.0, df=3.0, MU2=None, tau=None):
    MU = np.ascontiguousarray(MU, dtype=float)
    MU2 = MU if MU2 is None else np.ascontiguousarray(MU2, dtype=float)
    tau = n + 1 if tau is None else tau
    return k.run_batch(alg, np.asarray(prm, float), MU, MU2, tau, family, sigma, df, n,
                       np.asarray(cps, np.int64), seeds(tag, MU.shape[0]))


def mean_se(x, axis=0):
    return x.mean(axis), x.std(axis, ddof=1) / np.sqrt(x.shape[axis])


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ============================================================================ E1
def e1():
    instances = {
        "I1": np.array([0.9, 0.8]),
        "I2": np.array([0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]),
    }
    algs = {  # anytime algorithms only, so every checkpoint is a valid horizon
        "UCB1": (k.UCB1, [2.0, 0.0]),
        "KL-UCB (c=3, theory)": (k.KLUCB_BERN, [3.0, 0.0]),
        "KL-UCB (c=0, practice)": (k.KLUCB_BERN, [0.0, 0.0]),
        "TS-Beta": (k.TS_BETA, [0.0, 0.0]),
    }
    n, runs = 10 ** 6, 300
    cps = np.unique(np.round(np.logspace(2, 6, 41)).astype(np.int64))
    res = {"checkpoints": cps.tolist(), "instances": {}}
    for iname, mu in instances.items():
        MU = np.tile(mu, (runs, 1))
        entry = {"mu": mu.tolist(), "C_LR": bd.lai_robbins_bern(mu),
                 "ucb1_bound": [bd.ucb1_bound(mu, c) for c in cps], "algs": {}}
        for aname, (alg, prm) in algs.items():
            t0 = time.time()
            R = run(alg, prm, MU, n, cps, f"e1/{iname}/{aname}", family=k.BERNOULLI)
            m, se = mean_se(R)
            entry["algs"][aname] = {"mean": m.tolist(), "se": se.tolist(),
                                    "q90": np.quantile(R, 0.9, axis=0).tolist()}
            log(f"E1 {iname} {aname:24s} R_n={m[-1]:8.1f}  ({time.time() - t0:.0f}s)")
        res["instances"][iname] = entry

    # E1b: random instances -- how often is the gap-dependent bound informative at all?
    M, runs_b, nb = 200, 100, 10 ** 4
    rng = np.random.default_rng(1)
    mus = rng.uniform(0, 1, size=(M, 10))
    MU = np.repeat(mus, runs_b, axis=0)
    rb = {}
    for aname in ["UCB1", "TS-Beta", "KL-UCB (c=0, practice)"]:
        alg, prm = algs[aname]
        R = run(alg, prm, MU, nb, [nb], f"e1b/{aname}", family=k.BERNOULLI)[:, 0]
        rb[aname] = R.reshape(M, runs_b).mean(1)
    ub = np.array([bd.ucb1_bound(mu, nb) for mu in mus])
    trivial = nb * (mus.max(1) - mus.min(1))
    clr = np.array([bd.lai_robbins_bern(mu) for mu in mus]) * np.log(nb)
    res["random_instances"] = {
        "n": nb, "M": M,
        "ucb1_over_bound": (rb["UCB1"] / ub).tolist(),
        "bound_vacuous_frac": float(np.mean(ub >= trivial)),
        "bound_over_trivial": (ub / trivial).tolist(),
        "ts_over_LR": (rb["TS-Beta"] / clr).tolist(),
        "klucb_over_LR": (rb["KL-UCB (c=0, practice)"] / clr).tolist(),
    }
    (OUT / "e1.json").write_text(json.dumps(res))


# ============================================================================ E2
E2_ALGS = {
    # name: (code, params(n), anytime?)
    "UCB (delta=1/n^2)": (k.UCB_DELTA, lambda n: [1.0, 0.0], False),
    "KL-UCB-G (c=0)": (k.KLUCB_GAUSS, lambda n: [1.0, 0.0], True),
    "TS-Gauss": (k.TS_GAUSS, lambda n: [1.0, 0.0], True),
    "MOSS": (k.MOSS, lambda n: [1.0, 0.0], False),
    "ETC (m=(n/K)^(2/3))": (k.ETC, lambda n: [np.ceil((n / 10) ** (2 / 3)), 0.0], False),
    "eps-greedy (0.1)": (k.EPS_GREEDY, lambda n: [0.1, 0.0], True),
    "Greedy": (k.GREEDY, lambda n: [0.0, 0.0], True),
}


def hard_instances(K, gap_grid):
    """mu_1 = Delta, the other K-1 arms at 0: the family used in minimax lower-bound proofs."""
    MU = np.zeros((len(gap_grid), K))
    MU[:, 0] = gap_grid
    return MU


def e2():
    K, runs = 10, 400
    ns = np.array([1000, 3162, 10000, 31623, 100000], dtype=np.int64)
    gap_grid = np.geomspace(0.005, 1.0, 24)
    MU = np.repeat(hard_instances(K, gap_grid), runs, axis=0)
    G = len(gap_grid)
    res = {"K": K, "ns": ns.tolist(), "gaps": gap_grid.tolist(), "runs": runs, "algs": {}}
    for aname, (alg, prm_fn, anytime) in E2_ALGS.items():
        t0 = time.time()
        if anytime:
            R = run(alg, prm_fn(ns[-1]), MU, int(ns[-1]), ns, f"e2/{aname}")  # (G*runs, len(ns))
        else:
            R = np.column_stack([run(alg, prm_fn(n), MU, int(n), [n], f"e2/{aname}/{n}")[:, 0] for n in ns])
        R = R.reshape(G, runs, len(ns))
        m = R.mean(1)                                   # (G, len(ns))
        se = R.std(1, ddof=1) / np.sqrt(runs)
        q95 = np.quantile(R, 0.95, axis=1)
        res["algs"][aname] = {"mean": m.tolist(), "se": se.tolist(), "q95": q95.tolist()}
        worst = m.max(0)
        log(f"E2 {aname:22s} worst-case R/sqrt(Kn) = {np.round(worst / np.sqrt(K * ns), 3)}  ({time.time() - t0:.0f}s)")
    (OUT / "e2.json").write_text(json.dumps(res))


def e2b():
    """Is the one-parameter hard family really (close to) worst case?  Random search at n = 10^4."""
    K, n, runs, M = 10, 10000, 200, 300
    rng = np.random.default_rng(7)
    fam = json.loads((OUT / "e2.json").read_text())
    ns = fam["ns"]
    j = ns.index(n)
    out = {"n": n, "M": M, "algs": {}}
    for aname in ["UCB (delta=1/n^2)", "TS-Gauss", "MOSS", "KL-UCB-G (c=0)"]:
        alg, prm_fn, _ = E2_ALGS[aname]
        fam_mean = np.array(fam["algs"][aname]["mean"])[:, j]
        dstar = fam["gaps"][int(np.argmax(fam_mean))]
        # (a) perturbations around the worst member: K-1 gaps log-uniform in [dstar/4, 4 dstar]
        # (b) unstructured instances: mu ~ U(0,1)^K
        # (same location as the family: worst arm near 0, best arm at the largest sampled gap)
        gp = np.clip(dstar * np.exp(rng.uniform(np.log(0.25), np.log(4.0), size=(M, K - 1))), 0, 1)
        A = np.zeros((M, K))
        A[:, 0] = gp.max(1)
        A[:, 1:] = A[:, :1] - gp
        Bm = rng.uniform(0, 1, size=(M, K))
        best = {}
        for lab, mus in [("perturbed", A), ("uniform", Bm)]:
            R = run(alg, prm_fn(n), np.repeat(mus, runs, axis=0), n, [n], f"e2b/{aname}/{lab}")[:, 0]
            means = R.reshape(M, runs).mean(1)
            best[lab] = {"max": float(means.max()), "argmax_mu": mus[int(np.argmax(means))].tolist(),
                         "q90": float(np.quantile(means, 0.9))}
        out["algs"][aname] = {"family_max": float(fam_mean.max()), "family_dstar": dstar, **best}
        log(f"E2b {aname:20s} family max {fam_mean.max():7.1f} | perturbed max {best['perturbed']['max']:7.1f}"
            f" | uniform max {best['uniform']['max']:7.1f}")
    (OUT / "e2b.json").write_text(json.dumps(out))


# ============================================================================ E3
def e3():
    d, K, n, runs = 10, 50, 10000, 200
    R_noise, lam, S, delta = 1.0, 1.0, 1.0, 0.01
    rng = np.random.default_rng(3)
    th = rng.standard_normal((runs, d))
    th /= np.linalg.norm(th, axis=1, keepdims=True)
    X0 = rng.standard_normal((runs, K, d))
    X0 /= np.linalg.norm(X0, axis=2, keepdims=True)
    cps = np.unique(np.round(np.logspace(1, 4, 31)).astype(np.int64))
    v_theory = float(bd.lints_inflation(R_noise, d, 1.0 / np.log(n), delta))
    v_v4 = float(bd.lints_inflation_v4(R_noise, d, n, delta))
    policies = {
        "OFUL c=1 (theory)": (k.OFUL, 1.0),
        "OFUL c=0.3": (k.OFUL, 0.3),
        "OFUL c=0.1": (k.OFUL, 0.1),
        "OFUL c=0.03": (k.OFUL, 0.03),
        "OFUL c=0.01": (k.OFUL, 0.01),
        "Greedy (c=0)": (k.OFUL, 0.0),
        f"LinTS v={v_theory:.0f} (theory)": (k.LINTS, v_theory),
        f"LinTS v={v_v4:.0f} (theory, arXiv v4)": (k.LINTS, v_v4),
        "LinTS v=R (Bayes)": (k.LINTS, R_noise),
    }
    res = {"d": d, "K": K, "n": n, "runs": runs, "R": R_noise, "lam": lam, "S": S, "delta": delta,
           "v_theory": v_theory, "checkpoints": cps.tolist(),
           "theory_bound": bd.oful_regret_bound(cps, d, R_noise, lam, S, 1.0, delta).tolist(),
           "settings": {}}
    for setting, fresh in [("fixed arms", False), ("fresh contexts", True)]:
        out = {}
        for pname, (pol, scale) in policies.items():
            t0 = time.time()
            Rg, worst, wnoise = k.run_linear_batch(pol, scale, X0, th, R_noise, lam, S, delta, n, fresh,
                                                   cps, seeds(f"e3/{setting}/{pname}", runs))
            m, se = mean_se(Rg)
            out[pname] = {"mean": m.tolist(), "se": se.tolist(), "q95": np.quantile(Rg, 0.95, axis=0).tolist(),
                          "worst_ratio": worst.tolist(), "worst_noise_ratio": wnoise.tolist()}
            log(f"E3 {setting:14s} {pname:24s} R_n={m[-1]:8.1f}+-{1.96 * se[-1]:6.1f}  "
                f"slack q99={np.quantile(worst, 0.99):.3f} max={worst.max():.3f} "
                f"noise-only q99={np.quantile(wnoise, 0.99):.3f}  ({time.time() - t0:.0f}s)")
        res["settings"][setting] = out
    (OUT / "e3.json").write_text(json.dumps(res))


def e3b():
    """Robustness of the E3 findings: confidence slack and the best radius multiplier across d and noise."""
    K, n, runs, lam, S, delta = 50, 20000, 100, 1.0, 1.0, 0.01
    cs = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0]
    cps = np.array([n], dtype=np.int64)
    res = {"K": K, "n": n, "runs": runs, "cs": cs, "configs": []}
    for d in [5, 10, 20]:
        for R_noise in [0.1, 1.0]:
            rng = np.random.default_rng(100 + d)
            th = rng.standard_normal((runs, d))
            th /= np.linalg.norm(th, axis=1, keepdims=True)
            X0 = rng.standard_normal((runs, K, d))
            X0 /= np.linalg.norm(X0, axis=2, keepdims=True)
            row = {"d": d, "R": R_noise, "regret": [], "se": [], "slack_q99": [], "slack_med": [],
                   "noise_q99": [], "noise_med": []}
            for c in cs:
                Rg, worst, wnoise = k.run_linear_batch(k.OFUL, c, X0, th, R_noise, lam, S, delta, n, False, cps,
                                                       seeds(f"e3b/{d}/{R_noise}/{c}", runs))
                row["regret"].append(float(Rg[:, 0].mean()))
                row["se"].append(float(Rg[:, 0].std(ddof=1) / np.sqrt(runs)))
                row["slack_q99"].append(float(np.quantile(worst, 0.99)))
                row["slack_med"].append(float(np.median(worst)))
                row["noise_q99"].append(float(np.quantile(wnoise, 0.99)))
                row["noise_med"].append(float(np.median(wnoise)))
            best = int(np.argmin(row["regret"]))
            log(f"E3b d={d:2d} R={R_noise:3.1f}  regret by c: " + " ".join(f"{c}:{r:.0f}" for c, r in zip(cs, row["regret"]))
                + f" | best c={cs[best]}  theory/best={row['regret'][0] / row['regret'][best]:.1f}x"
                + f" | slack q99 {min(row['slack_q99']):.2f}-{max(row['slack_q99']):.2f}"
                + f" | noise-only q99 {min(row['noise_q99']):.2f}-{max(row['noise_q99']):.2f}")
            res["configs"].append(row)
    (OUT / "e3b.json").write_text(json.dumps(res))


# ============================================================================ E4
E4_ALGS = {
    "UCB1": (k.UCB1, [2.0, 0.0]),
    "TS-Gauss (sigma=1)": (k.TS_GAUSS, [1.0, 0.0]),
    "MOSS (sigma=1)": (k.MOSS, [1.0, 0.0]),
    "ETC (m=(n/K)^(2/3))": (k.ETC, None),
    "eps-greedy (0.1)": (k.EPS_GREEDY, [0.1, 0.0]),
    "Greedy": (k.GREEDY, [0.0, 0.0]),
}
E4_BERN_ONLY = {"KL-UCB (c=0)": (k.KLUCB_BERN, [0.0, 0.0]), "TS-Beta": (k.TS_BETA, [0.0, 0.0])}


def e4():
    n, M, runs = 10000, 40, 100
    rng = np.random.default_rng(4)
    fams = {
        "Bernoulli, K=10": dict(family=k.BERNOULLI, K=10, lo=0.0, hi=1.0),
        "Bernoulli low-mean (CTR-like), K=10": dict(family=k.BERNOULLI, K=10, lo=0.0, hi=0.1),
        "Bernoulli, K=100": dict(family=k.BERNOULLI, K=100, lo=0.0, hi=1.0),
        "Bernoulli, change point at n/2": dict(family=k.BERNOULLI, K=10, lo=0.0, hi=1.0, change=True),
        "Gaussian sigma=1, K=10": dict(family=k.GAUSSIAN, K=10, lo=0.0, hi=1.0, sigma=1.0),
        "Gaussian sigma=0.1 (over-estimated noise)": dict(family=k.GAUSSIAN, K=10, lo=0.0, hi=1.0, sigma=0.1),
        "Gaussian sigma=3 (under-estimated noise)": dict(family=k.GAUSSIAN, K=10, lo=0.0, hi=1.0, sigma=3.0),
        "Student-t df=2.5, var=1 (heavy tail)": dict(family=k.STUDENT_T, K=10, lo=0.0, hi=1.0, sigma=1.0),
    }
    res = {"n": n, "M": M, "runs": runs, "families": {}}
    for fname, f in fams.items():
        K = f["K"]
        mus = rng.uniform(f["lo"], f["hi"], size=(M, K))
        mus2 = mus.copy()
        if f.get("change"):
            ib, iw = mus.argmax(1), mus.argmin(1)
            r = np.arange(M)
            mus2[r, ib], mus2[r, iw] = mus[r, iw], mus[r, ib]
        MU, MU2 = np.repeat(mus, runs, axis=0), np.repeat(mus2, runs, axis=0)
        algs = dict(E4_ALGS)
        if f["family"] == k.BERNOULLI:
            algs.update(E4_BERN_ONLY)
        entry = {}
        for aname, (alg, prm) in algs.items():
            if alg == k.ETC:
                prm = [np.ceil((n / K) ** (2 / 3)), 0.0]
            R = run(alg, prm, MU, n, [n], f"e4/{fname}/{aname}", family=f["family"], sigma=f.get("sigma", 1.0),
                    df=2.5, MU2=MU2, tau=n // 2 if f.get("change") else None)[:, 0]
            Rm = R.reshape(M, runs)
            entry[aname] = {"mean": Rm.mean(1).tolist(), "se": (Rm.std(1, ddof=1) / np.sqrt(runs)).tolist()}
        res["families"][fname] = entry
        log(f"E4 {fname:42s} " + "  ".join(f"{a.split(' ')[0]}={np.mean(v['mean']):.0f}" for a, v in entry.items()))
    (OUT / "e4.json").write_text(json.dumps(res))


if __name__ == "__main__":
    todo = sys.argv[1:] or ["e1", "e2", "e2b", "e3", "e3b", "e4"]
    for name in todo:
        t0 = time.time()
        globals()[name]()
        log(f"{name} done in {time.time() - t0:.0f}s")
