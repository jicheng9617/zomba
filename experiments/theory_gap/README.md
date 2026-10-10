# Theory vs. practice pilot

A first measurement of how far published regret bounds are from what the
algorithms actually do, and which parts of the theory predict practice.

```bash
pip install numpy scipy matplotlib numba
python run_pilot.py            # all experiments (~17 min on 20 cores); or e.g.: python run_pilot.py e2 e3
python make_figures.py         # figures/*.png and RESULTS.md
```

| file | content |
|---|---|
| `kernels.py` | numba kernels: K-armed policies (UCB1, UCB(delta), UCB-AO, MOSS, KL-UCB, TS, ETC, eps-greedy, greedy) and linear policies (OFUL with scaled radius, LinTS) |
| `bounds.py` | the published bounds with their constants (UCB1, L&S Thms 7.1/7.2/8.1/9.1/15.2, Lai-Robbins, OFUL Thms 2/3, LinTS inflation) |
| `run_pilot.py` | experiments E1-E4, writes `results/*.json` |
| `make_figures.py` | figures and the tables in `RESULTS.md` |

## What is measured

* **E1 gap-dependent theory** (Bernoulli). Regret / log n against the UCB1 bound
  (Auer et al. 2002) and the Lai-Robbins constant, up to n = 10^6; plus 200 random
  instances to see how often the gap-dependent bound is informative at all.
* **E2 empirical minimax regret** (unit-variance Gaussian, K = 10). For each horizon,
  the worst mean regret over the family "best arm at Delta, the rest at 0"
  (24 gaps, 400 runs each), compared with the sqrt(Kn) upper bounds (MOSS, UCB) and
  the sqrt((K-1)n)/27 lower bound; fitted growth exponent in n. E2b checks with random
  search that this family is close to the worst case.
* **E3 linear bandits** (d = 10, K = 50). OFUL with the Theorem-2 radius scaled by
  c in {1, 0.3, 0.1, 0.03, 0.01, 0}, LinTS with the theoretical inflation and with
  v = R. Records max_t ||theta_hat_t - theta*||_{V_t} / sqrt(beta_t) per run: Theorem 2
  says it exceeds 1 with probability at most delta, so its distribution is the
  empirical slack of the confidence set; the same is recorded for the noise term alone
  (the self-normalized bound, Thm 1). E3b repeats the radius sweep for d in {5, 10, 20}
  and noise R in {0.1, 1}.
* **E4 generality**. One fixed configuration per algorithm, 8 environment families
  (Bernoulli, low-mean Bernoulli, many arms, change point, Gaussian with right /
  over- / under-estimated noise, heavy tails), 40 instances each; Dolan-More
  performance profiles of the regret ratio to the per-instance best.

## Headline findings (see RESULTS.md for all numbers)

Where the gap between a published bound and the best observed regret comes from:

| setting | bound / regret of the theory-configured algorithm | theory config / best practical algorithm | total |
|---|---:|---:|---:|
| E1 UCB1 gap bound, mu = (0.9, 0.8), n = 10^6 | 4.4x | 13.8x (vs TS) | 61x |
| E1 UCB1 gap bound, low-mean K = 10, n = 10^6 | 4.7x | 20.6x (vs TS) | 96x |
| E2 MOSS minimax bound, K = 10 | 28x | ~1x (MOSS is the best worst case) | 28x |
| E2 UCB(delta) minimax bound, K = 10 | 9x | 2.0x (vs MOSS) | 18x |
| E3 OFUL regret bound, fixed arms, n = 10^4 | 29x | 5.1x (vs c = 0.3) | 147x |
| E3 OFUL regret bound, fresh contexts, n = 10^4 | 132x | 1.2x | 159x |

* **Rates survive, constants do not.** Fitted exponents of the empirical worst-case
  regret match theory (MOSS 0.49, TS 0.50, UCB 0.55 ~ sqrt(n log n), ETC 0.66,
  eps-greedy / greedy ~1), and the hardest gap sits at a few sqrt(K/n). Random search did
  not find instances harder than the textbook family. The constants are another matter:
  the MOSS bound exceeds the trivial bound n until n = 1521 K, the OFUL bound
  (d = 10) until n ~ 3.7e4, and the lower-bound constant 1/27 is ~39x below the best
  worst case observed.
* **Asymptotics are far away.** At n = 10^6, TS is at 0.6x and KL-UCB (c = 0) at
  0.9x of the Lai-Robbins line; the version of KL-UCB the theorem covers (c = 3) is at 1.5x.
* **The confidence ellipsoid is tight; optimism with it is not.** For its shape (an
  ellipsoid in the V_t norm, valid uniformly in time) the self-normalized bound (Thm 1 of
  Abbasi-Yadkori et al.) is essentially tight: at delta = 1% it is exceeded in 15 of 3600
  runs (0.4%), and the whole Thm 2 radius is ~80% used at the 1 - delta quantile.
  Differently shaped valid sets can still be much smaller in the directions that matter
  (Emmenegger et al. 2023; arXiv:2309.14298).
  Yet the regret-optimal multiplier on that radius is c = 0.2-0.5 in all six (d, R)
  configurations of E3b, and the price of c = 1 grows with d (3.5x at d = 5, up to 8.3x
  at d = 20). Under-exploring is far worse (c <= 0.1 costs up to 400x with low noise,
  greedy locks in). With fresh contexts exploration hardly matters. LinTS with the
  theoretical inflation has near-linear regret, 15-17x that of v = R, both for the ICML
  2013 inflation (v ~ 101) and the arXiv v4 one (v ~ 35).
  A candidate explanation to test: the radius pays sqrt(d log n) for coverage of every
  direction, while choosing among K arms needs ~sqrt(log K); their ratio (0.3-0.6
  here) is close to the empirically best c.
* **No algorithm is general.** Across 8 families every generic algorithm is >= 4x off
  the per-instance best somewhere (MOSS best overall, geo-mean 2.0x, worst 8.5x).
  The dominant failure is a mis-specified noise scale; algorithms that adapt to the
  Bernoulli variance (TS-Beta, KL-UCB) are 4-5x better than sigma = 1 generic ones.
  TS breaks under a change point, greedy wins with many arms. Variance-adaptive UCB-V
  (Audibert, Munos & Szepesvari 2007) is the natural next candidate.

## Conventions

* Regret is the pseudo-regret sum_t Delta_{A_t}; its mean over runs estimates the
  expected regret that the theorems bound. CIs are mean +- 1.96 SE.
* Every run has its own seed (numba re-seeds the thread-local generator per run), so
  results are reproducible and independent of thread scheduling.
* Argmax ties are broken uniformly at random (np.argmax would favour arm 0, which is
  often the best arm in these instances).
* Horizon-dependent algorithms (UCB(delta), MOSS, ETC) are re-run for each horizon;
  anytime algorithms are read off at checkpoints of one long run.
* Known deviation: KL-UCB's level uses log(t + 1) instead of Garivier & Cappe's log t
  (an O(1/t) difference).

## Caveats

* Pilot scale: K = 10 (100 in one family), d <= 20, 100-400 runs per cell. Garivier &
  Cappe report that <= 1000 runs make "risky" algorithms look better than they are, so the
  greedy / small-c numbers here are, if anything, optimistic.
* The worst case in E2 is a maximum over one instance family plus random search, so it is
  a lower estimate of the true sup; adversarial search (CMA-ES / BO over instances) is
  the obvious next step.
* Bound constants were checked against the sources (ACF 2002 Thm 1; L&S Thms 7.1, 7.2,
  8.1, 9.1, 15.2, 16.2; Garivier & Cappe 2011; Abbasi-Yadkori et al. 2011 Thms 2-3;
  Agrawal & Goyal ICML 2013 and arXiv:1209.3352v4).

## Closest related work

Bound vs. regret, worst case, confidence sets:

* Flynn, Reeb, Kandemir & Peters, "PAC-Bayes Bounds for Bandit Problems: A Survey and
  Experimental Comparison" (TPAMI 2023, arXiv:2211.16110): the only explicit bound / regret
  comparison found (K = 10, n = 10^4; every bound > 10x the regret), PAC-Bayes bounds only.
* Kalvit & Zeevi, "A Closer Look at the Worst-case Behavior of Multi-armed Bandit
  Algorithms" (NeurIPS 2021, arXiv:2106.02126): analytic worst-case behaviour of UCB, close to E2.
* Han, Khamaru & Zhang, "UCB algorithms for multi-armed bandits: Precise regret and
  adaptive inference" (arXiv:2412.06126): exact UCB regret; the Lai-Robbins formula is only
  accurate for gaps >~ sigma sqrt(K log T / T).
* Emmenegger, Mutny & Krause, "Likelihood Ratio Confidence Sets for Sequential Decision
  Making" (NeurIPS 2023), and arXiv:2309.14298: valid confidence sets close to the size of
  the heuristically shrunk ones; to be reconciled with E3.
* Hamidi & Bayati (arXiv:2006.06790): LinTS without inflation can have linear regret,
  but their examples need d >~ log T and a prior / noise mismatch.

Theory-tuned exploration is too conservative (known; E3 adds where the slack sits):

* Hsu, Kveton, Meshi, Mladenov & Szepesvari, "Empirical Bayes Regret Minimization"
  (arXiv:1904.02664); Yang & Toni, "Differentiable Linear Bandit Algorithm"
  (arXiv:2006.03000); Kveton et al. (arXiv:2006.05094) and Boutilier et al. (NeurIPS 2020)
  on meta-learned bandit policies; Syndicated Bandits (NeurIPS 2022, arXiv:2106.02979) and
  Kang, Hsieh & Lee (TMLR, arXiv:2302.09440) on online tuning.

Empirical comparisons:

* Bietti, Agarwal & Langford, JMLR 2021 (RegCB best, greedy close second); Riquelme,
  Tucker & Snoek, ICLR 2018; Bayati, Hamidi, Johari & Khosravi, NeurIPS 2020 (greedy with
  many arms); Kuleshov & Precup 2014 and Vermorel & Mohri 2005 (eps-greedy hard to beat);
  Chapelle & Li, NIPS 2011; Garivier, Menard & Stoltz, Math. OR 2019 (true shape of regret).

Not found in the search (so likely open): systematic regret / bound ratios across
algorithm families, adversarial instance search for empirical worst-case regret, and
empirical coverage rates of OFUL ellipsoids. Among libraries only SMPyBandits overlays
bounds on regret plots (Lai-Robbins by default).
