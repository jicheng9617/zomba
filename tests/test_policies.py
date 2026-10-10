import numpy as np
import pytest

from zomba import envs, policies as P, simulate
from zomba.theory import kl_bernoulli


def _after_init(policy, counts, sums, horizon=None):
    """A policy whose statistics are set by hand, past the initial round robin."""
    counts, sums = np.atleast_2d(counts).astype(float), np.atleast_2d(sums).astype(float)
    policy.reset(counts.shape[1], counts.shape[0], horizon=horizon, seed=0)
    policy.counts, policy.sums = counts.copy(), sums.copy()
    policy.t = int(counts.sum(axis=1)[0])
    return policy


COUNTS, SUMS = [2.0, 3.0, 5.0], [1.0, 2.0, 4.5]
MEANS, N = np.array(SUMS) / np.array(COUNTS), np.array(COUNTS)


def test_ucb1_index():
    idx = _after_init(P.UCB1(), COUNTS, SUMS).index()[0]
    assert idx == pytest.approx(MEANS + np.sqrt(2 * np.log(10) / N))


def test_ucb_delta_index_uses_horizon():
    idx = _after_init(P.UCB(sigma=0.5), COUNTS, SUMS, horizon=100).index()[0]
    assert idx == pytest.approx(MEANS + 0.5 * np.sqrt(2 * np.log(100 ** 2) / N))
    idx = _after_init(P.UCB(delta=0.1), COUNTS, SUMS).index()[0]
    assert idx == pytest.approx(MEANS + np.sqrt(2 * np.log(10) / N))


def test_asymptotic_ucb_index():
    idx = _after_init(P.AsymptoticUCB(), COUNTS, SUMS).index()[0]
    f = 1 + 11 * np.log(11) ** 2
    assert idx == pytest.approx(MEANS + np.sqrt(2 * np.log(f) / N))


def test_moss_index():
    idx = _after_init(P.MOSS(), COUNTS, SUMS, horizon=60).index()[0]
    assert idx == pytest.approx(MEANS + np.sqrt(4 / N * np.log(np.maximum(1, 60 / (3 * N)))))


def test_klucb_indices():
    pol = _after_init(P.KLUCB(c=0), COUNTS, SUMS)
    q = pol.index()[0]
    level = np.log(10)
    assert N * kl_bernoulli(MEANS, q) == pytest.approx(level, rel=1e-9)
    assert np.all(N * kl_bernoulli(MEANS, q + 1e-6) > level)
    pol = _after_init(P.KLUCB(c=3, family="gaussian", sigma=2.0), COUNTS, SUMS)
    level = np.log(10) + 3 * np.log(np.log(10))
    assert pol.index()[0] == pytest.approx(MEANS + 2.0 * np.sqrt(2 * level / N))


def test_klucb_bernoulli_extremes():
    q = P.klucb_bernoulli(np.array([0.0, 1.0, 0.3, 0.3]), np.array([5.0, 5.0, 5.0, 1e9]), np.array([1.0, 1.0, 0.0, 1.0]))
    assert q[0] == pytest.approx(1 - np.exp(-0.2))   # kl(0, q) = -log(1 - q)
    assert q[1] == pytest.approx(1.0)
    assert q[2] == pytest.approx(0.3)                 # zero level: no optimism
    assert q[3] == pytest.approx(0.3, abs=1e-4)


def test_horizon_requirements():
    for pol in (P.MOSS(), P.UCB(), P.ExploreThenCommit()):
        with pytest.raises(ValueError):
            pol.reset(3)
    P.UCB(delta=0.05).reset(3)
    P.ExploreThenCommit(m=4).reset(3)
    with pytest.raises(ValueError):
        P.ExploreThenCommit(m=0)


def test_index_policies_play_each_arm_once_first():
    env = envs.BernoulliBandit([0.5] * 6)
    for pol in (P.UCB1(), P.MOSS(), P.KLUCB(), P.Greedy(), P.EpsilonGreedy(epsilon=1.0)):
        res = simulate(pol, env, 6, n_runs=10, seed=0)
        assert np.all(res.pulls == 1), pol


def test_explore_then_commit():
    res = simulate(P.ExploreThenCommit(m=7), envs.BernoulliBandit([0.9, 0.1, 0.5]), 21, n_runs=5, seed=0)
    assert np.all(res.pulls == 7)
    res = simulate(P.ExploreThenCommit(m=7), envs.BernoulliBandit([0.9, 0.1, 0.5]), 100, n_runs=5, seed=0)
    assert np.all(np.sort(res.pulls, axis=1)[:, :2] == 7)   # all later rounds go to one arm


def test_thompson_sampling_binarises_fractional_rewards():
    pol = P.ThompsonSampling().reset(2, 4000, seed=0)
    arms = np.zeros(4000, dtype=int)
    pol.update(arms, np.full(4000, 0.25))
    assert np.all(pol.successes[:, 0] + pol.failures[:, 0] == 1)
    assert pol.successes[:, 0].mean() == pytest.approx(0.25, abs=0.03)


def test_all_policies_learn_an_easy_instance():
    env, n = envs.BernoulliBandit([0.9, 0.5, 0.4]), 2000
    uniform = simulate(P.Uniform(), env, n, n_runs=32, seed=0).mean()[-1]
    for pol in (P.UCB1(), P.UCB(sigma=0.5), P.AsymptoticUCB(sigma=0.5), P.MOSS(sigma=0.5), P.KLUCB(), P.KLUCB(c=0),
                P.ThompsonSampling(), P.GaussianThompsonSampling(sigma=0.5), P.ExploreThenCommit(),
                P.EpsilonGreedy(epsilon=0.05), P.EpsilonGreedy(decay=5.0)):
        assert simulate(pol, env, n, n_runs=32, seed=0).mean()[-1] < 0.2 * uniform, pol
