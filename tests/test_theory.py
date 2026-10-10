import numpy as np
import pytest

from zomba import envs, policies as P, theory
from zomba.theory import bounds as B


def test_every_bound_cites_a_checked_source():
    found = [v for v in vars(B).values() if isinstance(v, theory.Bound)]
    assert len(found) >= 15
    for b in found:
        assert b.reference.key in theory.REFERENCES and b.location and b.checked, b.name
        assert b.kind in (theory.UPPER, theory.LOWER)


def test_ucb1_bound_value():
    v = B.UCB1_GAP.evaluate(means=[0.9, 0.8], horizon=10 ** 6)
    assert v == pytest.approx(8 * np.log(1e6) / 0.1 + (1 + np.pi ** 2 / 3) * 0.1)


def test_minimax_bounds():
    means = [0.5, 0.3, 0.3, 0.3]
    assert B.MOSS_MINIMAX.evaluate(means=means, horizon=1e4) == pytest.approx(39 * np.sqrt(4e4) + 0.6)
    assert B.MOSS_MINIMAX.evaluate(means=means, horizon=1e4, sigma=2.0) == pytest.approx(78 * np.sqrt(4e4) + 0.6)
    assert B.UCB_DELTA_MINIMAX.evaluate(means=means, horizon=1e4) == pytest.approx(
        8 * np.sqrt(4e4 * np.log(1e4)) + 1.8)
    assert B.MINIMAX_LOWER.evaluate(n_arms=10, horizon=1e4) == pytest.approx(np.sqrt(9e4) / 27)


def test_ucb_ao_closed_form_infimum_matches_a_grid_search():
    means, n, sigma = np.array([1.0, 0.7, 0.4, 0.35]), 5e4, 1.3
    lf = np.log(1 + n * np.log(n) ** 2)
    total = 0.0
    for d in means.max() - means[1:]:
        eps = d * np.linspace(1e-4, 1 - 1e-4, 200001)
        total += np.min(d * (1 + 5 * sigma ** 2 / eps ** 2 + 2 * sigma ** 2 * (lf + np.sqrt(np.pi * lf) + 1) / (d - eps) ** 2))
    assert B.UCB_AO_GAP.evaluate(means=means, horizon=n, sigma=sigma) == pytest.approx(total, rel=1e-7)


def test_asymptotic_constants():
    lr = B.LAI_ROBBINS_BERNOULLI.evaluate(means=[0.9, 0.8], horizon=np.e)
    assert lr == pytest.approx(0.1 / (0.8 * np.log(0.8 / 0.9) + 0.2 * np.log(0.2 / 0.1)))
    assert B.LAI_ROBBINS_GAUSSIAN.evaluate(means=[1.0, 0.5], horizon=np.e, noise_sigma=2.0) == pytest.approx(16.0)


def test_oful_regret_bound_value():
    v = B.OFUL_REGRET.evaluate(horizon=10000, dim=10, noise=1.0, lam=1.0, S=1.0, delta=0.01, arm_norm_bound=1.0)
    assert v == pytest.approx(32744.23, rel=1e-6)


def test_bounds_vectorise_over_runs():
    means = envs.repeat_instances([[0.9, 0.8], [0.6, 0.1]], 3)
    v = B.UCB1_GAP.evaluate(means=means, horizon=1000)
    assert v.shape == (6,) and v[0] == v[2] != v[3]


def test_policy_bounds_follow_the_analysed_configuration():
    env = envs.BernoulliBandit([0.9, 0.8]).reset(2)
    assert [b for b, _ in P.UCB1().bounds(env, 1000)] == [B.UCB1_GAP]
    assert P.UCB1(c=0.5).bounds(env, 1000) == []
    assert P.KLUCB(c=0).bounds(env, 1000) == []
    names = {b.name: v for b, v in P.KLUCB().bounds(env, 1000)}
    assert names[B.KLUCB_FINITE.name] is None and names[B.KLUCB_ASYMPTOTIC.name].shape == (2,)
    lin = envs.LinearBandit(5, 3).reset(1, seed=0)
    assert len(P.OFUL().bounds(lin, 1000)) == 1
    assert P.OFUL(radius_scale=0.3).bounds(lin, 1000) == []
    with pytest.raises(ValueError):
        B.TS_BETA_MINIMAX.evaluate(horizon=10)


def test_lower_bounds_follow_the_environment():
    bern = theory.lower_bounds(envs.BernoulliBandit([0.9, 0.8]).reset(1), 100)
    assert [b for b, _ in bern] == [B.LAI_ROBBINS_BERNOULLI]
    gauss = theory.lower_bounds(envs.GaussianBandit([1.0, 0.0]).reset(1), 100)
    assert [b for b, _ in gauss] == [B.LAI_ROBBINS_GAUSSIAN, B.MINIMAX_LOWER]
    assert theory.lower_bounds(envs.GaussianBandit([1.0, 0.0], sigma=2.0).reset(1), 100)[0][0] is B.LAI_ROBBINS_GAUSSIAN
    changing = envs.BernoulliBandit([0.9, 0.8], changes=[(50, [0.8, 0.9])]).reset(1)
    assert theory.lower_bounds(changing, 100) == []
