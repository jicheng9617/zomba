import numpy as np
import pytest

from zomba import envs, policies, simulate
from zomba.core import default_checkpoints, rand_argmax


def test_rand_argmax_picks_the_max_without_ties():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((50, 7))
    assert np.array_equal(rand_argmax(v, rng), v.argmax(axis=1))


def test_rand_argmax_breaks_ties_uniformly():
    rng = np.random.default_rng(0)
    picks = rand_argmax(np.zeros((40000, 4)), rng)
    counts = np.bincount(picks, minlength=4)
    assert np.all(np.abs(counts - 10000) < 400)


def test_rand_argmax_handles_infinite_values():
    rng = np.random.default_rng(0)
    v = np.array([[1.0, np.inf, np.inf], [np.inf, 0.0, 0.0], [-np.inf, -np.inf, -np.inf]])
    picks = np.array([rand_argmax(v, rng) for _ in range(300)])
    assert set(picks[:, 0]) == {1, 2}
    assert set(picks[:, 1]) == {0}
    assert set(picks[:, 2]) == {0, 1, 2}


def test_simulate_shapes_and_bookkeeping():
    res = simulate(policies.UCB1(), envs.BernoulliBandit([0.7, 0.5, 0.2]), horizon=300, n_runs=6, seed=1)
    assert res.regret.shape == (6, len(res.checkpoints))
    assert res.checkpoints[-1] == 300
    assert np.all(np.diff(res.regret, axis=1) >= 0)
    assert np.all(res.pulls.sum(axis=1) == 300)
    assert res.final.shape == (6,)


def test_simulate_is_reproducible_and_seed_none_is_recoverable():
    env, pol = envs.BernoulliBandit([0.6, 0.5]), policies.ThompsonSampling()
    a = simulate(pol, env, 500, n_runs=5, seed=11)
    b = simulate(pol, env, 500, n_runs=5, seed=11)
    c = simulate(pol, env, 500, n_runs=5, seed=12)
    assert np.array_equal(a.regret, b.regret)
    assert not np.array_equal(a.regret, c.regret)
    d = simulate(pol, env, 500, n_runs=5)
    e = simulate(pol, env, 500, n_runs=5, seed=d.seed)
    assert np.array_equal(d.regret, e.regret)


def test_online_loop_matches_simulate():
    env, pol, n = envs.GaussianBandit([0.3, 0.0, -0.2]), policies.AsymptoticUCB(), 400
    res = simulate(pol, env, n, n_runs=1, seed=5, checkpoints=[n])
    env_seed, pol_seed = np.random.SeedSequence(5).spawn(2)
    env.reset(1, seed=env_seed)
    pol.reset(env.n_arms, 1, horizon=n, seed=pol_seed)
    total = 0.0
    for _ in range(n):
        a = pol.select()
        total += env.regret(a)[0]
        pol.update(a, env.pull(a))
    assert total == pytest.approx(res.final[0])


def test_single_run_accepts_scalars():
    pol = policies.ThompsonSampling().reset(n_arms=3, seed=0)
    arm = pol.select()[0]
    pol.update(arm, 1.0)
    assert pol.successes.sum() == 1
    lin = policies.OFUL().reset(n_arms=4, dim=2, seed=0)
    x = np.ones((1, 4, 2))
    a = lin.select(x)[0]
    lin.update(a, 0.5, x)
    assert np.allclose(lin.b, 0.5)


def test_checkpoints():
    assert default_checkpoints(1000)[0] == 1 and default_checkpoints(1000)[-1] == 1000
    res = simulate(policies.Greedy(), envs.BernoulliBandit([0.5, 0.4]), 50, n_runs=2, seed=0, checkpoints=[10, 50])
    assert list(res.checkpoints) == [10, 50]
    with pytest.raises(ValueError):
        simulate(policies.Greedy(), envs.BernoulliBandit([0.5, 0.4]), 50, checkpoints=[60])


def test_common_random_numbers_across_policies():
    # the environment's noise stream does not depend on the policy
    env = envs.GaussianBandit([0.0, 0.0], sigma=1.0)
    first = []
    for pol in (policies.Greedy(), policies.Uniform()):
        env_seed, pol_seed = np.random.SeedSequence(3).spawn(2)
        env.reset(4, seed=env_seed)
        pol.reset(2, 4, seed=pol_seed)
        first.append(env.pull(pol.select()))
    assert np.allclose(first[0], first[1])
