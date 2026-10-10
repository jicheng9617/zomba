import numpy as np
import pytest

from zomba import envs


def test_bernoulli_rewards_and_regret():
    env = envs.BernoulliBandit([0.8, 0.3]).reset(20000, seed=0)
    r = env.pull(np.zeros(20000, dtype=int))
    assert set(np.unique(r)) <= {0.0, 1.0}
    assert r.mean() == pytest.approx(0.8, abs=0.01)
    assert np.allclose(env.regret(np.zeros(20000, dtype=int)), 0.0)
    assert np.allclose(env.regret(np.ones(20000, dtype=int)), 0.5)


def test_bernoulli_rejects_bad_means():
    with pytest.raises(ValueError):
        envs.BernoulliBandit([0.5, 1.2])


def test_gaussian_and_student_t_noise_scale():
    arms = np.zeros(200000, dtype=int)
    g = envs.GaussianBandit([1.0, 0.0], sigma=2.0).reset(200000, seed=1).pull(arms)
    assert g.mean() == pytest.approx(1.0, abs=0.02) and g.std() == pytest.approx(2.0, rel=0.01)
    t = envs.StudentTBandit([0.0, 0.0], df=8, sigma=1.5).reset(200000, seed=2).pull(arms)
    assert t.std() == pytest.approx(1.5, rel=0.03)
    with pytest.raises(ValueError):
        envs.StudentTBandit([0.0], df=2)


def test_per_run_instances():
    means = envs.repeat_instances([[0.9, 0.1], [0.2, 0.6]], runs_per_instance=3)
    env = envs.BernoulliBandit(means).reset(6, seed=0)
    assert np.allclose(env.regret(np.zeros(6, dtype=int)), [0, 0, 0, 0.4, 0.4, 0.4])
    with pytest.raises(ValueError):
        envs.BernoulliBandit(means).reset(5)


def test_change_points():
    env = envs.BernoulliBandit([0.9, 0.1], changes=[(3, [0.1, 0.9])]).reset(1, seed=0)
    regrets = []
    for _ in range(5):
        regrets.append(env.regret(np.array([0]))[0])
        env.pull(np.array([0]))
    assert regrets == pytest.approx([0, 0, 0, 0.8, 0.8])


def test_linear_bandit_fixed_and_fresh():
    env = envs.LinearBandit(n_arms=5, dim=3, noise=0.0).reset(4, seed=0)
    X1, X2 = env.observe(), env.observe()
    assert X1.shape == (4, 5, 3) and np.array_equal(X1, X2)
    assert np.allclose(np.linalg.norm(X1, axis=-1), 1.0)
    best = env.mean_rewards().argmax(axis=1)
    assert np.allclose(env.regret(best), 0.0)
    assert np.allclose(env.pull(best), env.mean_rewards()[np.arange(4), best])

    fresh = envs.LinearBandit(n_arms=5, dim=3, fresh_contexts=True).reset(2, seed=0)
    assert not np.array_equal(fresh.observe(), fresh.observe())

    theta = np.array([1.0, 0.0])
    arms = np.array([[1.0, 0.0], [0.0, 1.0]])
    env = envs.LinearBandit(2, 2, theta=theta, arms=arms).reset(3, seed=0)
    env.observe()
    assert np.allclose(env.mean_rewards(), [[1.0, 0.0]] * 3)
    assert env.info()["theta_norm_bound"] == pytest.approx(1.0)
