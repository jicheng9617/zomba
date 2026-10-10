import numpy as np
import pytest

from zomba import envs, policies as P, simulate
from zomba.theory import oful_radius


def _feed(pol, env, rounds):
    for _ in range(rounds):
        X = env.observe()
        a = pol.select(X)
        pol.update(a, env.pull(a), X)


def test_ridge_statistics_match_direct_computation():
    env = envs.LinearBandit(8, 4, noise=0.5).reset(3, seed=0)
    pol = P.OFUL(lam=2.0).reset(8, 3, dim=4, seed=1)
    X0 = env.observe()
    xs, rs = [], []
    for _ in range(1500):  # crosses the periodic re-inversion at 1000
        X = env.observe()
        a = pol.select(X)
        r = env.pull(a)
        pol.update(a, r, X)
        xs.append(X[np.arange(3), a])
        rs.append(r)
    xs, rs = np.stack(xs, 1), np.stack(rs, 1)
    for b in range(3):
        V = 2.0 * np.eye(4) + xs[b].T @ xs[b]
        assert np.allclose(pol.V[b], V)
        assert np.allclose(pol.Vinv[b], np.linalg.inv(V), atol=1e-10)
        assert np.allclose(pol.theta[b], np.linalg.solve(V, xs[b].T @ rs[b]))
        assert pol.logdet[b] == pytest.approx(np.linalg.slogdet(V)[1] - 4 * np.log(2.0))
    assert X0.shape == (3, 8, 4)


def test_oful_index_and_radius():
    env = envs.LinearBandit(6, 3).reset(2, seed=0)
    pol = P.OFUL(noise=0.5, S=2.0, lam=1.5, delta=0.05, radius_scale=0.3).reset(6, 2, dim=3, seed=0)
    _feed(pol, env, 20)
    X = env.observe()
    rad = oful_radius(pol.logdet, 0.5, 1.5, 2.0, 0.05)
    assert np.allclose(pol.radius(), rad)
    w = np.sqrt(np.einsum("bkd,bde,bke->bk", X, np.linalg.inv(pol.V), X))
    expected = np.einsum("bkd,bd->bk", X, pol.theta) + 0.3 * rad[:, None] * w
    assert np.allclose(pol.index(X), expected)
    assert np.allclose(P.LinGreedy().reset(6, 2, dim=3).index(X), 0.0)


def test_lints_inflation_settings():
    n, d, delta = 10000, 10, 0.01
    v = P.LinTS(inflation="icml2013").reset(5, 1, horizon=n, dim=d).v
    assert v == pytest.approx(np.sqrt(24 * np.log(n) * d * np.log(1 / delta)))
    assert P.LinTS(inflation="arxiv-v4").reset(5, 1, horizon=n, dim=d).v == pytest.approx(np.sqrt(9 * d * np.log(n / delta)))
    assert P.LinTS(noise=0.3, inflation="bayes").reset(5, 1, dim=d).v == pytest.approx(0.3)
    assert P.LinTS(inflation=2.5).reset(5, 1, dim=d).v == 2.5
    with pytest.raises(ValueError):
        P.LinTS(inflation="icml2013").reset(5, 1, dim=d)
    with pytest.raises(ValueError):
        P.OFUL().reset(5, 1)


def test_linear_policies_learn():
    env, n = envs.LinearBandit(10, 3, noise=0.3), 1500
    uniform = simulate(P.Uniform(), env, n, n_runs=16, seed=0).mean()[-1]
    for pol in (P.OFUL(noise=0.3), P.OFUL(noise=0.3, radius_scale=0.3), P.LinUCB(alpha=0.5), P.LinTS(noise=0.3, inflation="bayes")):
        assert simulate(pol, env, n, n_runs=16, seed=0).mean()[-1] < 0.3 * uniform, pol
