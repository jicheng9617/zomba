"""zomba: bandit algorithms with their theory attached.

Simulate policies in batches of independent runs, and compare what they do with
what the published bounds say::

    from zomba import envs, policies, simulate

    env = envs.BernoulliBandit([0.9, 0.8])
    res = simulate(policies.KLUCB(c=0), env, horizon=10_000, n_runs=500, seed=0)

The pre-0.2 API (``zomba.core`` agents, ``zomba.singleObjective``) lives in
``zomba.legacy``; ``zomba.multiObjective`` still builds on it.
"""
__version__ = "0.2.0.dev0"

from . import envs, policies, theory
from .core import Env, Policy, Result, simulate

__all__ = ["Env", "Policy", "Result", "simulate", "envs", "policies", "theory", "__version__"]
