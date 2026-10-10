"""Published bounds as data: formula, constants, source, assumptions.

Policies list the results about them in ``Policy.theory`` and evaluate the
applicable ones with ``Policy.bounds(env, horizon)``; problem-level lower bounds
come from ``lower_bounds(env, horizon)``.
"""
from . import bounds
from .bounds import (ASYMPTOTIC, GAP, HIGH_PROB, LOWER, MINIMAX, ORDER, UPPER, Bound, gaps, kl_bernoulli,
                     oful_radius)
from .references import REFERENCES, Reference


def lower_bounds(env, horizon):
    """Lower bounds on the regret of any (consistent) policy on ``env``, as ``(bound, value)`` pairs.

    ``env`` must have been reset. Lai-Robbins values are C log n, an asymptotic
    statement rather than a bound at finite n.
    """
    from ..envs import BernoulliBandit, GaussianBandit

    if getattr(env, "changes", ()):
        return []  # the stationary lower bounds do not apply to a piecewise-stationary problem
    found = []
    if isinstance(env, BernoulliBandit):
        found.append(bounds.LAI_ROBBINS_BERNOULLI)
    if isinstance(env, GaussianBandit):
        found.append(bounds.LAI_ROBBINS_GAUSSIAN)
        if env.sigma == 1.0:
            found.append(bounds.MINIMAX_LOWER)
    problem = {**env.info(), "horizon": horizon}
    return [(b, b.evaluate(**problem)) for b in found]


__all__ = ["Bound", "Reference", "REFERENCES", "bounds", "lower_bounds", "gaps", "kl_bernoulli", "oful_radius",
           "UPPER", "LOWER", "GAP", "MINIMAX", "ASYMPTOTIC", "HIGH_PROB", "ORDER"]
