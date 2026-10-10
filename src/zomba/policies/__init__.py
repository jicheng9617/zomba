"""Bandit policies. Defaults are the analysed versions; see each class for practical settings."""
from .linear import OFUL, LinGreedy, LinTS, LinUCB
from .mab import (MOSS, UCB, UCB1, AsymptoticUCB, EpsilonGreedy, ExploreThenCommit, GaussianThompsonSampling,
                  Greedy, KLUCB, ThompsonSampling, Uniform, klucb_bernoulli)

__all__ = ["UCB1", "UCB", "AsymptoticUCB", "MOSS", "KLUCB", "ThompsonSampling", "GaussianThompsonSampling",
           "ExploreThenCommit", "EpsilonGreedy", "Greedy", "Uniform", "klucb_bernoulli",
           "OFUL", "LinUCB", "LinGreedy", "LinTS"]
