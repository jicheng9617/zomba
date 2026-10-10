"""Simulated environments."""
from .linear import LinearBandit
from .mab import BernoulliBandit, GaussianBandit, KArmedBandit, StudentTBandit, repeat_instances

__all__ = ["KArmedBandit", "BernoulliBandit", "GaussianBandit", "StudentTBandit", "repeat_instances",
           "LinearBandit"]
