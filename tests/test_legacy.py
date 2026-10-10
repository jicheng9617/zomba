import importlib

import pytest


def test_legacy_core_imports():
    importlib.import_module("zomba.legacy.core")
    importlib.import_module("zomba.legacy.core.simulator")


@pytest.mark.parametrize("module", [
    "zomba.legacy.singleObjective.mab",
    "zomba.legacy.singleObjective.simulator",
    "zomba.legacy.singleObjective.stochastic",
    "zomba.legacy.singleObjective.applications.webRec",
])
def test_legacy_single_objective_imports(module):
    for dep in ("mpire", "matplotlib"):  # the package __init__ imports its experiment helpers
        pytest.importorskip(dep)
    importlib.import_module(module)


def test_multi_objective_still_imports():
    for dep in ("torch", "scipy", "sklearn", "networkx", "mpire", "matplotlib"):
        pytest.importorskip(dep)
    importlib.import_module("zomba.multiObjective.simulators")
    importlib.import_module("zomba.multiObjective.algorithms")
    importlib.import_module("zomba.multiObjective.algorithms.stochastic")
