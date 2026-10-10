"""The pre-0.2 API, kept so existing code keeps running: ``zomba.legacy.core`` agents and
environments and the ``zomba.legacy.singleObjective`` algorithms and simulators.

``zomba.multiObjective`` still builds on these classes. New code should use
``zomba.core``, ``zomba.policies`` and ``zomba.envs``. Known issues are left as
they were (for example, ``upperConfidenceBound.uncertainty`` grows with the
number of pulls, and ``ThompsonSampling.reset`` does not reset ``counts``).
"""
