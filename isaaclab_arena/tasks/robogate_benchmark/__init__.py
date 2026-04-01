"""RoboGate Benchmark for Isaac Lab-Arena.

68-scenario adversarial pick-and-place validation suite with 5 safety
metrics and deployment confidence scoring (0-100).

Usage with ArenaEnvBuilder::

    from robogate_benchmark.environments import RoboGateBenchmarkEnvironment
    env_def = RoboGateBenchmarkEnvironment()
    arena_env = env_def.get_env(args_cli)

Usage standalone::

    python -m scripts.run_benchmark --embodiment franka --config configs/robogate_68.yaml
"""

__version__ = "1.0.0"
__author__ = "Byungjin Kim"
