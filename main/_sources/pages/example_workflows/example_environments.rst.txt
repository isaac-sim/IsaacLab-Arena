Example Environments
====================

Isaac Lab Arena ships a catalog of ready-to-run environments under
``isaaclab_arena_environments/``. Environments can be provided in two ways:

* **Python registered environments**: small compositions of the building blocks
  introduced in :doc:`../concepts/environment/index` — **Scene**,
  **Embodiment**, and **Task** — wrapped in an ``ExampleEnvironmentBase``
  subclass and registered with the global ``EnvironmentRegistry``. The
  registered ``Task ID`` is passed as the positional ``example_environment``
  argument to scripts such as ``isaaclab_arena/evaluation/policy_runner.py``.
* **Environment graph YAML specs**: ``ArenaEnvGraphSpec`` files that describe the same
  scene, embodiment, task, objects, and relations declaratively. These are
  passed with ``--env_spec`` and can be generated from prompts by the
  :doc:`agentic_env_gen/index` workflow.

The environments are grouped into three catalogs:

Robolab-Inspired Benchmark
--------------------------

RoboLab environment graph YAMLs live under
``isaaclab_arena_environments/robolab/``. They are generated from natural-language
prompts and consumed with ``--env_spec`` instead of the positional
``example_environment`` name.

See :doc:`robolab_task_catalog` for the list of RoboLab tasks
currently supported in Arena.

Kitchen Benchmark
-----------------

Kitchen benchmark environment graph YAMLs live under
``isaaclab_arena_environments/kitchen_bench/``. They define DROID manipulation
tasks across Lightwheel RoboCasa and Replicator kitchen layouts.

See :doc:`kitchen_bench_catalog` for all 21 environment specs and their Pi
policy executions.

Python Environment Catalog
--------------------------

Python registered environments wrapped in an ``ExampleEnvironmentBase`` subclass
and consumed via the positional ``example_environment`` name. They span
pick-and-place, articulated-object manipulation, sorting, assembly,
goal-pose / lift (RL), sandbox, and sequential / composite tasks.

See :doc:`python_environment_catalog` for the full list with per-environment
Key Specifications tables.



See Also
--------

- :doc:`../concepts/environment/index` — the Scene / Embodiment / Task building blocks used by every environment listed here.
- :doc:`../quickstart/arena_env` — walkthrough of the ``pick_and_place_maple_table`` environment.
- :doc:`../arena_in_your_repo/index` — how to register your own ``ExampleEnvironmentBase`` subclass alongside the built-in ones.

.. toctree::
   :maxdepth: 1
   :hidden:

   robolab_task_catalog
   kitchen_bench_catalog
   python_environment_catalog
