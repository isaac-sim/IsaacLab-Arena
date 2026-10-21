Placement Success Benchmark
===========================

``isaaclab_arena/scripts/placement_success_benchmark.py`` measures how often
relation placement succeeds across the RoboLab and KitchenBench environment
specs. Run it before and after a change to the placement pipeline, then compare
the two runs to check for regressions.

What It Measures
----------------

For each environment spec, the benchmark builds the environment's placement
inputs the same way ``ArenaEnvBuilder`` does, including build-time variations
and task reachability constraints. It then places ``--num_layouts`` layouts for
every seed in ``--placement_seeds``, without stepping physics. Each seed solves
its own candidates. It reports two rates:

* **Layout success:** the share of layouts whose best candidate passes every
  required :doc:`pre-physics check <./validation>`. Each layout keeps the best of
  ``max_placement_attempts`` candidates, as ``ObjectPlacer.place()`` does. A
  full environment build fills its :doc:`placement pool <./pooled_placement>`
  differently, with retries and best-loss fallbacks, so this rate measures the
  solver and validators rather than the pool.
* **Valid candidates:** the share of all solved candidates that pass every
  required check. Best-of-candidates selection hides moderate changes, so this
  rate moves before layout success does.

Each environment is reported as scored, skipped or failed. Specs that place
only anchors and the robot are skipped. Specs that fail to build or place are
listed with their error instead of being dropped.

When a spec places the robot by relations, as the KitchenBench specs do, the
robot is part of every layout and its relations are checked too.

An environment is one task spec. Several RoboLab tasks share a scene, so they
place identically, and a suite's mean counts such a scene once per task.

Run the Benchmark
-----------------

Run inside the Arena container. With ``--output``, the run writes a KPI file
and rewrites it after every environment, so a stopped run keeps its results:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/placement_success_benchmark.py \
     --suite all \
     --output placement_main.json

Useful options:

* ``--suite robolab``, ``--suite kitchen_bench`` or ``--suite all`` selects the
  suite.
* ``--env 'kitchen_bench_replicator_*'`` keeps specs whose file stem matches the
  glob. Repeat it to select several patterns.
* ``--num_layouts`` and ``--placement_seeds`` set the sample size. The defaults
  are 50 layouts and seeds 0, 1 and 2.
* ``--random_yaw_init`` samples a random initial yaw for placed objects,
  including a robot placed by relations. Objects with ``FaceTo`` or a tilted
  ``RotateAroundSolution``, and ``ClutterOn`` objects, keep their own
  orientation rules. Without it, spec objects keep their fixed orientations.
* ``--collision_mode bbox`` or ``--collision_mode mesh`` overrides the default
  collision mode.

Compare Two Runs
----------------

Run the benchmark on the baseline and on the change, for example from two
checkouts, then compare the KPI files. Comparing does not start the simulator:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/placement_success_benchmark.py \
     --compare placement_main.json placement_change.json

The comparison reports each suite's rates in both runs over the environments
scored in both, with a 95% interval for the change. The intervals use the
Agresti-Caffo adjustment, so they need not be centered on the observed change.
It then lists the environments whose change stays significant after a Holm
adjustment over every environment and rate, and the environments that were not
scored in both runs. It warns when the two runs used different settings.

The KPI File
------------

The output follows the Arena KPI-tracking layout: a format version, the start
time, the commit and branch, and a flat list of KPI records. Each record has a
name, a value and tags for the suite and, where relevant, the environment.
Per-environment counts, failed check names and timings are stored alongside for
comparisons.

A suite has no rate records when any of its environments failed, and the
script exits with an error after writing the file.

The file also records whether the checkout had uncommitted changes. Only the
run's own output file is ignored, so keep other KPI files outside the checkout.
