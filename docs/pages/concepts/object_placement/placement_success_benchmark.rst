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
every seed in ``--placement_seeds``, without stepping physics. It reports two
rates:

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

An environment is one task spec. Each environment and seed solves its own
candidates, so tasks that share a scene, as several RoboLab tasks do, sample it
independently, and a suite's mean counts such a scene once per task. Rerunning
the same specs starts from the same candidates, but an asset that loads a
random variant in each process, such as the Lightwheel broccoli, can still
change results between runs. Adding or removing a spec changes the candidates
of the specs after it.

Run the Benchmark
-----------------

Run from the repository root inside the Arena container; see
:doc:`../../quickstart/installation`. With ``--output``, the run writes a results
file and rewrites it after every environment, so a stopped run keeps its
results. Write results outside the checkout, since the file records whether the
checkout had uncommitted changes:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/placement_success_benchmark.py \
     --suite all \
     --output /tmp/placement_main.json

For a quick check, place one spec with fewer layouts. The table it prints
should show one scored RoboLab environment and none failed:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/placement_success_benchmark.py \
     --env banana_on_plate \
     --num_layouts 5 \
     --placement_seeds 0 \
     --output /tmp/placement_quick_check.json

Useful options:

* ``--suite robolab``, ``--suite kitchen_bench`` or ``--suite all`` selects the
  suite.
* ``--env 'kitchen_bench_replicator_*'`` keeps specs whose file stem matches the
  glob. Repeat it to select several patterns.
* ``--num_layouts`` and ``--placement_seeds`` set the sample size. The defaults
  are 50 layouts and seeds 0, 1 and 2. Seeds must be distinct, and
  ``(seed + 1) * num_layouts * max_placement_attempts`` must not exceed
  ``2**20``, the candidate seeds reserved for each environment. The solver
  optimizes a seed's layouts as one batch, so ``--num_layouts`` changes
  individual candidates; compare only runs with the same value.
* ``--random_yaw_init`` samples a random initial yaw for placed objects,
  including a robot placed by relations. Objects with ``FaceTo`` or a tilted
  ``RotateAroundSolution``, and ``ClutterOn`` objects, keep their own
  orientation rules. Without it, spec objects keep their fixed orientations.
  KitchenBench places the robot next to a fixture, so turning the robot
  dominates its failures under this option.
* ``--collision_mode bbox`` or ``--collision_mode mesh`` overrides the default
  collision mode.

Compare Two Runs
----------------

Run the benchmark on the baseline and on the change, for example from two
checkouts, then compare the results files. Comparing does not start the
simulator:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/placement_success_benchmark.py \
     --compare /tmp/placement_main.json /tmp/placement_change.json

The comparison first names each run's commit, then reports each suite's rates
in both runs over the environments
scored in both, with a 95% interval for the change in percentage points. The
interval covers the sampling noise of these specs. A negative change is a
regression, and an interval that contains zero is inconclusive. The intervals
use the Agresti-Caffo adjustment, so they need not be centered on the observed
change.

It then lists the environments whose change stays significant after a Holm
adjustment over every environment and rate, and the environments that were not
scored in both runs. It warns when the two runs used different settings. It
also names the environments where a check that rejected candidates in either
run is missing from the other run or differs in being required. Their success
criteria changed, so they are left out of the suite rates and significance
tests. A check that never rejected a candidate does not change success, so
adding one does not exclude an environment.

``--compare`` only reports; its exit status does not signal a regression.

The Results File
----------------

The results file holds the start time, the commit and branch, the settings, a
flat list of metrics and one record per environment. Each metric has a name, a
value and tags for the suite and, where relevant, the environment.
Per-environment counts, failed check names, timings, the checks that ran, with
whether each is required, and how many candidates each check rejected are
stored alongside for comparisons.

A suite has no rate metrics when any of its environments failed, and the
script exits with an error after writing the file.

The file also records whether the checkout had uncommitted changes, ignoring
only the run's own output file. When git cannot read the checkout, for example
in an image without ``.git``, the run prints a warning and records no commit.
