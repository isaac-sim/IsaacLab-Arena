Placement Success Benchmark
===========================

``isaaclab_arena/scripts/placement_success_benchmark.py`` measures how often
relation placement succeeds across the RoboLab and KitchenBench environment
specs. Run it before and after a change to the placement pipeline, then compare
the two runs to check for regressions.

What It Measures
----------------

For each environment spec, the benchmark builds the environment's placement
inputs the same way ``ArenaEnvBuilder`` does, applying any enabled build-time
variations and the task's reachability constraints. It then places
``--num_layouts`` layouts for every seed in ``--placement_seeds``, without
stepping physics. It reports two rates:

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
nothing other than anchors are skipped. Specs that fail to build or place are
listed with their error instead of being dropped.

When a spec places the robot by relations, as the KitchenBench specs do, the
robot is part of every layout and its relations are checked too. Specs that
place only the robot, such as ``kitchen_bench_lightwheel_open_fridge``, are
scored on the robot's placement.

An environment is one task spec. Each environment and seed solves its own
candidates, so tasks that share a scene, as several RoboLab tasks do, sample it
independently, and a suite's mean counts such a scene once per task. Rerunning
the same specs starts from the same candidates, but an asset that loads a
random variant in each process, such as the Lightwheel broccoli, can still
change results between runs.

Run the Benchmark
-----------------

Run from the repository root inside the Arena container; see
:doc:`../../quickstart/installation`. Start with a quick check that places one
spec with few layouts. It takes under a minute, and the table it prints should
show one scored RoboLab environment and none failed:

.. code-block:: bash

   python isaaclab_arena/scripts/placement_success_benchmark.py \
     --env banana_on_plate \
     --num_layouts 5 \
     --placement_seeds 0 \
     --output outputs/placement_success/quick_check.json

Then run every spec at the defaults. On one RTX 6000 Ada GPU this takes about
25 minutes. The run prints each spec as it starts, then a table with one row
per suite. Its exit status is 0 when no spec failed and 1 otherwise:

.. code-block:: bash

   python isaaclab_arena/scripts/placement_success_benchmark.py \
     --output outputs/placement_success/before.json

With ``--output``, the run writes a results file and rewrites it after every
environment, so a run stopped with Ctrl+C keeps the environments it finished.
The script creates the file's folder, and Git ignores ``outputs/``. The table
shows suite rates; each environment's counts are in the results file.

The simulator sometimes does not exit after printing ``Closing simulation
app``. The results file is complete by then, so the process can be stopped.
For unattended runs, set ``ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE=1`` to skip
the simulator's shutdown; the exit status is unchanged.

Useful options:

* ``--suite robolab`` or ``--suite kitchen_bench`` selects one suite; the
  default, ``--suite all``, runs both.
* ``--env 'kitchen_bench_replicator_*'`` keeps specs whose file stem matches the
  glob. Repeat it to select several patterns.
* ``--num_layouts`` and ``--placement_seeds`` set the sample size. The defaults
  are 50 layouts and seeds 0, 1 and 2. Seeds must be distinct, and
  ``(seed + 1) * num_layouts * max_placement_attempts`` must not exceed
  ``2**20``, the candidate seeds reserved for each environment.
  ``max_placement_attempts`` comes from the spec's placer parameters and is 10
  by default. The solver optimizes a seed's layouts as one batch, so
  ``--num_layouts`` changes individual candidates; compare only runs with the
  same value.
* ``--random_yaw_init`` samples a random initial yaw for placed objects,
  including a robot placed by relations. Objects with ``FaceTo`` or a tilted
  ``RotateAroundSolution`` keep their own orientation rules. Without it, spec
  objects keep their fixed orientations.
* ``--collision_mode bbox`` or ``--collision_mode mesh`` overrides the specs'
  default collision mode. This can change results sharply: in a quick check
  with ``mesh``, every ``banana_on_plate`` candidate failed the no-overlap
  check.

Compare Two Runs
----------------

Run the benchmark before and after the change, for example on ``main`` and on
your branch. The first results file stays in place when you switch branches,
because Git ignores ``outputs/``:

.. code-block:: bash

   git checkout main
   python isaaclab_arena/scripts/placement_success_benchmark.py \
     --output outputs/placement_success/before.json
   git checkout <your-branch>
   python isaaclab_arena/scripts/placement_success_benchmark.py \
     --output outputs/placement_success/after.json

Then compare the two results files. Comparing does not start the simulator:

.. code-block:: bash

   python isaaclab_arena/scripts/placement_success_benchmark.py \
     --compare outputs/placement_success/before.json outputs/placement_success/after.json

The comparison first names each run's commit, then reports each suite's rates
in both runs over the environments scored in both, with a 95% interval for the
change, after minus before, in percentage points. A negative change is a
regression, and an interval that contains zero is inconclusive. Two full runs
of the same code gave suite changes of zero, with intervals of about ±1 point
for layout success and ±0.5 points for valid candidates. A random asset
variant, as noted above, can still move a single environment slightly.

The interval treats the set of specs as fixed and covers only the noise of
sampling their candidates. It also treats candidates as independent, although
the solver optimizes a seed's layouts as one batch and stops them together, so
it may be somewhat too narrow.

The comparison then lists the environments whose change is significant, and
the environments that were not scored in both runs. Testing every environment
and rate at 5% would flag a few by chance, so the Holm adjustment lowers each
threshold until the chance of flagging any unchanged environment is at most
5%. Each environment's interval uses the Agresti-Caffo adjustment, which adds
one success and one failure to each run so that rates of 0% and 100% still get
a sensible interval. That interval is therefore not always centered on the
observed change, while a suite's interval is.

The comparison warns when the two runs used different settings. It also names
the environments where a check required in only one run rejected candidates in
that run, and prints their rates without intervals. Their success criteria
differ, so they are left out of the suite rates and significance tests. A
missing or optional check does not change success, and neither does a required
check that never rejected a candidate.

``--compare`` only reports; its exit status does not signal a regression.

The Results File
----------------

The results file holds the start time, the commit and branch, whether the
checkout had uncommitted changes, the settings and one record per environment.
Changes count when they are to tracked files or are new files in
``isaaclab_arena`` or ``isaaclab_arena_environments``, such as a new spec;
other untracked files, such as local tool settings, do not. Each record holds
the environment's status, its layout and candidate counts, its wall time, the
checks that ran, with whether each is required, and how many candidates each
check rejected. A check that
runs only on candidates that pass the cheaper required checks, such as inverse
kinematics (IK) reachability, counts as rejecting the candidates it skipped.

A suite has no rates when any of its environments failed, and the script exits
with an error after writing the file.

When git cannot read the checkout, for example in an image without ``.git``,
the run prints a warning and records no commit.

``isaaclab_arena/offline_placement/placement_success.py`` reads, summarizes and
compares results files without the simulator, so other tools can use them:

.. code-block:: python

   from isaaclab_arena.offline_placement.placement_success import (
       PlacementBenchmarkResults,
       summarize_suites,
   )

   results = PlacementBenchmarkResults.load("outputs/placement_success/before.json")
   for summary in summarize_suites(results):
       print(summary.suite, summary.layout_success_rate, summary.valid_candidate_rate)

Add a Suite
-----------

A suite is a glob over environment graph-spec YAML files, listed in
``SUITE_SPEC_GLOBS`` in the script; environments defined only in Python are not
supported. To benchmark new scenes, add their specs and append a suite entry.
Each spec's seed block is its position among the specs of all suites, so a suite
appended at the end leaves the other specs' candidates unchanged, while adding
or removing a spec in an existing suite changes the candidates of the specs
after it. A run filtered with ``--env`` solves the same candidates for its specs
as a full run.
