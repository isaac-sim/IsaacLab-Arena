Placement Success Benchmark
===========================

The ``isaaclab_arena_benchmarking.object_placement`` package measures how often
relation placement succeeds across the RoboLab and KitchenBench environment
specs, for each spec and as a mean over each suite.

What It Measures
----------------

For each environment spec and each seed in ``--placement_seeds``, the benchmark
builds the environment the normal way, with ``ArenaEnvBuilder.build_env_cfg()``
for ``--num_envs`` environments, without stepping physics. The build applies
any enabled build-time variations and the task's reachability constraints, then
fills its :doc:`placement pool <./pooled_placement>`. The pool solves batches
of candidate layouts for every environment and stores valid ones in the
environments' empty slots, until every environment has enough. If the last
allowed batch gives an environment that still has empty slots no valid
candidate, and best-loss fallback is allowed, as it is by default, those slots
get the batch's best-loss layouts. The benchmark reads the placement
diagnostics the build records and reports:

* **Valid candidates:** the share of candidates in the pool's first batch that
  pass every required :doc:`pre-physics check <./validation>`. The first batch
  always holds ``num_envs * min_unique_layouts_per_env *
  max_placement_attempts`` candidates. A later batch runs only while some
  environment still has empty slots, and its size depends on how many are
  missing, so later batches are left out to keep the sample the same for every
  scene. This rate decides how much solving a build needs.
* **Valid environments:** the share of environments whose pool holds only
  valid layouts, without a best-loss fallback. This is what an environment
  build gets wrong when placement struggles.
* **Batches per build:** how many pool batches a build solved, shown in the
  results table.

Each environment is reported as scored, skipped or failed. Specs whose builds
solve no relations are skipped. Specs that fail to build or place are listed
with their error instead of being dropped; that includes a pool that cannot
find valid layouts and is not allowed to fall back.

When a spec places the robot by relations, as the KitchenBench specs do, the
robot is part of every layout and its relations are checked too. Specs that
place only the robot, such as ``kitchen_bench_lightwheel_open_fridge``, are
scored on the robot's placement.

An environment is one task spec. Each spec and seed builds from its own block
of candidate seeds, so tasks that share a scene, as several RoboLab tasks do,
sample it independently, and a suite's mean counts such a scene once per task.
Rerunning the same specs starts from the same candidates, but an asset that
loads a random variant in each process, such as the Lightwheel broccoli, can
still change results between runs.

Run the Benchmark
-----------------

Run from the repository root inside the Arena container; see
:doc:`../../quickstart/installation`. Start with a quick check that builds one
spec once with two environments. It takes under a minute, and the table it
prints should show one scored RoboLab environment and none failed:

.. code-block:: bash

   python -m isaaclab_arena_benchmarking.object_placement.run_placement_success_benchmark \
     --env banana_on_plate \
     --num_envs 2 \
     --placement_seeds 0 \
     --output outputs/placement_success/quick_check.json

Then run every spec at the defaults. On one RTX 6000 Ada GPU this takes about
50 minutes. The run prints each spec as it starts, then a table with one row
per suite. Its exit status is 0 when no spec failed and 1 otherwise:

.. code-block:: bash

   python -m isaaclab_arena_benchmarking.object_placement.run_placement_success_benchmark \
     --output outputs/placement_success/main.json

With ``--output``, the run writes a results file and rewrites it after every
environment, so a run stopped with Ctrl+C keeps the environments it finished.
The script creates the file's folder, and Git ignores ``outputs/``, so the file
stays in place when you switch branches. The table shows suite rates; each
environment's counts are in the results file.

The simulator sometimes does not exit after printing ``Closing simulation
app``. The results file is complete by then, so the process can be stopped.
For unattended runs, set ``ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE=1`` to skip
the simulator's shutdown; the exit status is unchanged.

Useful options:

* ``--suite robolab`` or ``--suite kitchen_bench`` selects one suite; the
  default, ``--suite all``, runs both.
* ``--env 'kitchen_bench_replicator_*'`` keeps specs whose file stem matches the
  glob. Repeat it to select several patterns.
* ``--num_envs`` and ``--placement_seeds`` set the sample size: one build of
  ``--num_envs`` environments per seed. The defaults are 10 environments and
  seeds 0, 1 and 2, which gives 1500 first-batch candidates per spec with the
  default placer parameters. Seeds must be distinct, and a pool can solve up to
  ``max_placement_attempts`` batches, so ``(seed + 1) * num_envs *
  min_unique_layouts_per_env * max_placement_attempts ** 2`` must not exceed
  ``2**20``, the candidate seeds reserved for each spec. The solver optimizes a
  batch together, so ``--num_envs`` changes individual candidates; compare only
  runs with the same value.
* ``--random_yaw_init`` samples a random initial yaw for placed objects,
  including a robot placed by relations. Objects with ``FaceTo`` or a tilted
  ``RotateAroundSolution`` keep their own orientation rules. Without it, each
  spec's own ``random_yaw_init`` setting applies.
* ``--collision_mode bbox`` or ``--collision_mode mesh`` overrides the specs'
  default collision mode. This can change results sharply: in a quick check
  with ``mesh``, every ``banana_on_plate`` candidate failed the no-overlap
  check.

The Results File
----------------

The results file holds the start time, the commit and branch, whether the
checkout had uncommitted changes, the settings and one record per environment.
Changes count when they are to tracked files or are new files in
``isaaclab_arena``, ``isaaclab_arena_benchmarking`` or
``isaaclab_arena_environments``, such as a new spec; other untracked files,
such as local tool settings, do not. Each record holds the environment's
status, its build, candidate, environment, fallback and batch counts, and its
wall time. For each check that ran, it holds whether the check is required and
how many first-batch candidates the check ran on and passed. A check that runs
only after the cheaper required checks pass, such as inverse kinematics (IK)
reachability, runs only on the candidates that passed them, so its counts
cover only those.

A suite has no rates when any of its environments failed, and the script exits
with an error after writing the file.

When git cannot read the checkout, for example in an image without ``.git``,
the run reads the commit and branch from the ``ARENA_GIT_SHA`` and
``ARENA_GIT_BRANCH`` environment variables. When they are unset or ``unknown``,
as in an image built without its commit, it prints a warning and records no
commit.

``isaaclab_arena_benchmarking/object_placement/results.py`` reads and
summarizes results files without the simulator, so other tools can use them:

.. code-block:: python

   from isaaclab_arena_benchmarking.object_placement.results import (
       PlacementBenchmarkResults,
       summarize_suites,
   )

   results = PlacementBenchmarkResults.load("outputs/placement_success/main.json")
   for summary in summarize_suites(results):
       print(summary.suite, summary.valid_candidate_rate, summary.valid_env_rate)

Add a Suite
-----------

A suite is a glob over environment graph-spec YAML files, listed in
``SUITE_SPEC_GLOBS`` in ``isaaclab_arena_benchmarking/object_placement/measurement.py``;
environments defined only in Python are not supported. To benchmark new scenes,
add their specs and append a suite entry. Each spec's seed block is its
position among the specs of all suites, so a suite appended at the end leaves
the other specs' candidates unchanged, while adding or removing a spec in an
existing suite changes the candidates of the specs after it. A run filtered
with ``--env`` solves the same candidates for its specs as a full run.
