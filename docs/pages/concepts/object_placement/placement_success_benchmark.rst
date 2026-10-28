Placement Success Benchmark
===========================

``isaaclab_arena/scripts/placement_success_benchmark.py`` measures how often
relation placement succeeds across the RoboLab and KitchenBench environment
specs. Run it before and after a change to the placement pipeline, then compare
the two runs to check for regressions.

What It Measures
----------------

For each environment spec and each seed in ``--placement_seeds``, the benchmark
builds the environment the normal way, with ``ArenaEnvBuilder.build_env_cfg()``
for ``--num_envs`` environments, without stepping physics. The build applies
any enabled build-time variations and the task's reachability constraints, then
fills its :doc:`placement pool <./pooled_placement>`: the pool solves batches
of candidate layouts until every environment has enough valid ones, and on its
last attempt gives an environment without any its best-loss layouts. The
benchmark reads the placement diagnostics the build records and reports:

* **Valid candidates:** the share of candidates in the pool's first batch that
  pass every required :doc:`pre-physics check <./validation>`. The first batch
  always holds ``num_envs * min_unique_layouts_per_env *
  max_placement_attempts`` candidates; later batches only refill the
  environments that lacked valid layouts, so they are left out to keep the
  sample the same for every scene. This rate decides how much solving a build
  needs.
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
sample it independently, and a suite's mean counts such a scene once per task. Rerunning
the same specs starts from the same candidates, but an asset that loads a
random variant in each process, such as the Lightwheel broccoli, can still
change results between runs.

Run the Benchmark
-----------------

Run from the repository root inside the Arena container; see
:doc:`../../quickstart/installation`. Start with a quick check that builds one
spec once with two environments. It takes under a minute, and the table it
prints should show one scored RoboLab environment and none failed:

.. code-block:: bash

   python isaaclab_arena/scripts/placement_success_benchmark.py \
     --env banana_on_plate \
     --num_envs 2 \
     --placement_seeds 0 \
     --output outputs/placement_success/quick_check.json

Then run every spec at the defaults. On one RTX 6000 Ada GPU this takes about
50 minutes. The run prints each spec as it starts, then a table with one row
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

The comparison first names each run's commit and warns about environments
that failed after the change. Each suite row then counts the environments
compared and the ones that were not, such as environments that failed in either
run, and reports the suite's rates in both runs with a 95% interval for the
change, after minus before, in percentage points. A negative change is a
regression, and an interval that contains zero is inconclusive. At the
defaults, a suite's interval is about ±0.5 points for valid candidates and
±1.5 points for valid environments.

The interval treats the set of specs as fixed and covers only the noise of
sampling their candidates. Both runs start from the same candidates, so for a
change that leaves the solver alone, such as a new validator, the two runs are
correlated and the interval is wider than it needs to be. Candidates in a
batch are solved together, which can make it somewhat too narrow.

The comparison then lists the environments whose change is significant, and
the environments that were not scored in both runs. With the 80 specs at the
defaults, a single environment is flagged once its valid-candidate rate changes
by about 6 points. Valid environments rest on 30 environments per spec, so a
single one is flagged only after a drop of about 40 points, and that rate is
mainly a suite-level guard against large regressions. Testing every environment
and rate at 5% would flag a few by chance, so the Holm adjustment lowers each
threshold until the chance of flagging any unchanged environment is at most
5%. Each environment's interval uses the Agresti-Caffo adjustment, which adds
one success and one failure to each run so that rates of 0% and 100% still get
a sensible interval. That interval is therefore not always centered on the
observed change, while a suite's interval is.

The comparison also warns when the two runs used different settings, and
names the environments whose required checks differ, since their success
criteria differ. Those environments stay in the suite rates, because a change
to the checks is often the change being measured. A check that one image lacks,
such as inverse kinematics (IK) reachability without cuRobo, also shows up
here.

``--compare`` only reports; its exit status does not signal a regression.

Regression Test
---------------

``isaaclab_arena/tests/test_placement_success_benchmark.py`` runs a short
version of the benchmark with the other tests in CI. It builds three
representative specs once with four environments: an easy RoboLab task, a
crowded bin and a spec that places only the robot. It compares them with
``isaaclab_arena/tests/test_data/placement_success_baseline.json``, and fails
when a spec fails, gains a best-loss fallback or its valid-candidate rate falls
more than 10 points below the baseline. The same seeds solve the same
candidates, so the tolerance only absorbs GPU differences and random asset
variants; smaller changes still need a full comparison.

After a change that is meant to move these rates, regenerate the baseline with
the benchmark and commit it with the change:

.. code-block:: bash

   python isaaclab_arena/scripts/placement_success_benchmark.py \
     --env banana_on_plate \
     --env mustard_in_left_bin \
     --env kitchen_bench_lightwheel_open_fridge \
     --num_envs 4 \
     --placement_seeds 0 \
     --output isaaclab_arena/tests/test_data/placement_success_baseline.json

The Results File
----------------

The results file holds the start time, the commit and branch, whether the
checkout had uncommitted changes, the settings and one record per environment.
Changes count when they are to tracked files or are new files in
``isaaclab_arena`` or ``isaaclab_arena_environments``, such as a new spec;
other untracked files, such as local tool settings, do not. Each record holds
the environment's status, its build, candidate, environment, fallback and batch
counts, its wall time, the checks that ran, with whether each is required, and
how many first-batch candidates each check ran on and rejected. A check that
runs only after the cheaper required checks pass, such as IK reachability, sees
only the candidates that passed them, so its rejections count out of those
candidates and are not charged with the ones it skipped.

A suite has no rates when any of its environments failed, and the script exits
with an error after writing the file.

When git cannot read the checkout, for example in an image without ``.git``,
the run reads the commit and branch from the ``ARENA_GIT_SHA`` and
``ARENA_GIT_BRANCH`` environment variables, if they are set; otherwise it prints
a warning and records no commit.

``isaaclab_arena/offline_placement/placement_success.py`` reads, summarizes and
compares results files without the simulator, so other tools can use them:

.. code-block:: python

   from isaaclab_arena.offline_placement.placement_success import (
       PlacementBenchmarkResults,
       summarize_suites,
   )

   results = PlacementBenchmarkResults.load("outputs/placement_success/before.json")
   for summary in summarize_suites(results):
       print(summary.suite, summary.valid_candidate_rate, summary.valid_env_rate)

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
