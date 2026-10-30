Placement Success Benchmark
===========================

The placement success benchmark measures how often relation placement finds
valid object layouts for the RoboLab and KitchenBench environment specs. Each
suite is an :doc:`Arena Experiment <../concept_arena_experiments>` with one Run
per spec. A Run builds its spec, which places the objects, and writes its
placement diagnostics to its output directory. For each Run, and as a mean over
the Experiment, the benchmark reports how many candidate layouts were valid and
how many environments got only valid layouts.

What It Measures
----------------

An environment build places the objects that have relations before physics
starts. Each of the ``num_envs`` parallel environments has
``min_unique_layouts_per_env`` slots, each holding one valid layout, so that its
resets can start from different layouts. The build fills the slots in batches.
A batch solves candidate layouts for every environment, runs the
:doc:`pre-physics checks <./validation>` on each, and stores each environment's
valid candidates in its own empty slots. Another batch runs only while some
environment still has empty slots, up to ``max_placement_attempts`` batches; see
:doc:`./pooled_placement`.

When the last batch gives an environment no valid candidate, its empty slots get
that batch's lowest-loss candidates, which fail at least one check. This
best-loss fallback is on by default. When the last batch gives an environment
some valid candidates but too few, or none while fallback is off, the build
fails.

The experiment runner writes the placement diagnostics of each build that solves
relation placement to ``placement_diagnostics_rebuild<i>.json`` in the Run
directory, whatever the Run's policy. The file holds the number of environments,
the environments that fell back, and for each batch the number of candidates,
the number of valid ones, and how many candidates each check ran on and passed,
which shows which check rejects candidates. A build that fails, or that replays
recorded layouts, writes none. From these files the benchmark reports for each
Run:

* **Valid candidates** (``valid_candidate_rate_first_batch``): the share of
  candidates in the first batch that pass every required check. The first batch
  always holds ``num_envs * min_unique_layouts_per_env *
  max_placement_attempts`` candidates. A later batch's size depends on how many
  slots are still empty, so later batches are left out to keep the sample the
  same for every scene. This rate moves when the solver gets better or worse.
* **Valid environments** (``valid_env_rate_after_retries``): the share of
  environments that needed no best-loss fallback once the last batch ran. It
  drops below 100% only when the last batch gives an environment no valid
  candidate.
* **Batches per build** (``batches_per_build``): the mean number of batches a
  build solved.

The Experiment's rates are the means over its Runs, with every Run weighted
equally. Several RoboLab tasks share a scene, and the mean counts such a scene
once per task. A Run that failed, for example because its build failed, leaves
the Experiment without rates, since it could have lowered them.

When a spec places the robot by relations, as the KitchenBench specs do, the
robot is part of every layout and its relations are checked too. Specs that
place only the robot, such as ``kitchen_bench_lightwheel_open_fridge``, are
scored on the robot's placement.

Each Run builds its spec once, for 30 environments, which gives 1500
first-batch candidates. It builds only once because the rebuilds of a Run get
the same placement seed and would repeat the same candidates. Each Run has its
own placement seed, 100000 apart from the others, so no two Runs share candidate
seeds and tasks that share a scene solve different candidates. Rerunning the
same code solves the same candidates, although an asset that loads a random
variant in each process, such as the Lightwheel broccoli, can still change
results.

The inverse kinematics (IK) reachability check runs only for embodiments with a
registered cuRobo config, and a spec can make it optional, as the RoboLab tasks
on the ``butter_raisin_box`` scene do. It counts toward a rate only where it is
required.

Run the Benchmark
-----------------

Run from the repository root inside the Arena container; see
:doc:`../../quickstart/installation`. Each command runs one suite:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_benchmarking/experiment_configs/robolab_placement_experiment.yaml \
     --experiment_output_directory outputs/placement_success/robolab \
     --continue_on_error

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_benchmarking/experiment_configs/kitchen_bench_placement_experiment.yaml \
     --experiment_output_directory outputs/placement_success/kitchen_bench \
     --continue_on_error

The output directory must be new or empty. ``--continue_on_error`` keeps a
failed Run from stopping the later ones, so the runner can exit with status 0
even when some Runs failed. To change the sample size, add
``shared.environment_builder.num_envs=<n>`` to the command; the seed spacing
allows up to 200 environments. The solver optimizes a batch together, so compare
only results with the same ``num_envs``.

Then print each Run's placement success and the Experiment's means. This reads
the output directory and does not start the simulator. Check the scored and
failed counts in the last row: every spec should be scored. A Run that completed
without diagnostics solved no relations or replayed recorded layouts; the table
lists it, and the means leave it out:

.. code-block:: bash

   python -m isaaclab_arena_benchmarking.object_placement.placement_success \
     outputs/placement_success/robolab

Add Specs
---------

A placement Experiment lists one Run per spec, named after the spec's file stem.
To benchmark a new spec, add a Run for it to its suite's Experiment, with a
placement seed 100000 above the largest one there. For a new folder of specs,
add an Experiment and matching cases to
``isaaclab_arena_benchmarking/tests/test_placement_success.py``, which checks
that each Experiment lists every spec in its folder and that no two Runs share
candidate seeds.
