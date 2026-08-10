.. _run-an-arena-experiment:

Run an Arena Experiment
=======================

An Arena Experiment groups one or more named Runs into one evaluation. Each Run combines an
environment, a policy, and a rollout limit. Its **Experiment Definition** is a YAML file that you
can use on one machine or submit to OSMO.

The earlier environment examples launched several versions of the Maple-table scene with separate
commands. Here, those choices become one Experiment. The zero-action policy keeps the example
quick and needs no model weights.


Define the Experiment
---------------------

Arena includes this Experiment Definition:

.. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/getting_started_experiment.yaml
   :language: yaml
   :start-at: shared:

There are three ideas to notice:

* Values below ``shared`` are used by every Run.
* The keys below ``runs`` are the Run names.
* ``baseline: {}`` uses the shared values as written. The other Runs list only what they change.

The four Runs make the same changes that the earlier commands made, then add one scaled-up Run:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: ``baseline``
      :class-card: sd-shadow-sm

      .. image:: ../../../images/default_srl_pnp.png
         :alt: Baseline DROID pick-and-place environment with a Rubik's cube and bowl

      Shared scene · 1 environment · 50 steps

   .. grid-item-card:: ``swap_objects``
      :class-card: sd-shadow-sm

      .. image:: ../../../images/swap_objects.gif
         :alt: DROID pick-and-place environment with a mustard bottle and wooden bowl

      Mustard bottle and wooden bowl · 1 environment · 50 steps

   .. grid-item-card:: ``change_background_hdr``
      :class-card: sd-shadow-sm

      .. image:: ../../../images/swap_hdr.gif
         :alt: DROID pick-and-place environment with a billiard-hall background

      Billiard-hall background · 1 environment · 50 steps

   .. grid-item-card:: ``parallel_envs``
      :class-card: sd-shadow-sm

      .. image:: ../../../images/scale_up.gif
         :alt: Many copies of the DROID pick-and-place environment running in parallel

      Shared scene · 64 parallel environments · 100 steps


Run the Experiment locally
--------------------------

Start or enter the Base Docker container from the repository root:

:docker_run_default:

Then run the Experiment inside the container:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/getting_started_experiment.yaml

The Experiment Runner loads the Runs in YAML order and reuses one SimulationApp. It builds a fresh
environment for every Run, then closes that environment before starting the next one.

.. note::

   The four Runs execute one after another on your machine. The 64 environments in
   ``parallel_envs`` are different: they are copies inside that one Run, and they step in parallel.


Find the output
---------------

A successful Experiment creates a timestamped directory below ``outputs``:

.. code-block:: text

   outputs/
   └── YYYY-MM-DD_HH-MM-SS/
       ├── index.html
       ├── baseline/
       │   └── episode_results_rebuild0.jsonl
       ├── swap_objects/
       │   └── episode_results_rebuild0.jsonl
       ├── change_background_hdr/
       │   └── episode_results_rebuild0.jsonl
       └── parallel_envs/
           └── episode_results_rebuild0.jsonl

Open ``index.html`` to view the Experiment report. Run names become directory names and labels in
the report. This short, step-limited example may end before a full episode finishes, so its JSONL
files can be empty. Episode-based policy evaluations populate them with results and metrics.


Change values from the command line
-----------------------------------

You can adjust declared values without editing the YAML. This command shortens the shared rollout
and reduces the number of environments in one Run:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/getting_started_experiment.yaml \
     shared.rollout_limit.num_steps=20 \
     runs.parallel_envs.environment_builder.num_envs=8

The ``shared.*`` override changes ``baseline``, ``swap_objects``, and
``change_background_hdr`` to 20 steps. ``parallel_envs`` remains at 100 steps because that Run has
its own value in the YAML. The ``runs.parallel_envs.*`` override changes only its number of
environments.

See :ref:`arena-experiments-and-runs` for the full precedence order and configuration rules.


Use the same YAML with OSMO
---------------------------

The Experiment Definition says *what* to evaluate. The executor decides *where* it runs:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Local
      :class-card: sd-shadow-sm

      The Experiment Runner executes Runs in order in one SimulationApp.

   .. grid-item-card:: OSMO
      :class-card: sd-shadow-sm

      OSMO schedules every Run independently, then collects them into one Experiment output.

Preview the OSMO workflow without submitting it:

.. code-block:: bash

   python -m osmo.submit_arena_experiment \
     --experiment_cfg isaaclab_arena_environments/experiment_configs/getting_started_experiment.yaml \
     --dry_run \
     osmo.workflow_name=getting-started

OSMO uses the same override paths with an ``experiment_cfg.`` prefix. For example:

.. code-block:: text

   local:  runs.parallel_envs.environment_builder.num_envs=8
   OSMO:   experiment_cfg.runs.parallel_envs.environment_builder.num_envs=8

Before an actual submission, install and authenticate the OSMO client and choose a compute pool,
platform, and output location for your cluster. Follow :doc:`Multi-node Evaluation
<../../example_workflows/multi_node_evaluation/multi_node_evaluation>` to configure and submit the
workflow.


Next steps
----------

* Continue to :doc:`exploring_variations` to sample controlled environment changes.
* Continue to :doc:`running_a_real_policy/index` to evaluate a trained policy.
* Read :ref:`arena-experiments-and-runs` for all Run fields and execution choices.
