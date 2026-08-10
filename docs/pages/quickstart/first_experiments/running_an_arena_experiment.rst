.. _run-an-arena-experiment:

Run an Arena Experiment
=======================

An Arena Experiment groups one or more named Runs into one evaluation. Each Run combines an
environment, a policy, and a rollout limit. Its **Experiment Definition** is the YAML file that
records those Runs.

The previous page launched four versions of the Maple-table scene with separate commands. Here,
those choices become one Experiment. The zero-action policy keeps the example quick and needs no
model weights.


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

The four Runs collect the same choices that the earlier commands launched separately:

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


Change values from the command line
-----------------------------------

You can adjust declared values without editing the YAML. This command reduces the number of
parallel environments in the ``parallel_envs`` Run:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/getting_started_experiment.yaml \
     runs.parallel_envs.environment_builder.num_envs=8

This changes only ``environment_builder.num_envs`` in the ``parallel_envs`` Run. All other values
remain as written in the YAML.

See :doc:`Arena Experiments and Runs <../../concepts/concept_arena_experiments>` for the full
precedence order and configuration rules.


Next steps
----------

* Continue to :doc:`exploring_variations` to sample controlled environment changes.
* Continue to :doc:`running_a_real_policy/index` to evaluate a trained policy.
* Read :doc:`Arena Experiments and Runs <../../concepts/concept_arena_experiments>` for all Run
  fields and execution choices.
* To submit an Experiment to OSMO, follow :doc:`Multi-node Evaluation
  <../../example_workflows/multi_node_evaluation/multi_node_evaluation>`.
