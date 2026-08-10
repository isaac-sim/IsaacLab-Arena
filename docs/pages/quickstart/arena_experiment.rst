.. _first-arena-experiment:

First Arena Experiment
======================

The previous page launched three versions of the Maple-table scene and previewed 64 parallel
copies. Here, those four setups become four named Runs in one YAML file. The zero-action policy
keeps the example quick and needs no model weights.

Together, those Runs form an Arena Experiment. The YAML file is its **Experiment Definition**.


Define the Experiment
---------------------

The four Runs collect the three setups you launched and the parallel setup you previewed:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: ``baseline``
      :class-card: sd-shadow-sm

      .. image:: ../../images/default_srl_pnp.png
         :alt: Baseline DROID pick-and-place environment with a Rubik's cube and bowl

      Uses the shared environment settings without changes.

   .. grid-item-card:: ``swap_objects``
      :class-card: sd-shadow-sm

      .. image:: ../../images/swap_objects.gif
         :alt: DROID pick-and-place environment with a mustard bottle and wooden bowl

      Replaces the Rubik's cube and bowl with a mustard bottle and wooden bowl.

   .. grid-item-card:: ``change_background_hdr``
      :class-card: sd-shadow-sm

      .. image:: ../../images/swap_hdr.gif
         :alt: DROID pick-and-place environment with a billiard-hall background

      Replaces the home-office background with a billiard hall.

   .. grid-item-card:: ``parallel_envs``
      :class-card: sd-shadow-sm

      .. image:: ../../images/scale_up.gif
         :alt: Many copies of the DROID pick-and-place environment running in parallel

      Runs 64 copies of the baseline environment in parallel.

The four setups above are defined together in one YAML file:

.. literalinclude:: ../../../isaaclab_arena_environments/experiment_configs/getting_started_experiment.yaml
   :language: yaml
   :start-at: shared:

There are three ideas to notice:

* Values below ``shared`` are used by every Run.
* The keys below ``runs`` are the Run names.
* ``baseline: {}`` uses the shared values as written. The other Runs list only what they change.

The shared ``episode_length_s: 1.5`` value is a short timeout in simulated time. Together with
the one-episode rollout limit, it lets this zero-action example finish quickly and produce episode
results.


Run the Experiment locally
--------------------------

Start or enter the Base Docker container from the repository root:

:docker_run_default:

Then run it inside the container:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/getting_started_experiment.yaml

The Experiment Runner loads the Runs in YAML order and reuses one SimulationApp. It builds a fresh
environment for every Run, then closes that environment before starting the next one.

.. note::

   The four Runs execute one after another on your machine. The 64 environments in
   ``parallel_envs`` are different: they are copies inside that one Run, and they step in parallel.

By default, Arena saves the result as
``outputs/YYYY-MM-DD_HH-MM-SS/arena_experiment_result.json`` and prints the exact path. The file
records every Run, its status, and its episode results.


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

See :doc:`Arena Experiments <../concepts/concept_arena_experiments>` for the full
precedence order and configuration rules.


Next steps
----------

* Continue to :doc:`environment_variations` to sample controlled environment changes.
* Continue to :doc:`running_a_real_policy/index` to evaluate a trained policy.
