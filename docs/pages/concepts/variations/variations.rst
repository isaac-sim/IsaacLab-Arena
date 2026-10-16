Variations
==========

Variations are a structured way of introducing randomization into simulated environments.

Variations are automatically available in Arena-defined environments.
Activating the variation causes that particular source of randomness to be
injected into the environment.

.. _build-time-run-time-variations:

Build-time and run-time variations
----------------------------------

Some properties must be chosen before the environment is created. Others can change whenever
an environment resets during policy rollouts. Arena calls these *build-time* and *run-time* variations.

.. list-table::
   :header-rows: 1
   :widths: 20 25 35 20

   * - Type
     - When it changes
     - Where the drawn value applies
     - Examples
   * - Build-time
     - Before the environment is built
     - Every parallel environment and episode in that build
     - Background image, lighting changes.
   * - Run-time
     - When an environment resets
     - One episode in one parallel environment
     - Camera extrinsics, camera intrinsics

This distinction matters when planning an evaluation. To collect several values of a
build-time variation, the environment must be rebuilt; the Experiment
Runner gives each rebuild its own seed. Set ``num_rebuilds`` in the Run; with the default of 1,
each enabled build-time variation draws one value for the whole Run. A run-time variation can
produce a new value on each
reset without rebuilding the scene. Both kinds draw their values from the variation's own
generator, seeded from ``--seed`` (see :doc:`../environment/env_builder`).

.. _asset-selection-variation:

In-order asset selection (``RigidAssetSelectionVariation`` in :ref:`available-variations`) is the
exception: rebuild k spawns the k-th name of its ``registry_names``, so set ``num_rebuilds`` to the
length of the list to spawn each name once. To give each name its own episode count, list the
counts in the same order in ``rollout_limit.num_episodes_per_rebuild``; for example,
``registry_names: [cracker_box, sugar_box]`` with ``num_rebuilds: 2`` and
``num_episodes_per_rebuild: [20, 5]`` runs 20 episodes with the cracker box and 5 with the sugar box.

A :doc:`RigidObjectSet <../scene/concept_rigid_object_set>` fits neither row: it assigns one member
to each parallel environment when the environment is built and keeps it for every episode of that
build. It is part of the environment definition rather than a variation, so it has no
``asset_selection``.

.. _discovering-available-variations:

Discovering available variations
---------------------------------

Pass ``--list_variations`` to ``experiment_runner.py`` to print every Hydra-configurable variation
for the environment of each Run, and then exit before rollout. This command lists the variations of
the example Experiment from :ref:`enabling-variations`:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --list_variations \
     --experiment_config isaaclab_arena_environments/experiment_configs/concepts/maple_table_variations_experiment.yaml

The output lists each asset (scene asset or embodiment), the variation name, whether it is
run-time or build-time, the Hydra path to enable it, and all tunable fields. The printed values
include the Run's own ``variations`` entries, so a variation that the Run enables shows
``(default: True)``:

.. code-block:: text

   === Variations for run 'maple_table' ===
   Variations (Hydra-configurable)
   ================================

   ...

   Asset: pick_up_object
     mass (ObjectMassVariation, run-time)
       Enable: pick_up_object.mass.enabled=true  (default: False)
       Fields:
         pick_up_object.mass.recompute_inertia = True
         pick_up_object.mass.sampler_cfg.high = [2.0]
         pick_up_object.mass.sampler_cfg.low = [0.05]

     ...

     asset_selection (RigidAssetSelectionVariation, build-time)
       Enable: pick_up_object.asset_selection.enabled=true  (default: False)
       Fields:
         pick_up_object.asset_selection.registry_names = []
         pick_up_object.asset_selection.sampler_cfg.random_choice = False

   ...

.. _enabling-variations:

Enabling variations
-------------------

A Run enables variations in its ``variations`` field. The field maps each dotted Hydra path from the
listing to its value. For example, the entry ``light.hdr_image.enabled: true`` matches the listed
override ``light.hdr_image.enabled=true``. This Experiment enables the HDR image and camera
extrinsics variations with default control parameters:

.. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/concepts/maple_table_variations_experiment.yaml
   :language: yaml
   :start-after: SPDX-License-Identifier: Apache-2.0

Run it with:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/concepts/maple_table_variations_experiment.yaml

The same run with tunable variation control parameters spelled out as command-line overrides:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/concepts/maple_table_variations_experiment.yaml \
     "+runs.maple_table.variations.light.hdr_image.hdr_names=[home_office_robolab,billiard_hall_robolab,garage_robolab]" \
     "+runs.maple_table.variations.droid_abs_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.low=[-0.01,-0.01,-0.01]" \
     "+runs.maple_table.variations.droid_abs_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.high=[0.01,0.01,0.01]"

Each override starts with ``+`` because it adds an entry that the Run's ``variations`` field does
not declare yet. The ``hdr_names`` list restricts HDR sampling to the three named maps instead of
the full registered set. The ``sampler_cfg.low`` / ``sampler_cfg.high`` vectors widen the camera
extrinsics jitter range to ±10 mm per axis. To keep these values with the Run, write the same
paths and values in its ``variations`` field instead.

To see the available variations and control parameters for a specific environment,
see :ref:`discovering-available-variations`.

The :doc:`Exploring Environment Variations <../../quickstart/environment_variations>` quickstart
also sets ``num_rebuilds`` so that one Run draws several build-time values.

With Isaac Lab's ``record_demos.py`` and Arena's ``environment_registration_callback``,
append variation overrides as ``<path>=<value>`` tokens to the recording command, for example
``pick_up_object.mass.enabled=true`` for ``pick_and_place_maple_table``.
Arena consumes its variation overrides and returns the remaining arguments to the Isaac Lab script.
See :doc:`Teleoperation Data Collection <../../example_workflows/sequential_static_manipulation/step_2_teleoperation>`
for recording setup and the full command using the callback.

.. _available-variations:

Available variations
--------------------

The variations shipped in ``isaaclab_arena/variations/`` are listed below.  Run-time variations
are realised via an event term and resampled during simulation (e.g. per reset); build-time
variations are sampled once and applied to asset configs before the environment is composed.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variation
     - Type
     - Description
   * - ``RigidAssetSelectionVariation``
     - build-time
     - Picks the registered asset a rigid object spawns from a list of names. Rebuild k spawns the
       k-th name, or a random one with ``sampler_cfg.random_choice: true``. Episode rows record the
       name. See the ``swap_objects`` Run of the :ref:`first Arena Experiment <first-arena-experiment>`.
   * - ``CameraExtrinsicsVariation``
     - run-time
     - Adds a small sampled offset to a camera's nominal local position on every reset.
   * - ``CameraIntrinsicsBuildTimeVariation``
     - build-time
     - Perturbs a pinhole camera's focal lengths and principal point when the environment is built.
   * - ``CameraIntrinsicsRunTimeVariation``
     - run-time
     - Perturbs a pinhole camera's focal lengths and principal point on every reset.
   * - ``HDRImageVariation``
     - build-time
     - Samples a single HDR and attaches it to a dome light.
   * - ``LightColorTemperatureVariation``
     - build-time
     - Samples a white-point color temperature (Kelvin) and applies it to a light.
   * - ``LightColorVariation``
     - build-time
     - Samples an RGB color and applies it to a light.
   * - ``LightDirectionVariation``
     - build-time
     - Samples a continuous orientation and applies it to a directional light.
   * - ``LightIntensityVariation``
     - build-time
     - Samples a single intensity and applies it to a light.
