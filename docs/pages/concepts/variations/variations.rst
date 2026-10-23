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
generator, seeded from ``--seed`` (see :doc:`../environment/env_builder`). The exception is
in-order :ref:`asset selection <asset-selection-variation>`, which follows the rebuild index.

.. _discovering-available-variations:

Discovering available variations
---------------------------------

Pass ``--list_variations`` to print every Hydra-configurable variation for the selected
environment and then exit before rollout:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --list_variations \
     pick_and_place_maple_table

The output lists each asset (scene asset or embodiment), the variation name, whether it is
run-time or build-time, the Hydra path to enable it, and all tunable fields with their current
defaults:

.. code-block:: text

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

     asset_selection (AssetSelectionVariation, build-time)
       Enable: pick_up_object.asset_selection.enabled=true  (default: False)
       Fields:
         pick_up_object.asset_selection.random_choice = False
         pick_up_object.asset_selection.registry_names = []
         pick_up_object.asset_selection.sampler_cfg = {}

   ...

Enabling variations
-------------------

To enable a variation, with default control parameters append its ``enabled=true`` override token
after the environment subcommand. For example, to enable the HDR image and camera extrinsics variations
run:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type zero_action \
     --num_steps 50 \
     --enable_cameras \
     pick_and_place_maple_table \
     light.hdr_image.enabled=true \
     droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true


The same run with tunable variation control parameters spelled out:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type zero_action \
     --num_steps 50 \
     --enable_cameras \
     pick_and_place_maple_table \
     light.hdr_image.enabled=true \
     "light.hdr_image.hdr_names=[home_office_robolab,billiard_hall_robolab,garage_robolab]" \
     droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true \
     "droid_abs_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.low=[-0.01,-0.01,-0.01]" \
     "droid_abs_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.high=[0.01,0.01,0.01]"

The ``hdr_names`` list restricts HDR sampling to the three named maps instead of the full
registered set.  The ``sampler_cfg.low`` / ``sampler_cfg.high`` vectors widen the camera
extrinsics jitter range to ±10 mm per axis.

To see the available variations and control parameters for a specific environment,
see :ref:`discovering-available-variations`.

With Isaac Lab's ``record_demos.py`` and Arena's ``environment_registration_callback``,
append the same variation overrides to the recording command, for example
``pick_up_object.mass.enabled=true`` for ``pick_and_place_maple_table``.
Arena consumes its variation overrides and returns the remaining arguments to the Isaac Lab script.
See :doc:`Teleoperation Data Collection <../../example_workflows/static_manipulation/step_2_teleoperation>`
for recording setup and the full command using the callback.


Configuring variations in an experiment config
----------------------------------------------

When running experiments with ``experiment_runner.py``, variations are configured per run via a
dedicated ``variations`` field instead of command-line override tokens.  The field maps each
dotted Hydra path to its value.  For example, the entry ``light.hdr_image.enabled: true`` is
equivalent to the command-line override ``light.hdr_image.enabled=true``.

The example config ``isaaclab_arena_environments/experiment_configs/droid_pnp_variations_experiment.yaml``
enables three variations on a single run:

.. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/droid_pnp_variations_experiment.yaml
   :language: yaml

Run it with:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_variations_experiment.yaml

``--list_variations`` works with ``experiment_runner.py`` too, printing the variations catalogue for
each run's environment:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --list_variations \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_variations_experiment.yaml

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
   * - ``AssetSelectionVariation``
     - build-time
     - Picks the registered asset a rigid object spawns, from a list of names, when the environment is built.
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

.. _asset-selection-variation:

Asset selection
---------------

To evaluate a Run on several objects, enable the object's ``asset_selection`` and list registered
rigid objects in its ``registry_names``. The first rebuild of the Run spawns the first name in every
parallel environment, the second rebuild the second name, and so on. Set ``num_rebuilds`` to the
length of ``registry_names`` to spawn each entry once. With fewer rebuilds, a name listed only after
the entry of the last rebuild never spawns.

The example config ``isaaclab_arena_environments/experiment_configs/asset_selection_experiment.yaml``
spawns three pick-up objects in ``pick_and_place_maple_table``, one per rebuild, each in four parallel
environments:

.. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/asset_selection_experiment.yaml
   :language: yaml

Every rigid ``Object`` has a disabled ``asset_selection`` variation, for example the
``pick_up_object`` of ``pick_and_place_maple_table``. While it is off, the object spawns its own
asset. When it is on, each build chooses one name from ``registry_names``, and every parallel
environment of that build spawns that asset. For different objects in the
environments of one build, use a :doc:`RigidObjectSet <../scene/concept_rigid_object_set>`.

With more rebuilds than entries, the order starts over after the last entry. The seed does not change
this order. With ``random_choice: true``, each build instead draws a name at random, seeded from
``environment_builder.seed`` and the variation key, and names can repeat. To select a name more
often, list it more than once: in order, it spawns in more rebuilds, and with ``random_choice: true``
it is drawn more often.

The object keeps its ``name``, so its variation keys stay the same. Its ``registry_name`` becomes the
selected name. The default descriptions of ``PickAndPlaceTask`` and ``AssemblyTask`` name the
selected asset. Other tasks, such as ``GearInsertionTask`` and ``PlaceUprightTask``, still use the
object's ``name``.

Every episode row records the chosen name, for example
``"pick_up_object.asset_selection": "mustard_bottle_hot3d_robolab"``, and
:doc:`recorded variation replay <../experiment_record_replay>` spawns the recorded asset.
