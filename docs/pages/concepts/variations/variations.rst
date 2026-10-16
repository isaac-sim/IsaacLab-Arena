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
build-time variation, the environment must be rebuilt with different seeds; the Experiment
Runner gives each rebuild its own seed. A run-time variation can produce a new value on each
reset without rebuilding the scene.

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

   Asset: droid_abs_joint_pos
     camera_extrinsics_wrist_camera (CameraExtrinsicsVariation, run-time)
       Enable: droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true  (default: False)
       Fields:
         droid_abs_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.high = [0.005,0.005,0.005]
         droid_abs_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.low = [-0.005,-0.005,-0.005]

   Asset: light
     hdr_image (HDRImageVariation, build-time)
       Enable: light.hdr_image.enabled=true  (default: False)
       Fields:
         light.hdr_image.hdr_names = []

   Asset: bowl_ycb_robolab
     (no variations)

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
     - Chooses which of several candidate assets a rigid object spawns.
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

``AssetSelectionVariation`` chooses which asset a rigid ``Object`` spawns. It chooses one
candidate per build, and every parallel environment of that build spawns it. Each rebuild can
choose another candidate. For different objects in the environments of one build, use a
:doc:`RigidObjectSet <../scene/concept_rigid_object_set>`.

The environment's Python code declares the candidates as ``(name, spawn configuration)`` pairs.
``AssetRegistry.get_asset_definition()`` returns the pair for a registered rigid object:

.. code-block:: python

   from isaaclab_arena.assets.object import Object
   from isaaclab_arena.assets.registries import AssetRegistry
   from isaaclab_arena.variations.asset_selection_variation import (
       AssetSelectionVariation,
       AssetSelectionVariationCfg,
   )

   asset_registry = AssetRegistry()
   cracker_box_asset = asset_registry.get_asset_definition("cracker_box")
   sugar_box_asset = asset_registry.get_asset_definition("sugar_box")
   mustard_bottle_asset = asset_registry.get_asset_definition("mustard_bottle")

   pick_up_object = Object(name="pickup")
   pick_up_object.add_variation(
       AssetSelectionVariation(
           asset_candidates=[cracker_box_asset, sugar_box_asset, mustard_bottle_asset],
           cfg=AssetSelectionVariationCfg(enabled=True),
       )
   )

``Object(name="pickup")`` has no asset of its own, so the build fails unless its selection is
enabled. ``Object(name="pickup", asset=cracker_box_asset)`` spawns the cracker box while the
selection is disabled.

``asset_names`` lists the candidate names to choose from, in order. Empty means all candidates in
declaration order, and one name pins that candidate. By default, a build takes name number
``seed mod n`` of the ``n`` names, counting from zero, where ``seed`` is ``--seed``. The
Experiment Runner adds the rebuild index to the seed, so ``n`` rebuilds take each name once. With
``random_choice: true``, each build draws a name at random, and a name can repeat across rebuilds.

This run takes each candidate once:

.. code-block:: yaml

   runs:
     one_object_per_build:
       num_rebuilds: 3
       variations:
         pickup.asset_selection.enabled: true
         pickup.asset_selection.asset_names: [cracker_box, sugar_box, mustard_bottle]

With the default seed 42, the three rebuilds spawn ``cracker_box``, ``sugar_box`` and
``mustard_bottle``. Every episode row records the name, for example
``"pickup.asset_selection": "sugar_box"``, and
:doc:`recorded variation replay <../experiment_record_replay>` spawns the recorded asset.

Add the selection to a plain ``Object``, not to a library object such as ``Mug``, whose
methods were written for its declared asset.

The task description does not name the selected asset: ``PickAndPlaceTask`` builds it from the
object's name when the task is built. For a language-conditioned policy, pin one name per run
with ``asset_names`` and set ``environment_builder.language_instruction``.
