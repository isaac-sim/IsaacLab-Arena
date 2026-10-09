Variations
==========

Variations change selected environment properties through configured samplers.
Assets declare the variations they support. Each variation starts disabled and takes effect
when enabled.

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
     - Shared across environments or fixed per environment, for every episode in that build
     - Asset selection, background image, lighting changes
   * - Run-time
     - When an environment resets
     - One episode in one parallel environment
     - Camera extrinsics, camera intrinsics

To change a build-time choice after scene creation, rebuild the environment. A run-time
variation can produce a new value on each reset without rebuilding the scene.

Sampling scope is separate from this lifetime. The common ``sample_per_environment`` setting
controls whether environments receive separate samples (``true``) or share one sample (``false``).
Lighting and HDR variations support only ``false``. The existing camera, mass, and disappearance
variations support only ``true``. ``AssetSelectionVariation`` supports both and defaults to ``true``.
An unsupported setting is rejected when the configuration is applied or the variation is built.

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


.. _asset-selection-variation:

Selecting assets at build time
------------------------------

Use the same generic ``Object`` for a fixed rigid asset or for selection across environments.
``AssetRegistry.get_asset_definition()`` returns a reusable ``tuple[str, SpawnerCfg]``:
the requested registry name and a copy of the native spawn settings. Supply it through
``asset`` for a normal default:

.. code-block:: python

   from isaaclab_arena.assets.object import Object
   from isaaclab_arena.assets.registries import AssetRegistry

   asset_registry = AssetRegistry()
   cracker_box_asset = asset_registry.get_asset_definition("cracker_box")
   sugar_box_asset = asset_registry.get_asset_definition("sugar_box")
   pick_up_object = Object(name="pick_up_object", asset=cracker_box_asset)

For selection without a default, create the object without ``asset`` and let an enabled
variation supply it:

.. code-block:: python

   from isaaclab_arena.variations.asset_selection_variation import (
       AssetSelectionVariation,
       AssetSelectionVariationCfg,
   )

   pick_up_object = Object(name="pick_up_object")
   pick_up_object.add_variation(
       AssetSelectionVariation(
           asset_candidates=[cracker_box_asset, sugar_box_asset],
           cfg=AssetSelectionVariationCfg(enabled=True),
       )
   )

Both examples use the same scene object API. Tasks and placement relations keep their
reference to ``pick_up_object``, whose scene name stays fixed. Selection can also be attached
to an object with a default. When selection is disabled, that object uses its default.
An object without a default needs an enabled variation to assign an asset; missing or disabled
selection fails before placement. Arena does not silently use the first candidate as a default.

Candidate definitions must describe concrete rigid assets with unique, nonempty names.
The variation copies the names and native spawn settings when constructed, so later edits
to a supplied configuration do not change the candidates. Names serve as recording IDs.

The registry adapter calls the existing rigid library constructor with any supplied keyword
overrides and snapshots its native spawn settings. The returned definition keeps the requested
registry ID, even when an ``instance_name`` override changes the temporary object's name.
For differently configured copies of the same registered asset, provide distinct tuple IDs:

.. code-block:: python

   _, small_sugar_box_spawn_cfg = asset_registry.get_asset_definition(
       "sugar_box", scale=(0.8, 0.8, 0.8)
   )
   small_sugar_box_asset = ("small_sugar_box", small_sugar_box_spawn_cfg)

The adapter does not transfer subclass behavior, poses, relations, or attached variations.
Existing library constructors remain available; the adapter introduces no new asset class
or library format. Settings on the enclosing Isaac
Lab asset configuration remain arguments to ``Object``, through ``asset_cfg_addon`` as before.

The default ``SequentialChoiceSamplerCfg`` cycles through candidates in declaration order.
With the two candidates above, four environments receive cracker box, sugar box, cracker box,
and sugar box. For independent random choices, replace the configuration above with:

.. code-block:: python

   from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg

   selection_cfg = AssetSelectionVariationCfg(
       enabled=True,
       sampler_cfg=ChoiceSamplerCfg(),
   )

Pass ``cfg=selection_cfg`` when constructing the variation. Set
``sample_per_environment=False`` to share one choice across all environments. With the
sequential sampler, that shared choice is always the first candidate.

The builder applies experiment overrides and variations, then checks that every object has
an assigned asset before placement and scene construction. ``Object`` provides generic
asset assignment and geometry access without depending on the selection variation class.
Selection happens once per build. Every episode in an environment keeps the same selected
asset, and its episode records repeat that candidate's
name under the target's variation key, such as ``pick_up_object.asset_selection``. Resets do
not resample asset identities. Create fresh target objects and variations for another build;
reusing an already resolved selection is rejected.

This first implementation supports Python-authored candidates. YAML candidate authoring,
selection build manifests, and selection replay are not supported yet. Recorded variation
replay and recorded placement replay reject enabled asset selection because candidate IDs
alone cannot reconstruct the build. References into a target with enabled selection and
other enabled build-time variations on that target are also rejected.


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
     - Chooses a fixed rigid asset per environment, or one asset shared across the build.
   * - ``CameraExtrinsicsVariation``
     - run-time
     - Adds a small sampled offset to a camera's nominal local position on every reset.
   * - ``CameraIntrinsicsVariation``
     - run-time
     - Perturbs a pinhole camera's focal lengths on every reset.
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
   * - ``ObjectMassVariation``
     - run-time
     - Samples a rigid object's absolute mass on every reset.
   * - ``ObjectDisappearVariation``
     - run-time
     - Moves an object away from the scene with a sampled probability on every reset.


Writing a variation
-------------------

Use ``BuildTimeVariationBase`` for choices fixed during a build and ``RunTimeVariationBase``
for values sampled during simulation. Attach each variation to one asset with
``asset.add_variation(variation)``. A variation with an explicit target must attach to the
asset that owns that target.

The builder supplies a frozen ``VariationBuildContext`` containing ``num_envs``, ``seed``,
and ``variation_key`` to ``configure_at_build_time(context)``. Both
``_prepare_at_build_time`` and ``_realize_at_build_time`` accept
``context: VariationBuildContext | None = None``. The optional argument preserves manual
calls that do not need build settings; asset selection requires a context.

A run-time variation's sampler supplies per-environment values during simulation. Each draw
must include the resetting environment IDs so the recorder can associate it with the current
episodes. Its build-time preparation hook can set deterministic prerequisites, such as
choosing untiled cameras, but must not draw from that run-time sampler. A sampled build-time
prerequisite belongs in a separate build-time variation.
