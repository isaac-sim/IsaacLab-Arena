Variations
==========

Variations are a structured way of introducing randomization into simulated environments.

Variations are attached to assets, travelling with them in a deactivated state.
A user can enable a variation in any environment that contains an
asset with that variation attached.

Variations are enabled by appending a Hydra override to the command line.

.. _build-time-run-time-variations:

Build-time and run-time variations
----------------------------------

Some properties must be chosen before the environment is created. Others can change whenever
an episode resets. Arena calls these *build-time* and *run-time* variations.

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
     - Background image, light intensity
   * - Run-time
     - When an environment resets
     - One episode in one parallel environment
     - Wrist-camera position offset

This distinction matters when planning an evaluation. To collect several values of a
build-time variation, the environment must be rebuilt several times. A run-time variation can
produce a new value on each reset without rebuilding the scene.

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

Enabling variations (minimal)
------------------------------

To enable both variations, append their ``enabled=true`` override tokens after the environment
subcommand.  All other parameters stay at their defaults — the HDR is sampled uniformly from
every registered HDR, and the camera extrinsics offset is drawn from ``[-0.005, 0.005]`` m per
axis on every reset:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type zero_action \
     --num_steps 50 \
     --enable_cameras \
     pick_and_place_maple_table \
     light.hdr_image.enabled=true \
     droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true

Enabling variations with explicit parameters
---------------------------------------------

The same run with every tunable parameter spelled out, showing the full override syntax:

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

How Hydra override paths are structured
-----------------------------------------

All override paths follow the pattern::

   <asset>.<variation_name>.<cfg_field>

where:

* ``<asset>`` is the scene asset name (e.g. ``light``) or the embodiment name
  (e.g. ``droid_abs_joint_pos``).
* ``<variation_name>`` is the name under which the variation is registered on that asset
  (e.g. ``hdr_image``, ``camera_extrinsics_wrist_camera``).
* ``<cfg_field>`` is any attribute path within the variation's ``*Cfg`` dataclass
  (e.g. ``enabled``, ``sampler_cfg.low``).

Use ``--list_variations`` on any environment to discover the exact paths available.

Configuring variations in an eval jobs config
---------------------------------------------

When running batches of jobs with ``experiment_runner.py``, variations are configured per job via a
dedicated ``variations`` field instead of command-line override tokens.  The field is a nested
dict that mirrors the dotted Hydra paths: each level of nesting corresponds to one segment of
the ``<asset>.<variation_name>.<cfg_field>`` path.  For example, the nested entry
``{"light": {"hdr_image": {"enabled": true}}}`` is equivalent to the command-line override
``light.hdr_image.enabled=true``.

The example config ``isaaclab_arena_environments/eval_jobs_configs/droid_pnp_variations_config.json``
enables three variations on a single job:

.. code-block:: json

   {
       "jobs": [
           {
               "name": "variations_demo",
               "arena_env_args": {
                   "environment": "pick_and_place_maple_table",
                   "embodiment": "droid_rel_joint_pos",
                   "pick_up_object": "rubiks_cube_hot3d_robolab",
                   "destination_location": "bowl_ycb_robolab",
                   "hdr": "home_office_robolab"
               },
               "num_steps": 10,
               "num_rebuilds": 5,
               "policy_type": "zero_action",
               "policy_config_dict": {},
               "variations": {
                   "light": {
                       "hdr_image": { "enabled": true },
                       "intensity": { "enabled": true }
                   },
                   "droid_rel_joint_pos": {
                       "camera_extrinsics_wrist_camera": { "enabled": true }
                   }
               }
           }
       ]
   }

Run it with:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --enable_cameras \
     --viz kit \
     --eval_jobs_config isaaclab_arena_environments/eval_jobs_configs/droid_pnp_variations_config.json

``--list_variations`` works with ``experiment_runner.py`` too, printing the variations catalogue for
each job's environment:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --list_variations \
     --eval_jobs_config isaaclab_arena_environments/eval_jobs_configs/droid_pnp_variations_config.json

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
