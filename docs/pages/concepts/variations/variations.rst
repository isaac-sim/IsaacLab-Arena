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
build-time variation, the environment must be rebuilt several times. A run-time variation can
produce a new value on each reset without rebuilding the scene.

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

For agents and scripts, write JSON directly to a file. Redirecting stdout also
captures simulator messages and does not produce a clean JSON document:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --list_variations --variations_format json \
     --variations_output outputs/variations.json \
     pick_and_place_maple_table

Discovery starts SimulationApp and constructs the selected environment's assets,
so it requires the simulation runtime and asset access even though it does not
run a policy. ``--variations_output`` requires ``--list_variations``, writes the
selected text or JSON format, creates missing parent directories, and replaces
an existing file. Console output is unchanged.

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

To save its machine-readable catalogue, add ``--variations_format json
--variations_output outputs/experiment_variations.json`` to that discovery
command. The JSON document contains ``schema_version`` and ``runs``, with each
run name mapped to its environment's catalogue. The policy runner writes one
environment catalogue directly.

Reproducible draws and replay
-----------------------------

Set ``environment_builder.variation_seed`` to give enabled variations an
independent random stream. In an Experiment Definition, for example:

.. code-block:: yaml

   shared:
     environment_builder:
       seed: 42
       variation_seed: 73

The built-in uniform, choice and Bernoulli samplers derive each draw from the
variation seed, its qualified ``asset.variation`` path, and either a build-time
key or the runtime ``(env_id, episode_in_env)`` pair. Unrelated variations and
global Torch RNG calls do not change those draws. Reordered or partial resets
give the same value for the same environment and episode. Experiment rebuilds
offset the variation seed, just as they offset the environment seed.

This guarantee does not assign logical cases across different environment
counts or worker shards: changing the environment or episode ID changes the
key. It also does not guarantee identical physics trajectories across devices
or simulator versions. With ``variation_seed: null`` and no replay file,
samplers retain their existing global-RNG behavior; build-time draws are then
not fixed by the environment seed.

Every accepted draw is recorded by the existing variation recorder in the
episode JSONL's ``variations`` field. The experiment runner also exports a
complete ``variation_samples_rebuild<N>.jsonl`` trace, including initializations
drawn by the final autoreset after the last completed episode. To apply
recorded values instead of sampling, set
``environment_builder.variation_replay_path`` to this companion trace from one
Run and one rebuild. Runtime draws require the exact recorded environment and
episode IDs. Set ``num_rebuilds: 1`` for replay; use a separate Run for each
recorded rebuild's trace. Multiple rebuilds with one replay path are rejected.

Existing ``episode_results_rebuild<N>.jsonl`` files are also accepted. They
must include every requested reset, including a lookahead episode if the run
autoresets at completion. Such files contain only completed episodes; use the
companion trace to replay a complete evaluation with its final autoreset.
Build-time values in an episode-results file must be present and identical in
every row.

Replay never falls back to random sampling. Missing values, duplicate episode
keys, wrong numeric shapes, nonfinite numbers, incorrect categorical types,
and values outside the configured domain are rejected. Configure the same
variation domains as the recorded run, and do not request resets beyond those
in the trace. The normal variation event applies the value and the
recorder receives one notification for the complete accepted sample batch.

Variation replay restores sampled parameters. It does not restore robot
joints, object poses, contacts, task history, or the complete simulator state.
Use placement replay or trajectory/state recording when those are needed,
and keep asset versions, task configuration, and simulator settings with the
evaluation artifacts.

Custom variations can implement ``validate_cfg()`` to reject invalid physical
domains before construction. Custom continuous samplers retain their legacy
``_sample`` implementation. ``ContinuousSampler.validate_range()`` checks the
declared shape without sampling and registers physical bounds to check on
every realized batch before recording or application. Custom samplers can
override that hook, call ``super()``, and add distribution-specific preflight
checks; the built-in uniform sampler also validates its configured bounds.
For Hydra composition, a replacement sampler config must match the variation
config's declared ``sampler_cfg`` type. Subclass the variation config and
override that annotation when replacing its default ``UniformSamplerCfg``
with another distribution's config.
Opt into keyed sampling by implementing
``_sample_with_generator(generator)`` for one sample row. This method must use
the supplied CPU generator without changing global RNG state. The base class
handles row attribution and listener notification. Custom replay domains can
override ``_replay_value(value)`` and call the base implementation for numeric
shape and finiteness checks.

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
   * - ``CameraIntrinsicsVariation``
     - run-time
     - Perturbs a pinhole camera's focal lengths on every reset; uses untiled cameras.
   * - ``ObjectMassVariation``
     - run-time
     - Sets a rigid object's absolute mass, optionally scaling its inertia from the nominal value.
   * - ``ObjectDisappearVariation``
     - run-time
     - Parks optional objects away from the workcell; tasks may prohibit removal of required objects.
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
