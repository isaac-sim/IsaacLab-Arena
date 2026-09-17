Offline Clutter Placement
=========================

Arena's offline generator creates physics-settled layouts from ``ClutterOn``
relations. ``ObjectPlacer`` generates release poses; the generator steps physics
and validates rest and support containment before saving a companion pose file.

Declare clutter in the environment
----------------------------------

Add relations to the environment YAML:

.. code-block:: yaml

   relations:
   - kind: is_anchor
     subject: table
   - kind: clutter_on
     subject: cube_0
     reference: table
     params:
       spread: 0.2
       clearance_m: 0.01
       gap_m: 0.03
       random_yaw: true

``spread`` scales the release footprint about the support center. Release
solving and validation use that smaller region. After physics settling,
containment uses the whole support. ``clearance_m`` is the initial surface gap;
``gap_m`` separates overlapping bounds during initialization; the solver's
collision clearance is used when larger. Objects start in
free space in asset order, then the solver applies their relations and the
shared collision clearance. ``ClutterOn`` can combine with ``AtPosition`` or
other spatial constraints. Use Arena's ``RotateAroundSolution`` relation to
author a base rotation; sampled world-Z yaw preserves its roll and pitch. Set
``random_yaw: false`` to retain that rotation.

Without a cache, the same environment uses ``ObjectPlacer`` to initialize the
release column. The objects fall when simulation starts. No online settling or
settled-layout validation runs during environment construction or reset.

Generate a companion cache
--------------------------

Run inside the Arena development container, from ``/workspaces/isaaclab_arena``:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/generate_clutter_scene.py \
       --env_spec isaaclab_arena_examples/relations/clutter/clutter_scene.yaml \
       --output outputs/clutter/placements.yaml --num_envs 4 --num_layouts 100 \
       --seed 42 --viz none

One environment file and one companion pose file describe all layouts.
``--num_envs`` controls parallel generation; ``--num_layouts`` controls the total
(default: one per environment). ``--attempts`` limits retries per environment;
``--timeout_s`` limits simulated time per trial. Only rejected environments are
retried. Generation fails if any requested layout cannot be validated, and
writes output only after all layouts pass. Only release candidates that pass
placement validation are simulated. Existing output files
are not overwritten.

The source YAML's ``placement_validators`` settings apply to release poses.
``no_overlap`` and ``on_relation`` are always enabled and required for a safe
release. ``on_relation`` is the shared validator for ``On`` and ``ClutterOn``.
For ``ClutterOn``, it checks that the object's XY bounds fit inside the release
region and its bottom is at least ``clearance_m`` above the support, within
tolerance. There is no upper height limit or contact requirement, so objects
in a release column can pass. Ordinary ``On`` still requires the object's
bottom to lie in a narrow band near the support surface.

Additional enabled checks run with the configured required/optional status.
Unknown or unavailable requested checks fail generation. The Python API
accepts the same configuration through ``placer_params``, including
``max_placement_attempts`` candidates per environment in each physics trial.
The generator's ``--attempts`` controls the number of physics trials.

Release validation cannot certify a pose after physics moves it. Requests for
``ik_reachable``, ``physics_settled``, or ``RequiresReachability`` are rejected
by this offline workflow. Settling uses its own rest, containment and passive
motion checks; it does not certify task reachability.

Other placement must already be resolved to fixed anchors. Object sets are not
supported by this cache format. Downstream packages can register assets and
tasks with ``--register package.module:register_components``. The generator uses
Arena's standard graph loader and environment builder. ``default_physics_backend``
and ``env_cfg_override`` apply during generation and replay, including when loaded
from ``external_yaml``. ``--presets`` overrides the graph's backend default;
backend-specific configuration and asset physics must remain compatible with
that selection. The supplied office-table example uses PhysX. For Newton,
use assets with valid MuJoCo inertias and a robot configuration that stays
within the passive-motion tolerances, or generate without an embodiment.
Keep task-specific physics settings in the environment YAML, as in the CAP gear
environments, so generation and replay use the same configuration.

The four-cube example lives in ``isaaclab_arena_examples/relations/clutter`` and
uses only Arena assets. CAP-specific assets, registration, and task behavior
remain in ``isaaclab_arena_environments/isaac_cap``.

Load at runtime
---------------

Pass the companion file explicitly:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/environment_runner.py \
       --env_spec isaaclab_arena_examples/relations/clutter/clutter_scene.yaml \
       --placement_layouts outputs/clutter/placements.yaml

Alternatively, add ``placement_layouts: placements.yaml`` to the environment
YAML. That path is relative to the environment file; a command-line override is
relative to the working directory and takes precedence.

For a registered Python environment, pass ``--placement_layouts`` before the
environment subcommand. Keys must be runtime scene names instead of graph IDs.
Python callers can supply ``PlacementLayouts.from_yaml(path)`` through the
``IsaacLabArenaEnvironment(placement_layouts=...)`` constructor. The same
coverage and pose-reset ownership checks apply to both environment sources.

The companion file maps graph object IDs to lists of environment-local poses,
with positions in metres and quaternions in ``[x, y, z, w]`` order:

.. code-block:: yaml

   cube_0:
   - position_xyz: [0.1, 0.2, 0.8]
     rotation_xyzw: [0.0, 0.0, 0.0, 1.0]
   - position_xyz: [-0.1, 0.2, 0.8]
     rotation_xyzw: [0.0, 0.0, 0.0, 1.0]

Every object has the same number of poses. An index selects a complete layout
across all objects. Environment ``i`` starts at index ``i % num_layouts``; each
reset selects the next layout and wraps at the end. This is independent cycling,
not a shared queue: environments may reuse the same layout concurrently, and
the cache is never exhausted. Partial resets
leave other environments unchanged. Replay restores exact root poses and zero
velocities, bypassing relation solving and settling. Pose-changing variations
or custom reset callbacks can alter the layout and should be disabled for exact
replay. Cached assets must not have their own explicit pose-reset events;
the complete-layout event owns their pose resets.

Loading validates pose data and object coverage, not physical placement.
``placement_validators`` configure generated placements; they are not rerun
when replaying a cache. The same environment YAML can therefore generate and
replay layouts without removing its validator settings. A manually edited
cache, or one generated for different geometry, is not certified by a
successful load. Use the same scene geometry, fixed poses and physics settings
as generation. Replay does not certify reachability.

``PlacementLayouts.get_layout(index)`` retrieves one complete layout without
advancing reset cursors. ``write_scene_poses_to_sim`` applies supplied pose
tensors without drawing another layout or invoking the solver. This is also
the root-pose writer used by ordinary placement; compound assets expand their
poses through ``layout_pose_to_scene_writes`` before application.

Use from Python
---------------

.. code-block:: python

   from isaaclab_arena.relations.clutter.settle import settle_clutter
   from isaaclab_arena.relations.clutter.validation import ClutterSettleParams

   env.reset()
   layouts = settle_clutter(
       env,
       list(arena_env.scene.assets.values()),
       seed=42,
       params=ClutterSettleParams(timeout_s=15.0),
   )

Assets must carry their ``ClutterOn`` and ``IsAnchor`` relations. Their fixed
poses must match the constructed scene. ``layouts[env_id]`` maps dynamic rigid
object scene keys to ``Pose`` values. The helper restores scene state and
actuator targets on success and failure.

Checks and limits
-----------------

- Supports must be horizontal, axis aligned or turned by a multiple of 90
  degrees, with static or kinematic spawned geometry.
- Clutter members must be dynamic rigid objects with gravity enabled. Passive
  rigid neighbors need fixed poses and collision geometry. All dynamic rigid
  objects are monitored and cached; the YAML exporter rejects unmapped bodies.
- Rest requires consecutive quiet pose windows. Full rotated bounds must stay
  above and inside the support. Moving, non-finite, or escaped objects reject
  the trial. Rest thresholds use ``--move_thresh_m`` and ``--turn_thresh_deg``.
- Neighbors, supports and robot links must remain within
  ``--passive_move_thresh_m`` and ``--passive_turn_thresh_deg`` of their initial
  poses. Articulation configurations are not stored in the pose cache.
- Conservative bounding boxes do not describe concave interiors. Use a fixed
  reference identifying the usable support surface inside a fixture.
- Release placement uses Arena's geometry validators. Settled layouts are
  checked for rest and containment; task reachability is not certified.
- Replay assumes the same assets and physics configuration. Exact starting
  poses do not guarantee identical trajectories across simulator versions.

Run the tests
-------------

.. code-block:: bash

   /isaac-sim/python.sh -m pytest -sv \
       isaaclab_arena/tests/clutter/test_settled_scene.py \
       isaaclab_arena/tests/clutter/test_clutter_validation.py \
       isaaclab_arena/tests/test_clutter_on.py \
       isaaclab_arena/tests/test_placement_layouts.py

   /isaac-sim/python.sh -m pytest -sv isaaclab_arena/tests/clutter/test_clutter_cli.py
