Record Initial Placement Poses
==============================

Use ``record_placement_layouts.py`` to prepare reusable initial poses before
running a policy. It solves existing placement relations, advances physics, and
saves the final poses of layouts that remain close to the solved arrangement.
The saved poses can then be loaded on reset without solving again.

.. code-block:: text

   Environment YAML -> solver and required validation -> physics
                          -> velocity and pose-shift checks -> poses.jsonl
   Environment YAML + poses.jsonl -> restore poses on reset -> run policy

1. Record a scene
-----------------

Start with a working Arena container from :doc:`../../quickstart/installation`.
Run the commands below from the repository root inside that container.
Recording and the policy check run without a window; visual replay requires a
workstation display available to the container. This example uses CPU PhysX.
The Franka task is to pick up the cube and place it into the bowl.
Both objects start ``On`` the office table with ``clearance_m: 0.001``.
An ``AtPosition`` relation keeps their X coordinate in front of the robot while
the solver varies their Y positions. The same environment YAML is used for
recording, replay and policy evaluation:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/record_placement_layouts.py \
       env_spec=isaaclab_arena_environments/office_table/franka_pick_cube_into_bowl_office_table.yaml \
       output=outputs/placements/poses.jsonl \
       num_envs=2 layouts_per_env=5 seed=42 \
       settle.num_steps=120 --device cpu --viz none

Expect all three default checks to print ``ENABLED``: ``physics_settled``,
``pose_shift`` and ``articulation_link_shift``. The final line reports
``Saved K/N accepted layouts: outputs/placements/poses.jsonl``. This command
requests five candidates in each of two environments; the accepted count may be
smaller if candidates fail validation. A successful run writes at least one layout.

An existing output file is never overwritten. For another recording, choose a
new ``output`` path and use that same path in the inspection and replay commands.

Recording settings use Hydra ``key=value`` syntax. Isaac Lab launcher settings
retain their ``--flag`` syntax. Use ``presets=newton`` for a Newton scene, and
``render=true --viz kit`` to watch the recording pass. Every enabled source
relation remains part of the same solver problem; no special relation is required.

2. Check the recording
----------------------

Each line contains poses keyed by runtime scene name under
``variations["scene.relation_placement"]["poses"]``. Check the layout count
and inspect the first record:

.. code-block:: bash

   wc -l outputs/placements/poses.jsonl
   head -n 1 outputs/placements/poses.jsonl | /isaac-sim/python.sh -m json.tool

The line count must match the accepted count printed by the recorder.
``source``, ``poses`` and ``validation`` are all fields inside
``variations["scene.relation_placement"]``. For example, post-physics reports are at
``variations["scene.relation_placement"]["validation"]["post_physics"]``.
For this example, each record should have:

* ``source: "settled"`` and poses for ``cube``, ``bowl`` and ``robot``;
* finite XYZ positions and XYZW unit quaternions;
* ``passed: true`` for all three entries in ``validation.post_physics``.

Positions are in metres in the local environment frame. ``validation.pre_physics``
contains solver verdicts for the initial candidate. Post-physics reports include
check names, implementation paths, effective settings, results and reasons.
Simulation duration is under ``validation.sampling``. Other scenes can contain
skipped reports (``passed: null``), for example when they have no articulations.

3. Replay visually
------------------

Load the saved poses with the evaluation runner and its zero-action policy:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
       --env_spec isaaclab_arena_environments/office_table/franka_pick_cube_into_bowl_office_table.yaml \
       --placement_layouts outputs/placements/poses.jsonl \
       --policy_type zero_action --num_episodes 3 \
       --num_envs 1 --device cpu --viz kit \
       --output_base_dir outputs/placements/visual_replay

Expect a Franka beside the cube and bowl on the table. The cube starts outside
the bowl, and the objects should remain near their loaded poses without a visible
release drop. The runner applies zero actions: it does not perform the pick-and-place
task. Leave the objects untouched when checking stability.

With one environment, each episode reset loads the next record in file order and
wraps after the last one. This task times out after 70 seconds of simulation,
which may run faster than wall-clock time. The command exits after three episodes;
press Ctrl-C to stop early. Restarting the command loads the first record again.
See :doc:`../object_placement/relations` for parallel and partial-reset selection.

Exact reset poses do not imply identical future trajectories. Keep the environment
YAML, robot reset configuration, backend and device unchanged when comparing runs.

Headless alternative
--------------------

Without a workstation display, use the same runner without Kit. This completes
one episode using the saved poses:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
       --env_spec isaaclab_arena_environments/office_table/franka_pick_cube_into_bowl_office_table.yaml \
       --placement_layouts outputs/placements/poses.jsonl \
       --policy_type zero_action --num_episodes 1 \
       --num_envs 1 --device cpu --viz none \
       --output_base_dir outputs/placements/evaluation

Expect ``num_episodes: 1``, ``success_rate: 0.0`` and ``object_moved_rate: 0.0``.
Zero task success is expected because the policy takes no action. The runner prints
the path to its evaluation report. Open
``outputs/placements/evaluation/<timestamp>/index.html`` in a browser to inspect
the results. The adjacent ``episode_results_rank0.jsonl`` contains the per-episode
results. Include these files and the input ``poses.jsonl`` when reporting a discrepancy.
This checks loading, stepping and resetting; it does not measure a manipulation
policy's performance.

If recording fails
------------------

No file is written when fewer than ``settle.min_layouts`` candidates pass. Inspect
the rejection summary before retrying:

* ``missing required solver checks``: make the named validator available or fix
  the scene configuration. Do not remove a required check to make recording pass.
* ``physics_settled``: bodies still exceed the final velocity limits. Check contacts
  and the source arrangement; allow more ``settle.num_steps`` if motion is transient.
* ``pose_shift``: an object moved beyond the allowed displacement or rotation.
  Check release clearance, support and collisions.
* ``articulation_link_shift``: links moved too far from their reset configuration.
  Joint states are not recorded, so root-pose replay cannot reproduce that configuration.

Keep the default validators enabled for this example. Save the command, log and
JSONL file when reporting a discrepancy.

Acceptance checks
-----------------

Each recorded candidate must pass all required solver checks and every enabled,
applicable post-physics validator. Candidates with missing explicitly required
solver results are rejected before simulation, including unavailable IK checks.
All validators share one physics pass. Disabled or inapplicable validators produce
skipped reports, not successful verdicts.
All rigid and articulation roots are recorded, including fixed rigid roots.

``settle.num_steps`` controls duration in environment steps (default 5), each
containing ``decimation`` physics substeps. The example uses 120 to give the scene
time to settle. ``settle.min_layouts`` sets the minimum accepted count needed to
write output (default 1).

The default validators are:

* ``physics_settled``: existing final-velocity check, with ``lin_vel_thresh=0.1`` m/s
  and ``ang_vel_thresh=0.1`` rad/s.
* ``pose_shift``: maximum root displacement of ``max_translation_m=0.002`` metres
  and rotation of ``max_rotation_deg=2`` degrees from the initial pose.
* ``articulation_link_shift``: the same displacement and rotation limits for links
  relative to their root. Skipped when the scene has no articulations.

Settings live under ``settle.validators.<check>``. For example, append
``settle.validators.pose_shift.max_translation_m=0.001`` to tighten the root limit.
To disable a check explicitly, use
``settle.validators.pose_shift.enabled=false``. Its skipped report remains in the
record; that record no longer certifies the disabled condition. At least one
applicable validator must remain enabled. Keep the defaults for the SQA example.

With the default checks, a layout that comes to rest after a large drop is rejected.
``On`` defaults to 1 cm of release clearance, which exceeds the recording shift
limit. For an initial arrangement intended to remain still, use a smaller clearance,
as in the sample, rather than weakening the recording limit. The accepted final
pose, including a small permitted adjustment, is what gets recorded. Joint
states are not saved, so excessive link motion also rejects the layout.

The solver's IK and geometric checks are not repeated after physics. The default
post-physics checks certify the velocity and shift limits above, not exact
preservation of every relation or a new IK solution at the measured poses.

Python use and scope
--------------------

For an initialized environment with a placement pool:

.. code-block:: python

   from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams
   from isaaclab_arena.offline_placement.settled_placement import collect_settled_pool_layouts
   from isaaclab_arena.relations.placement_events import get_placement_pool

   result = collect_settled_pool_layouts(
       env, get_placement_pool(env), PlacementRecordingParams(num_steps=120),
       scene_assets=arena_env.get_placement_assets(),
   )
   result.layouts.write_episode_jsonl(
       "poses.jsonl", source="settled", validation=result.validation,
   )

Collection does not consume the pool or change its validation results. Before each
batch, it restores scene roots, joints and actuator targets so earlier candidates
cannot change the next candidates' starting conditions. It also restores that
state on completion or failure for Python callers. This does not restore validator
internals, task managers or all simulator state.

The recorder and ``run_placement_pool_validation.py`` share the same physics loop; a separate
validation run is unnecessary. Pool validation lives in
``isaaclab_arena.offline_placement.pool_validation``. The velocity-only pool
validator also supports deformables; root-pose recording does not.

Use concrete assets with writable rigid or articulation roots. Object sets and
``RandomAroundSolution`` are unsupported. Recording checks replay compatibility
before stepping physics: recorded assets must allow pose resets, have zero initial
velocity, and have no randomized or per-environment pose-reset policy. Replay
requires the same assets and robot joint reset configuration. Disable pose-changing
variations and callbacks
when exact pose restoration is required. This tool records object and robot root
poses, not general variations or joint states.

Custom checks
-------------

Custom post-physics checks subclass ``PostPhysicsPlacementValidator`` in
``isaaclab_arena.offline_placement.post_physics_validation``. Define a dataclass with a unique
``check`` name and implement ``validate(PostPhysicsState)`` to return one
``PlacementValidatorReport`` per ``env_ids`` entry, in order. Use ``self.report``
to retain the effective settings. Enabled, applicable checks must return pass/fail;
``skip_reason`` describes scene-level inapplicability.

For an importable ``my_project.validators.SupportValidator`` with
``check = "support"``, add it to the existing checks with:

.. code-block:: bash

   +settle.validators.support._target_=my_project.validators.SupportValidator

The shared ``PlacementValidator`` base lives in ``relations.placement_validation``.
Existing solver validators keep their batch API; post-physics implementations
live under ``offline_placement``. The online placement path does not import the
offline package.
