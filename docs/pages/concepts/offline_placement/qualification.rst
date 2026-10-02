.. _offline-placement-verification:

Verify Recorded Layouts
========================

Check the output count, validation reports, and measured replay poses before
using a recording in an evaluation. A zero exit status, an existing file, or a
viewer screenshot alone does not establish that the requested layouts were
recorded and replayed correctly.

Start with the recipes in :doc:`clutter`. Keep each recording's source YAML,
command, backend, device, and full log alongside its JSONL in the ignored
``outputs/clutter/`` directory. Use the same runtime for recording and replay.

Check Every Recorded Row
-------------------------

Save this standard-library-only checker as ``outputs/clutter/check_jsonl.py``.
It reads the whole file, checks finite poses and unit quaternions, and requires
the release and post-physics checks used by the clutter recipes. It can run on
the host without starting Isaac Sim.

.. code-block:: python

   import json
   import math
   import sys
   from pathlib import Path


   path = Path(sys.argv[1])
   expected_count = int(sys.argv[2])
   assert expected_count > 0
   records = []
   for line_number, line in enumerate(path.read_text().splitlines(), 1):
       if line.strip():
           record = json.loads(line)
           records.append((line_number, record["variations"]["scene.relation_placement"]))
   assert len(records) == expected_count, (len(records), expected_count)
   expected_roots = set(records[0][1]["poses"])
   assert expected_roots

   for line_number, record in records:
       assert record["source"] == "settled", line_number
       assert set(record["poses"]) == expected_roots, line_number
       for root, pose in record["poses"].items():
           for field, size in (("position_xyz", 3), ("rotation_xyzw", 4)):
               values = pose[field]
               assert isinstance(values, list) and len(values) == size, (
                   f"row {line_number}, root {root!r}: {field}={values!r}; "
                   f"expected a list of {size} values"
               )
               assert all(
                   isinstance(value, (int, float))
                   and not isinstance(value, bool)
                   and math.isfinite(value)
                   for value in values
               ), f"row {line_number}, root {root!r}: non-finite or nonnumeric {field}={values!r}"
           norm_squared = sum(value * value for value in pose["rotation_xyzw"])
           assert math.isclose(norm_squared, 1.0, rel_tol=0, abs_tol=1e-4), (
               f"row {line_number}, root {root!r}: quaternion squared norm "
               f"{norm_squared!r}, expected 1 within 1e-4; "
               f"rotation_xyzw={pose['rotation_xyzw']!r}"
           )

       validation = record["validation"]
       for check in ("no_overlap", "clutter_on_relation"):
           assert validation["pre_physics"][check] is True, (line_number, check)
       reports = validation["post_physics"]
       assert reports
       checks = {}
       for report in reports:
           check = report["check"]
           checks[check] = report
           assert report["passed"] is True or report["passed"] is None, (line_number, check)
           if report["passed"] is None:
               assert report["reason"], (line_number, check, "Missing skip reason")
       for check in ("physics_settled", "pose_shift", "support_containment"):
           assert checks[check]["passed"] is True, (line_number, check)

   print(f"PASS: {len(records)} settled layouts; roots: {sorted(expected_roots)}")

Run it from the repository root for each completed recipe:

.. code-block:: bash

   python3 outputs/clutter/check_jsonl.py outputs/clutter/tools_on_table.jsonl 1
   python3 outputs/clutter/check_jsonl.py outputs/clutter/tools_on_table_batch.jsonl 10
   python3 outputs/clutter/check_jsonl.py outputs/clutter/three_cubes_in_bowl.jsonl 1

The expected count is the requested ``min_layouts``, not the number that happened
to be written. A partial recording therefore fails this target check even when
each saved row is valid. No file is expected after zero accepts.

``passed: null`` means a check was skipped; inspect its reason and configuration.
The checker permits justified skips for other checks but requires the three
listed clutter checks to pass. Also inspect ``validation.sampling`` for the
intended settling steps, decimation, and physics timestep, and the containment
report's ``minimum_resting_heights_m`` for the intended support-local height.

Consistent root keys across rows do not prove that all roots in the intended
scene were recorded. The replay check below compares the file with the actual
runtime roots. For an external environment, construct that environment through
its maintained factory so this comparison uses its full scene configuration.

Measure Replay Across Resets
-----------------------------

Save the following as ``outputs/clutter/check_replay.py`` and run it in the same
Arena runtime, with the same backend and asset configuration as recording. This
example checks the ten-layout table file in two environments. It compares every
root immediately after reset, exercises queue wraparound and a partial reset,
and checks zero root velocities without advancing physics or a policy. Before
each reset it displaces every recorded root. It sets nonzero velocities only
for dynamic rigid bodies and floating-base articulations, using their spawned
physics properties. Kinematic bodies and fixed-base articulations still have
their poses restored and observed velocities checked, but receive no velocity
perturbation.

.. code-block:: python

   import argparse
   from dataclasses import replace

   from isaaclab.app import AppLauncher
   from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

   parser = argparse.ArgumentParser()
   AppLauncher.add_app_launcher_args(parser)
   args = parser.parse_args()

   with SimulationAppContext(args):
       import torch

       from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
       from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
       from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
       from isaaclab_arena.offline_placement.clutter_geometry import (
           spawned_rigid_body_is_dynamic,
       )
       from isaaclab_arena.relations.placement_layouts import PlacementLayouts

       source = "isaaclab_arena_environments/clutter/franka_three_hammers_and_clamp_no_task.yaml"
       path = "outputs/clutter/tools_on_table_batch.jsonl"
       layouts = PlacementLayouts.from_episode_jsonl(path)
       arena_env = ArenaEnvGraphSpec.from_yaml(source).to_arena_env()
       if arena_env.placer_params is not None:
           arena_env.placer_params = replace(
               arena_env.placer_params, placement_seed=None, resolve_on_reset=True
           )
       env = ArenaEnvBuilder(
           arena_env,
           ArenaEnvBuilderCfg(
               num_envs=2, device=args.device, presets="physx",
               placement_layouts_path=path, resolve_on_reset=True,
           ),
       ).make_registered()
       try:
           base = env.unwrapped
           runtime_roots = set(base.scene.rigid_objects) | set(base.scene.articulations)
           assert set(layouts.poses) == runtime_roots, (set(layouts.poses), runtime_roots)

           velocity_roots = {
               name for name in base.scene.rigid_objects
               if spawned_rigid_body_is_dynamic(base.scene, name)
           }
           velocity_roots.update(
               name for name, body in base.scene.articulations.items()
               if not body.is_fixed_base
           )
           print("Velocity perturbation roots:", sorted(velocity_roots))

           def check(env_ids, first_layout):
               for name, poses in layouts.poses.items():
                   expected = torch.stack([
                       poses[(first_layout + offset) % layouts.num_layouts].to_tensor(base.device)
                       for offset in range(len(env_ids))
                   ])
                   actual = base.arena_world.get_pose_e(name)[env_ids]
                   torch.testing.assert_close(actual, expected, atol=2e-5, rtol=0)
                   velocity = base.scene[name].data.root_vel_w.torch[env_ids]
                   torch.testing.assert_close(velocity, torch.zeros_like(velocity), atol=0, rtol=0)

           def displace(env_ids):
               for name in layouts.poses:
                   body = base.scene[name]
                   pose = body.data.root_pose_w.torch[env_ids].clone()
                   pose[:, 2] += 0.25
                   body.write_root_pose_to_sim(pose, env_ids=env_ids)
                   if name in velocity_roots:
                       body.write_root_velocity_to_sim(
                           torch.ones((len(env_ids), 6), device=base.device), env_ids=env_ids
                       )

           next_layout = 0
           all_ids = torch.arange(base.num_envs, device=base.device)
           for _ in range(layouts.num_layouts + 1):
               displace(all_ids)
               env.reset()
               check(all_ids, next_layout)
               next_layout = (next_layout + base.num_envs) % layouts.num_layouts

           untouched = {
               name: base.arena_world.get_pose_e(name)[0].clone() for name in layouts.poses
           }
           reset_ids = torch.tensor([1], device=base.device)
           displace(reset_ids)
           base._reset_idx(reset_ids)
           check(reset_ids, next_layout)
           for name, expected in untouched.items():
               torch.testing.assert_close(
                   base.arena_world.get_pose_e(name)[0], expected, atol=2e-5, rtol=0
               )
           print(f"PASS: {layouts.num_layouts} layouts, complete roots, wraparound and partial reset")
       finally:
           env.close()

For the PhysX CPU recipes, run from the repository root in the
:doc:`selected Arena runtime <recording>`:

.. code-block:: bash

   python outputs/clutter/check_replay.py --viz none --device cpu

For the one-layout table recording, change ``path`` to
``outputs/clutter/tools_on_table.jsonl``. For the bowl recording, also change
``source`` to
``isaaclab_arena_environments/clutter/franka_three_cubes_in_bowl_no_task.yaml``
and ``path`` to ``outputs/clutter/three_cubes_in_bowl.jsonl``. Keep ``presets``
and the device consistent with the recording being checked.

The absolute pose-component tolerance ``2e-5`` and zero relative tolerance match
``test_companion_cache_round_trip`` and ``test_recording_writes_complete_layouts``.
This is a numerical reset comparison, not a tolerance for later physical drift.
The partial-reset call uses Isaac Lab's internal ``_reset_idx`` entry point, as
the existing replay test does. It is used here to verify reset behavior, not as
an application API.

Run Controlled Failure and Compatibility Checks
------------------------------------------------

Run these targeted tests from the repository root using ``python`` in your
selected Arena runtime. Keep tests that launch child simulations in a separate
pytest process. These commands cover the named behaviors; they are not a claim
that the full suite passed.

.. code-block:: bash

   python -m pytest -sv \
       -m 'not with_cameras and not with_subprocess' \
       isaaclab_arena/tests/clutter/test_recording.py \
       isaaclab_arena/tests/clutter/test_clutter_collection.py::test_settling_rejects_object_sets_before_reset \
       isaaclab_arena/tests/test_settled_placement.py::test_record_placements_to_jsonl_writes_partial_acceptance \
       isaaclab_arena/tests/test_placement_layout_replay.py::test_cache_rejects_conflicting_configuration \
       isaaclab_arena/tests/test_placement_layout_replay.py::test_companion_cache_round_trip

   python -m pytest -sv -m with_subprocess \
       isaaclab_arena/tests/test_settled_placement.py::test_recording_cli_writes_partial_acceptance \
       isaaclab_arena/tests/clutter/test_record_placement_layouts.py

``test_post_physics_checks_gate_output`` deliberately rejects candidates and
checks that no file is created. ``test_recording_cli_writes_partial_acceptance``
accepts one candidate and rejects another, then checks that exactly one row and
a shortfall message are produced even though the CLI exits normally.
``test_recording_writes_complete_layouts`` checks complete recordings, replay,
refusal to rerecord cached layouts, and byte-for-byte preservation of an existing
output. The compatibility tests cover unresolved object sets, conflicting seeds,
incomplete required roots, two layout sources, and incompatible root resets.

The primitive CLI test parametrizes PhysX and Newton; the maintained table CLI
test uses its environment's default backend. These tests do not establish that
the real table or bowl recipe has passed on every backend or device. Run each
advertised recipe itself and apply the file and measured-replay checks above.

.. _clutter-physx-diagnostics:

Interpret PhysX Diagnostics
---------------------------

The supplied CPU PhysX recipes can emit these messages during normal recording
and replay, independently of the checker's deliberate perturbations:

* ``PxRigidDynamic::setLinearVelocity`` or ``setAngularVelocity`` reports
  ``Body must be non-kinematic!``. The background and pose-reset paths attempt
  zero-velocity writes to the kinematic table and bowl. PhysX rejects those
  writes. Restricting the checker's nonzero writes to dynamic roots does not
  remove diagnostics from these normal resets.
* During bowl setup, ``PxRigidBody::setRigidBodyFlag()`` reports
  ``kinematic bodies with CCD enabled are not supported! CCD will be ignored.``
  PhysX ignores continuous collision detection for that kinematic body. This
  message does not mean that CCD is disabled for every object in the scene.

These are known limitations of the supplied scenes and reset paths, not a
reason to ignore PhysX errors generally. Retain the log, confirm the fixed
support remains in place, and require the output-count, validation-report and
measured-replay checks to pass. Investigate other errors, changed scene settings,
unexpected movement, or failed assertions before accepting a run. A checker
``PASS`` reports its assertions; it does not certify an error-free simulation.

Retain Reproducible Evidence
-----------------------------

For each run, retain the Arena commit and dirty state, dependency/runtime
versions, OS/GPU, backend/device, exact command and seed, source YAML or factory
configuration, validator settings, full log, accepted/attempted counts, rejection
summary, JSONL, and checker output. Keep this material in an ignored output
directory or your artifact store; avoid committing datasets or recordings.

Record GUI inspection separately: the scene should open with the expected
arrangement, and settled clutter should visibly rest on its support or inside
its container. A one-layout ``NoTask`` viewer run checks appearance; the measured
reset procedure checks file ordering and root restoration. Record failures and
partial results explicitly, and report only the configurations actually run.
