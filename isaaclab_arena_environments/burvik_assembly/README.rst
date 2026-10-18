BURVIK assembly with bimanual YAM
================================

``burvik_assembly_yam`` is a registered Python environment factory, following the
maintained Maple-table factory. It uses the previously generated 38 cm BURVIK tray,
four legs, handle, and 2.0 x 1.5 x 0.75 m workbench at their original scale.

Local USD payloads and recordings are not committed. The environment defaults to
``artifacts/ikea_burvik/assets`` for the six parts and
``.cache/tool_hanging_mdl_patch/assets/industrial__i2rt_yam/i2rt_yam_default.usda``
for each YAM arm. Retain the robot's referenced payload directory and the parts'
textures beside their USDs. The design specification is committed separately;
reusable generation and audit scripts live in ``tools/burvik_assets``.

The shared YAM embodiment was brought from ``main_old`` commit
``c8d04e2199b86abbbb301bb22cec0effd83c4e63``, under
``isaaclab_arena/embodiments/yam``.
Its class now lives in ``yam.py`` to keep package initializers free of imports.
The embodiment supports ``joint_position``, ``ik_pose``, and ``ik_delta`` action
modes. BURVIK defaults to ``ik_pose``. The original displacement demonstration
explicitly selects ``joint_position`` and parks the wrists beside the work area.

YAM pose control
----------------

``ik_pose`` has 16 actions: left TCP ``x, y, z, qx, qy, qz, qw``, left gripper,
right TCP ``x, y, z, qx, qy, qz, qw``, right gripper. Each TCP pose is expressed
in that arm's own articulation base frame. Grippers use 0 for open and 1 for
closed. Quaternions are normalized; nonfinite commands and zero quaternions
are rejected. ``absolute_pose_actions`` converts world targets into this layout.

``ik_delta`` uses base-frame translation increments and an axis-angle rotation
vector per arm, followed by its gripper (14 actions total), following Isaac
Lab's differential IK convention. Joint
control retains its original 14-value layout. The tool frame is offset from
``link_6`` by (0, -0.044, 0.13) m. The IK action uses Isaac Lab's damped least
squares controller, corrects the rotated TCP lever arm in its Jacobian, and
bounds joint targets by both step size and the imported joint limits.
The default tracking gain is 1.0. Model gravity is compensated through a
joint-position bias, so the existing drives still enforce their effort limits.
The PhysX dynamics buffer
can include six constrained-base entries; the action accounts for those entries
when selecting the arm's gravity torques.

The registered ``burvik_yam_assembly`` policy reads simulation state and returns
only pose/gripper actions. Its states cover pickup, contact grasp, measured lift,
alignment, insertion, physical retention, release, bimanual turning, handle
insertion, and placement. State failures raise an error rather than advancing
the task. After release, the policy holds for at least five seconds and waits up
to ten more seconds for Arena's current settled-success condition. It does not
finish merely because the ordered milestones were previously recorded. The policy currently supports one environment and uses fixed grasp
candidates for this layout; it does not provide general collision planning.
While carrying the shared table, it commands 20% of the measured TCP pose error
to reduce opposing-arm loads. Pickup and insertion use the full pose target.
Finger commands ramp during both grasp and release. Side placement stops on
measured bench support before the right hand releases the table. The open jaws
are re-centered on the settled leg before withdrawal. The empty right hand
then follows raised clearance waypoints before picking up the handle.

The IK layout raises the robot base origins to world height 0.90 m. The
132.5 mm pedestals leave clearance for the base colliders below those origins.
Leg grasps are 220 mm below their pin seats. Two 35 mm presentation
rails support the thin handle above the workbench,
leaving clearance for a full finger grasp. The handle pickup rotates the jaws
15 degrees about the tube to retain wrist joint-limit margin through transfer
and insertion. Handle transfer waypoints account
for its bounds to clear the supports, bench, and tray face. Two-pin insertion
advances only below 0.3 mm lateral and 0.2 degree angular error, using a slower
compliant pose command and bounded integral correction throughout the approach.
After releasing the handle, the right arm follows FK-generated TCP waypoints
to fold its open wrist and clear the tray. These remain pose actions; no joint
states are written. The left wrist turns through its neutral angle before
regrasping for the final table turn. Oblique
regrasps respect the wrists' joint limits. Final carrying uses a common table
trajectory and fixed measured grasp transforms, avoiding accumulated wrist
rotation around the round tubes. A shared vertical integral correction, bounded
to 10 mm, compensates static payload droop at carrying waypoints. The final grasps move to approximately 120 mm
below the pin seats to reduce the load's lever arm, then close firmly for carrying.
The policy checks both grasps and all retained legs while moving. The
PhysX configuration uses extra solver iterations and resolves articulation
contacts last to improve convergence of loaded bimanual grasp constraints.
Pose control uses 128 position iterations and 16 velocity iterations, with
gravity applied once per step. In this runtime, applying gravity every solver
iteration improved resting velocity reports but destabilized physical grasps.
Arm drives use stiffness 3000 and damping 80; finger drives use stiffness 20000,
damping 120, and a 100 N effort limit. Loaded TCP waypoints allow 12 mm tracking
error; the separate insertion and final assembly predicates retain their
millimetre seating tolerances.

Run the robot policy, optionally recording the YAM top camera::

    /isaac-sim/python.sh -m isaaclab_arena_environments.burvik_assembly.robot_demo \
        --video --output artifacts/ikea_burvik/yam_policy

For faster validation, omit ``--video`` and subsequently render the recorded
physical trajectory through the same Isaac Sim camera::

    /isaac-sim/python.sh -m isaaclab_arena_environments.burvik_assembly.render_rollout \
        artifacts/ikea_burvik/yam_policy

This playback requires a successful validation report, writes ``rendering.json``,
and does not rerun dynamics or establish task success. The saved physics rollout
supplies every part pose and robot joint position used in the video.

``--stop-after-leg 1`` is a partial diagnostic, never reported as full success.
Every run writes ``validation.json`` with measured Arena completion and state
transitions. The IK scene uses tuned simulation servo gains; these gains and
gripper force limits have not been calibrated against YAM hardware.

The complete pose-policy rollout in ``artifacts/ikea_burvik/yam_ik_assembly_v10``
passed all five ordered steps and the current final conditions. The released
table remained supported for 5.3375 seconds with maximum part drift of 0.147 mm,
maximum seating position error of 0.334 mm, and 25.68 N upward bench support.
``validation.json`` records source hashes, measured errors, and Arena events;
``yam_top.mp4`` is a camera rendering of that saved physical trajectory.

Original displacement demonstration
-----------------------------------

Run inside this checkout's existing Docker container, as the host user::

    cd /workspaces/isaaclab_arena
    /isaac-sim/python.sh -m isaaclab_arena_environments.burvik_assembly.scripted_demo \
        --video --check-reachability --output artifacts/ikea_burvik/arena_results

Record the actual Kit viewport camera and YAM top sensor simultaneously::

    /isaac-sim/python.sh -m isaaclab_arena_environments.burvik_assembly.scripted_demo \
        --video --record-cameras viewport top --output artifacts/ikea_burvik/arena_camera_recording

This produces ``burvik_arena_viewport.mp4`` and ``burvik_arena_top.mp4`` from the
same rollout at 1280 x 720, 24 fps. The viewport uses ``/OmniverseKit_Persp`` through
Kit's public frame-capture API. The imported YAM ``top_camera`` retains its optics
and orientation, with a task-specific mount at (0, 0.06, 2.60) m to keep the loose
parts and assembled handle in view. Only the requested sensors are enabled.
``--record-cameras overview`` selects the original overview stream and remains
the default for ``--video``. The validation report records camera paths, poses,
and frame counts. All streams are written incrementally during simulation.

Use ``--smoke --num-envs 2`` for a bounded build/reset/settling check. The factory's
``asset_directory`` and ``robot_usd_path`` are configurable; their defaults point
to the existing local BURVIK artifacts and cached Isaac Cap YAM asset. Keep both
asset directories outside version control. For direct Python construction::

    # After SimulationApp startup:
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.burvik_assembly_environment import (
        BurvikAssemblyEnvironment, BurvikAssemblyEnvironmentCfg,
    )

    description = BurvikAssemblyEnvironment().build(BurvikAssemblyEnvironmentCfg())
    env = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(
        device="cpu", solve_relations=False, disable_fabric=True,
    )).make_registered()
    env.reset()

The layout is fixed; ``solve_relations=False`` preserves the measured layout.
``--check-reachability`` reuses Isaac Lab's differential IK controller and the live
YAM Jacobians to check all six pickup points and five insertion grasp points
against joint limits. This is positional TCP reachability; grasp orientation,
collision-free robot paths, and grasp execution are not certified.

Task semantics
--------------

The task reuses ``CompositeTaskBase`` with sequential subtasks and all five current
final states required: leg 1, leg 2, leg 3, leg 4, then handle. It reuses
``TrueForConsecutiveStepsCfg``, ``objects_below_velocity_thresholds``,
``contact_force_is_upward_support``, the dropped-object termination, and Arena's
progress and episode recorders. No separate progress counter or temporal runner
is implemented.

The small ``InstallBurvikPartTask`` extension is needed because the existing
``AssemblyTask`` tests root proximity only. Each new subtask checks its actual
tray-relative pose and enabled retention joint. Smooth-pin seating within
0.3 mm radially, 1.3 mm axially for legs (0.3 mm for the handle), and 1 degree
engages a part-to-part fixed joint at the measured relative pose, without a world
anchor. The handle target includes a measured 1.5 mm axial stand-off so its tube
shoulders clear the tray surface; the original USD geometry is preserved.
This approximates push-fit retention; it does not
model frictional interference. Seated connections use ideal fixed retention with
no fracture model; the previously estimated break limits were not supported by
material-strength data. Part mass and inertia remain estimates from the original assets.

Leg conditions must hold for 24 steps (0.1 s). The final handle condition must
hold for 120 steps (0.5 s), including upright orientation, correct standing height,
low part velocities, and at least 20 N upward support measured by four contact
sensors filtered against the workbench. All previous attachments must still hold.
Settling measures each part's translation and full quaternion rotation between
consecutive physics steps, with limits of 0.02 m/s and 0.05 rad/s. Reported linear
speed must also remain below 0.02 m/s. This avoids treating TGS angular-velocity
residuals in the fixed leg joints as physical movement: a standing diagnostic
showed sub-micrometre pose drift while reporting occasional 0.26 rad/s residuals.
Actual rotation, including axial leg rotation, still prevents settling.
Repeated predicate calls share one measurement per step; reset invalidates only
the affected environments' pose samples. Arena retains the consecutive-step hold.
Reset removes only the resetting environments' connection joints. Arena owns
the progress, temporal counters, and their per-environment reset.

Scripted demonstration
----------------------

The original joint-action displacement demonstration retains 128 position /
16 velocity iterations with external forces applied once per step.

The demo calls the real wrapped Arena ``env.step`` with the imported YAM's joint
actions and guides dynamic parts with bounded velocity commands. It installs the
four legs upside down, turns the assembly sideways to insert the handle, then
turns it upright and releases all part guidance. Five seconds of free-standing
stability are measured after release. Robot and part collisions remain enabled;
joined part pairs are collision-filtered by their retention joints.

This validates the assembly environment and success semantics using the requested
scripted part displacement approach. The YAM remains parked during assembly; the
video is not a robot-grasp policy demonstration. PhysX is the validated runtime;
dynamic retention has not been implemented or validated for Newton.

The output folder contains the full MP4, reachability residuals, trajectory,
validation report, final image, and USD scene. The report records ordered Arena
progress events, final subtask states, measured seating errors, support force,
and released drift. Focused tests in ``test_burvik_assembly_task.py`` cover false
seating, wrong orientation, unsupported/moving tables, ordering, interrupted
holds, current final-condition loss, and partial reset isolation.
