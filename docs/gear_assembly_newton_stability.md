# Stable Gear Assembly with Newton

This document describes the current Arena Gear Assembly setup for the Newton backend. It explains what is backend-specific, why it is needed, and how stability is checked.

## Environment

The environment contains:

- the DROID Franka arm and Robotiq 2F-85 gripper;
- the small, medium, and large Factory gears;
- the matching three-peg base; and
- the maple table and robot stand.

Newton is the default physics backend. The default embodiment is `droid_differential_ik`.

## Robot control

Runtime arm motion uses Isaac Lab's existing `DifferentialInverseKinematicsAction` and `DifferentialIKController`. Arena does not implement another IK solver. The seven-dimensional action is:

```text
[dx, dy, dz, dRx, dRy, dRz, gripper]
```

The first six values are relative Cartesian pose commands. They use axis-angle rotation, a scale of `0.5`, and Isaac Lab's damped-least-squares controller with `lambda=0.01`. The last value selects the gear-specific open or closed gripper target.

Newton exposes the DROID arm Jacobian columns at articulation joint indices `0:7`. Isaac Lab's standard action normally offsets those indices by six floating-base columns. For this asset that selects the wrong columns and produces no Cartesian motion.

`NewtonDroidDifferentialInverseKinematicsAction` changes only the Jacobian column selection. Pose processing, differential IK, joint target generation, and action application remain Isaac Lab code. The adapter is installed only for the Newton DROID differential-IK embodiment, so the PhysX path is unchanged.

The reset event is separate from runtime control. It uses finite-difference IK to place the robot at the selected gear's grasp pose because the upstream reset helper makes the same floating-base Jacobian assumption. Runtime rollout motion still uses Isaac Lab differential IK.

## Gripper setup

The grasp reference is the Robotiq `base_link`, with a `0.107 m` tool offset. The grasp is rotated by 30 degrees so the pads meet flat faces of the collision proxy. All six Robotiq mimic joints are commanded with the correct signs.

| Gear | Open target | Closed target | Grasp offset [m] |
|---|---:|---:|---|
| Small | 0.50 | 0.650 | `[-0.16245, 0.0, 0.0]` |
| Medium | 0.30 | 0.461 | `[-0.16085, 0.0, 0.0]` |
| Large | 0.24 | 0.412 | `[-0.15985, 0.0, 0.0]` |

Gripper targets are rate limited to avoid an instantaneous closing impulse. The Newton gripper actuator uses a `5.0` effort limit, `1.0` velocity limit, `20.0` stiffness, `5.0` damping, and `0.1` armature.

The source Robotiq collision meshes have nested transforms that do not import at the visible fingertip locations in this Newton version. Newton therefore uses one small link-local convex pad on each inner finger. The render meshes remain unchanged, so the visible fingers and physical pads move together.

## Collision geometry

Physics replication stays enabled. Disabling it makes replicated multi-environment startup impractical and is not needed for these assets.

Replication can convexify a concave mesh leaf. Applying it directly to a complete gear would fill the center bore and make insertion impossible. The Newton collision representation avoids that problem by making every replicated leaf convex before import:

- the base is one box platform plus three independent 12-sided convex pegs;
- each gear is split into a plate tier and a hub tier;
- each tier is decomposed into six convex annular leaves, preserving the bore; and
- the maple tabletop uses one explicit finite box collider.

Because every active gear leaf is already convex, replication cannot fill the bore. Newton 1.2.1 does not need to interpret the source SDF collider schema, and Arena does not need a runtime SDF construction callback. This is smaller and proved more stable in gripper contact than a runtime SDF collider.

The original source collision meshes are disabled. The proxy meshes use USD purpose `guide`, so they participate in physics without covering the detailed render meshes.

## Physics materials

The Factory USD files contain nested collision meshes. Assigning a material only to a rigid-object root does not reliably reach the Newton runtime shapes. In addition, Newton's per-articulation material view assumes a uniform contiguous shape layout, while this scene mixes gears, a base, and a robot with different shape counts. Using that view allowed gear friction to leak onto robot shapes and left some gear shapes with default material values.

The startup event now resolves each selected asset's global Newton shape labels and writes the runtime material arrays directly. The test reads those same global arrays back, rather than trusting a per-asset view.

| Shapes | Friction | Restitution |
|---|---:|---:|
| Gears and base | 0.75 | 0.0 |
| Finger pads | 2.0 | 0.0 |

Newton uses one friction coefficient for these runtime shapes, so the source static and dynamic value of `0.75` maps to that coefficient. Correct material assignment removed the sliding and bouncing that initially looked like a solver-frequency problem.

## Other physics settings

| Setting | Value |
|---|---:|
| Simulation step | `1 / 120 s` |
| Control decimation | 4 |
| Control rate | 30 Hz |
| Newton substeps | 12 |
| Integrator | `implicitfast` |
| Solver iterations | 100 |
| Line-search iterations | 15 |
| CCD iterations | 35 |
| Constraint capacity (`njmax`) | 512 |
| MuJoCo contact path | Disabled |
| Default shape gap | 0.0 |
| Gear contact offset | `1e-4 m` |
| Environment spacing | 1.5 m |

The gears keep zero artificial linear and angular damping. Stability comes from valid collision geometry, correct materials, and controlled contact rather than hiding motion with damping. The robot USD compatibility step also repairs invalid mass and inertia values, adds fingertip pads, and enables per-body gravity compensation without disabling gravity for loose objects.

## Seated success

Newton's proxy geometry places the physically seated gear root `0.0075 m` above the base root for all three gear sizes. The success term uses the base platform as its support surface; the taller peg colliders are deliberately excluded from the support-height bound.

Only the selected gear can complete the episode. It must meet all conditions for 10 consecutive environment steps:

- root XY position is within `0.015 m` of the matching peg;
- root height is within `0.01 m` of the seated height;
- its local Z axis is within 15 degrees of the base local Z axis;
- linear speed is at most `0.05 m/s`;
- angular speed is at most `0.5 rad/s`; and
- its bottom surface is within `0.005 m` of the base platform.

This prevents a gear that is falling through or briefly crossing the target volume from being reported as assembled.

## Rollout procedure

The validation rollout starts from the task's deterministic grasp reset and then:

1. closes the gripper gradually;
2. lifts the selected gear by 0.10 m;
3. transports it above the matching peg;
4. descends with the existing pose differential-IK action;
5. applies a shallow 5 mm downward preload so the gear is supported before release;
6. opens the gripper and retreats; and
7. waits for the task success term.

The preload matters because very small relative Cartesian commands do not build enough arm position-target error under grasp load. A 3 mm descent command tracks reliably while remaining slow at contact. This is a rollout command choice, not a second controller or a change to Newton.

A rollout is accepted only if robot and object tensors remain finite, the grasp is retained through transport, the selected gear passes the seated-success check after release, and the video visibly shows the gripper and gear. The 3 by 3 artifact is composed from nine labeled, independent one-environment runs: three executions for each of the three gear sizes. It is a visualization matrix, not a claim that nine environments were simulated in one batch.

## Automated validation

Run the focused regression file inside the Arena development container:

```bash
/isaac-sim/python.sh -m pytest -q \
  isaaclab_arena/tests/test_gear_assembly_environment.py
```

The checks cover scene composition, active collision leaves, disabled source colliders, global runtime materials, finite two-environment replicated simulation, gear-specific gripper closure, deterministic reset accuracy, seated success, and Newton DROID differential IK. The controller regression requires a finite `1 x 6 x 7` arm Jacobian, more than 15 mm of commanded upward motion, less than 2 mm of lateral drift, and finite joint state.

## Code map

| Area | Responsibility |
|---|---|
| `isaaclab_arena/embodiments/droid/actions.py` | Newton Jacobian column adapter and Robotiq action behavior. |
| `isaaclab_arena/embodiments/droid/droid.py` | DROID scene and action configuration. |
| `isaaclab_arena/tasks/gear_assembly/actions.py` | Gear-specific, rate-limited gripper targets. |
| `isaaclab_arena/tasks/gear_assembly/assets.py` | Gear, base, fingertip-pad, and tabletop collision proxies. |
| `isaaclab_arena/tasks/gear_assembly/events.py` | Reset placement and Newton runtime material assignment. |
| `isaaclab_arena/tasks/gear_assembly/specs.py` | Poses, grasp geometry, materials, and success thresholds. |
| `isaaclab_arena/tasks/gear_assembly/terminations.py` | Seated-and-settled success evaluation. |
| `isaaclab_arena_environments/gear_assembly_environment.py` | Default embodiment and Newton-specific composition. |
| `isaaclab_arena/tests/test_gear_assembly_environment.py` | Focused configuration and runtime regressions. |
