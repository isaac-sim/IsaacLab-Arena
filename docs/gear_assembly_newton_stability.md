# Gear Assembly on Newton: Stability Engineering Note

This note explains how the Arena Gear Assembly environment was made stable enough for repeatable pick-and-lift rollouts with the Newton backend. It records the problems we found, the changes that fixed them, and the checks used to decide that a result is stable.

> [!IMPORTANT]
> The current result is stable for independent, single-environment rollouts of the small, medium, and large gears. The 3×3 rollout video is a grid of nine independent runs, not one simulation containing nine replicated environments. Replicated multi-environment Newton pickup still has a contact-instability issue, described under [Known limitation](#known-limitation).

## Goal and acceptance criteria

The first milestone is deliberately smaller than full gear assembly: the DROID robot must start in a valid pose, close the Robotiq gripper around one gear, and lift it cleanly from the table.

We call a rollout stable when all of the following are true:

- The robot, gripper, gear, base, and table start without a visible jump.
- Every pose, joint state, and velocity remains finite.
- The arm stays near its command while the gripper closes.
- Both finger pads follow the Robotiq mimic-joint relationship.
- The gear does not slide or get ejected during grasp closure.
- The gear rises at least 80 mm during the lift and stays below the approach height plus 30 mm, which rejects contact explosions.
- The rendered gripper uses the original visual meshes and is clearly visible.

## Why the PhysX setup did not transfer directly

The source Gear Assembly task works with PhysX, but Newton imports and solves some parts of the scene differently:

- The gear meshes are concave. Newton's convex approximation filled important spaces such as the center bore and stepped hub.
- The Robotiq pad collision meshes are nested under transformed USD prims. Their imported contact position did not reliably match the visible fingertips.
- Some DROID rigid bodies did not contain complete positive mass and inertia data.
- Disabling gravity on the articulated robot affected the imported scene more broadly than intended, so the gears also lost gravity.
- The source reset uses an end-effector Jacobian that is not exposed in the same way by this Newton integration.
- A single gripper command was not enough. The three hub sizes need different open and closed targets, and an instantaneous command creates a large contact impulse.

These are asset-import and contact-model differences. Treating them only as a solver-tuning problem made the environment harder to diagnose and did not fix the root causes.

## Stabilization process

The work was done from the lowest layer upward. Each layer was checked before moving to the next one.

### 1. Check the scene at rest

The base, three gears, table, and robot were spawned without an action. This separated bad initial placement or collision geometry from gripper-control problems.

The movable gears remain dynamic, the base is kinematic, and a simple tabletop box provides an explicit Newton collision surface. Inactive gears are parked at separated positions on that surface.

### 2. Give Newton simple, explicit collision shapes

The original render meshes are still used for appearance. Separate guide-purpose meshes are used only for Newton collision:

- The base is one platform and three independent convex pegs.
- Each gear uses two height tiers. Each tier is a six-segment ring, giving 12 convex pieces per gear. This keeps the bore open and follows the visible gear footprint without passing a large concave mesh to Newton.
- The gripper uses two small link-local convex boxes, one on each inner finger. The original nested Robotiq collision leaves are disabled.

The proxy meshes have USD purpose `guide`. This keeps them active for physics without rendering colored boxes over the real gripper. The collision roots are also de-instanced before editing so the full Robotiq render hierarchy remains available.

### 3. Repair the DROID USD at load time

`ensure_newton_compatible_droid_usd` creates a cached `_newton_droid.usd` next to the resolved source asset. It makes only the Newton-specific corrections:

- Missing or invalid rigid-body mass is clamped to 0.02 kg.
- Missing diagonal inertia is clamped to `1e-5`.
- Invalid center-of-mass and principal-axis values are repaired.
- `mjc:gravcomp=1` is authored per robot rigid body.
- The two direct fingertip collision proxies are added.

Per-body gravity compensation keeps the arm supported while preserving normal gravity for the loose gears. The cache is validated before reuse, so a stale or partially generated asset is rebuilt automatically.

### 4. Make reset deterministic and backend-compatible

The Newton path uses this fixed, collision-free Franka seed:

```text
(0.98, -0.47, -1.73, -1.42, -1.28, 2.71, 1.35)
```

Robot joint randomization is disabled because the DROID Gear Assembly specification does not request it. A small finite-difference inverse-kinematics solver computes the reset pose without depending on the unavailable Newton end-effector Jacobian. The reset is accepted only when position and rotation error are each below `1e-3`.

The end-effector reference is the Robotiq `base_link`. The grasp is rotated 30 degrees so the parallel pads meet flat faces of the six-segment hub instead of closing against corners. Gear-specific offsets place the center of the pads on each raised hub.

### 5. Close the real gripper smoothly

All six Robotiq joints are commanded with their correct mimic signs. Open and closed positions are selected by gear size, and the action term limits each target change to `0.5 * control_dt`. This slew limit avoids asking the fingers to jump directly into the gear.

| Gear | Open target | Closed target | Grasp offset [m] |
|---|---:|---:|---|
| Small | 0.50 | 0.650 | `[-0.16245, 0.0, 0.0]` |
| Medium | 0.30 | 0.461 | `[-0.16085, 0.0, 0.0]` |
| Large | 0.24 | 0.412 | `[-0.15985, 0.0, 0.0]` |

The Newton gripper actuator uses an effort limit of 5, velocity limit of 1, stiffness 20, damping 5, and armature 0.1. These values let the fingers settle on the hub instead of driving through it or oscillating.

### 6. Tune contact only after fixing geometry and control

The final contact and simulation settings are:

| Setting | Value | Reason |
|---|---:|---|
| Physics step | `1 / 120 s` | Resolves the short contact events around the small hubs. |
| Control decimation | 4 | Preserves the task's 30 Hz control rate. |
| Newton substeps | 12 | Gives the contact solver smaller internal steps. |
| Shape gap | 0.0 | Avoids adding extra separation to millimeter-scale geometry. |
| Gear contact offset | `1e-4 m` | Keeps a small contact margin without noticeably closing the bore. |
| CCD iterations | 35 | Reduces missed contacts during closure and lift. |
| Constraint capacity (`njmax`) | 512 | The default 300 was too small for nine simultaneous grasps. |
| MuJoCo contact path | Disabled | The internal contact path hung while importing this mesh-heavy scene. |
| Gear and base friction | 0.75 static/dynamic | Keeps the loose objects stable without excessive sticking. |
| Robot friction | 2.0 static/dynamic | Prevents the medium and large hubs from slipping out of the pads. |
| Restitution | 0.0 | Avoids bounce during contact. |
| Environment spacing | 1.5 m | Keeps small collision coordinates closer to the world origin. |

Artificial linear and angular damping on the gears is zero. Stability therefore comes from valid geometry, controlled motion, contact, and support—not from hiding motion with heavy damping.

### 7. Measure physical success and visual correctness separately

The success term does more than test whether a gear is near the base. It checks the correct lateral offset and height for the selected gear, upright orientation, low linear and angular velocity, support height, and 10 consecutive successful steps. This prevents a fast fly-through from counting as assembly.

For pickup videos, the camera is moved to `(1.6, 1.2, 1.0)`. The original view placed the hub-aligned jaws nearly edge-on, which made a physically present gripper look like a narrow black column. Camera placement is only a visualization change; the stability checks use simulation state.

## Validation

### Automated test

Run the focused test inside the Arena development container:

```bash
/isaac-sim/python.sh -m pytest -q \
  isaaclab_arena/tests/test_gear_assembly_environment.py
```

The test checks the composed Newton configuration, gear and fingertip proxy geometry, preservation of all 11 Robotiq render meshes, finite state, IK accuracy, joint-limit margin, mimic-joint consistency, controlled closure, gear settling, and the consecutive-step success rule. The current result is `2 passed`.

### Pickup rollout matrix

The visual check uses three independent seeds for each gear. Accepted lift heights were measured relative to the starting gear height:

| Gear | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|
| Small | 94.736 mm | 95.176 mm | 95.282 mm |
| Medium | 96.343 mm | 90.374 mm | 92.178 mm |
| Large | 94.464 mm | 92.730 mm | 98.096 mm |

Some harder grasps used a slower rollout schedule: 75 close steps and 120 lift steps instead of 45 and 75. One medium rollout that reached 340.5 mm was rejected as a contact ejection and rerun; a large rollout with almost no lift was also rejected and rerun. Failed trials are useful diagnostics and are not counted as successful merely because the gear moved.

The final MP4 and GIF are generated development artifacts under `artifacts/newton_gear_pickup/`. They are not production code, and no rollout-only policy or video binary is added to the package.

## Experiments that did not solve the problem

Recording unsuccessful approaches avoids repeating them:

- Raising `njmax` fixed the known constraint-capacity limit, but did not remove replicated-world contact spikes.
- Changing gains, friction alone, lift speed, Cartesian waypoints, or grasp height did not repair invalid collision geometry.
- The MuJoCo internal contact path hung during scene creation with these meshes.
- Disabling physics replication failed because actuator defaults were not expanded from one environment to nine.
- The conjugate-gradient solver produced non-finite articulation state.
- Removing mimic-joint references prevented the follower fingers from reaching the required grasp pose.

## Known limitation

Independent single-world pickup is repeatable, but a true nine-world replicated Newton run is not yet stable. World 0 often succeeds while later worlds can see roughly 90–170 N contact spikes and eject a gear. Contact-to-world mapping is correct, and observed active constraints remain below `njmax=512`, so this is not the original capacity overflow.

Until that issue is isolated in Newton/MuJoCo-Warp, the 3×3 video must be described as a composition of single-world runs. It must not be used as evidence that batched Newton simulation is stable.

## Code organization

The implementation keeps backend-specific behavior close to the layer that owns it:

| Area | Responsibility |
|---|---|
| `isaaclab_arena/utils/usd/newton.py` | Cached DROID USD repair and fingertip collision authoring. |
| `isaaclab_arena/tasks/gear_assembly/assets.py` | Newton gear, base, and tabletop collision proxies. |
| `isaaclab_arena/tasks/gear_assembly/specs.py` | Backend-specific poses, grasp geometry, widths, and materials. |
| `isaaclab_arena/tasks/gear_assembly/actions.py` | Gear-specific, rate-limited gripper targets. |
| `isaaclab_arena/tasks/gear_assembly/events.py` | Finite-difference IK reset and safe inactive-gear parking. |
| `isaaclab_arena/tasks/gear_assembly/terminations.py` | Settled and supported assembly success test. |
| `isaaclab_arena_environments/gear_assembly_environment.py` | Newton composition and solver settings. |
| `isaaclab_arena/tests/test_gear_assembly_environment.py` | Focused configuration, asset, reset, closure, and settling regression test. |

This structure avoids changing the PhysX path and avoids adding a production example whose only purpose is to generate one video. The next engineering step is to reduce the replicated-world failure to a small Newton reproducer before adding training-scale environment counts.
