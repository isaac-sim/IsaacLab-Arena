# CAP USB-C insertion

The task definitions match Isaac-cap main `587c7b047c635f64404f4b6a1a88333b362253df`.
Both variants use bimanual YAM, a table background, a bench, and a full-size plug:

| Variant | Receiver | Cables | Reset randomization |
| --- | --- | --- | --- |
| Easy | Fixed chamfered port | Plug: 16 links / 240 mm | Plug: ±12 mm XY, ±8° yaw |
| Medium | Free bulkhead on front/rear cradles | Plug plus 8-link / 100 mm bulkhead cable | Plug: ±18 mm XY, ±50° yaw; bulkhead: ±4 mm XY, ±3° yaw |

## Placement and success

The YAML graphs use Arena's ordinary placement and reset system. The table's
`placement_surface` is an object reference; the table itself is a background.
The bench and cradles are anchored kinematic objects. Medium places the bulkhead
`on` the front cradle with `overlap: true`, translating CAP's older `allow_overhang: true`
API. The rear cradle supports the barrel physically; it deliberately has no second
whole-body `on` constraint. Placement requires valid support relations without
best-loss fallbacks.

Both tasks reuse `depth_in_range`, `lateral_in_proximity`, and
`velocity_below_threshold` from Arena's shared spatial predicates. They require
depth ≥10.4 mm, CAP's calibrated lateral error ≤8.7931792 mm, and plug speed
≤0.05 m/s.
Neither current CAP variant adds a tilt gate or maximum depth.
Arena's shared `gripper_released` predicate checks measured jaw clearance, and
`gripper_distance_from_object_exceeds_threshold` checks the embodiment-owned
work-hand TCP against the plug (>40 mm for easy, >50 mm for medium). The YAM
embodiment owns the robot joint and frame details; the task graphs contain only
the grasp geometry and success thresholds. Both predicates must pass;
`gripper_released` is included only when `require_released` is enabled.
Success is not just geometric alignment while a hand still holds the plug.

The shared graph `env_cfg_override` selects Newton and configures 60 Hz control
with sixteen solver substeps per control step,
`implicitfast`, an elliptic friction cone, `impratio=10`, exact authored connector
meshes, and the collision pipeline. A small runtime hook installs CAP's custom
cleanup manager and applies robot actuator tuning. Embodiment and object spawn
addons apply the full-hand, passive-jaw, connector, bench, and table contact
properties before Newton imports the assets. Both arms use stiffness 1600 /
damping 70; driven grippers use stiffness 40000 / damping 40 with a 160 N limit.

Cables are registered assets with their own builder hooks and reset events.
Reset restores zero bend coordinates and velocities in both Newton state buffers
for only the selected environments. No extra task-level cable reset is needed.
Arena's general `Cable` asset cannot be used here because it is a standalone,
unwelded VBD articulation. CAP requires an MJWarp chain whose first link is
jointed directly to the dynamic plug or bulkhead so the cable stays attached and
moves with its connector.

## Assets

Assets are registered through `@register_asset` under `usbc_insertion_` names.
The active variants use `easy_plug`, `easy_port`, `medium_plug`, `bulkhead`,
`bench`, `cradle_front`, `cradle_rear`, `yam_table`, lighting, and `connector_cable`.

All USDs resolve from Arena's staging S3 bucket under
`temp_newton_envs/usbc_insertion/assets`, matching the hosted-asset convention used
by the cable-routing environments. The factories and demo do not expose a local
asset-root override.

## Behavior demo

Like the gear demo, this is a task-validation tool, not a robot policy. It uses a
predefined teleport trajectory that starts with every predicate false, then makes
depth, lateral alignment, low velocity, gripper release, and TCP withdrawal pass
one at a time. Each waypoint asserts the exact cumulative predicate state. Once
all five pass, the demo advances the ordinary environment, requires success
termination, and verifies that automatic reset increments the episode index and
clears the success state. In both variants, the receiver remains at its initial
workstation pose, the plug advances through local mating keyframes, and the work
gripper teleports with the plug before opening and withdrawing. Both task variants
use their unchanged thresholds.

Run inside this checkout's Arena container as the host user:

```bash
/isaac-sim/python.sh isaaclab_arena_environments/isaac_cap/usbc_insertion/usbc_env_behaviour_demo.py easy --cycles 2
/isaac-sim/python.sh isaaclab_arena_environments/isaac_cap/usbc_insertion/usbc_env_behaviour_demo.py medium --cycles 2
```

The waypoint previews render without advancing physics. Only the final all-true
state enters task termination, so intermediate predicate inspection cannot trigger
an early reset. Use `--cycles 1 --pause-steps 1 --no-real-time --viz none` for a
finite headless run.
