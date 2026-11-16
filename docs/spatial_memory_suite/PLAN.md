# Spatial-memory task suite: plan

Status: Phase 0 (orientation). This note records how Arena composes environments, how each of the ten
tasks maps onto what the repository offers, and the open questions. The user asked for the work to
continue through all phases without stopping, so each open question below lists the default that
was taken; review can overturn any of them.

## 1. How an Arena environment is composed

Read for this note: `README.md`, `AGENTS.md`, the `author-environment` skill,
`docs/pages/concepts/environment/*`, `docs/pages/concepts/task/concept_progress_tracking_design.rst`,
`isaaclab_arena/environments/{arena_environment_factory,isaaclab_arena_environment,arena_env_builder,
arena_env_builder_cfg,isaaclab_arena_manager_based_env}.py`, `isaaclab_arena/tasks/{task_base,
task_termination_cfg,lift_object_task,pick_and_place_task}.py`, `isaaclab_arena/embodiments/
{embodiment_base,franka/franka,g1/g1}.py`, `isaaclab_arena/assets/{object,object_base,object_library,
register,registries}.py`, `isaaclab_arena/metrics/*`, `isaaclab_arena/recording/*`,
`isaaclab_arena/policy/policy_base.py`, `isaaclab_arena/evaluation/policy_runner.py`,
`isaaclab_arena/utils/cameras.py`, `isaaclab_arena_environments/{__init__,cli,
franka_put_and_close_door_environment}.py` and the tests that build environments
(`test_close_door.py`, `test_all_environments.py`, `test_g1_locomanip_apple_to_plate.py`).

**Composition.** An environment is `IsaacLabArenaEnvironment(scene, embodiment, task, ...)`.
`ArenaEnvBuilder.build_env_cfg()` merges config fragments from the three parts with
`combine_configclass_instances`:

| Isaac Lab manager | Scene | Embodiment | Task |
| --- | --- | --- | --- |
| scene (assets, sensors) | each asset's `object_cfg` | robot, frame transformers, cameras when `enable_cameras` | `get_scene_cfg()` (e.g. contact sensors) |
| observations | - | `policy` group (proprioception) + `camera_obs` group | `get_observation_cfg()` |
| events | per-asset reset events | joint-noise + root-pose reset | `get_events_cfg()` |
| terminations | - | - | `get_termination_cfg()` -> `TaskTerminationCfg(success=[CompletionCriteria], failures={...}, timeout_s)` |
| metrics/recorders | - | - | `get_metrics()` -> `MetricBase` (a recorder term + an offline compute function) |

`IsaacLabArenaEnvironment` also takes `env_cfg_callback` (patch the compiled `ManagerBasedRLEnvCfg`),
`episode_recorder_terms` (per-episode JSONL fields) and `placer_params`.

**Registration.** A Python environment is an `ArenaEnvironmentFactory[Cfg]` subclass with a `name`,
a dataclass `Cfg(ArenaEnvironmentCfg)` whose fields become CLI flags, and `build(cfg)`; it is
decorated with `@register_environment`. Only top-level modules of `isaaclab_arena_environments/` are
auto-imported, and they must defer Isaac Sim imports into `build()` (they are imported during pytest
collection). `test_all_environments.py` boots every registered environment with zero actions, so
every config field needs a default.

**Success.** `TaskTerminationCfg.success` holds `CompletionCriteria` built from predicates:
callables `(env, **params) -> BoolTensor[num_envs]`, ordered within a sequence. The builder installs
one `success` termination fed by the `ProgressTracker`. `failures` are ordinary
`TerminationTermCfg`s; any of them ends the episode as a failure.

**Observations and cameras.** Embodiment cameras are `ArenaCameraCfg` configclasses (tiled by
default) that are only spawned when `enable_cameras=True`. `make_camera_observation_cfg` turns each
camera data type into a term in the `camera_obs` group. Isaac Lab's `ObservationGroupCfg.history_length`
provides frame history natively. The Franka has exactly one camera, `wrist_cam` on `panda_hand`
(84x84 RGB). The G1 has one head camera.

**Randomisation.** Arena offers relation-based placement (`On`, `NextTo`, ... solved into a layout
pool), `PoseRange` reset events, and opt-in run-time/build-time variations. The suite needs coupled
per-episode geometry (a door angle and a shelf height change several boxes at once), so each task owns
one class-based reset event that samples a layout and writes every part's pose, following the
`ManagerTermBase` event pattern used in `isaaclab_arena/terms/events.py`.

**Policies and runners.** A policy is `PolicyBase[Cfg]` with `get_action(env, obs)` and `reset(env_ids)`.
`policy_runner.py --policy_type <dotted.path.Class>` imports unregistered policies, so the oracle and
blind controllers need no registry change. Control runs at 15 Hz (sim dt 1/120, decimation 8).

**Tests.** Simulation tests use the inner/outer `run_function_with_persistent_simulation_app` pattern;
markers `with_cameras` and `with_subprocess` split the three CI phases.

## 2. Shared infrastructure (Phase 1)

Location: `isaaclab_arena_environments/spatial_memory/` (a sub-package, so it is not auto-imported as
an environment) plus one top-level `spatial_memory_<task>_environment.py` per task for registration.
Tests: `isaaclab_arena/tests/test_spatial_memory_*.py`.

- **Fixtures.** Static scene geometry is built from kinematic `CuboidCfg` rigid objects (one Arena
  `Object` per board), because the per-episode geometry (door angle, shelf height, slot width) must
  differ per environment and per reset. Being boxes, their exact geometry is known to the oracle map,
  the visibility model and the oracle controller.
- **Observation variants** (`observation_variant` config field):
  `current` (wrist/head image + proprioception), `history_k` (Isaac Lab `history_length` on the
  `policy` and `camera_obs` groups; `history_k` defaults to `ceil(3.0 s / step_dt)` = 45 frames),
  `oracle_map` (adds an `oracle_map` group: a fixed-size labelled point cloud sampled on every fixture
  and object surface, including hidden ones, expressed in the robot base frame), `mapped` (adds a
  `mapped` group fed by an `OnlineMapProvider` protocol; the default provider returns an empty map and
  carries the nvblox-next TODO).
- **Visibility logging.** Per step and per task-relevant surface: sample points on the surface, test them
  against the egocentric camera frustum (pose from the camera's mount body and offset, intrinsics from
  the camera config), and ray-cast each point against every box in the scene (fixtures, objects and a
  box model of the hand and fingers) for occlusion. A rendered cross-check test compares the model
  with instance segmentation from the real camera.
- **Collision term.** One unfiltered contact sensor over the arm links (any contact on links 1-7 is a
  collision), and filtered sensors on the hand, each finger and each carried object against the
  fixture boards. Contacts the task needs (fingers on the object, the object on its start or goal
  support) are excluded per task. A collision ends the episode as a failure; the first contact
  location (body and world position) is logged.
- **Difficulty knob.** One `difficulty` field per task config (documented per task).
- **Scripted oracle / blind controller.** `SpatialMemoryOraclePolicy` and `SpatialMemoryBlindPolicy`
  read the privileged layout and drive the embodiment's own action space through Cartesian waypoints
  (Franka: joint-position targets from damped IK with a null-space posture term, plus the binary
  gripper). The blind controller uses the
  same skills but replaces the hidden-geometry-aware part of the plan with a straight-line motion or a
  nominal (mean-layout) guess.
- **Metrics.** Success rate (existing `SuccessRateMetric`), collision rate, mean time to success and a
  searched-again rate, each with a recorder term and an offline compute function; plus per-episode
  JSONL fields through `episode_recorder_terms`.

## 3. The ten tasks

Common choices: Franka on its stand for the fixed-arm tasks, wrist camera only; ground plane
(`ground_plane`, 0.8755 m below the base) and default `light` assets; primitive-shape stand-ins
for fixtures and items. Robot base frame `B` = `panda_link0` = environment origin, `x` forward,
`z` up; counters have their top at `z = 0`.

Embodiment decision (made in Phase 1, after measuring): the suite uses `franka_joint_pos` with the
high-gain, gravity-compensated arm that `franka_ik` uses. With `franka_ik` the scripted controllers
cannot choose the elbow (the action's damped IK has no posture term) and the arm swung through the
fixtures; the oracle now runs its own damped IK with a null-space posture term and limit handling.
Measured workspace: a forward-horizontal hand is only reachable well above the counter (joint 6
limit), so items are grasped and inserted with the hand pitched 45 degrees down, and cabinet
compartments are sized for that approach.

| # | Env name | Scene and assets | Embodiment / camera | Hidden at action time | Knob | Missing / substitute |
| - | --- | --- | --- | --- | --- | --- |
| 1 | `spatial_memory_mug_past_door` | counter slab, wall cabinet (carcass, 2 shelves, hinged door at a fixed open angle), mug on counter | Franka, wrist | door panel and the front edges of the target compartment once the mug is inside the opening | door angle (deg) | no repositionable cabinet asset; boxes. Mug: cylinder stand-in |
| 4 | `spatial_memory_dishwasher_rack` | counter with overhang, under-counter rack with dividers pulled out, open door lying flat | Franka, wrist | dividers and door once the item is under the counter edge | slot clearance (m) | no dishwasher asset; low drawer with dividers (boxes) |
| 7 | `spatial_memory_machine_tending` | table, enclosure with a framed front opening, fixture cradle, two clamp blocks inside, part on the table | Franka, wrist | opening frame and clamps once the hand is inside | opening size (m) | boxes |
| 2 | `spatial_memory_fetch_behind` | tiered open shelf to the robot's left with N coloured mugs, counter in front | Franka, wrist | every mug position at the start of the reach (instruction is revealed only after the shelf has been out of view for 3.5 s > 3 s history) | N mugs (2-8) | cabinet replaced by an open tiered shelf so mugs can be grasped top-down |
| 8 | `spatial_memory_kitting` | rack of N open bins to the left, each holding a distinct coloured part; kit tray in front | Franka, wrist | rack while at the tray, tray while at the rack | N bins | bins and tray from boxes |
| 9 | `spatial_memory_restock_slot` | cubby shelf (grid) in front, most cubbies filled; new item on a side counter | Franka, wrist | which cubbies are free once the held item is in front of the camera | empty cubbies | boxes |
| 5 | `spatial_memory_back_of_shelf` | deep shelf, a front row of tall items with one gap, a short target behind the gap | Franka, wrist | target and gap edges once the hand enters the front row | gap width (m) | boxes |
| 6 | `spatial_memory_high_shelf` | high shelf with a front lip above the working area, existing items on it, item on the counter | Franka, wrist | shelf surface, lip and existing items once the item is over the shelf | number of existing items | boxes |
| 3 | `spatial_memory_tray_carry` | floor, counter corner and chair blocks on the route, destination table, a tray/box load | G1 WBC (`g1_wbc_joint`), head camera | lower part of the route (obstacles) while the load is carried | load size (m) | humanoid carrying; see open question Q3 |
| 10 | `spatial_memory_door_closed` | task 1 scene; the door may change angle while the robot faces away | Franka, wrist | the new door pose, except for a re-observation window | re-observation window (s) | - |

Episode phases, randomisation, success and failure terms per task are defined in the task module
docstrings and summarised in `README.md` (Phase 4).

## 4. Open questions (with the default taken)

- **Q1 Primitive stand-ins.** The registry has kitchen backgrounds with fixed cabinets
  (`kitchen`, `lightwheel_robocasa_kitchen`, `replicator_kitchen_*`), a microwave articulation, mugs
  (`mug`, `mug_ycb_robolab`, `ceramic_mug_hot3d_robolab`), bins (`bin_*_vomp_robolab`,
  `grey_bin_robolab`) and shelves (`wireshelving_a01_vomp_robolab`, `rack_l04_vomp_robolab`). None
  of them can change door angle, shelf height, slot width or cabinet position per environment and per
  reset, and their metric geometry is not available to an oracle map without mesh processing.
  *Default:* box fixtures and primitive items with distinct colours; existing `ground_plane` and
  `light` assets. Real assets can later replace the items one by one.
- **Q2 Searched-again.** A wrist camera necessarily sees the shelf during the final approach.
  *Default:* "searched again" = after the hidden phase starts, task-relevant items are visible for
  more than `reobservation_budget_s` (default 1.0 s) before the first gripper close.
- **Q3 Task 3 embodiment.** The G1 WBC embodiment walks with a navigate command, but no scripted
  bimanual tray grasp exists. *Default:* try G1 WBC with a two-handed squeeze grasp; if that is not
  reliable, report task 3 as blocked rather than fake it.
- **Q4 Task 10 re-observation window.** *Default:* the wrist image is blanked (zeros) outside the
  initial look phase and outside a window of `difficulty` seconds after the door re-enters view.
- **Q5 Contacts that count.** *Default:* any arm-link contact, any hand/finger contact with a fixture
  board and any carried-object contact with a board other than its start and goal supports.
