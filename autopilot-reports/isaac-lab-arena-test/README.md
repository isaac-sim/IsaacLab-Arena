# Isaac Lab Arena Test Report

**Date:** 2026-03-24
**Tester:** Autopilot (Claude)
**Arena version:** 1.0.0 (release/0.1.1)
**Isaac Lab version:** v3.0.0-beta
**Isaac Sim version:** 6.0.0-rc.22
**GPU:** NVIDIA L40 (49,140 MB VRAM)
**Driver:** 570.158.01

---

## Summary

| Criterion | Status |
|-----------|--------|
| Arena installs on Isaac Lab 3 | ✅ PASS |
| Core modules import (isaaclab_arena, _g1, _gr00t) | ✅ PASS |
| SimulationApp starts headless | ✅ PASS |
| Arena environment instantiation | ✅ PASS |
| 100-step rollout × 64 envs completes | ✅ PASS |
| Throughput benchmark measured | ✅ PASS |
| Unit test suite | ⚠️ PARTIAL (16/47 pass) |

---

## Installation

Isaac Lab Arena v1.0.0 installs cleanly into the Isaac Lab 3 Python environment:

```bash
# Pre-installed environment at /tmp/IsaacLab (v3.0.0-beta)
~/Workspace/IsaacLab/isaaclab.sh -p -m pip install -e .
```

All three sub-packages import successfully:
- `isaaclab_arena` ✅
- `isaaclab_arena_g1` ✅
- `isaaclab_arena_gr00t` ✅

---

## Available Arena Environments

Arena uses the `ArenaEnvBuilder` pattern (not standard `gym.register` at import time). 5 built-in example environments are available via `ExampleEnvironments`:

| Environment | Class |
|-------------|-------|
| `gr1_open_microwave` | `Gr1OpenMicrowaveEnvironment` |
| `kitchen_pick_and_place` | `KitchenPickAndPlaceEnvironment` |
| `galileo_pick_and_place` | `GalileoPickAndPlaceEnvironment` |
| `galileo_g1_locomanip_pick_and_place` | `GalileoG1LocomanipPickAndPlaceEnvironment` |
| `press_button` | `PressButtonEnvironment` |

---

## Rollout Benchmark

**Task:** Franka Panda arm + packing table + coffee machine (Dummy task, zero-action policy)
**Configuration:**

```
Num envs: 64
Num steps: 100 (+ 5 warm-up steps)
Action space: (64, 7)  # 7-DOF Franka
Obs keys: ['policy']
```

**Results:**

| Metric | Value |
|--------|-------|
| Total time (100 steps × 64 envs) | 4.05 seconds |
| **Throughput** | **1,578 env-steps/sec** |
| Reward shape | torch.Size([64]) |
| Reward mean | 0.0000 |
| GPU utilization | 46% |
| GPU memory used | 3,256 MB / 49,140 MB |

Observations and rewards were correctly returned for all 64 environments.

---

## Test Suite Results

**Command:**
```bash
cd /tmp/IsaacLab-Arena
/tmp/IsaacLab/isaaclab.sh -p -m pytest isaaclab_arena/tests/ -v --timeout=300
```

**Overall:** `27 failed, 16 passed, 2 skipped, 2 errors` (47 total, 75.66s)

### Passing Tests (16)

| Test | Module |
|------|--------|
| test_combine_configclasses_with_multiple_inheritance | test_configclass |
| test_combine_configclasses_with_inheritance | test_configclass |
| test_combine_configclasses_with_post_init | test_configclass |
| test_pose_composition | test_pose |
| test_affordance_base | test_affordance_base |
| test_default_assets_registered | test_asset_registry |
| test_all_assets_in_registry | test_asset_registry |
| test_detect_object_type | test_detect_object_type |
| test_detect_object_type_for_all_objects | test_detect_object_type |
| test_object_configuration | test_object_configuration |
| test_object_of_type_base | test_object_of_type_base |
| test_scene_to_usd | test_scene_to_usd |
| test_simulation_app_context | test_simulation_app_context |
| test_run_simulation_app_function_with_arg | test_simulation_app_context |
| test_get_prim_pose_in_default_prim_frame | test_usd_pose_helpers |
| test_pytorch_cuda_compatibility | test_gr00t_deps |

### Failed Tests — By Category

#### GR00T optional dependencies not installed (expected)
- `test_flash_attn_import` — `flash_attn` not installed
- `test_gr00t_package_directory_exists` — GR00T model package absent
- `test_gr00t_package_import` — GR00T model package absent
- `test_g1_convert_hdf5_to_lerobot` — LeRobot dependency missing
- `test_g1_locomanip_replay_lerobot_policy_runner_single_env` — LeRobot missing
- `test_g1_locomanip_gr00t_closedloop_policy_runner_*` (2 ERRORs) — GR00T model absent

#### Camera test (headless/no-display, expected)
- `test_camera_observation` — requires cameras enabled

#### API compatibility: Arena 1.0.0 vs Isaac Lab v3.0.0-beta
These tests encounter runtime errors due to API changes between Isaac Lab 2.x (Arena's target) and 3.0.0-beta:

| Test | Error |
|------|-------|
| test_auto_object_type | `'array' object has no attribute 'clone'` |
| test_all_devices_in_registry | `Invalid index: tensor([0], device='cuda:0', dtype=torch.int32)` |
| test_door_moved_rate_metric | Simulation step failure |
| test_set_object_post_per_env_event | Simulation step failure |
| test_wbc_joint_standing_idle_actions_single_env | WBC/Pink IK failure |
| test_wbc_pink_standing_idle_actions_single_env | WBC/Pink IK failure |
| test_object_on_destination_termination | `'RecorderManager' has no attribute '_failed_episode_dataset_file_handler'` |
| test_open_door_microwave | Simulation step failure |
| test_open_door_microwave_multiple_envs | Simulation step failure |
| test_open_door_microwave_reset_condition | Simulation step failure |
| test_press_button_coffee_machine | Simulation step failure |
| test_press_button_coffee_machine_multiple_envs | Simulation step failure |
| test_reference_objects | Simulation step failure |
| test_reference_objects_with_transform | Simulation step failure |
| test_robot_initial_position | Simulation step failure |
| test_success_rate_metric | Simulation step failure |
| test_zero_action_policy_* (4 tests) | Simulation step failure |
| test_replay_policy_gr1_open_microwave | Simulation step failure |

**Root causes:**
1. `'array' object has no attribute 'clone'` — Isaac Lab 3.0 returns numpy arrays in some data paths where Arena expects PyTorch tensors
2. `Invalid index: tensor(...)` — Fabric/physics indexing API changed in v3.0
3. `RecorderManager._failed_episode_dataset_file_handler` — new attribute in RecorderManager not present in Arena 1.0.0

---

## Key Findings

1. **Installation**: Arena 1.0.0 installs without errors on Isaac Lab v3.0.0-beta. All package imports succeed.

2. **SimulationApp**: Starts correctly headless. The `omni.gpu_foundation_factory.plugin` warning ("The default graphics plugin cannot be set!") is non-fatal; the app runs normally. Physics simulation (PhysX 110.0.7) works.

3. **Rollout**: Core simulation loop runs. A 100-step headless rollout with 64 Franka environments achieves **1,578 env-steps/sec** on an NVIDIA L40, using 46% GPU utilization and 3.2 GB VRAM.

4. **API compatibility gap**: Arena release/0.1.1 was authored against Isaac Lab 2.x. Isaac Lab v3.0.0-beta introduced breaking changes that cause ~16 simulation-based tests to fail. The primary issues are tensor vs. numpy array handling and updated RecorderManager/Physics APIs.

5. **Optional dependencies**: GR00T model weights and `flash_attn` are not installed — GR00T closed-loop policy tests are skipped/errored as expected in a base environment.

---

## Recommendations

1. **For production use**: Pin to Isaac Lab 2.x until Arena is updated for v3.0.0 compatibility, or apply targeted fixes for the numpy/tensor conversion issues.
2. **Quick fix path**: The `'array' object has no attribute 'clone'` errors suggest switching numpy arrays to torch tensors in a few data-path locations in Arena.
3. **GR00T testing**: Install `flash_attn` and GR00T model weights to enable policy evaluation tests.
