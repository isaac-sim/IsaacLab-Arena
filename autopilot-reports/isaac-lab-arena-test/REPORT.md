# Isaac Lab Arena Validation Report

## Environment
- **Isaac Lab version:** v3.0.0-beta (isaaclab 4.5.22)
- **Isaac Lab Arena version:** v1.0.0 (release/0.1.1)
- **Isaac Sim version:** 6.0.0.0 (pip)
- **GPU:** NVIDIA L40 (49140 MiB VRAM)
- **OS:** Ubuntu 22.04.5 LTS, Kernel 5.15.0-113-generic
- **CPU:** Intel Xeon Platinum 8362 @ 2.80GHz (16 cores available)
- **Python:** 3.12.13
- **Date:** 2026-03-24

## Installation Summary

### Isaac Lab 3 (v3.0.0-beta)
- Installed via `./isaaclab.sh --install`
- Required Python 3.12 (enforced by CLI)
- All extensions installed: isaaclab, isaaclab_assets, isaaclab_contrib, isaaclab_experimental, isaaclab_mimic, isaaclab_newton, isaaclab_ov, isaaclab_physx, isaaclab_rl, isaaclab_tasks, isaaclab_tasks_experimental, isaaclab_teleop, isaaclab_visualizers

### Isaac Lab Arena (release/0.1.1)
- Installed via `isaaclab.sh -p -m pip install -e .`
- Version 1.0.0
- Sub-packages: isaaclab_arena, isaaclab_arena_g1, isaaclab_arena_gr00t

### Isaac Sim
- Isaac Lab v3.0.0-beta requires isaacsim 5.1.0, which is not available on PyPI
- Used isaacsim 6.0.0.0 (latest available for Python 3.12)
- Required additional packages: onnxruntime-gpu, lightwheel_sdk

## Import Validation

| Module | Status |
|--------|--------|
| isaaclab_arena | OK |
| isaaclab_arena_g1 | OK |
| isaaclab_arena_gr00t | OK |

## Available Arena Environments

Discovered via CLI argument parser (Arena uses its own environment builder, not gymnasium registry):

1. `gr1_open_microwave` - GR1 robot opening a microwave
2. `kitchen_pick_and_place` - Kitchen pick and place task
3. `galileo_pick_and_place` - Galileo pick and place task
4. `galileo_g1_locomanip_pick_and_place` - G1 locomotion + manipulation
5. `press_button` - Franka robot pressing a coffee machine button

## Unit Test Results

### Pure Python tests (no isaacsim required)
- `test_configclass.py::test_combine_configclasses_with_multiple_inheritance` - PASSED
- `test_configclass.py::test_combine_configclasses_with_inheritance` - PASSED
- `test_configclass.py::test_combine_configclasses_with_post_init` - PASSED
- `test_pose.py::test_pose_composition` - FAILED (quaternion wxyz/xyzw convention mismatch in Arena code)

**Result: 3 passed, 1 failed**

### Simulation-dependent tests
- 23 tests require `isaacsim` runtime and could not be collected without the full simulation context
- These tests use `SimulationAppContext` subprocess utilities

## Smoke Test: press_button (Headless Rollout)

### Configuration
- Task: `press_button` (Franka robot + coffee machine)
- Embodiment: Franka Panda
- Parallel environments: 64
- Steps: 100
- Policy: zero_action
- Device: cuda:0

### Environment Details
- Physics step-size: 0.005
- Rendering step-size: 0.01
- Environment step-size: 0.02
- Decimation: 4
- Action dimensions: 7 (6 arm + 1 gripper)
- Observation dimensions: 25 (actions:7, joint_pos:9, joint_vel:9, eef_pos:3, eef_quat:4, gripper_pos:2)

### Results
- **100 steps x 64 envs completed successfully**
- **Throughput: 819 env-steps/sec**
- **Elapsed time: 7.81s**
- **Success rate: 100%** (3264 episodes)
- Reward sum: 0.0 (zero-action policy, expected)

## Compatibility Issues Found

Arena release/0.1.1 was designed for isaacsim 5.1.0 but was tested with isaacsim 6.0.0.0. Two compatibility fixes were needed:

### 1. Warp array vs torch.Tensor (observations.py)
Isaac Sim 6.0 returns Warp arrays from `articulation.data.joint_pos` instead of torch tensors. Fixed by adding `torch.as_tensor()` conversion.

### 2. Warp array indexing and dtype (joint_utils.py)
- `articulation.data.joint_pos_limits` returns a Warp array that doesn't support multi-dimensional fancy indexing. Fixed by converting to torch first.
- `write_joint_position_to_sim` requires int32 indices (Warp dtype constraint). Fixed by adding explicit `dtype=torch.int32`.
- Batch size mismatch: `write_joint_position_to_sim` expects (num_envs, num_joints) shape. Fixed by using `torch.full()` with correct batch size.

## GPU Utilization
- GPU: NVIDIA L40
- VRAM total: 49140 MiB
- VRAM used during test: ~568 MiB (post-test idle measurement)
- GPU utilization: 0% (measured post-test)
