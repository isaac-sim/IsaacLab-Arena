# VR Teleoperation World Frame to Base Frame Transformation

## Overview

This document describes the implementation of automatic coordinate transformation for VR controller inputs (Quest motion controllers) in the G1 WBC Pink embodiment.

## Problem

VR controllers (like Meta Quest) provide end-effector poses in **world coordinates**. However, the robot's IK controller expects poses relative to the **robot's base frame**. Without this transformation:
- When the robot moves, the hands try to stay at their original world position
- The hands drift away from the robot's body
- Teleoperation becomes unusable

## Solution

We implemented a new action term that automatically transforms wrist poses from world frame to robot base frame before passing them to the IK controller.

## Implementation

### 1. New Action Term: `G1DecoupledWBCPinkWorldFrameAction`

**File**: `isaaclab_arena_g1/g1_env/mdp/actions/g1_decoupled_wbc_pink_world_frame_action.py`

This action term extends `G1DecoupledWBCPinkAction` and adds coordinate transformation in the `process_actions()` method.

**Key method**: `_transform_wrist_poses_to_base_frame()`
- Transforms wrist positions: `wrist_pos_base = R^(-1) * (wrist_pos_world - base_pos)`
- Transforms wrist orientations: `wrist_quat_base = base_quat^(-1) * wrist_quat_world`
- Processes both wrists in batch for efficiency

### 2. Configuration: `G1DecoupledWBCPinkWorldFrameActionCfg`

**File**: `isaaclab_arena_g1/g1_env/mdp/actions/g1_decoupled_wbc_pink_world_frame_action_cfg.py`

Adds a new config option:
```python
transform_to_base_frame: bool = True
```

### 3. Wrapper Action Configuration

**File**: `isaaclab_arena/embodiments/g1/g1.py`

Created `G1WBCPinkWorldFrameActionCfg` class that wraps `G1DecoupledWBCPinkWorldFrameActionCfg` with proper initialization:
```python
@configclass
class G1WBCPinkWorldFrameActionCfg:
    g1_action: ActionTermCfg = G1DecoupledWBCPinkWorldFrameActionCfg(
        asset_name="robot", 
        joint_names=[".*"]
    )
```

This ensures the action term has all required fields (`asset_name` and `joint_names`) set.

### 4. G1 Embodiment Update

**File**: `isaaclab_arena/embodiments/g1/g1.py`

Updated `G1WBCPinkEmbodiment.__init__()` to accept:
```python
use_world_frame_actions: bool = False
```

When `True`, uses `G1WBCPinkWorldFrameActionCfg` instead of the standard `G1WBCPinkActionCfg`.

### 5. Environment Auto-Detection

**File**: `isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py`

Automatically enables world frame actions when:
- Teleop device is `motion_controllers` or `openxr`
- AND embodiment is `g1_wbc_pink`

```python
use_world_frame_actions = (
    args_cli.teleop_device in ["motion_controllers", "openxr"]
    and args_cli.embodiment == "g1_wbc_pink"
)
```

### 6. Dummy Torso Retargeter

**File**: `isaaclab_arena/assets/retargeter_library.py`

Added `DummyTorsoRetargeter` that returns 3 zeros for torso orientation (roll, pitch, yaw).

Updated `G1WbcPinkMotionControllersRetargeter` to return:
- Upper body retargeter: 16 dims `[gripper(2), left_wrist(7), right_wrist(7)]`
- Lower body retargeter: 4 dims `[nav_cmd(3), hip_height(1)]`
- Dummy torso retargeter: 3 dims `[torso_rpy(3)]`
- **Total**: 23 dims (matches G1 WBC Pink action space)

## Usage

### Command Line
```bash
python isaaclab_arena/scripts/imitation_learning/teleop.py \
  --xr \
  --num_env 1 \
  galileo_g1_locomanip_pick_and_place \
  --teleop_device motion_controllers \
  --embodiment g1_wbc_pink
```

The world frame transformation is automatically enabled!

### Programmatic
```python
from isaaclab_arena.embodiments.g1 import G1WBCPinkEmbodiment

# For VR/motion controllers
embodiment = G1WBCPinkEmbodiment(
    use_world_frame_actions=True
)

# For other teleop devices
embodiment = G1WBCPinkEmbodiment(
    use_world_frame_actions=False  # default
)
```

## Technical Details

### Action Layout (23 dimensions)
```
[0:1]   left_hand_state (0=open, 1=close)
[1:2]   right_hand_state (0=open, 1=close)
[2:5]   left_wrist_pos (x,y,z)
[5:9]   left_wrist_quat (w,x,y,z)
[9:12]  right_wrist_pos (x,y,z)
[12:16] right_wrist_quat (w,x,y,z)
[16:19] navigate_cmd (x, y, angular_z)
[19:20] base_height
[20:23] torso_orientation_rpy
```

### Transformation Math

**Position transformation**:
```
wrist_pos_translated = wrist_pos_world - robot_base_pos
wrist_pos_base = quat_apply_inverse(robot_base_quat, wrist_pos_translated)
```

**Orientation transformation**:
```
robot_base_quat_inv = quat_inv(robot_base_quat)
wrist_quat_base = quat_mul(robot_base_quat_inv, wrist_quat_world)
```

Both wrists are processed in batch for efficiency.

## VR Camera Viewport Configuration

The Quest VR headset viewport automatically follows the robot's first-person view through proper XR configuration.

### G1 Embodiment XR Config

**File**: `isaaclab_arena/embodiments/g1/g1.py`

The G1 embodiment's XR configuration is set up to:
```python
self.xr: XrCfg = XrCfg(
    anchor_pos=(0.0, 0.0, -1.0),
    anchor_rot=(0.70711, 0.0, 0.0, -0.70711),
    anchor_prim_path="/World/envs/env_0/Robot/pelvis",  # Track robot's pelvis
    fixed_anchor_height=True,  # Keep height fixed for comfort
)
```

### Motion Controllers Device Config

**File**: `isaaclab_arena/assets/device_library.py`

The motion controllers device automatically:
1. Retrieves the XR config from the embodiment
2. Sets anchor rotation mode to `FOLLOW_PRIM_SMOOTHED` for smooth camera following
3. Passes it to `OpenXRDeviceCfg`

```python
def get_device_cfg(self, retargeters, embodiment) -> OpenXRDeviceCfg:
    xr_cfg = embodiment.get_xr_cfg()
    xr_cfg.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED
    return OpenXRDeviceCfg(
        retargeters=retargeters,
        sim_device=self.sim_device,
        xr_cfg=xr_cfg,  # Camera follows robot!
    )
```

**Result**: The VR headset viewport now tracks the robot's pelvis, rotating smoothly as the robot moves and turns, providing a natural first-person view.

## Benefits

1. **Automatic**: No manual coordinate transformation needed
2. **Clean architecture**: Proper action term instead of ad-hoc padding
3. **Efficient**: Batch processing of both wrists
4. **Configurable**: Can be toggled on/off via config
5. **Auto-detection**: Environment automatically enables it for VR devices
6. **First-person view**: VR camera follows robot's pelvis for immersive teleoperation

## Files Changed

1. `isaaclab_arena_g1/g1_env/mdp/actions/g1_decoupled_wbc_pink_world_frame_action.py` (new)
2. `isaaclab_arena_g1/g1_env/mdp/actions/g1_decoupled_wbc_pink_world_frame_action_cfg.py` (new)
3. `isaaclab_arena_g1/g1_env/mdp/actions/__init__.py` (updated exports)
4. `isaaclab_arena/embodiments/g1/g1.py` (added `G1WBCPinkWorldFrameActionCfg` wrapper and `use_world_frame_actions` parameter)
5. `isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py` (auto-detection logic)
6. `isaaclab_arena/assets/retargeter_library.py` (added `DummyTorsoRetargeter`)
7. `isaaclab_arena/scripts/imitation_learning/teleop.py` (removed dimension padding hack)
