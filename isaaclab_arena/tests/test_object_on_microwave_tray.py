# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the microwave-tray contact fires a pick-and-place success termination.

Teleports the dex_cube just above the microwave turntable, lets it fall under gravity
with zero robot actions, and checks that resting on the tray registers contact and
fires the task's ``success`` termination. The contact filter targets the
``Microwave039_Disc001`` rigid body, which relies on the
``/physics/tensors/recursiveLeafPatternMatch`` carb workaround (IsaacLab #6424) so a body
with multiple collision shapes still resolves to a single filter entry.
"""

import torch
import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 120
HEADLESS = True

_MICROWAVE_POS = (0.4, -0.00586, 0.22773)
_MICROWAVE_ROT_XYZW = (0.0, 0.0, -0.7071068, 0.7071068)
_DISC_BODY_SUBPATH = "microwave/Microwave039_Disc001"
_DROP_HEIGHT = 0.06


def _tray_world_pos(env, env_index: int = 0) -> torch.Tensor:
    """World-frame position of the tray collision prim for ``env_index``."""
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom

    env_ns = env.unwrapped.scene.env_regex_ns.replace(".*", str(env_index))
    prim_path = f"{env_ns}/{_DISC_BODY_SUBPATH}"
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsValid(), f"Tray collision prim not found: {prim_path}"
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    t = transform.ExtractTranslation()
    return torch.tensor([t[0], t[1], t[2]], device=env.unwrapped.device, dtype=torch.float32)


def _test_object_on_microwave_tray_termination(simulation_app) -> bool:
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    microwave = asset_registry.get_asset_by_name("microwave")()
    dex_cube = asset_registry.get_asset_by_name("dex_cube")()

    microwave.set_initial_pose(Pose(position_xyz=_MICROWAVE_POS, rotation_xyzw=_MICROWAVE_ROT_XYZW))

    # Destination reference targeting the microwave turntable rigid body (the filter under test).
    destination_ref = ObjectReference(
        name="microwave_disc",
        parent_asset=microwave,
        prim_path="{ENV_REGEX_NS}/" + _DISC_BODY_SUBPATH,
        object_type=ObjectType.RIGID,
    )

    scene = Scene(assets=[background, microwave, dex_cube])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="microwave_tray",
        embodiment=FrankaIKEmbodiment(),
        scene=scene,
        task=PickAndPlaceTask(dex_cube, destination_ref, background),
    )

    env = ArenaEnvBuilder(isaaclab_arena_environment, args_cli).make_registered()
    env.reset()

    try:
        # Teleport the cube just above the tray and drop it (zero velocity, zero actions).
        cube_asset = env.unwrapped.scene[dex_cube.name]
        target_pos = _tray_world_pos(env)
        target_pos[2] += _DROP_HEIGHT
        root_pose = torch.zeros((1, 7), device=env.unwrapped.device)
        root_pose[0, :3] = target_pos
        root_pose[0, 3] = 1.0  # identity quaternion (w, x, y, z)
        cube_asset.write_root_pose_to_sim(root_pose)
        cube_asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.unwrapped.device))

        # Open the microwave door so the cube drops onto the tray.
        microwave.open(env, env_ids=None)

        success_vec = []
        terminated_vec = []
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(NUM_STEPS):
            with torch.inference_mode():
                _, _, terminated, _, _ = env.step(actions)
                success_vec.append(env.unwrapped.termination_manager.get_term("success").clone())
                terminated_vec.append(terminated.item())
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()

    print("Checking the cube was not on the tray at the first step")
    assert not success_vec[0].item(), "Cube registered success before it could fall onto the tray"
    print("Checking the cube landed on the tray and fired the success termination")
    assert any(s.item() for s in success_vec), "Cube on the tray never fired the success termination"
    print("Checking the task terminated")
    assert any(terminated_vec), "The task was not terminated"

    return True


def test_object_on_microwave_tray_termination():
    result = run_simulation_app_function(_test_object_on_microwave_tray_termination, headless=HEADLESS)
    assert result, "Test failed"


if __name__ == "__main__":
    test_object_on_microwave_tray_termination()
