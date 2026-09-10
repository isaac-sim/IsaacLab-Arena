# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A camera-pose write must move both the RTX render and ``camera.data.pos_w``, on PhysX and Newton.

These tests write the wrist camera to two offsets and check that the render (pixels) and
``camera.data.pos_w`` follow. Set ``ISAACLAB_ARENA_SAVE_TEST_IMAGES=1`` to dump the compared frames under
``IMAGE_OUTPUT_DIR/<test name>/``.
"""

import os
import torch

import pytest

from isaaclab_arena.patches.camera_render_pose import CameraPoseWriter
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True
ENABLE_CAMERAS = True

# The two parent-frame camera offsets [m] whose renders and reported poses are compared.
COMPARED_OFFSETS = ([0.0, 0.0, 0.0], [0.3, 0.25, 0.0])
WARMUP_STEPS = 2
RENDER_ITERS = 3
RENDER_DIFF_THRESHOLD = 5.0
POSE_SHIFT_THRESHOLD_M = 0.05
SAVE_IMAGES = os.environ.get("ISAACLAB_ARENA_SAVE_TEST_IMAGES") == "1"
IMAGE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Physics backend name -> the --presets value that selects it (None is the PhysX default).
BACKEND_PRESETS = {"physx": None, "newton": "newton"}


def _disable_joint_randomization(env_cfg):
    """Null the reset-time joint randomization so the robot pose is deterministic across steps."""
    env_cfg.events.randomize_franka_joint_state = None
    return env_cfg


def _build_env(presets: str | None, disable_fabric: bool = False):
    """Build a gym-wrapped Arena env with the Franka wrist camera on the given physics backend.

    Franka (rather than the Robotiq-gripper DROID) is used so the arm builds under Newton.

    Args:
        presets: The ``--presets`` value selecting the physics backend, or None for the PhysX default.
        disable_fabric: Whether to build with Fabric off, so the render reads poses from USD.
    """
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    embodiment = FrankaIKEmbodiment(enable_cameras=ENABLE_CAMERAS)
    embodiment.camera_config.wrist_cam.height = 180
    embodiment.camera_config.wrist_cam.width = 240

    # A lit table under the wrist camera: without lights and geometry the RTX render is all black.
    asset_registry = AssetRegistry()
    scene = Scene(
        assets=[
            asset_registry.get_asset_by_name("maple_table_robolab")(),
            asset_registry.get_asset_by_name("light")(),
            asset_registry.get_asset_by_name("directional_light")(),
        ]
    )
    arena_env = IsaacLabArenaEnvironment(
        name="test_camera_extrinsics_variation_render",
        embodiment=embodiment,
        scene=scene,
        env_cfg_callback=_disable_joint_randomization,
    )

    cli_args = ["--num_envs", "1", "--enable_cameras"]
    if presets is not None:
        cli_args += ["--presets", presets]
    if disable_fabric:
        cli_args.append("--disable_fabric")
    args_cli = get_isaaclab_arena_cli_parser().parse_args(cli_args)

    return ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()


def _apply_camera_offset(pose_writer, nominal_translation, offset, device, env_ids) -> None:
    """Write ``nominal + offset`` (parent-frame translation) to the camera via the CameraPoseWriter."""
    offset_tensor = torch.tensor(offset, device=device).unsqueeze(0).expand(len(env_ids), 3)
    target = nominal_translation[env_ids] + offset_tensor
    pose_writer.set_local_translations(translations=target, env_ids=env_ids)


def _render_wrist_at_offsets(simulation_app, *, presets, disable_fabric, out) -> bool:
    """Render the wrist camera at each compared offset, appending (offset, mean, nonzero, rgb) to ``out``."""
    env = _build_env(presets, disable_fabric)
    env.reset()

    camera = env.unwrapped.scene["wrist_cam"]
    device = env.unwrapped.device
    sim = env.unwrapped.sim
    env_ids = torch.arange(env.unwrapped.num_envs, device=device)
    pose_writer = CameraPoseWriter(camera)
    nominal_translation = camera._view.get_local_poses()[0].torch.detach().clone()
    zero_actions = torch.zeros(env.action_space.shape, device=device)

    with torch.inference_mode():
        # Bring up the sensors and render non-black content once; both offsets render this same
        # frozen physics state, so the only difference between them is the camera pose.
        for _ in range(WARMUP_STEPS):
            env.step(zero_actions)

        for offset in COMPARED_OFFSETS:
            _apply_camera_offset(pose_writer, nominal_translation, offset, device, env_ids)
            # Render only (no physics advance) so scene motion cannot contaminate the comparison.
            for _ in range(RENDER_ITERS):
                sim.render()
            camera.update(dt=0.0, force_recompute=True)
            rgb = camera.data.output["rgb"].detach().float().cpu()
            out.append((offset, rgb.mean().item(), (rgb > 0).float().mean().item(), rgb))

    env.close()
    return True


def _read_camera_pose_at_offsets(simulation_app, *, presets, out) -> bool:
    """Read the camera pose at each compared offset, appending (offset, camera_pos_w, view_pos_w) to ``out``."""
    env = _build_env(presets)
    env.reset()

    camera = env.unwrapped.scene["wrist_cam"]
    view = camera._view
    assert view is not None, "Camera FrameView was not initialized."

    device = env.unwrapped.device
    env_ids = torch.arange(env.unwrapped.num_envs, device=device)
    pose_writer = CameraPoseWriter(camera)
    nominal_translation = view.get_local_poses()[0].torch.detach().clone()
    zero_actions = torch.zeros(env.action_space.shape, device=device)

    with torch.inference_mode():
        for _ in range(WARMUP_STEPS):
            env.step(zero_actions)

        for offset in COMPARED_OFFSETS:
            _apply_camera_offset(pose_writer, nominal_translation, offset, device, env_ids)
            # camera.data.pos_w is the world pose the renderer consumes; the writer's camera.reset pushes
            # the FrameView write here, so it should follow the pose.
            camera_pos_w = camera.data.pos_w.detach().float().cpu().clone()
            # The FrameView's own world pose reflects the write; captured as a positive control.
            view_pos_w = view.get_world_poses()[0].torch.detach().float().cpu().clone()
            out.append((offset, camera_pos_w, view_pos_w))

    env.close()
    return True


def _save_offset_images(renders: list, output_subdir: str) -> None:
    """Write one PNG per camera offset plus their absolute difference into ``IMAGE_OUTPUT_DIR/<subdir>/``."""
    from PIL import Image

    out_dir = os.path.join(IMAGE_OUTPUT_DIR, output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    images = {f"offset{index}": rgb[0].to(torch.uint8) for index, (_offset, _mean, _nonzero, rgb) in enumerate(renders)}
    images["difference"] = (renders[1][3][0] - renders[0][3][0]).abs().to(torch.uint8)
    for tag, image in images.items():
        output_path = os.path.join(out_dir, f"wrist_cam-{tag}.png")
        Image.fromarray(image.numpy()).save(output_path)
        print(f"Wrote {output_path}", flush=True)


def _mean_render_difference(label: str, presets: str | None, disable_fabric: bool, output_subdir: str) -> float:
    """Return the mean absolute pixel difference between the two camera-offset renders."""
    renders: list = []
    assert run_function_with_persistent_simulation_app(
        _render_wrist_at_offsets,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
        presets=presets,
        disable_fabric=disable_fabric,
        out=renders,
    ), "Failed to render the wrist camera."

    if SAVE_IMAGES:
        _save_offset_images(renders, output_subdir)

    for offset, mean, nonzero, _ in renders:
        print(f"[{label}] offset={offset} rgb mean={mean:.2f} nonzero_frac={nonzero:.3f}")
    difference = (renders[1][3] - renders[0][3]).abs().mean().item()
    print(f"[{label}] mean render difference between offsets: {difference:.4f}")
    return difference


def _camera_pose_shift(label: str, presets: str | None) -> tuple[float, float]:
    """Return the world-space shift [m] of ``camera.data.pos_w`` and of the FrameView pose across offsets."""
    poses: list = []
    assert run_function_with_persistent_simulation_app(
        _read_camera_pose_at_offsets,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
        presets=presets,
        out=poses,
    ), "Failed to read the wrist camera pose."

    camera_shift = (poses[1][1] - poses[0][1]).norm(dim=-1).max().item()
    view_shift = (poses[1][2] - poses[0][2]).norm(dim=-1).max().item()
    print(f"[{label}] FrameView world-pose shift: {view_shift:.4f} m; camera.data.pos_w shift: {camera_shift:.4f} m")
    return camera_shift, view_shift


@pytest.mark.with_cameras
@pytest.mark.parametrize("backend", list(BACKEND_PRESETS))
# Fabric changes which view the render reads, so both settings are covered.
@pytest.mark.parametrize("disable_fabric", [True, False], ids=["fabric_off", "fabric_on"])
def test_camera_extrinsics_variation_moves_render(backend, disable_fabric, request):
    """Changing the camera's local pose moves the rendered wrist-camera image.

    Guards the Newton regression: without the USD write the ``NewtonSiteFrameView`` pose never reaches the
    renderer and the image stays put.
    """
    label = f"{backend}-{'fabric_off' if disable_fabric else 'fabric_on'}"
    difference = _mean_render_difference(
        label, presets=BACKEND_PRESETS[backend], disable_fabric=disable_fabric, output_subdir=request.node.name
    )
    assert difference > RENDER_DIFF_THRESHOLD, (
        f"Expected the {label} render to change with the camera pose (mean diff > {RENDER_DIFF_THRESHOLD}); "
        f"got {difference:.4f}. The camera pose write is not reaching the renderer."
    )


@pytest.mark.with_cameras
@pytest.mark.parametrize("backend", list(BACKEND_PRESETS))
def test_camera_extrinsics_variation_moves_camera_pose(backend):
    """The camera's reported world pose (camera.data.pos_w) follows the FrameView write.

    The FrameView pose is checked as a positive control; camera.data.pos_w only follows because the
    applied offset pushes the write with ``camera.reset``.
    """
    camera_shift, view_shift = _camera_pose_shift(backend, presets=BACKEND_PRESETS[backend])
    assert view_shift > POSE_SHIFT_THRESHOLD_M, (
        "Sanity check failed: the FrameView world pose should move with the offset "
        f"(> {POSE_SHIFT_THRESHOLD_M} m); got {view_shift:.4f} m."
    )
    assert camera_shift > POSE_SHIFT_THRESHOLD_M, (
        f"Expected camera.data.pos_w to follow the camera pose (> {POSE_SHIFT_THRESHOLD_M} m); "
        f"got {camera_shift:.4f} m. The variation's FrameView write is not pushed to the camera pose."
    )
