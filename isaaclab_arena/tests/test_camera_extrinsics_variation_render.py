# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for setting camera poses across physics backends, at the pixel and pose level.

The camera-extrinsics variation moves a camera by writing its FrameView local pose
(``view.set_local_poses(...)``) and then pushing that write into the camera sensor and renderer with
``camera.reset()`` (``Camera._update_poses``). These tests apply an offset the same way. Without the
``camera.reset`` push the FrameView pose changes but the render and ``camera.data.pos_w`` keep the old
pose -- under Newton the ``NewtonSiteFrameView`` only updates in-memory Warp state, so nothing else
moves the rendered image. This is the regression these tests guard.

Two test sets share one env build path:

- ``_moves_render_*`` compares two rendered wrist-camera frames and asserts the image follows the pose.
- ``_moves_camera_pose_*`` reads ``camera.data.pos_w`` directly (deterministic, no denoiser noise) and
  asserts the reported pose follows the FrameView write, with the FrameView pose itself as a positive
  control.

Set ``SAVE_IMAGES = True`` to dump the compared renders under ``IMAGE_OUTPUT_DIR/<test name>/``.
"""

import os
import torch

import pytest

from isaaclab_arena.patches import CameraPoseWriter
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True
ENABLE_CAMERAS = True

CAMERA_NAME = "wrist_cam"
# Two camera offsets (parent-frame translation [m]); large enough that the wrist view visibly shifts.
VARIANT_OFFSETS = ([0.0, 0.0, 0.0], [0.3, 0.25, 0.0])
# Physics steps taken once before capturing, to bring up the sensors and render non-black content.
WARMUP_STEPS = 2
# Render-only iterations per variant (no physics advance) to converge the RTX image at the new pose.
RENDER_ITERS = 3
# Reduced wrist-camera resolution keeps the render cheap for a test.
CAMERA_HEIGHT = 180
CAMERA_WIDTH = 240
# Mean absolute per-channel difference (uint8 scale) above which we consider the image "moved".
# Observed: PhysX moves the render by ~47; Newton (bug) leaves only ~1.2 of denoiser noise.
RENDER_DIFF_THRESHOLD = 5.0
# World-space camera pose shift [m] above which we consider the reported pose to have "moved".
# The offsets shift the camera by ~0.25 m; the bug leaves camera.data.pos_w at ~0.
POSE_SHIFT_THRESHOLD_M = 0.05
# Set True to dump the compared renders as PNGs into IMAGE_OUTPUT_DIR/<test name>/, created on demand.
SAVE_IMAGES = False
IMAGE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _disable_joint_randomization(env_cfg):
    """Null the reset-time joint randomization so the robot pose is deterministic across steps."""
    if getattr(env_cfg.events, "randomize_franka_joint_state", None) is not None:
        env_cfg.events.randomize_franka_joint_state = None
    return env_cfg


def _build_env(presets: str | None):
    """Build one gym-wrapped Arena env with the Franka wrist camera on the given physics backend.

    The same PhysX-tuned Franka embodiment is used for both backends; the camera-pose bug depends only on
    the physics backend (Newton selects ``NewtonSiteFrameView``) and RTX rendering, not on Newton-tuned
    dynamics. Franka (rather than the Robotiq-gripper DROID) is used so the arm builds under Newton.
    """
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    embodiment = FrankaIKEmbodiment(enable_cameras=ENABLE_CAMERAS)
    embodiment.camera_config.wrist_cam.height = CAMERA_HEIGHT
    embodiment.camera_config.wrist_cam.width = CAMERA_WIDTH

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
    args_cli = get_isaaclab_arena_cli_parser().parse_args(cli_args)

    return ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()


def _apply_camera_offset(pose_writer, nominal_translation, offset, device, env_ids) -> None:
    """Write ``nominal + offset`` (parent-frame translation) to the camera via the CameraPoseWriter."""
    offset_tensor = torch.tensor(offset, device=device).unsqueeze(0).expand(len(env_ids), 3)
    target = nominal_translation[env_ids] + offset_tensor
    pose_writer.set_local_poses(translations=target, orientations=None, env_ids=env_ids)


def _render_wrist_at_offsets(simulation_app, *, presets, out) -> bool:
    """Render the wrist camera at each variant offset, appending (offset, mean, nonzero, rgb) to ``out``."""
    env = _build_env(presets)
    env.reset()

    camera = env.unwrapped.scene[CAMERA_NAME]
    device = env.unwrapped.device
    sim = env.unwrapped.sim
    env_ids = torch.arange(env.unwrapped.num_envs, device=device)
    pose_writer = CameraPoseWriter(camera)
    nominal_translation = camera._view.get_local_poses()[0].torch.detach().clone()
    zero_actions = torch.zeros(env.action_space.shape, device=device)

    with torch.inference_mode():
        # Bring up the sensors and render non-black content once; both variants render this same
        # frozen physics state, so the only difference between them is the camera pose.
        for _ in range(WARMUP_STEPS):
            env.step(zero_actions)

        for offset in VARIANT_OFFSETS:
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
    """Read the camera pose at each variant offset, appending (offset, camera_pos_w, view_pos_w) to ``out``."""
    env = _build_env(presets)
    env.reset()

    camera = env.unwrapped.scene[CAMERA_NAME]
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

        for offset in VARIANT_OFFSETS:
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
        output_path = os.path.join(out_dir, f"{CAMERA_NAME}-{tag}.png")
        Image.fromarray(image.numpy()).save(output_path)
        print(f"Wrote {output_path}", flush=True)


def _mean_render_difference(label: str, presets: str | None, output_subdir: str) -> float:
    """Return the mean absolute pixel difference between the two camera-offset renders."""
    renders: list = []
    assert run_function_with_persistent_simulation_app(
        _render_wrist_at_offsets,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
        presets=presets,
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
def test_camera_extrinsics_variation_moves_render_physx(request):
    """PhysX: changing the camera's FrameView pose moves the rendered wrist-camera image."""
    difference = _mean_render_difference("physx", presets=None, output_subdir=request.node.name)
    assert difference > RENDER_DIFF_THRESHOLD, (
        f"Expected the PhysX render to change with the camera pose (mean diff > {RENDER_DIFF_THRESHOLD}); "
        f"got {difference:.4f}."
    )


@pytest.mark.with_cameras
def test_camera_extrinsics_variation_moves_render_newton(request):
    """Newton: changing the camera pose (with the camera.reset push) moves the rendered wrist image.

    Guards the Newton regression: without the push the ``NewtonSiteFrameView`` write never reaches the
    renderer and the image stays put.
    """
    difference = _mean_render_difference("newton", presets="newton", output_subdir=request.node.name)
    assert difference > RENDER_DIFF_THRESHOLD, (
        f"Expected the Newton render to change with the camera pose (mean diff > {RENDER_DIFF_THRESHOLD}); "
        f"got {difference:.4f}. Under Newton the FrameView pose write is not pushed to the renderer."
    )


@pytest.mark.with_cameras
def test_camera_extrinsics_variation_moves_camera_pose_physx():
    """PhysX: the camera's reported world pose (camera.data.pos_w) follows the FrameView write.

    The FrameView pose is checked as a positive control; camera.data.pos_w only follows because the
    applied offset pushes the write with ``camera.reset``.
    """
    camera_shift, view_shift = _camera_pose_shift("physx", presets=None)
    assert view_shift > POSE_SHIFT_THRESHOLD_M, (
        "Sanity check failed: the FrameView world pose should move with the offset "
        f"(> {POSE_SHIFT_THRESHOLD_M} m); got {view_shift:.4f} m."
    )
    assert camera_shift > POSE_SHIFT_THRESHOLD_M, (
        f"Expected camera.data.pos_w to follow the camera pose (> {POSE_SHIFT_THRESHOLD_M} m); "
        f"got {camera_shift:.4f} m. The variation's FrameView write is not pushed to the camera pose."
    )


@pytest.mark.with_cameras
def test_camera_extrinsics_variation_moves_camera_pose_newton():
    """Newton: the camera's reported world pose (camera.data.pos_w) follows the FrameView write.

    The FrameView pose is checked as a positive control; camera.data.pos_w only follows because the
    applied offset pushes the write with ``camera.reset``.
    """
    camera_shift, view_shift = _camera_pose_shift("newton", presets="newton")
    assert view_shift > POSE_SHIFT_THRESHOLD_M, (
        "Sanity check failed: the FrameView world pose should move with the offset "
        f"(> {POSE_SHIFT_THRESHOLD_M} m); got {view_shift:.4f} m."
    )
    assert camera_shift > POSE_SHIFT_THRESHOLD_M, (
        f"Expected camera.data.pos_w to follow the camera pose (> {POSE_SHIFT_THRESHOLD_M} m); "
        f"got {camera_shift:.4f} m. The variation's FrameView write is not pushed to the camera pose."
    )
