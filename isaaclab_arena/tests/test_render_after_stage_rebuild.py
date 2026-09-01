# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for camera rendering across an experiment-runner stage rebuild."""

import os
import torch

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True
ENABLE_CAMERAS = True
# Steps taken after reset before images are captured, so the arm reaches its commanded pose and the
# RTX renderer finishes accumulating. Comparing settled frames keeps run-to-run sampling noise low.
NUM_STEPS = 30
# Mean per-channel 0-255 difference tolerated per camera. Settled renders agree to well under 1.0;
# geometry missing from a render moves this into the hundreds.
MAX_MEAN_ABSOLUTE_DIFFERENCE = 5.0
# Per-channel difference above which a pixel counts as changed, and the fraction of such pixels
# tolerated. Catches a single missing part that is too small to move the mean much.
PIXEL_DIFFERENCE_TOLERANCE = 8
MAX_CHANGED_PIXEL_FRACTION = 0.10
# Minimum per-image standard deviation, so a pair of blank renders cannot pass the comparison vacuously.
MIN_IMAGE_STD = 1.0
# Set True to dump the compared renders as PNGs into IMAGE_OUTPUT_DIR, which is created on demand.
SAVE_IMAGES = True
IMAGE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _build_droid_env():
    """Build a single-env, camera-enabled DROID environment on a lit packing table."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    asset_registry = AssetRegistry()
    scene = Scene(
        assets=[
            asset_registry.get_asset_by_name("light")(),
            asset_registry.get_asset_by_name("packing_table")(),
        ]
    )
    arena_env = IsaacLabArenaEnvironment(
        name="test_render_after_stage_rebuild",
        embodiment=DroidAbsoluteJointPositionEmbodiment(enable_cameras=ENABLE_CAMERAS),
        scene=scene,
    )
    # TODO(alexmillane, 2026-08-31): [lab-render-after-rebuild-bug] Remove --disable_fabric once the
    # render after rebuild bug is solved in Lab. Until then rebuilds render correctly only with Fabric
    # off (see execute_experiment), so this test builds the same way.
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras", "--disable_fabric"])
    # Both builds share the builder's default seed, so reset-time joint randomization draws the same
    # offsets and the two builds are expected to render the same scene.
    builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli))
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    # Arena turns this off by default. Without it a reset can return while assets are still streaming,
    # which blanks or misplaces geometry in the first frames and would be misread as a rebuild failure.
    env_cfg.wait_for_textures = True
    return builder.make_registered(env_cfg, env_kwargs)


def _render_camera_images(env) -> dict[str, torch.Tensor]:
    """Reset, step to settle the scene, and return a host copy of each camera's RGB image."""
    env.reset()
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(NUM_STEPS):
            obs, _, _, _, _ = env.step(actions)
    return {name: image.cpu().clone() for name, image in obs["camera_obs"].items()}


def _save_comparison_images(
    images_before_rebuild: dict[str, torch.Tensor],
    images_after_rebuild: dict[str, torch.Tensor],
) -> None:
    """Write a before, after, and absolute-difference PNG per camera into IMAGE_OUTPUT_DIR."""
    from PIL import Image

    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    for camera_name in sorted(images_before_rebuild.keys() & images_after_rebuild.keys()):
        before = images_before_rebuild[camera_name][0]
        after = images_after_rebuild[camera_name][0]
        images_to_save = {
            "before": before,
            "after": after,
            "difference": (before.float() - after.float()).abs().to(torch.uint8),
        }
        for tag, image in images_to_save.items():
            output_path = os.path.join(IMAGE_OUTPUT_DIR, f"{camera_name}-{tag}.png")
            Image.fromarray(image.numpy()).save(output_path)
            print(f"Wrote {output_path}", flush=True)


def _test_render_after_stage_rebuild(simulation_app) -> bool:
    from isaaclab_arena.evaluation.resource_cleanup import close_environment

    env = _build_droid_env()
    try:
        images_before_rebuild = _render_camera_images(env)
    finally:
        # The same teardown the experiment runner performs between rebuilds.
        close_environment(env)

    env = _build_droid_env()
    try:
        images_after_rebuild = _render_camera_images(env)
    finally:
        close_environment(env)

    # Written before the assertions so a failing rebuild still leaves images to look at.
    if SAVE_IMAGES:
        _save_comparison_images(images_before_rebuild, images_after_rebuild)

    assert set(images_before_rebuild) == set(images_after_rebuild), (
        "The rebuilt environment exposes a different set of cameras; "
        f"before: {sorted(images_before_rebuild)}, after: {sorted(images_after_rebuild)}."
    )
    for camera_name, before in images_before_rebuild.items():
        after = images_after_rebuild[camera_name]
        assert before.dtype == torch.uint8, f"Expected '{camera_name}' to render 0-255 RGB; got {before.dtype}."
        image_std = float(before.float().std())
        assert image_std > MIN_IMAGE_STD, (
            f"'{camera_name}' rendered an almost uniform image (std {image_std:.3f}), "
            "so comparing it across the rebuild would be vacuous."
        )

        absolute_difference = (before.float() - after.float()).abs()
        mean_absolute_difference = float(absolute_difference.mean())
        changed_pixel_fraction = float(absolute_difference.gt(PIXEL_DIFFERENCE_TOLERANCE).float().mean())
        rebuild_diagnostics = (
            f"'{camera_name}' renders differently after the stage rebuild "
            f"(mean absolute difference {mean_absolute_difference:.3f}/255, "
            f"{changed_pixel_fraction:.1%} of pixels changed by more than {PIXEL_DIFFERENCE_TOLERANCE}/255). "
            "Scene geometry likely failed to render into the rebuilt stage."
        )
        assert mean_absolute_difference <= MAX_MEAN_ABSOLUTE_DIFFERENCE, rebuild_diagnostics
        assert changed_pixel_fraction <= MAX_CHANGED_PIXEL_FRACTION, rebuild_diagnostics
    return True


@pytest.mark.with_cameras
def test_render_after_stage_rebuild():
    assert run_function_with_persistent_simulation_app(
        _test_render_after_stage_rebuild,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )
