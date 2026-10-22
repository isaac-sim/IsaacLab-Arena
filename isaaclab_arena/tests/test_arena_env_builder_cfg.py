# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the typed Arena environment-builder configuration boundary."""

from types import SimpleNamespace

import pytest

from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.utils.physics_backend import PhysicsBackend


# TODO(cvolk, 2026-07-03): [typed-config-migration] Delete the argparse adapter tests below with
# arena_env_builder_cfg_from_argparse once runners pass ArenaEnvBuilderCfg directly.
def test_cli_defaults_match_builder_configuration():
    """Keep the manual CLI defaults aligned with the typed builder configuration."""
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])

    assert not hasattr(args_cli, "headless")
    assert args_cli.visualizer is None
    assert arena_env_builder_cfg_from_argparse(args_cli) == ArenaEnvBuilderCfg()


def test_argparse_adapter_maps_builder_configuration():
    """Translate only builder-owned command-line values into the typed config."""
    parser = get_isaaclab_arena_cli_parser()
    add_policy_runner_arguments(parser)
    args_cli = parser.parse_args([
        "--policy_type",
        "zero_action",
        "--num_envs",
        "3",
        "--env_spacing",
        "2.5",
        "--seed",
        "7",
        "--no_solve_relations",
        "--placement_seed",
        "11",
        "--no-resolve_on_reset",
        "--disable_fabric",
        "--mimic",
        "--presets",
        "newton",
        "--device",
        "cpu",
        "--language_instruction",
        "pick up the cube",
    ])

    cfg = arena_env_builder_cfg_from_argparse(args_cli)

    assert cfg == ArenaEnvBuilderCfg(
        num_envs=3,
        env_spacing=2.5,
        seed=7,
        solve_relations=False,
        placement_seed=11,
        resolve_on_reset=False,
        disable_fabric=True,
        mimic=True,
        presets=PhysicsBackend.NEWTON,
        device="cpu",
        language_instruction="pick up the cube",
    )
    assert cfg.presets is PhysicsBackend.NEWTON


def test_builder_configuration_requires_positive_num_envs():
    """Reject configurations that cannot build any environment instances."""
    with pytest.raises(AssertionError, match="num_envs must be greater than zero"):
        ArenaEnvBuilderCfg(num_envs=0)


def test_builder_configuration_rejects_unknown_physics_backend():
    """Reject arbitrary strings at the typed builder boundary."""
    with pytest.raises(ValueError, match="unknown_backend"):
        ArenaEnvBuilderCfg(presets="unknown_backend")


def test_builder_configuration_normalizes_physics_backend_strings():
    """Normalize legacy configuration input to the typed backend enum."""
    cfg = ArenaEnvBuilderCfg(presets="newton")

    assert cfg.presets is PhysicsBackend.NEWTON


@pytest.mark.parametrize("camera_resolution", ["hd", "auto", "126x224", "1x1"])
def test_builder_configuration_accepts_camera_resolution_modes(camera_resolution):
    """Accept the rig default, the policy-fitted mode, and explicit dimensions."""
    cfg = ArenaEnvBuilderCfg(camera_resolution=camera_resolution)

    assert cfg.camera_resolution == camera_resolution
    assert cfg.camera_resolution_is_auto == (camera_resolution == "auto")


@pytest.mark.parametrize("camera_resolution", ["", "HD", "360", "360x", "x640", "360x640x3", "360 x 640", "360.0x640"])
def test_builder_configuration_rejects_malformed_camera_resolution(camera_resolution):
    """Reject anything other than 'hd', 'auto', or '<height>x<width>'."""
    with pytest.raises(AssertionError, match="camera_resolution must be"):
        ArenaEnvBuilderCfg(camera_resolution=camera_resolution)


@pytest.mark.parametrize("camera_resolution", ["0x640", "360x0"])
def test_builder_configuration_requires_positive_camera_resolution(camera_resolution):
    """Reject camera dimensions that Isaac Lab cannot render."""
    with pytest.raises(AssertionError, match="greater than zero"):
        ArenaEnvBuilderCfg(camera_resolution=camera_resolution)


def test_explicit_camera_resolution_parses_height_then_width():
    """Read '<height>x<width>' in the (height, width) order Isaac Lab cameras use."""
    assert ArenaEnvBuilderCfg(camera_resolution="126x224").explicit_camera_resolution_hw() == (126, 224)


def _test_builder_applies_camera_resolution(simulation_app):
    # ArenaEnvBuilder imports Isaac Sim modules, so it is imported after the SimulationApp starts.
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    class FakeCameraRig:
        def __init__(self):
            self.calls = []

        def set_resolution(self, height, width):
            self.calls.append(("set", height, width))

        def fit_resolution(self, max_height, max_width):
            self.calls.append(("fit", max_height, max_width))

    def arena_environment(enable_cameras=True):
        return SimpleNamespace(embodiment=SimpleNamespace(enable_cameras=enable_cameras, camera_config=FakeCameraRig()))

    hd_environment = arena_environment(enable_cameras=False)
    ArenaEnvBuilder(hd_environment, ArenaEnvBuilderCfg(camera_resolution="hd"))
    assert hd_environment.embodiment.camera_config.calls == []

    explicit_environment = arena_environment()
    ArenaEnvBuilder(explicit_environment, ArenaEnvBuilderCfg(camera_resolution="126x224"))
    assert explicit_environment.embodiment.camera_config.calls == [("set", 126, 224)]

    auto_environment = arena_environment()
    ArenaEnvBuilder(auto_environment, ArenaEnvBuilderCfg(camera_resolution="auto"), policy_image_size=(224, 224))
    assert auto_environment.embodiment.camera_config.calls == [("fit", 224, 224)]

    with pytest.raises(AssertionError, match="requires a policy that declares the image size"):
        ArenaEnvBuilder(arena_environment(), ArenaEnvBuilderCfg(camera_resolution="auto"))

    with pytest.raises(AssertionError, match="camera_resolution requires embodiment cameras to be enabled"):
        ArenaEnvBuilder(arena_environment(enable_cameras=False), ArenaEnvBuilderCfg(camera_resolution="126x224"))
    return True


def test_builder_applies_camera_resolution():
    """Apply each camera_resolution mode to the embodiment camera rig, or reject it without cameras."""
    result = run_function_with_persistent_simulation_app(_test_builder_applies_camera_resolution)
    assert result
