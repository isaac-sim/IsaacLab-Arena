# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the typed Arena environment-builder configuration boundary."""

from types import SimpleNamespace
from unittest import mock

import pytest

from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg, CameraResolutionMode
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


@pytest.mark.parametrize("camera_resolution_mode", ["embodiment_default", "policy_default"])
def test_builder_configuration_accepts_camera_resolution_mode_strings(camera_resolution_mode):
    """Convert plain value strings into the CameraResolutionMode enum, as done for the physics backend."""
    cfg = ArenaEnvBuilderCfg(camera_resolution_mode=camera_resolution_mode)

    assert cfg.camera_resolution_mode is CameraResolutionMode(camera_resolution_mode)


def test_builder_configuration_defaults_to_the_embodiment_camera_resolution():
    """Keep the embodiment's declared camera dimensions unless a Run opts into the policy's size."""
    assert ArenaEnvBuilderCfg().camera_resolution_mode is CameraResolutionMode.EMBODIMENT_DEFAULT


@pytest.mark.parametrize("camera_resolution_mode", ["", "hd", "auto", "126x224", "POLICY_DEFAULT"])
def test_builder_configuration_rejects_unknown_camera_resolution_modes(camera_resolution_mode):
    """Reject anything other than the two CameraResolutionMode values."""
    with pytest.raises(ValueError):
        ArenaEnvBuilderCfg(camera_resolution_mode=camera_resolution_mode)


def _test_builder_applies_camera_resolution_mode(simulation_app):
    # ArenaEnvBuilder imports Isaac Sim modules, so it is imported after the SimulationApp starts.
    from isaaclab_arena.environments import arena_env_builder
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg

    class FakeCameraRig:
        def __init__(self):
            self.calls = []

        def fit_resolution(self, max_height, max_width):
            self.calls.append(("fit", max_height, max_width))

    def arena_environment(enable_cameras=True):
        return SimpleNamespace(embodiment=SimpleNamespace(enable_cameras=enable_cameras, camera_config=FakeCameraRig()))

    class ImagePolicyCfg(PolicyCfg):
        pass

    class ImagePolicy(PolicyBase[ImagePolicyCfg]):
        @classmethod
        def default_image_size(cls, config):
            return (224, 224)

        def get_action(self, env, observation):
            raise NotImplementedError

    class FakePolicyRegistry:
        def get_policy_type_for_cfg(self, policy_cfg):
            return ImagePolicy

    policy_default_cfg = ArenaEnvBuilderCfg(camera_resolution_mode="policy_default")

    embodiment_default_environment = arena_environment(enable_cameras=False)
    ArenaEnvBuilder(embodiment_default_environment, ArenaEnvBuilderCfg(camera_resolution_mode="embodiment_default"))
    assert embodiment_default_environment.embodiment.camera_config.calls == []

    with mock.patch.object(arena_env_builder, "PolicyRegistry", FakePolicyRegistry):
        policy_default_environment = arena_environment()
        ArenaEnvBuilder(policy_default_environment, policy_default_cfg, policy_cfg=ImagePolicyCfg())
        assert policy_default_environment.embodiment.camera_config.calls == [("fit", 224, 224)]

        with pytest.raises(AssertionError, match="requires embodiment cameras to be enabled"):
            ArenaEnvBuilder(arena_environment(enable_cameras=False), policy_default_cfg, policy_cfg=ImagePolicyCfg())

    with pytest.raises(AssertionError, match="requires the policy config"):
        ArenaEnvBuilder(arena_environment(), policy_default_cfg)

    # The zero-action policy is registered and declares no image size.
    with pytest.raises(AssertionError, match="ZeroActionPolicy declares none"):
        ArenaEnvBuilder(arena_environment(), policy_default_cfg, policy_cfg=ZeroActionPolicyCfg())
    return True


def test_builder_applies_camera_resolution_mode():
    """Fit the embodiment camera rig to the policy only in policy_default mode, and only with cameras."""
    result = run_function_with_persistent_simulation_app(_test_builder_applies_camera_resolution_mode)
    assert result
