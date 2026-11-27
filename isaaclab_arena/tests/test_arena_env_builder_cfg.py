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
from isaaclab_arena.utils.camera_renderer import CameraRenderer
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
        "--camera_renderer",
        "newton_warp",
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
        camera_renderer=CameraRenderer.NEWTON_WARP,
        device="cpu",
        language_instruction="pick up the cube",
    )
    assert cfg.presets is PhysicsBackend.NEWTON
    assert cfg.camera_renderer is CameraRenderer.NEWTON_WARP


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


@pytest.mark.parametrize("camera_renderer", ["isaac_rtx", "newton_warp", "ovrtx"])
def test_builder_configuration_accepts_camera_renderer_strings(camera_renderer):
    """Convert plain value strings into the CameraRenderer enum, as done for the physics backend."""
    assert ArenaEnvBuilderCfg(camera_renderer=camera_renderer).camera_renderer is CameraRenderer(camera_renderer)


def test_builder_configuration_defaults_to_isaac_rtx_cameras():
    """Keep rendering the cameras with Isaac RTX unless a Run opts into Newton Warp."""
    assert ArenaEnvBuilderCfg().camera_renderer is CameraRenderer.ISAAC_RTX


@pytest.mark.parametrize("camera_renderer", ["", "rtx", "warp", "NEWTON_WARP"])
def test_builder_configuration_rejects_unknown_camera_renderers(camera_renderer):
    """Reject anything other than the two CameraRenderer values."""
    with pytest.raises(ValueError):
        ArenaEnvBuilderCfg(camera_renderer=camera_renderer)


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
    ArenaEnvBuilder(
        embodiment_default_environment, ArenaEnvBuilderCfg(camera_resolution_mode="embodiment_default")
    )._apply_camera_resolution_mode()
    assert embodiment_default_environment.embodiment.camera_config.calls == []

    with mock.patch.object(arena_env_builder, "PolicyRegistry", FakePolicyRegistry):
        policy_default_environment = arena_environment()
        policy_default_builder = ArenaEnvBuilder(
            policy_default_environment, policy_default_cfg, policy_cfg=ImagePolicyCfg()
        )
        # The rig is fitted when the environment config is built, not when the builder is constructed.
        assert policy_default_environment.embodiment.camera_config.calls == []
        policy_default_builder._apply_camera_resolution_mode()
        assert policy_default_environment.embodiment.camera_config.calls == [("fit", 224, 224)]

        with pytest.raises(AssertionError, match="requires embodiment cameras to be enabled"):
            ArenaEnvBuilder(
                arena_environment(enable_cameras=False), policy_default_cfg, policy_cfg=ImagePolicyCfg()
            )._apply_camera_resolution_mode()

    with pytest.raises(AssertionError, match="requires the policy config"):
        ArenaEnvBuilder(arena_environment(), policy_default_cfg)._apply_camera_resolution_mode()

    # The zero-action policy is registered and declares no image size.
    with pytest.raises(AssertionError, match="ZeroActionPolicy declares none"):
        ArenaEnvBuilder(
            arena_environment(), policy_default_cfg, policy_cfg=ZeroActionPolicyCfg()
        )._apply_camera_resolution_mode()
    return True


def test_builder_applies_camera_resolution_mode():
    """Fit the embodiment camera rig to the policy only in policy_default mode, and only with cameras."""
    result = run_function_with_persistent_simulation_app(_test_builder_applies_camera_resolution_mode)
    assert result


def _test_builder_applies_camera_renderer(simulation_app):
    # ArenaEnvBuilder imports Isaac Sim modules, so it is imported after the SimulationApp starts.
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    class FakeCameraRig:
        def __init__(self):
            self.renderers = []

        def set_renderer(self, renderer):
            self.renderers.append(renderer)

    def arena_environment(default_physics_backend):
        return SimpleNamespace(
            embodiment=SimpleNamespace(camera_config=FakeCameraRig()), default_physics_backend=default_physics_backend
        )

    # Isaac RTX, the default, renders with either physics backend.
    for physics_backend in PhysicsBackend:
        environment = arena_environment(physics_backend)
        ArenaEnvBuilder(environment, ArenaEnvBuilderCfg())._apply_camera_renderer()
        assert environment.embodiment.camera_config.renderers == [CameraRenderer.ISAAC_RTX]

    newton_warp_cfg = ArenaEnvBuilderCfg(camera_renderer="newton_warp")
    newton_environment = arena_environment(PhysicsBackend.NEWTON)
    ArenaEnvBuilder(newton_environment, newton_warp_cfg)._apply_camera_renderer()
    assert newton_environment.embodiment.camera_config.renderers == [CameraRenderer.NEWTON_WARP]

    with pytest.raises(AssertionError, match="requires the Newton physics backend"):
        ArenaEnvBuilder(arena_environment(PhysicsBackend.PHYSX), newton_warp_cfg)._apply_camera_renderer()
    # This test runs in Kit, which OVRTX cannot share a process with.
    with pytest.raises(ValueError, match="cannot run with Kit"):
        ArenaEnvBuilder(
            arena_environment(PhysicsBackend.NEWTON), ArenaEnvBuilderCfg(camera_renderer="ovrtx")
        )._apply_camera_renderer()
    return True


def test_builder_applies_camera_renderer():
    """Pass the selected renderer to the camera rig; reject Newton Warp without Newton physics and OVRTX in Kit."""
    assert run_function_with_persistent_simulation_app(_test_builder_applies_camera_renderer)
