# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

HEADLESS = True

TEST_EVENT_NAME = "test_runtime_variation"
TEST_ASSET_NAME = "sphere"


def noop_test_variation_event(env, env_ids, asset_cfg):  # noqa: ARG001
    """No-op event term for runtime variation wiring tests."""


@configclass
class TestRunTimeVariationCfg(VariationBaseCfg):
    __test__ = False

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=[0.0], high=[1.0]),
    )


class TestRunTimeVariation(RunTimeVariationBase):
    __test__ = False

    cfg: TestRunTimeVariationCfg

    def __init__(self, asset_name: str, cfg: TestRunTimeVariationCfg | None = None, name: str = "test_runtime"):
        super().__init__(cfg=cfg if cfg is not None else TestRunTimeVariationCfg(), name=name)
        self.asset_name = asset_name

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        event_cfg = EventTermCfg(
            func=noop_test_variation_event,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg(self.asset_name)},
        )
        return TEST_EVENT_NAME, event_cfg


def get_test_environment(*, enabled: bool):
    """Build a minimal arena env with an optional enabled run-time test variation."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    sphere = AssetRegistry().get_asset_by_name(TEST_ASSET_NAME)()
    assert sphere.name == TEST_ASSET_NAME

    variation = TestRunTimeVariation(sphere.name)
    if enabled:
        variation.enable()
    assert variation.enabled is enabled
    sphere.add_variation(variation)

    return IsaacLabArenaEnvironment(
        name="test_runtime_variations",
        scene=Scene(assets=[sphere]),
    )


def _test_disabled_runtime_variation_not_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(enabled=False)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, args_cli).compose_manager_cfg()

    assert not hasattr(env_cfg.events, TEST_EVENT_NAME), (
        f"Disabled variation must not add '{TEST_EVENT_NAME}' to env_cfg.events; "
        f"got event fields: {sorted(vars(env_cfg.events))}."
    )
    return True


def _test_enabled_runtime_variation_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(enabled=True)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, args_cli).compose_manager_cfg()

    assert hasattr(
        env_cfg.events, TEST_EVENT_NAME
    ), f"Expected env_cfg.events to contain '{TEST_EVENT_NAME}'; got event fields: {sorted(vars(env_cfg.events))}."
    event_cfg = getattr(env_cfg.events, TEST_EVENT_NAME)
    assert event_cfg.func is noop_test_variation_event
    assert event_cfg.mode == "reset"
    assert event_cfg.params["asset_cfg"].name == TEST_ASSET_NAME
    return True


def test_disabled_runtime_variation_not_in_events_cfg():
    assert run_simulation_app_function(
        _test_disabled_runtime_variation_not_in_events_cfg,
        headless=HEADLESS,
    )


def test_enabled_runtime_variation_in_events_cfg():
    assert run_simulation_app_function(
        _test_enabled_runtime_variation_in_events_cfg,
        headless=HEADLESS,
    )
