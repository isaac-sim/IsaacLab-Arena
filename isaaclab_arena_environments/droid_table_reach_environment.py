# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID Maple-table environments with and without the WithinReach placement relation.

The robot stands back from the table, so random placement on the table leaves many items beyond
the arm's reach. ``droid_table_unconstrained_reach`` places the items with ``On`` only;
``droid_table_within_reach`` adds ``WithinReach`` so the solver keeps every item reachable.

Compare the two layouts with:

    python isaaclab_arena/evaluation/policy_runner.py --viz kit --policy_type zero_action \
        --num_envs 4 --env_spacing 3.0 --placement_seed 42 --num_steps 500 droid_table_within_reach
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


REACH_DEMO_OBJECTS = [
    "orange_01_fruits_veggies_robolab",
    "lemon_01_fruits_veggies_robolab",
    "ketchup_bottle_hope_robolab",
    "alphabet_soup_can_hope_robolab",
    "canned_peaches_hope_robolab",
    "spoon_handal_robolab",
    "sugar_box_ycb_robolab",
]


@dataclass
class DroidTableReachEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the DROID Maple-table reach demonstration."""

    embodiment: str = "droid_abs_joint_pos"
    light_intensity: float = 500.0
    episode_length_s: float | None = None
    robot_x_m: float = -0.25
    """Robot base X in the environment frame; the default table spans X from 0.2 m to 0.9 m."""
    reach_scale: float = 0.9
    """WithinReach margin factor applied to the robot's published reach."""


@dataclass
class DroidTableUnconstrainedReachEnvironmentCfg(DroidTableReachEnvironmentCfg):
    """Configure the reach demonstration that places items with On only."""


@dataclass
class DroidTableWithinReachEnvironmentCfg(DroidTableReachEnvironmentCfg):
    """Configure the reach demonstration that also places items WithinReach of the robot."""


@register_environment
class DroidTableUnconstrainedReachEnvironment(ArenaEnvironmentFactory[DroidTableUnconstrainedReachEnvironmentCfg]):
    """Place items anywhere on a Maple table, ignoring whether the robot can reach them."""

    name: str = "droid_table_unconstrained_reach"
    _legacy_argparse_cfg_type = DroidTableUnconstrainedReachEnvironmentCfg

    def build(self, cfg: DroidTableUnconstrainedReachEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        return _build_droid_table_reach_environment(self, cfg, use_within_reach=False)


@register_environment
class DroidTableWithinReachEnvironment(ArenaEnvironmentFactory[DroidTableWithinReachEnvironmentCfg]):
    """Place items on a Maple table where the robot can reach them."""

    name: str = "droid_table_within_reach"
    _legacy_argparse_cfg_type = DroidTableWithinReachEnvironmentCfg

    def build(self, cfg: DroidTableWithinReachEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        return _build_droid_table_reach_environment(self, cfg, use_within_reach=True)


def _build_droid_table_reach_environment(
    factory: ArenaEnvironmentFactory,
    cfg: DroidTableReachEnvironmentCfg,
    use_within_reach: bool,
) -> IsaacLabArenaEnvironment:
    """Build the reach demonstration, adding WithinReach to every item when use_within_reach is True."""
    from isaaclab.envs.common import ViewerCfg

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.relations import IsAnchor, On, WithinReach
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    background = factory.asset_registry.get_asset_by_name("maple_table_robolab")()
    table_reference = ObjectReference(
        name="table",
        prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
        parent_asset=background,
        object_type=ObjectType.RIGID,
    )
    table_reference.add_relation(IsAnchor())

    # Bound the robot by its stand only, so the arm's box does not push items out of its own reach.
    embodiment = factory.asset_registry.get_asset_by_name(cfg.embodiment)(
        enable_cameras=cfg.enable_cameras,
        initial_pose=Pose(position_xyz=(cfg.robot_x_m, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        placement_bbox_stand_only=True,
    )
    embodiment.add_relation(IsAnchor())

    items = [factory.asset_registry.get_asset_by_name(name)() for name in REACH_DEMO_OBJECTS]
    for item in items:
        item.add_relation(On(table_reference))
        if use_within_reach:
            item.add_relation(WithinReach(embodiment, reach_scale=cfg.reach_scale))

    light = factory.asset_registry.get_asset_by_name("light")()
    light.set_intensity(cfg.light_intensity)
    directional_light = factory.asset_registry.get_asset_by_name("directional_light")()

    scene = Scene(assets=[background, light, directional_light, table_reference, *items])

    episode_length_s = cfg.episode_length_s

    def _configure_viewer_and_timeout(env_cfg):
        env_cfg.viewer = ViewerCfg(eye=(1.6, 0.0, 2.0), lookat=(0.3, 0.0, 0.0))
        if episode_length_s is not None:
            import isaaclab.envs.mdp as mdp_isaac_lab
            from isaaclab.managers import TerminationTermCfg

            env_cfg.episode_length_s = episode_length_s
            env_cfg.terminations.time_out = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)
        return env_cfg

    name = DroidTableWithinReachEnvironment.name if use_within_reach else DroidTableUnconstrainedReachEnvironment.name
    return IsaacLabArenaEnvironment(
        name=name,
        embodiment=embodiment,
        scene=scene,
        task=NoTask(),
        env_cfg_callback=_configure_viewer_and_timeout,
    )
