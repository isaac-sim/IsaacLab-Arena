# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Assets for the spring-platform ballast-matching environment: the spring-platform rack and the load blocks."""

from __future__ import annotations

import itertools
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.spring_platform_ballast_match_usd import (
    DEFAULT_GEOMETRY,
    NUM_PLATFORMS,
    get_spring_platform_rack_usd_path,
    platform_joint_name,
)
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

GRAVITY_M_S2 = 9.81


@dataclass(frozen=True)
class SpringClass:
    """A spring class: the load it is matched to and the spring rate that deflects that load to the target."""

    matched_mass_kg: float
    stiffness_n_m: float


SPRING_CLASSES = {
    # stiffness = matched mass * g / 22 mm, so every matched pair deflects to the band center.
    "light": SpringClass(matched_mass_kg=0.10, stiffness_n_m=44.59),
    "medium": SpringClass(matched_mass_kg=0.20, stiffness_n_m=89.18),
    "heavy": SpringClass(matched_mass_kg=0.40, stiffness_n_m=178.36),
}

SPRING_CLASS_PERMUTATIONS = ["/".join(classes) for classes in itertools.permutations(SPRING_CLASSES)]
"""Every assignment of the three spring classes to platforms 0, 1, 2, written as 'class_0/class_1/class_2'."""


def spring_rest_position_m(spring_class: str, top_mass_kg: float = DEFAULT_GEOMETRY.top_mass_kg) -> float:
    """Return the spring's rest offset: the drive target at which the unloaded top rests at level (joint position 0)."""
    return top_mass_kg * GRAVITY_M_S2 / SPRING_CLASSES[spring_class].stiffness_n_m


def set_spring_rest_targets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    joint_names: list[str],
    rest_positions_m: list[float],
) -> None:
    """Set each spring's drive target to its rest offset, so each unloaded top rests at joint position 0."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    joint_ids = [asset.data.joint_names.index(name) for name in joint_names]
    targets = torch.tensor(rest_positions_m, device=env.device, dtype=torch.float32).repeat(len(env_ids), 1)
    asset.actuators.target_command.set_position_index(value=targets, joint_ids=joint_ids, env_ids=env_ids)


def parse_spring_classes(spring_classes: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return one spring class per platform from a list or a 'class_0/class_1/class_2' string."""
    if isinstance(spring_classes, str):
        spring_classes = spring_classes.split("/")
    spring_classes = tuple(str(name).strip() for name in spring_classes)
    assert len(spring_classes) == NUM_PLATFORMS, f"Give one spring class per platform, got {spring_classes}."
    for name in spring_classes:
        assert name in SPRING_CLASSES, f"Unknown spring class '{name}'; use one of {sorted(SPRING_CLASSES)}."
    return spring_classes


@configclass
class SpringClassesVariationCfg(VariationBaseCfg):
    """Configuration for SpringClassesVariation."""

    choices: list[str] = field(default_factory=lambda: list(SPRING_CLASS_PERMUTATIONS))
    """Spring-class assignments to draw from, each 'class_0/class_1/class_2' (platform 0 is at +Y). Give one entry to
    fix the assignment for a run, for example 'heavy/light/medium'."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Uniform choice over ``choices``, seeded by the build seed."""


class SpringClassesVariation(BuildTimeVariationBase):
    """Assign hidden spring classes to the rack's platforms for one environment build.

    Args:
        rack: The spring-platform rack whose springs the variation sets.
        cfg: Assignments to draw from. By default the variation is off and the rack keeps its spring_classes param.
        name: Name of the variation on the rack.
    """

    cfg: SpringClassesVariationCfg

    def __init__(
        self, rack: SpringPlatformRack, cfg: SpringClassesVariationCfg | None = None, name: str = "spring_classes"
    ):
        super().__init__(cfg=cfg if cfg is not None else SpringClassesVariationCfg(), name=name)
        self._rack = rack

    def _realize_at_build_time(self) -> None:
        choices = list(self.cfg.choices)
        assert choices, "SpringClassesVariation needs at least one choice."
        for choice in choices:
            parse_spring_classes(choice)
        selected = self.sampler.sample(num_samples=1, choices=choices)[0]
        self._rack.set_spring_classes(parse_spring_classes(selected))


@register_asset
class SpringPlatformRack(LibraryObject):
    """Three look-alike spring platforms on a fixed rack, with hidden spring rates.

    The USD is generated procedurally on first use; see spring_platform_ballast_match_usd.py for the frame and
    dimensions. Each top rides on a prismatic joint whose spring rate and damping come from an implicit actuator.
    A reset event sets each spring's drive target to its rest offset, which cancels the top's own weight, so every
    unloaded top rests at the same level (joint position 0) and the spring rate is not visible at rest.
    """

    registry_name = "spring_platform_rack"
    tags = ["object", "articulation", "procedural", "spring_platform"]
    object_type = ObjectType.ARTICULATION

    def __init__(
        self,
        spring_classes: list[str] | str = ("light", "medium", "heavy"),
        damping_ratio: float = 0.7,
        **kwargs,
    ):
        """Create the rack.

        Args:
            spring_classes: Spring class of platforms 0, 1 and 2 (platform 0 at +Y), each 'light', 'medium' or
                'heavy', as a list or a 'class_0/class_1/class_2' string. The spring_classes variation overrides it.
            damping_ratio: Damping ratio of each spring when it carries its matched load.
            **kwargs: Arguments passed to ``LibraryObject``.
        """
        assert damping_ratio > 0.0, "damping_ratio must be positive."
        self.usd_path = get_spring_platform_rack_usd_path()
        super().__init__(**kwargs)
        self.damping_ratio = damping_ratio
        self.object_cfg.init_state.joint_pos = {".*": 0.0}
        self.object_cfg.init_state.joint_vel = {".*": 0.0}
        self.spring_classes: tuple[str, ...] = ()
        self.set_spring_classes(spring_classes)
        self.add_variation(SpringClassesVariation(self))

    def set_spring_classes(self, spring_classes: list[str] | tuple[str, ...] | str) -> None:
        """Set the spring class of every platform and rebuild the spring actuators."""
        from isaaclab.actuators import ImplicitActuatorCfg

        self.spring_classes = parse_spring_classes(spring_classes)
        actuators = {}
        for index, class_name in enumerate(self.spring_classes):
            spring = SPRING_CLASSES[class_name]
            loaded_mass_kg = DEFAULT_GEOMETRY.top_mass_kg + spring.matched_mass_kg
            damping = 2.0 * self.damping_ratio * (spring.stiffness_n_m * loaded_mass_kg) ** 0.5
            actuators[f"spring_{index}"] = ImplicitActuatorCfg(
                joint_names_expr=[platform_joint_name(index)],
                stiffness=spring.stiffness_n_m,
                damping=damping,
                effort_limit_sim=100.0,
            )
        self.object_cfg.actuators = actuators

    def get_event_cfgs(self) -> list[tuple[str, EventTermCfg]]:
        """Return the pose and joint reset events, then the event that sets each spring's rest offset."""
        event_cfgs = super().get_event_cfgs()
        rest_targets_cfg = EventTermCfg(
            func=set_spring_rest_targets,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(self.name),
                "joint_names": [platform_joint_name(index) for index in range(NUM_PLATFORMS)],
                "rest_positions_m": [spring_rest_position_m(name) for name in self.spring_classes],
            },
        )
        event_cfgs.append((f"{self.name}_spring_rest_targets", rest_targets_cfg))
        return event_cfgs


@register_asset
class SpringLoadBlock(Object):
    """A painted steel load block; every block looks the same whatever its mass."""

    registry_name = "spring_load_block"
    tags = ["object", "graspable", "procedural", "spring_platform"]

    def __init__(
        self,
        mass_kg: float = 0.2,
        size_m: float = 0.05,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
    ):
        """Create a load block.

        Args:
            mass_kg: Simulated mass in kilograms. The object's mass variation overrides it at reset.
            size_m: Edge length of the cube in meters.
            name: Name of the object in the scene. Defaults to ``load_block``.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the object.
        """
        assert mass_kg > 0.0, "mass_kg must be positive."
        spawner_cfg = sim_utils.CuboidCfg(
            size=(size_m, size_m, size_m),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.52, 0.47), metallic=0.6, roughness=0.5),
        )
        super().__init__(
            name=name if name is not None else "load_block",
            prim_path=prim_path,
            object_type=ObjectType.RIGID,
            spawner_cfg=spawner_cfg,
            initial_pose=initial_pose,
            tags=list(self.tags),
        )
