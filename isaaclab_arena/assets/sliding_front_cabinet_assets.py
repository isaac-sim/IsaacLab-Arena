# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Assets for the sliding-front cabinet environment: the cabinet with hidden front modes, the tokens and the tray."""

from __future__ import annotations

import itertools
from dataclasses import field

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.sliding_front_cabinet_usd import (
    FRONT_MODES,
    NUM_COMPARTMENTS,
    front_joint_name,
    get_sliding_front_cabinet_usd_path,
    get_tray_usd_path,
)
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

FRONT_MODE_COMBINATIONS = ["/".join(modes) for modes in itertools.product(FRONT_MODES, repeat=NUM_COMPARTMENTS)]
"""Every assignment of opening modes to fronts 1, 2 and 3 (bottom to top), written as 'mode_1/mode_2/mode_3'."""


def parse_front_modes(front_modes: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return one opening mode per front from a list or a 'mode_1/mode_2/mode_3' string."""
    if isinstance(front_modes, str):
        front_modes = front_modes.split("/")
    front_modes = tuple(str(mode).strip() for mode in front_modes)
    assert len(front_modes) == NUM_COMPARTMENTS, f"Give one mode per front, got {front_modes}."
    for mode in front_modes:
        assert mode in FRONT_MODES, f"Unknown front mode '{mode}'; use one of {FRONT_MODES}."
    return front_modes


@configclass
class FrontModesVariationCfg(VariationBaseCfg):
    """Configuration for FrontModesVariation."""

    choices: list[str] = field(default_factory=lambda: list(FRONT_MODE_COMBINATIONS))
    """Mode assignments to draw from, each 'mode_1/mode_2/mode_3' for fronts 1 (bottom) to 3 (top). Give one entry
    to fix the assignment for a run, for example 'slide_right/slide_left/fold_down'."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Uniform choice over ``choices``, seeded by the build seed."""


class FrontModesVariation(BuildTimeVariationBase):
    """Assign hidden opening modes to the cabinet's fronts for one environment build.

    Args:
        cabinet: The cabinet whose fronts the variation sets.
        cfg: Assignments to draw from. By default the variation is off and the cabinet keeps its front_modes param.
        name: Name of the variation on the cabinet.
    """

    cfg: FrontModesVariationCfg

    def __init__(
        self, cabinet: SlidingFrontCabinet, cfg: FrontModesVariationCfg | None = None, name: str = "front_modes"
    ):
        super().__init__(cfg=cfg if cfg is not None else FrontModesVariationCfg(), name=name)
        self._cabinet = cabinet

    def _realize_at_build_time(self) -> None:
        choices = list(self.cfg.choices)
        assert choices, "FrontModesVariation needs at least one choice."
        for choice in choices:
            parse_front_modes(choice)
        selected = self.sampler.sample(num_samples=1, choices=choices)[0]
        self._cabinet.set_front_modes(parse_front_modes(selected))


@register_asset
class SlidingFrontCabinet(LibraryObject):
    """A light three-compartment cabinet whose look-alike fronts open by hidden modes.

    The USD is generated procedurally per mode assignment; see sliding_front_cabinet_usd.py for the frame, joints and
    dimensions. Fronts are passive: their actuators only add viscous damping. Every front is closed at joint
    position 0, and its openness is the joint position's magnitude over its travel.
    """

    registry_name = "sliding_front_cabinet"
    tags = ["object", "articulation", "procedural", "cabinet"]
    object_type = ObjectType.ARTICULATION

    def __init__(
        self,
        front_modes: list[str] | str = ("slide_left", "fold_down", "slide_right"),
        fixed_base: bool = False,
        slide_damping: float = 0.5,
        fold_damping: float = 0.01,
        **kwargs,
    ):
        """Create the cabinet.

        Args:
            front_modes: Opening mode of fronts 1, 2 and 3 (bottom to top), each 'slide_left', 'slide_right' or
                'fold_down', as a list or a 'mode_1/mode_2/mode_3' string. The front_modes variation overrides it.
            fixed_base: Whether to fix the carcass to the world. By default the light cabinet stands free.
            slide_damping: Viscous damping of a sliding front, in N*s/m.
            fold_damping: Viscous damping of a fold-down front, in N*m*s/rad.
            **kwargs: Arguments passed to ``LibraryObject``.
        """
        self.fixed_base = fixed_base
        self.slide_damping = slide_damping
        self.fold_damping = fold_damping
        self.front_modes = parse_front_modes(front_modes)
        self.usd_path = get_sliding_front_cabinet_usd_path(self.front_modes, fixed_base)
        super().__init__(**kwargs)
        self.object_cfg.init_state.joint_pos = {".*": 0.0}
        self.object_cfg.init_state.joint_vel = {".*": 0.0}
        self.set_front_modes(self.front_modes)
        self.add_variation(FrontModesVariation(self))

    def set_front_modes(self, front_modes: list[str] | tuple[str, ...] | str) -> None:
        """Set every front's opening mode: select the matching USD and rebuild the front actuators."""
        from isaaclab.actuators import ImplicitActuatorCfg

        self.front_modes = parse_front_modes(front_modes)
        self.usd_path = get_sliding_front_cabinet_usd_path(self.front_modes, self.fixed_base)
        self.object_cfg.spawn.usd_path = self.usd_path
        actuators = {}
        for index, mode in enumerate(self.front_modes, start=1):
            damping = self.fold_damping if mode == "fold_down" else self.slide_damping
            actuators[f"front_{index}"] = ImplicitActuatorCfg(
                joint_names_expr=[front_joint_name(index)], stiffness=0.0, damping=damping
            )
        self.object_cfg.actuators = actuators


@register_asset
class CabinetToken(Object):
    """A small brass token, an upright cylinder a gripper can pinch from the side, that fits in a compartment."""

    registry_name = "cabinet_token"
    tags = ["object", "graspable", "procedural", "cabinet"]

    def __init__(
        self,
        radius_m: float = 0.015,
        height_m: float = 0.03,
        mass_kg: float = 0.02,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
    ):
        """Create a token.

        Args:
            radius_m: Cylinder radius in meters.
            height_m: Cylinder height in meters.
            mass_kg: Simulated mass in kilograms.
            name: Name of the object in the scene. Defaults to ``cabinet_token``.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the object.
        """
        spawner_cfg = sim_utils.CylinderCfg(
            radius=radius_m,
            height=height_m,
            axis="Z",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.65, 0.20), metallic=0.8, roughness=0.3),
        )
        super().__init__(
            name=name if name is not None else "cabinet_token",
            prim_path=prim_path,
            object_type=ObjectType.RIGID,
            spawner_cfg=spawner_cfg,
            initial_pose=initial_pose,
            tags=list(self.tags),
        )


@register_asset
class ProceduralTray(LibraryObject):
    """An open rectangular tray (14 x 16 cm, 2 cm rim) generated procedurally; see sliding_front_cabinet_usd.py."""

    registry_name = "procedural_tray"
    tags = ["object", "container", "procedural"]
    object_type = ObjectType.RIGID

    def __init__(self, **kwargs):
        self.usd_path = get_tray_usd_path()
        super().__init__(**kwargs)
