# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registered Arena interfaces for the authored service geometry and joint affordances."""

from __future__ import annotations

import math
from copy import copy
from pathlib import Path
from typing import Literal, get_args

from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.agentic_environment_generation.authoring_metadata import AuthoringMetadata, ParameterMetadata
from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.light_color_variation import LightColorVariation
from isaaclab_arena.variations.light_intensity_variation import LightIntensityVariation

from .assets import prepare_assets

ComponentName = Literal[
    "vacuum_body",
    "dust_cup",
    "battery",
    "battery_decoy",
    "filter",
    "filter_decoy",
    "crevice_tool",
    "brush_tool",
    "obstruction",
    "airflow_adapter",
    "debris",
]
FixtureName = Literal[
    "bench",
    "floor",
    "work_order",
    "battery_bin",
    "filter_bin",
    "waste_bin",
    "spare_rack",
    "parking_tray",
    "release_panel",
    "airflow_test_panel",
]
InstrumentName = Literal["cradle", "battery_tester", "airflow_tester"]
CaseJoint = Literal["hinge", "latch"]


@register_asset
class ServiceBench(Background):
    """The authored service bench exposed as an Arena graph background."""

    name = "return_to_service_bench"
    tags = ["background", "service", "fixed"]
    object_type = ObjectType.BASE
    authoring_metadata = AuthoringMetadata(
        constraints=("Uses the existing Blender bench; initial pose Z is the authored work-surface height.",),
        reset_semantics="The bench remains fixed at its configured initial pose.",
    )

    def __init__(
        self,
        asset_root: str | Path | None = None,
        initial_pose: Pose | None = None,
        instance_name: str | None = None,
    ):
        prepared = prepare_assets(asset_root)
        initial_pose = initial_pose or Pose((0.48, 0.0, 0.78))
        super().__init__(
            name=instance_name or self.name,
            usd_path=str(prepared.asset_usd("bench", "static")),
            object_min_z=initial_pose.position_xyz[2] - 0.2,
            initial_pose=initial_pose,
            reset_nested_physics=False,
            tags=list(self.tags),
        )


@register_asset
class ServiceComponent(Object):
    """A movable component using the Blender-authored geometry and mass."""

    name = "return_to_service_component"
    tags = ["object", "service", "movable"]
    object_type = ObjectType.RIGID
    authoring_metadata = AuthoringMetadata(
        provides=("service_component",),
        constraints=(
            "Requires the existing Blender asset bundle; discovery never creates geometry.",
            "Manifest grasp and socket poses are expressed in the component root frame.",
        ),
        reset_semantics="Arena restores the configured root pose; the service task owns fault and connector resets.",
    )

    def __init__(
        self,
        component: ComponentName = "battery",
        asset_root: str | Path | None = None,
        instance_name: str | None = None,
        initial_pose: Pose | None = None,
    ):
        assert component in get_args(ComponentName), f"Unknown movable service component: {component!r}"
        prepared = prepare_assets(asset_root)
        super().__init__(
            name=instance_name or self.name,
            tags=list(self.tags),
            object_type=self.object_type,
            usd_path=str(prepared.asset_usd(component, "rigid")),
            initial_pose=initial_pose,
        )
        self.component = component


@register_asset
class ServiceFixture(Object):
    """A fixed work surface, bin, or fixture from the authored service asset bundle."""

    name = "return_to_service_fixture"
    tags = ["object", "service", "fixed"]
    object_type = ObjectType.BASE
    authoring_metadata = AuthoringMetadata(
        constraints=(
            "Fixed fixture geometry is not a movable pick-up target.",
            "Container interior_bounds and poses come from the existing asset manifest, in the fixture root frame.",
        ),
        reset_semantics="The fixture remains fixed at its configured initial pose.",
    )

    def __init__(
        self,
        component: FixtureName = "battery_bin",
        asset_root: str | Path | None = None,
        instance_name: str | None = None,
        initial_pose: Pose | None = None,
    ):
        assert component in get_args(FixtureName), f"Unknown static service fixture: {component!r}"
        prepared = prepare_assets(asset_root)
        super().__init__(
            name=instance_name or self.name,
            tags=list(self.tags),
            object_type=self.object_type,
            usd_path=str(prepared.asset_usd(component, "static")),
            initial_pose=initial_pose,
        )
        self.component = component


@register_asset
class ServiceInstrument(Object):
    """A fixed kinematic service instrument retaining its authored interaction geometry."""

    name = "return_to_service_instrument"
    tags = ["object", "service", "instrument"]
    object_type = ObjectType.RIGID
    authoring_metadata = AuthoringMetadata(
        provides=("service_instrument",),
        constraints=(
            "Kinematic instrument roots cannot be picked up; instrument buttons are separate Pressable assets.",
        ),
        reset_semantics="Restores the instrument root pose; readings are maintained by the service task runtime.",
    )

    def __init__(
        self,
        component: InstrumentName = "battery_tester",
        asset_root: str | Path | None = None,
        instance_name: str | None = None,
        initial_pose: Pose | None = None,
    ):
        assert component in get_args(InstrumentName), f"Unknown service instrument: {component!r}"
        prepared = prepare_assets(asset_root)
        super().__init__(
            name=instance_name or self.name,
            tags=list(self.tags),
            object_type=self.object_type,
            usd_path=str(prepared.asset_usd(component, "kinematic")),
            initial_pose=initial_pose,
        )
        self.component = component


@register_asset
class ServiceButton(Object, Pressable):
    """A spring-return service button exposing Arena's Pressable interface."""

    name = "return_to_service_button"
    tags = ["object", "service", "button"]
    object_type = ObjectType.ARTICULATION
    authoring_metadata = AuthoringMetadata(
        parameters={"pressedness_threshold": ParameterMetadata(units="fraction", minimum=0.0, maximum=1.0)},
        constraints=(
            "The press joint moves from zero toward negative Z; Arena's joint utilities account for this polarity.",
            (
                "Pressable detects physical depression; instrument results and release interlocks belong to the service"
                " task."
            ),
        ),
        reset_semantics="Restores press joint position and velocity to zero; spring force returns an unheld cap.",
    )

    def __init__(
        self,
        asset_root: str | Path | None = None,
        instance_name: str | None = None,
        initial_pose: Pose | None = None,
        pressedness_threshold: float = 2.0 / 3.0,
    ):
        assert 0.0 <= pressedness_threshold <= 1.0, "Pressedness threshold must lie in [0, 1]"
        prepared = prepare_assets(asset_root)
        super().__init__(
            name=instance_name or self.name,
            tags=list(self.tags),
            object_type=self.object_type,
            usd_path=str(prepared.button_usd()),
            initial_pose=initial_pose,
            pressable_joint_name="press",
            pressedness_threshold=pressedness_threshold,
        )
        self.object_cfg.actuators = {
            "spring": ImplicitActuatorCfg(
                joint_names_expr=["press"], stiffness=180.0, damping=1.5, joint_effort_limit=8.0
            )
        }
        self.object_cfg.init_state.joint_pos = {"press": 0.0}
        self.object_cfg.init_state.joint_vel = {"press": 0.0}


@register_asset
class ServiceCase(Object, Openable):
    """An anchored service case exposing its selected lid or latch joint as Openable."""

    name = "return_to_service_case"
    tags = ["object", "service", "container"]
    object_type = ObjectType.ARTICULATION
    authoring_metadata = AuthoringMetadata(
        parameters={"openable_threshold": ParameterMetadata(units="fraction", minimum=0.0, maximum=1.0)},
        constraints=(
            "The hinge and latch are independent passive joints; Openable describes the selected joint only.",
            "Use for_joint('hinge') and for_joint('latch') as task views of one case in Python composites.",
            "Physical packing, lid clearance, and latch interlocks are checked by the service task.",
        ),
        reset_semantics="Restores hinge=-1.8 rad, latch=1.4 rad, and zero joint velocities.",
    )

    def __init__(
        self,
        asset_root: str | Path | None = None,
        instance_name: str | None = None,
        initial_pose: Pose | None = None,
        openable_joint_name: CaseJoint = "hinge",
        openable_threshold: float = 0.5,
    ):
        assert openable_joint_name in ("hinge", "latch"), "Case joint must be hinge or latch"
        assert 0.0 <= openable_threshold <= 1.0, "Openness threshold must lie in [0, 1]"
        prepared = prepare_assets(asset_root)
        super().__init__(
            name=instance_name or self.name,
            tags=list(self.tags),
            object_type=self.object_type,
            usd_path=str(prepared.case_usd()),
            initial_pose=initial_pose,
            openable_joint_name=openable_joint_name,
            openable_threshold=openable_threshold,
        )
        self.object_cfg.actuators = {
            "passive": ImplicitActuatorCfg(
                joint_names_expr=["hinge", "latch"], stiffness=0.0, damping=0.08, joint_effort_limit=10.0
            )
        }
        self.object_cfg.init_state.joint_pos = {"hinge": -1.8, "latch": 1.4}
        self.object_cfg.init_state.joint_vel = {"hinge": 0.0, "latch": 0.0}

    def for_joint(self, joint: CaseJoint) -> ServiceCase:
        """Return a task-only affordance view of one joint, sharing this case's scene entity.

        Args:
            joint: ``hinge`` or ``latch`` to use with an existing Openable task.

        Returns:
            A shallow view with the same scene key and asset config. Add only the original case
            to the scene; the view introduces no additional articulation or reset event.
        """
        assert joint in ("hinge", "latch"), "Case joint must be hinge or latch"
        view = copy(self)
        view.openable_joint_name = joint
        return view


@register_asset
class ServiceLighting(Object):
    """The authored studio dome with existing Arena intensity and color variations."""

    name = "return_to_service_lighting"
    tags = ["object", "light", "service"]
    object_type = ObjectType.BASE
    authoring_metadata = AuthoringMetadata(
        provides=("lighting",),
        constraints=(
            (
                "Intensity and color edits add a cached USD opinion over the Blender source; source geometry remains"
                " unchanged."
            ),
        ),
        reset_semantics="Lighting variations apply once at build time and persist across resets.",
    )

    def __init__(
        self,
        asset_root: str | Path | None = None,
        instance_name: str | None = None,
        initial_pose: Pose | None = None,
        prim_path: str | None = None,
    ):
        self._prepared = prepare_assets(asset_root)
        self._intensity: float | None = None
        self._color: tuple[float, float, float] | None = None
        super().__init__(
            name=instance_name or self.name,
            tags=list(self.tags),
            object_type=self.object_type,
            usd_path=str(self._prepared.asset_usd("lighting", "static")),
            initial_pose=initial_pose,
            prim_path=prim_path,
        )
        self.add_variation(LightIntensityVariation(self))
        self.add_variation(LightColorVariation(self))

    def set_intensity(self, intensity: float) -> None:
        """Set the dome's USD intensity through a cached build-time opinion."""
        assert math.isfinite(intensity) and intensity >= 0.0, "Light intensity must be finite and non-negative"
        self._intensity = float(intensity)
        self._update_lighting_opinion()

    def set_color(self, color: tuple[float, float, float]) -> None:
        """Set the dome's linear RGB color through a cached build-time opinion."""
        assert len(color) == 3 and all(
            math.isfinite(channel) and 0.0 <= channel <= 1.0 for channel in color
        ), "Light color must contain three finite channels in [0, 1]"
        self._color = tuple(float(channel) for channel in color)
        self._update_lighting_opinion()

    def _update_lighting_opinion(self) -> None:
        self.usd_path = str(self._prepared.lighting_usd(intensity=self._intensity, color=self._color))
        self.object_cfg.spawn.usd_path = self.usd_path
