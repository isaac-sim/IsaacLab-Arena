# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared physical-shape containment in fixed and measured region frames."""

from __future__ import annotations

import math
import torch
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from .collision_geometry import read_collision_primitives


@dataclass(frozen=True)
class FixedRegion:
    """Describe a box fixed in the local environment frame."""

    center_xyz: tuple[float, float, float]
    half_extents_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class BoxRegion:
    """Describe an immutable box in a live parent body's frame."""

    parent_name: str
    """Configured scene object whose measured pose supplies the parent frame P."""

    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    """Strictly ordered lower and upper corners in region frame R, in meters."""

    position_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation t_P_R from the parent origin to the region origin."""

    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Unit quaternion q_P_R mapping region axes into the parent frame."""

    floor_allowance_m: float = 0.0
    """Permitted penetration at the lower Z face only; this does not establish support."""

    def __post_init__(self) -> None:
        assert isinstance(self.parent_name, str) and self.parent_name, "A region requires a parent scene name"
        bounds = tuple(tuple(float(value) for value in corner) for corner in self.bounds)
        position = tuple(float(value) for value in self.position_xyz)
        rotation = tuple(float(value) for value in self.rotation_xyzw)
        assert len(bounds) == 2 and all(len(corner) == 3 for corner in bounds), "Region bounds must have shape (2, 3)"
        assert len(position) == 3 and len(rotation) == 4, "A region pose requires XYZ and XYZW"
        assert all(math.isfinite(value) for corner in bounds for value in corner), "Region bounds must be finite"
        assert all(bounds[0][axis] < bounds[1][axis] for axis in range(3)), "Region bounds must have positive volume"
        assert all(math.isfinite(value) for value in (*position, *rotation)), "Region pose must be finite"
        assert abs(math.sqrt(sum(value * value for value in rotation)) - 1.0) <= 1e-4, "Region quaternion must be unit"
        assert math.isfinite(self.floor_allowance_m) and self.floor_allowance_m >= 0.0, "Invalid floor allowance"
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "position_xyz", position)
        object.__setattr__(self, "rotation_xyzw", rotation)


@dataclass(frozen=True)
class ContainmentMeasurement:
    """Measure all six support planes; invalid live frames cannot certify containment."""

    valid_pose: torch.Tensor
    """Validity of both independently checked incoming frames, shape (...,)."""

    contained: torch.Tensor
    """Whether every configured primitive satisfies all six faces, shape (...,)."""

    T_R_O: torch.Tensor
    """Normalized object-to-region poses, shape (..., 7); NaN for invalid frames."""

    occupied_bounds_R: torch.Tensor
    """Primitive-union bounds in the region frame, shape (..., 2, 3); NaN for invalid poses."""

    raw_face_margins_m: torch.Tensor
    """Unadjusted margins in lower XYZ then upper XYZ order, shape (..., 6)."""

    effective_face_margins_m: torch.Tensor
    """Margins including floor-only allowance and one micrometer numerical tolerance."""


class RegionContainment:
    """Cache immutable configured geometry while measuring fresh object and region poses."""

    def __init__(
        self,
        scene_cfg,
        regions: Mapping[str, FixedRegion] | None = None,
        *,
        floor_allowance_m: float = 0.0,
        unit_scale_regions: tuple[str, ...] = (),
    ) -> None:
        """Bind configured component geometry and optional rigid-region scale contracts.

        Args:
            scene_cfg: Spawn configuration supplying physical component USDs.
            regions: Named fixed region frames and half extents in environment coordinates.
            floor_allowance_m: Lower-Z contact allowance for the named fixed-region adapter.
            unit_scale_regions: Configured region objects whose metadata requires unit spawn scale.
        """
        assert math.isfinite(floor_allowance_m) and floor_allowance_m >= 0.0
        self.scene_cfg = scene_cfg
        self.regions = {} if regions is None else regions
        self.floor_allowance_m = floor_allowance_m
        self._geometry = {}
        self._regions = {}
        self._bounds = {}
        self._tensor_type: tuple[torch.dtype, torch.device] | None = None
        self._region_configurations = {}
        for name in unit_scale_regions:
            configuration = self._configuration(name)
            assert configuration[1] == (1.0, 1.0, 1.0), "Region metadata requires unit spawn scale"
            self._region_configurations[name] = configuration

    def _configuration(self, name: str) -> tuple[str, tuple[float, ...]]:
        spawn = getattr(self.scene_cfg, name).spawn
        for field in ("variants", "collision_props", "prim_physics", "deformable_props"):
            assert not getattr(spawn, field, None), f"Containment cannot apply spawn.{field}"
        scale = (1.0, 1.0, 1.0) if spawn.scale is None else tuple(spawn.scale)
        assert len(scale) == 3 and all(math.isfinite(value) and value > 0 for value in scale), "Invalid spawn scale"
        identifier = str(spawn.usd_path)
        # USD resolvers own URI identifiers. Filesystem normalization would turn
        # https:// or omniverse:// identifiers into invalid local paths.
        if not urlsplit(identifier).scheme:
            identifier = str(Path(spawn.usd_path).expanduser().resolve())
        return identifier, scale

    def _bind_tensor(self, pose: torch.Tensor) -> None:
        assert isinstance(pose, torch.Tensor) and pose.ndim >= 1 and pose.shape[-1] == 7, "Expected (..., 7) pose"
        assert pose.dtype in (torch.float32, torch.float64), "Containment requires float32 or float64 poses"
        tensor_type = (pose.dtype, pose.device)
        if self._tensor_type is None:
            self._tensor_type = tensor_type
        assert tensor_type == self._tensor_type, "Construct a new containment checker for another dtype or device"

    @staticmethod
    def _validated_pose(pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm = torch.linalg.vector_norm(pose[..., 3:], dim=-1)
        valid = torch.isfinite(pose).all(dim=-1) & ((norm - 1.0).abs() <= 1e-4)
        identity = torch.zeros_like(pose)
        identity[..., 6] = 1.0
        safe = torch.where(valid[..., None], pose, identity)
        # Only accepted small quaternion drift is normalized, into owned temporaries.
        normalized = safe[..., 3:] / torch.linalg.vector_norm(safe[..., 3:], dim=-1, keepdim=True)
        return torch.cat((safe[..., :3], normalized), dim=-1), valid

    def measure_in_frame(
        self,
        name: str,
        T_F_O: torch.Tensor,
        T_F_R: torch.Tensor,
        bounds_R,
        *,
        floor_allowance_m: float = 0.0,
    ) -> ContainmentMeasurement:
        """Measure configured primitives against a box in an explicit live region frame.

        Args:
            name: Configured rigid component O to measure.
            T_F_O: Object-to-common-frame XYZ+XYZW poses, shape (..., 7).
            T_F_R: Region-to-common-frame poses on the same dtype/device; scalar or matching batch.
            bounds_R: Finite strictly ordered lower/upper corners, shape (2, 3), in region frame R.
            floor_allowance_m: Physical contact allowance on lower Z only; all faces retain 1 micrometer.

        Returns:
            Six-face diagnostics and containment for each broadcast pose. Each live frame must
            independently have a finite unit quaternion within 1e-4. Invalid entries return false
            validity/containment and NaN diagnostics. Scene geometry must remain immutable: changed
            spawn paths/scales/overrides require a new checker, as do external USD/stage mutations.
            Cached USDs are read once per component; no live pose or containment result is cached.
        """
        self._bind_tensor(T_F_O)
        self._bind_tensor(T_F_R)
        assert (
            T_F_O.shape[:-1] == T_F_R.shape[:-1] or T_F_O.ndim == 1 or T_F_R.ndim == 1
        ), "Frames must have identical batch dimensions or one scalar frame"
        assert math.isfinite(floor_allowance_m) and floor_allowance_m >= 0.0, "Invalid floor allowance"
        for region, original in self._region_configurations.items():
            assert self._configuration(region) == original, "Region configuration changed; construct a new checker"
        configuration = self._configuration(name)
        if name not in self._geometry:
            geometry = read_collision_primitives(
                configuration[0], scale=configuration[1], device=T_F_O.device, dtype=T_F_O.dtype
            )
            self._geometry[name] = (configuration, geometry)
        original, geometry = self._geometry[name]
        assert configuration == original, "Component configuration changed; construct a new checker"
        if isinstance(bounds_R, torch.Tensor):
            assert (bounds_R.dtype, bounds_R.device) == self._tensor_type, "Bounds must match pose dtype and device"
            values = bounds_R.detach().tolist()
        else:
            values = bounds_R
        assert len(values) == 2 and all(len(corner) == 3 for corner in values), "Region bounds must have shape (2, 3)"
        entries = []
        for corner in values:
            entries.extend(float(value) for value in corner)
        key = tuple(entries)
        assert all(math.isfinite(value) for value in key) and all(
            key[axis + 3] > key[axis] for axis in range(3)
        ), "Invalid region bounds"
        if key not in self._bounds:
            self._bounds[key] = T_F_O.new_tensor(key).reshape(2, 3)
        bounds = self._bounds[key]
        object_pose, object_valid = self._validated_pose(T_F_O)
        region_pose, region_valid = self._validated_pose(T_F_R)
        object_pose, region_pose = torch.broadcast_tensors(object_pose, region_pose)
        valid = object_valid & region_valid
        q_R_F = quat_conjugate(region_pose[..., 3:])
        T_R_O = torch.cat(
            (
                quat_apply(q_R_F, object_pose[..., :3] - region_pose[..., :3]),
                quat_mul(q_R_F, object_pose[..., 3:]),
            ),
            dim=-1,
        )
        occupied = geometry.bounds_in_frame(T_R_O)
        occupied = torch.where(valid[..., None, None], occupied, torch.full_like(occupied, torch.nan))
        raw = torch.cat((occupied[..., 0, :] - bounds[0], bounds[1] - occupied[..., 1, :]), dim=-1)
        effective = raw + 1e-6
        effective[..., 2] += floor_allowance_m
        contained = valid & (effective >= 0.0).all(dim=-1)
        T_R_O = torch.where(valid[..., None], T_R_O, torch.full_like(T_R_O, torch.nan))
        return ContainmentMeasurement(valid, contained, T_R_O, occupied, raw, effective)

    def contains(self, name: str, T_E_O: torch.Tensor, region_name: str) -> torch.Tensor:
        """Measure a component against a configured fixed environment-relative region.

        Args:
            name: Configured rigid component to measure.
            T_E_O: Component-to-environment XYZ+XYZW poses, shape (..., 7).
            region_name: Fixed region whose lower Z face supports resting components.

        Returns:
            Containment from the shared live-frame measurement kernel.
        """
        self._bind_tensor(T_E_O)
        region = self.regions[region_name]
        definition = (tuple(region.center_xyz), tuple(region.rotation_xyzw), tuple(region.half_extents_xyz))
        if region_name not in self._regions:
            T_E_R = T_E_O.new_tensor((*definition[0], *definition[1]))
            _, valid = self._validated_pose(T_E_R)
            assert bool(valid), "Invalid fixed region pose"
            half = T_E_O.new_tensor(definition[2])
            assert half.shape == (3,) and bool(torch.isfinite(half).all()) and bool((half > 0).all())
            self._regions[region_name] = (definition, T_E_R, torch.stack((-half, half)))
        original, T_E_R, bounds = self._regions[region_name]
        assert definition == original, "Fixed region definition changed; construct a new checker"
        return self.measure_in_frame(name, T_E_O, T_E_R, bounds, floor_allowance_m=self.floor_allowance_m).contained

    def measure_region(
        self, name: str, T_F_O: torch.Tensor, T_F_P: torch.Tensor, region: BoxRegion
    ) -> ContainmentMeasurement:
        """Measure a component against a box attached to a live parent body.

        Args:
            name: Configured component O to measure.
            T_F_O: Object-to-common-frame XYZ+XYZW poses, scalar or batched.
            T_F_P: Measured parent-to-common-frame poses on the same dtype and device.
            region: Region frame and bounds relative to the configured parent P.

        Returns:
            Containment and face margins in region frame R. Invalid parent poses fail closed
            before composition. Parent spawn scale must stay unit because bounds are in meters.
        """
        self._bind_tensor(T_F_O)
        self._bind_tensor(T_F_P)
        configuration = self._configuration(region.parent_name)
        assert configuration[1] == (1.0, 1.0, 1.0), "Region metadata requires unit spawn scale"
        if region.parent_name not in self._region_configurations:
            self._region_configurations[region.parent_name] = configuration
        parent, valid = self._validated_pose(T_F_P)
        local, local_valid = self._validated_pose(T_F_P.new_tensor((*region.position_xyz, *region.rotation_xyzw)))
        assert bool(local_valid), "Invalid region pose"
        local = local.expand_as(parent)
        T_F_R = torch.cat(
            (parent[..., :3] + quat_apply(parent[..., 3:], local[..., :3]), quat_mul(parent[..., 3:], local[..., 3:])),
            dim=-1,
        )
        T_F_R = torch.where(valid[..., None], T_F_R, torch.full_like(T_F_R, torch.nan))
        return self.measure_in_frame(name, T_F_O, T_F_R, region.bounds, floor_allowance_m=region.floor_allowance_m)
