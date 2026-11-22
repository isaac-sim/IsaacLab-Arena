# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Procedural USD for a five-layer block tower in which some dot-marked blocks are fused to the layer above.

Every block is its own rigid body. Layers alternate direction: odd layers run along X, even layers along Y. Each
layer holds three blocks named ``L<layer>_a``, ``L<layer>_b`` (middle) and ``L<layer>_c``; ``a`` sits on the
negative side of the layer's spread axis. The six outer blocks of layers 1 to 3 carry dot marks. Every dot-marked
block except the two free ones is held to the middle block of the layer above by a fixed joint ``fuse_<block>``
anchored at the crossing point, so pulling it drags that block, and the layers it carries, along. Layer 5 is a plain
cap: it keeps layer 4 pressed down, so sliding a free layer-3 block out does not drag layer-4 blocks off the tower.

Asset frame T: Z up, origin at the bottom center of the tower.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

NUM_LAYERS = 5
BLOCK_SLOTS = ("a", "b", "c")
CANDIDATE_BLOCK_NAMES = ("L1_a", "L1_c", "L2_a", "L2_c", "L3_a", "L3_c")
"""Dot-marked blocks; each is fused to the middle block of the layer above unless it is in the free pair."""

BLOCK_NAMES = tuple(f"L{layer}_{slot}" for layer in range(1, NUM_LAYERS + 1) for slot in BLOCK_SLOTS)

ROOT_PRIM_NAME = "FusedBlockTower"


@dataclass(frozen=True)
class FusedBlockTowerGeometry:
    """Dimensions (m), mass (kg) and contact material of the procedural tower."""

    block_length_m: float = 0.105
    block_width_m: float = 0.034
    block_height_m: float = 0.020
    block_gap_m: float = 0.0015
    """Clearance between neighbouring blocks of a layer."""

    block_mass_kg: float = 0.045
    static_friction: float = 0.6
    dynamic_friction: float = 0.4

    dot_radius_m: float = 0.006

    @property
    def block_pitch_m(self) -> float:
        """Distance between the centers of neighbouring blocks in a layer."""
        return self.block_width_m + self.block_gap_m


DEFAULT_GEOMETRY = FusedBlockTowerGeometry()

_WOOD_BY_LAYER_PARITY = ((0.74, 0.55, 0.34), (0.86, 0.70, 0.48))
"""Block colors for even and odd layers, so the layers read apart."""
_DOT = (0.55, 0.06, 0.05)
_USD_FORMAT_VERSION = 3
"""Part of the cache key; bump it when the written USD changes for the same free pair and geometry."""


def pair_key(free_pair: tuple[str, str] | list[str]) -> tuple[str, str]:
    """Return a free pair as a sorted tuple after checking it names two dot-marked blocks in different layers."""
    assert len(free_pair) == 2, f"A free pair names exactly two blocks, got {list(free_pair)}"
    first, second = sorted(free_pair)
    for block_name in (first, second):
        assert block_name in CANDIDATE_BLOCK_NAMES, f"'{block_name}' is not one of {CANDIDATE_BLOCK_NAMES}"
    assert first[:2] != second[:2], f"The free blocks must be in different layers, got {first} and {second}"
    return first, second


ALL_FREE_PAIRS: tuple[tuple[str, str], ...] = tuple(
    pair for pair in itertools.combinations(CANDIDATE_BLOCK_NAMES, 2) if pair[0][:2] != pair[1][:2]
)
"""The 12 valid free pairs: two dot-marked blocks in different layers."""


def fuse_target_name(candidate_name: str) -> str:
    """Return the middle block of the layer above a dot-marked block, which a fuse joint holds it to."""
    layer = int(candidate_name[1])
    return f"L{layer + 1}_b"


def block_pose_in_tower(block_name: str, geometry: FusedBlockTowerGeometry = DEFAULT_GEOMETRY) -> tuple:
    """Return (t_T_B, yaw_deg) of a block's center B in the tower frame T."""
    layer = int(block_name[1])
    slot_offset = (BLOCK_SLOTS.index(block_name[3]) - 1) * geometry.block_pitch_m
    z = (layer - 0.5) * geometry.block_height_m
    if layer % 2 == 1:
        # Odd layers run along X and spread along Y.
        return (0.0, slot_offset, z), 0.0
    # Even layers run along Y and spread along X.
    return (slot_offset, 0.0, z), 90.0


def _yaw_quat(yaw_deg: float) -> Gf.Quatf:
    return Gf.Quatf(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw_deg).GetQuat())


def _define_physics_material(stage: Usd.Stage, geometry: FusedBlockTowerGeometry) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, f"/{ROOT_PRIM_NAME}/Looks/wood_physics")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(geometry.static_friction)
    physics_material.CreateDynamicFrictionAttr(geometry.dynamic_friction)
    physics_material.CreateRestitutionAttr(0.0)
    return material


def _add_dot(stage: Usd.Stage, path: str, center_B: tuple[float, float, float], geometry: FusedBlockTowerGeometry):
    """Add a visual-only dot on a block's end face, at a center in the block frame B."""
    dot = UsdGeom.Cylinder.Define(stage, path)
    dot.CreateRadiusAttr(geometry.dot_radius_m)
    dot.CreateHeightAttr(0.001)
    dot.CreateAxisAttr("X")
    UsdGeom.Xformable(dot).AddTranslateOp().Set(Gf.Vec3d(*center_B))
    dot.CreateDisplayColorAttr([Gf.Vec3f(*_DOT)])


def _define_block(
    stage: Usd.Stage,
    block_name: str,
    geometry: FusedBlockTowerGeometry,
    material: UsdShade.Material,
) -> None:
    """Define one block as a rigid body with a box collider, plus end-face dots on dot-marked blocks."""
    g = geometry
    t_T_B, yaw_deg = block_pose_in_tower(block_name, g)
    block_path = f"/{ROOT_PRIM_NAME}/{block_name}"
    block = UsdGeom.Xform.Define(stage, block_path)
    block.AddTranslateOp().Set(Gf.Vec3d(*t_T_B))
    block.AddOrientOp().Set(_yaw_quat(yaw_deg))
    UsdPhysics.RigidBodyAPI.Apply(block.GetPrim())
    UsdPhysics.MassAPI.Apply(block.GetPrim()).CreateMassAttr(g.block_mass_kg)

    box = UsdGeom.Cube.Define(stage, f"{block_path}/box")
    box.CreateSizeAttr(1.0)
    UsdGeom.Xformable(box).AddScaleOp().Set(Gf.Vec3f(g.block_length_m, g.block_width_m, g.block_height_m))
    box.CreateDisplayColorAttr([Gf.Vec3f(*_WOOD_BY_LAYER_PARITY[int(block_name[1]) % 2])])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    binding = UsdShade.MaterialBindingAPI.Apply(box.GetPrim())
    binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

    if block_name in CANDIDATE_BLOCK_NAMES:
        half_length = g.block_length_m / 2
        _add_dot(stage, f"{block_path}/dot_front", (-half_length - 0.0004, 0.0, 0.0), g)
        _add_dot(stage, f"{block_path}/dot_back", (half_length + 0.0004, 0.0, 0.0), g)


def _define_fuse_joint(stage: Usd.Stage, candidate_name: str, geometry: FusedBlockTowerGeometry) -> None:
    """Hold a dot-marked block to the middle block above it with a fixed joint at their crossing point."""
    target_name = fuse_target_name(candidate_name)
    t_T_C, yaw_C_deg = block_pose_in_tower(candidate_name, geometry)
    t_T_U, yaw_U_deg = block_pose_in_tower(target_name, geometry)
    # The crossing point X lies on the candidate's top face, below the middle block above it.
    t_T_X = Gf.Vec3d(t_T_C[0], t_T_C[1], t_T_C[2] + geometry.block_height_m / 2)
    q_T_C = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw_C_deg)
    q_T_U = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw_U_deg)
    t_U_X = q_T_U.GetInverse().TransformDir(t_T_X - Gf.Vec3d(*t_T_U))
    t_C_X = q_T_C.GetInverse().TransformDir(t_T_X - Gf.Vec3d(*t_T_C))
    joint = UsdPhysics.FixedJoint.Define(stage, f"/{ROOT_PRIM_NAME}/fuse_{candidate_name}")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/{ROOT_PRIM_NAME}/{target_name}")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/{ROOT_PRIM_NAME}/{candidate_name}")])
    # Both joint frames are aligned with the tower frame T.
    joint.CreateLocalPos0Attr(Gf.Vec3f(*t_U_X))
    joint.CreateLocalRot0Attr(Gf.Quatf(q_T_U.GetInverse().GetQuat()))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*t_C_X))
    joint.CreateLocalRot1Attr(Gf.Quatf(q_T_C.GetInverse().GetQuat()))


def write_fused_block_tower_usd(
    path: Path,
    free_pair: tuple[str, str],
    geometry: FusedBlockTowerGeometry = DEFAULT_GEOMETRY,
) -> None:
    """Write the tower to a USD file, fusing every dot-marked block except the free pair.

    Args:
        path: Output USD path; overwritten if it exists.
        free_pair: The two dot-marked blocks without a fuse joint.
        geometry: Dimensions, mass and contact material of the blocks.
    """
    free_pair = pair_key(free_pair)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{ROOT_PRIM_NAME}").GetPrim()
    stage.SetDefaultPrim(root)
    material = _define_physics_material(stage, geometry)
    for block_name in BLOCK_NAMES:
        _define_block(stage, block_name, geometry, material)
    for candidate_name in CANDIDATE_BLOCK_NAMES:
        if candidate_name not in free_pair:
            _define_fuse_joint(stage, candidate_name, geometry)
    stage.GetRootLayer().Save()


def get_fused_block_tower_usd_path(
    free_pair: tuple[str, str] | list[str], geometry: FusedBlockTowerGeometry = DEFAULT_GEOMETRY
) -> str:
    """Return a cached USD path for the tower with the given free pair, writing the file on first use."""
    free_pair = pair_key(free_pair)
    cache_dir = Path(os.environ.get("ISAACLAB_ARENA_PROCEDURAL_CACHE", Path(tempfile.gettempdir()) / "isaaclab_arena"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(repr((_USD_FORMAT_VERSION, free_pair, geometry)).encode()).hexdigest()[:16]
    path = cache_dir / f"fused_block_tower_{digest}.usda"
    if not path.exists():
        # Write to a temporary file first so parallel processes never read a partial file.
        tmp_path = path.with_suffix(f".{os.getpid()}.usda")
        write_fused_block_tower_usd(tmp_path, free_pair, geometry)
        os.replace(tmp_path, path)
    return str(path)
