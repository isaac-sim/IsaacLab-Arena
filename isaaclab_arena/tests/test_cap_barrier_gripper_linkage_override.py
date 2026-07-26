# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab_arena.integrations.cap_barrier.franka_env import (
    FrankaSimulationAdapter,
)
from isaaclab_arena.integrations.cap_barrier.gripper_linkage_override import (
    _ALL_ORIGINAL_COLLISION_SUBPATHS,
    _EXPECTED_SOURCE_URI_SUFFIX,
    _LINKAGE_COLLISION_SUBPATHS,
    _MAX_PROXY_PADDING_M,
    _PROXY_BOX_SPECS,
    _PROXY_CONTACT_OFFSET_M,
    _RETAINED_COLLISION_SUBPATHS,
    _author_proxy_box,
    GripperLinkageCollisionOverride,
    apply_grocery_gripper_linkage_override,
    validate_live_grocery_gripper_collision_contract,
)


@dataclass(frozen=True)
class _PinnedSourceGeometry:
    lower_m: tuple[float, float, float]
    upper_m: tuple[float, float, float]
    translation_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]


# Structural values read from the pinned S3 robot asset
# (sha256 c8d72259834e2e5290754f8580b37efbc0dec079ac6a98b27b167efe6461eb2).
# The non-identity transforms are load-bearing: BBoxCache.ComputeLocalBound()
# applies each source prim's transform once and expresses the result in the
# owning rigid-body frame.
_PINNED_SOURCE_GEOMETRY = {
    _RETAINED_COLLISION_SUBPATHS[0]: _PinnedSourceGeometry(
        lower_m=(
            -0.042443349957466125,
            -0.003356433939188719,
            -0.03750000149011612,
        ),
        upper_m=(
            0.042443353682756424,
            0.09000010788440704,
            0.03750000149011612,
        ),
        translation_m=(
            -0.00024200000000064448,
            -5.104250355714157e-14,
            9.865145355429384e-15,
        ),
        orientation_wxyz=(
            -0.7071067811865475,
            4.3843338211951236e-16,
            -4.3843338211951216e-16,
            0.7071067811865476,
        ),
    ),
    _RETAINED_COLLISION_SUBPATHS[1]: _PinnedSourceGeometry(
        lower_m=(
            -0.07480880618095398,
            0.09144018590450287,
            -0.013500000350177288,
        ),
        upper_m=(
            -0.04490762576460838,
            0.11942079663276672,
            0.013500000350177288,
        ),
        translation_m=(
            0.0006797541058696664,
            -0.0010508959287422717,
            1.278421450231001e-10,
        ),
        orientation_wxyz=(
            0.7071069290788005,
            1.5708081984147143e-16,
            7.650710870356492e-16,
            -0.7071066332942779,
        ),
    ),
    _RETAINED_COLLISION_SUBPATHS[2]: _PinnedSourceGeometry(
        lower_m=(
            -0.06270762532949448,
            0.11041999608278275,
            -0.010999999940395355,
        ),
        upper_m=(
            -0.043548423796892166,
            0.14842018485069275,
            0.010999999940395355,
        ),
        translation_m=(
            0.0006797541058690697,
            -0.001050895928744506,
            2.0104105262867812e-14,
        ),
        orientation_wxyz=(
            0.7071069290787926,
            1.767837047204845e-16,
            7.688530711582171e-16,
            -0.7071066332942857,
        ),
    ),
    _RETAINED_COLLISION_SUBPATHS[3]: _PinnedSourceGeometry(
        lower_m=(
            -0.07480880618095398,
            0.09144018590450287,
            -0.013500000350177288,
        ),
        upper_m=(
            -0.04490762576460838,
            0.11942079663276672,
            0.013500000350177288,
        ),
        translation_m=(
            0.0006797541053413883,
            0.0010508959278896506,
            -9.774969571848016e-15,
        ),
        orientation_wxyz=(
            -1.0556339700730048e-15,
            0.7071069290760581,
            0.7071066332970178,
            1.1180409949514395e-16,
        ),
    ),
    _RETAINED_COLLISION_SUBPATHS[4]: _PinnedSourceGeometry(
        lower_m=(
            -0.06270762532949448,
            0.11041999608278275,
            -0.010999999940395355,
        ),
        upper_m=(
            -0.043548423796892166,
            0.14842018485069275,
            0.010999999940395355,
        ),
        translation_m=(
            0.0006797541052276689,
            0.0010508984242193833,
            -1.3318820883567101e-14,
        ),
        orientation_wxyz=(
            -1.0648313981991159e-15,
            0.707106929076103,
            0.7071066332969895,
            1.1448899282735203e-16,
        ),
    ),
}


def _source_path(tmp_path: Path) -> Path:
    return tmp_path / _EXPECTED_SOURCE_URI_SUFFIX.lstrip("/")


def _write_source(
    path: Path,
    *,
    default_prim_path: str = "/panda",
    root_scale: tuple[float, float, float] | None = None,
    missing_target: str | None = None,
    target_without_api: str | None = None,
    disabled_target: str | None = None,
    missing_retained: str | None = None,
    disabled_retained: str | None = None,
    transform_drift_retained: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    robot = UsdGeom.Xform.Define(stage, "/panda").GetPrim()
    if root_scale is not None:
        UsdGeom.Xformable(robot).AddScaleOp().Set(Gf.Vec3d(*root_scale))
    stage.SetDefaultPrim(
        robot if default_prim_path == "/panda" else stage.DefinePrim(default_prim_path)
    )
    for subpath in _LINKAGE_COLLISION_SUBPATHS:
        if subpath == missing_target:
            continue
        prim = stage.DefinePrim(f"/panda/{subpath}")
        if subpath != target_without_api:
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(
                subpath != disabled_target
            )
    for subpath in _RETAINED_COLLISION_SUBPATHS:
        if subpath == missing_retained:
            continue
        geometry = _PINNED_SOURCE_GEOMETRY[subpath]
        mesh = UsdGeom.Mesh.Define(stage, f"/panda/{subpath}")
        lower = geometry.lower_m
        upper = geometry.upper_m
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(x, y, z)
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ]
        )
        translation = geometry.translation_m
        if subpath == transform_drift_retained:
            translation = (
                translation[0] + 0.001,
                translation[1],
                translation[2],
            )
        xformable = UsdGeom.Xformable(mesh)
        xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*translation)
        )
        orientation = geometry.orientation_wxyz
        xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(
                orientation[0],
                Gf.Vec3d(*orientation[1:]),
            )
        )
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(
            subpath != disabled_retained
        )
    stage.GetRootLayer().Save()


def _aligned_bounds(
    box: Gf.BBox3d,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
]:
    aligned = box.ComputeAlignedBox()
    return (
        tuple(float(value) for value in aligned.GetMin()),
        tuple(float(value) for value in aligned.GetMax()),
    )


def _scene_config(source_usd_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        robot=SimpleNamespace(
            spawn=SimpleNamespace(
                usd_path=source_usd_path,
                unrelated_setting="preserved",
            )
        )
    )


def test_pinned_nonidentity_source_bounds_are_owning_body_local(
    tmp_path: Path,
) -> None:
    source_path = _source_path(tmp_path)
    _write_source(source_path)
    stage = Usd.Stage.Open(str(source_path))
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        ["default", "render", "proxy"],
        useExtentsHint=False,
    )

    for spec in _PROXY_BOX_SPECS:
        source = stage.GetPrimAtPath(f"/panda/{spec.source_collision_subpath}")
        body = stage.GetPrimAtPath(f"/panda/{spec.body_subpath}")
        untransformed_lower, untransformed_upper = _aligned_bounds(
            cache.ComputeUntransformedBound(source)
        )
        local_lower, local_upper = _aligned_bounds(cache.ComputeLocalBound(source))
        relative_lower, relative_upper = _aligned_bounds(
            cache.ComputeRelativeBound(source, body)
        )

        assert (local_lower, local_upper) != (
            untransformed_lower,
            untransformed_upper,
        )
        assert local_lower == pytest.approx(relative_lower, abs=1e-12)
        assert local_upper == pytest.approx(relative_upper, abs=1e-12)
        assert all(
            proxy <= source
            for proxy, source in zip(
                spec.lower_m,
                relative_lower,
                strict=True,
            )
        )
        assert all(
            source <= proxy
            for source, proxy in zip(
                relative_upper,
                spec.upper_m,
                strict=True,
            )
        )
        maximum_padding = max(
            *(
                source - proxy
                for source, proxy in zip(
                    relative_lower,
                    spec.lower_m,
                    strict=True,
                )
            ),
            *(
                proxy - source
                for source, proxy in zip(
                    relative_upper,
                    spec.upper_m,
                    strict=True,
                )
            ),
        )
        assert math.isfinite(maximum_padding)
        assert maximum_padding <= _MAX_PROXY_PADDING_M


def test_override_is_instance_local_verified_and_owned_until_close(
    tmp_path: Path,
) -> None:
    source_path = _source_path(tmp_path)
    _write_source(source_path)
    shared_robot = _scene_config(str(source_path)).robot
    grocery_scene = SimpleNamespace(robot=shared_robot)
    stock_scene = SimpleNamespace(robot=shared_robot)

    override = apply_grocery_gripper_linkage_override(grocery_scene)
    override_path = Path(override.override_usd_path)

    assert grocery_scene.robot is not shared_robot
    assert grocery_scene.robot.spawn is not shared_robot.spawn
    assert grocery_scene.robot.spawn.unrelated_setting == "preserved"
    assert grocery_scene.robot.spawn.usd_path == str(override_path)
    assert stock_scene.robot.spawn.usd_path == str(source_path)
    assert override_path.is_file()

    source_stage = Usd.Stage.Open(str(source_path))
    override_stage = Usd.Stage.Open(str(override_path))
    assert str(override_stage.GetDefaultPrim().GetPath()) == "/panda"
    reference_spec = override_stage.GetRootLayer().GetPrimAtPath("/panda")
    assert [
        reference.assetPath
        for reference in reference_spec.GetInfo("references").GetAppliedItems()
    ] == [str(source_path)]
    assert len(_ALL_ORIGINAL_COLLISION_SUBPATHS) == 11
    for subpath in _ALL_ORIGINAL_COLLISION_SUBPATHS:
        prim_path = f"/panda/{subpath}"
        assert (
            UsdPhysics.CollisionAPI(source_stage.GetPrimAtPath(prim_path))
            .GetCollisionEnabledAttr()
            .Get()
        )
        assert (
            UsdPhysics.CollisionAPI(override_stage.GetPrimAtPath(prim_path))
            .GetCollisionEnabledAttr()
            .Get()
            is False
        )
    assert len(_PROXY_BOX_SPECS) == 5
    for spec in _PROXY_BOX_SPECS:
        proxy_path = f"/panda/{spec.proxy_subpath}"
        proxy = override_stage.GetPrimAtPath(proxy_path)
        assert proxy.GetTypeName() == "Cube"
        assert UsdPhysics.CollisionAPI(proxy).GetCollisionEnabledAttr().Get()
        assert not proxy.HasAPI(UsdPhysics.MeshCollisionAPI)

    override.close()
    assert not override_path.exists()
    override.close()


def test_adapter_keeps_override_alive_through_environment_close(
    tmp_path: Path,
) -> None:
    source_path = _source_path(tmp_path)
    _write_source(source_path)
    override = GripperLinkageCollisionOverride(str(source_path))
    override_path = Path(override.override_usd_path)
    close_observations: list[bool] = []

    class _Environment:
        def close(self) -> None:
            close_observations.append(override_path.is_file())

    adapter = object.__new__(FrankaSimulationAdapter)
    adapter._environment = _Environment()
    adapter._owned_resources = (override,)

    adapter.close()

    assert close_observations == [True]
    assert not override_path.exists()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"default_prim_path": "/robot"}, "default prim must be /panda"),
        (
            {"missing_target": _LINKAGE_COLLISION_SUBPATHS[0]},
            "missing=.*left_outer_knuckle",
        ),
        (
            {"target_without_api": _LINKAGE_COLLISION_SUBPATHS[1]},
            "missing_PhysicsCollisionAPI=.*left_outer_finger",
        ),
        (
            {"disabled_target": _LINKAGE_COLLISION_SUBPATHS[2]},
            "wrong_collisionEnabled=.*left_inner_knuckle",
        ),
        (
            {"missing_retained": _RETAINED_COLLISION_SUBPATHS[0]},
            "missing=.*base_link",
        ),
        (
            {"disabled_retained": _RETAINED_COLLISION_SUBPATHS[1]},
            "wrong_collisionEnabled=.*left_inner_finger",
        ),
        (
            {"transform_drift_retained": _RETAINED_COLLISION_SUBPATHS[2]},
            "outside proxy",
        ),
        (
            {"root_scale": (1.01, 1.0, 1.0)},
            "metric transform drifted",
        ),
    ],
)
def test_override_rejects_source_asset_contract_drift(
    tmp_path: Path,
    mutation: dict[str, object],
    match: str,
) -> None:
    source_path = _source_path(tmp_path)
    _write_source(source_path, **mutation)

    with pytest.raises(RuntimeError, match=match):
        GripperLinkageCollisionOverride(str(source_path))


def test_override_rejects_unrecognized_robot_asset(tmp_path: Path) -> None:
    source_path = tmp_path / "different_robot.usd"
    _write_source(source_path)

    with pytest.raises(RuntimeError, match="requires the pinned DROID robot USD"):
        GripperLinkageCollisionOverride(str(source_path))


def test_live_contract_requires_disabled_originals_and_analytic_proxies(
    tmp_path: Path,
) -> None:
    stage_path = tmp_path / "live.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    robot_prim_path = "/World/envs/env_0/Robot"
    stage.DefinePrim(robot_prim_path)
    for subpath in _ALL_ORIGINAL_COLLISION_SUBPATHS:
        prim = stage.DefinePrim(f"{robot_prim_path}/{subpath}")
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(False)
    for spec in _PROXY_BOX_SPECS:
        _author_proxy_box(stage, prim_root=robot_prim_path, spec=spec)

    assert validate_live_grocery_gripper_collision_contract(
        stage,
        robot_prim_path=robot_prim_path,
    ) == (11, 5)

    original = stage.GetPrimAtPath(
        f"{robot_prim_path}/{_RETAINED_COLLISION_SUBPATHS[-1]}"
    )
    UsdPhysics.CollisionAPI(original).GetCollisionEnabledAttr().Set(True)
    with pytest.raises(RuntimeError, match="wrong_collisionEnabled=.*fingertips"):
        validate_live_grocery_gripper_collision_contract(
            stage,
            robot_prim_path=robot_prim_path,
        )

    UsdPhysics.CollisionAPI(original).GetCollisionEnabledAttr().Set(False)
    proxy = stage.GetPrimAtPath(
        f"{robot_prim_path}/{_PROXY_BOX_SPECS[-1].proxy_subpath}"
    )
    proxy.GetAttribute("physxCollision:contactOffset").Set(0.01)
    with pytest.raises(RuntimeError, match="offsets"):
        validate_live_grocery_gripper_collision_contract(
            stage,
            robot_prim_path=robot_prim_path,
        )

    proxy.GetAttribute("physxCollision:contactOffset").Set(_PROXY_CONTACT_OFFSET_M)
    extra = UsdGeom.Cube.Define(
        stage,
        f"{robot_prim_path}/Gripper/Robotiq_2F_85/unexpected_collision",
    ).GetPrim()
    UsdPhysics.CollisionAPI.Apply(extra).CreateCollisionEnabledAttr(True)
    with pytest.raises(RuntimeError, match="enabled gripper collision set mismatch"):
        validate_live_grocery_gripper_collision_contract(
            stage,
            robot_prim_path=robot_prim_path,
        )


def test_live_contract_rejects_inherited_robot_scale(tmp_path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(tmp_path / "live_scaled.usda"))
    robot_prim_path = "/World/envs/env_0/Robot"
    robot = UsdGeom.Xform.Define(stage, robot_prim_path)
    UsdGeom.Xformable(robot).AddScaleOp().Set(Gf.Vec3d(1.0, 1.01, 1.0))
    for subpath in _ALL_ORIGINAL_COLLISION_SUBPATHS:
        prim = stage.DefinePrim(f"{robot_prim_path}/{subpath}")
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(False)
    for spec in _PROXY_BOX_SPECS:
        _author_proxy_box(stage, prim_root=robot_prim_path, spec=spec)

    with pytest.raises(RuntimeError, match="metric transform drifted"):
        validate_live_grocery_gripper_collision_contract(
            stage,
            robot_prim_path=robot_prim_path,
        )
