# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pxr import Usd, UsdPhysics

from isaaclab_arena.integrations.cap_barrier.franka_env import (
    FrankaSimulationAdapter,
)
from isaaclab_arena.integrations.cap_barrier.gripper_linkage_override import (
    _EXPECTED_SOURCE_URI_SUFFIX,
    _LINKAGE_COLLISION_SUBPATHS,
    _RETAINED_COLLISION_SUBPATHS,
    GripperLinkageCollisionOverride,
    apply_grocery_gripper_linkage_override,
    validate_live_grocery_gripper_collision_contract,
)


def _source_path(tmp_path: Path) -> Path:
    return tmp_path / _EXPECTED_SOURCE_URI_SUFFIX.lstrip("/")


def _write_source(
    path: Path,
    *,
    default_prim_path: str = "/panda",
    missing_target: str | None = None,
    target_without_api: str | None = None,
    disabled_target: str | None = None,
    missing_retained: str | None = None,
    disabled_retained: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    robot = stage.DefinePrim("/panda")
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
        prim = stage.DefinePrim(f"/panda/{subpath}")
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(
            subpath != disabled_retained
        )
    stage.GetRootLayer().Save()


def _scene_config(source_usd_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        robot=SimpleNamespace(
            spawn=SimpleNamespace(
                usd_path=source_usd_path,
                unrelated_setting="preserved",
            )
        )
    )


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
    assert len(_LINKAGE_COLLISION_SUBPATHS) == 6
    for subpath in _LINKAGE_COLLISION_SUBPATHS:
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
    assert len(_RETAINED_COLLISION_SUBPATHS) == 5
    for subpath in _RETAINED_COLLISION_SUBPATHS:
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
        )

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
    ],
)
def test_override_rejects_source_asset_contract_drift(
    tmp_path: Path,
    mutation: dict[str, str],
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


def test_live_contract_requires_disabled_linkages_and_retained_colliders(
    tmp_path: Path,
) -> None:
    stage_path = tmp_path / "live.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    robot_prim_path = "/World/envs/env_0/Robot"
    stage.DefinePrim(robot_prim_path)
    for subpath in _LINKAGE_COLLISION_SUBPATHS:
        prim = stage.DefinePrim(f"{robot_prim_path}/{subpath}")
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(False)
    for subpath in _RETAINED_COLLISION_SUBPATHS:
        prim = stage.DefinePrim(f"{robot_prim_path}/{subpath}")
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)

    assert validate_live_grocery_gripper_collision_contract(
        stage,
        robot_prim_path=robot_prim_path,
    ) == (6, 5)

    retained = stage.GetPrimAtPath(
        f"{robot_prim_path}/{_RETAINED_COLLISION_SUBPATHS[-1]}"
    )
    UsdPhysics.CollisionAPI(retained).GetCollisionEnabledAttr().Set(False)
    with pytest.raises(RuntimeError, match="wrong_collisionEnabled=.*fingertips"):
        validate_live_grocery_gripper_collision_contract(
            stage,
            robot_prim_path=robot_prim_path,
        )
