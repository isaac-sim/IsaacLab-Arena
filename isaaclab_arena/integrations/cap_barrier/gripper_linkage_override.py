# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pre-parse collision correction for the CAP grocery DROID gripper."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

_EXPECTED_SOURCE_URI_SUFFIX = (
    "/Arena/assets/robot_library/droid/franka_robotiq_2f_85_flattened.usd"
)
_ROBOT_DEFAULT_PRIM_PATH = "/panda"
_LINKAGE_COLLISION_SUBPATHS = (
    "Gripper/Robotiq_2F_85/left_outer_knuckle/Defeatured_2F_85_PAD_OPEN_Finger1step_01",
    "Gripper/Robotiq_2F_85/left_outer_finger/Defeatured_2F_85_PAD_OPEN_finger2step_01",
    "Gripper/Robotiq_2F_85/left_inner_knuckle/Defeatured_2F_85_PAD_OPEN_finger3step_01",
    "Gripper/Robotiq_2F_85/right_outer_knuckle/Defeatured_2F_85_PAD_OPEN_Finger1step_01",
    "Gripper/Robotiq_2F_85/right_outer_finger/Defeatured_2F_85_PAD_OPEN_finger2step_01",
    "Gripper/Robotiq_2F_85/right_inner_knuckle/Defeatured_2F_85_PAD_OPEN_finger3step_01",
)


def _require_source_identity(source_usd_path: str) -> None:
    normalized_path = source_usd_path.replace("\\", "/")
    if not normalized_path.endswith(_EXPECTED_SOURCE_URI_SUFFIX):
        raise RuntimeError(
            "CAP grocery linkage override requires the pinned DROID robot USD; "
            f"got {source_usd_path!r}"
        )


def _require_default_prim(stage: Any, *, label: str) -> Any:
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        raise RuntimeError(f"{label} has no valid default prim")
    actual_path = str(default_prim.GetPath())
    if actual_path != _ROBOT_DEFAULT_PRIM_PATH:
        raise RuntimeError(
            f"{label} default prim must be {_ROBOT_DEFAULT_PRIM_PATH}, got {actual_path}"
        )
    return default_prim


def _require_linkage_collision_state(stage: Any, *, enabled: bool, label: str) -> None:
    from pxr import UsdPhysics

    missing: list[str] = []
    missing_api: list[str] = []
    wrong_state: list[str] = []
    for subpath in _LINKAGE_COLLISION_SUBPATHS:
        prim_path = f"{_ROBOT_DEFAULT_PRIM_PATH}/{subpath}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or not prim.IsDefined():
            missing.append(prim_path)
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            missing_api.append(prim_path)
            continue
        actual_enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if actual_enabled is not enabled:
            wrong_state.append(f"{prim_path}={actual_enabled!r}")
    if missing or missing_api or wrong_state:
        raise RuntimeError(
            f"{label} linkage collision contract mismatch: "
            f"missing={missing}, missing_PhysicsCollisionAPI={missing_api}, "
            f"wrong_collisionEnabled={wrong_state}"
        )


def _open_and_validate_source(source_usd_path: str) -> Any:
    from pxr import Usd

    _require_source_identity(source_usd_path)
    try:
        stage = Usd.Stage.Open(source_usd_path)
    except Exception as exc:
        raise RuntimeError(
            f"failed to open CAP grocery DROID robot USD {source_usd_path!r}"
        ) from exc
    if stage is None:
        raise RuntimeError(
            f"failed to open CAP grocery DROID robot USD {source_usd_path!r}"
        )
    _require_default_prim(stage, label="CAP grocery DROID source USD")
    _require_linkage_collision_state(
        stage,
        enabled=True,
        label="CAP grocery DROID source USD",
    )
    return stage


def _write_override_layer(source_usd_path: str, override_usd_path: Path) -> None:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.CreateNew(str(override_usd_path))
    if stage is None:
        raise RuntimeError(
            f"failed to create linkage override layer {override_usd_path}"
        )
    robot_prim = stage.DefinePrim(_ROBOT_DEFAULT_PRIM_PATH)
    robot_prim.GetReferences().AddReference(assetPath=source_usd_path)
    stage.SetDefaultPrim(robot_prim)
    for subpath in _LINKAGE_COLLISION_SUBPATHS:
        prim = stage.OverridePrim(f"{_ROBOT_DEFAULT_PRIM_PATH}/{subpath}")
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False).Set(False)
    stage.GetRootLayer().Save()


def _validate_override_layer(
    override_usd_path: Path,
    *,
    source_usd_path: str,
) -> None:
    from pxr import Usd

    try:
        stage = Usd.Stage.Open(str(override_usd_path))
    except Exception as exc:
        raise RuntimeError(
            f"failed to reopen linkage override layer {override_usd_path}"
        ) from exc
    if stage is None:
        raise RuntimeError(
            f"failed to reopen linkage override layer {override_usd_path}"
        )
    default_prim = _require_default_prim(stage, label="CAP grocery linkage override")
    default_prim_spec = stage.GetRootLayer().GetPrimAtPath(default_prim.GetPath())
    if default_prim_spec is None or not default_prim_spec.HasInfo("references"):
        raise RuntimeError(
            "CAP grocery linkage override root layer has no robot reference"
        )
    authored_references = tuple(
        default_prim_spec.GetInfo("references").GetAppliedItems()
    )
    if (
        len(authored_references) != 1
        or authored_references[0].assetPath != source_usd_path
    ):
        raise RuntimeError(
            "CAP grocery linkage override must contain exactly one reference to "
            f"{source_usd_path!r}; got {authored_references!r}"
        )
    _require_linkage_collision_state(
        stage,
        enabled=False,
        label="CAP grocery linkage override",
    )


class GripperLinkageCollisionOverride:
    """Own one verified temporary DROID gripper linkage override layer."""

    def __init__(self, source_usd_path: str):
        if not isinstance(source_usd_path, str) or not source_usd_path:
            raise RuntimeError("CAP grocery DROID robot spawn is missing its USD path")
        _open_and_validate_source(source_usd_path)
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="cap_grocery_gripper_linkage_"
        )
        self.source_usd_path = source_usd_path
        self.override_usd_path = str(
            Path(self._temporary_directory.name)
            / "franka_robotiq_2f_85_flattened.cap_grocery_override.usda"
        )
        try:
            _write_override_layer(source_usd_path, Path(self.override_usd_path))
            _validate_override_layer(
                Path(self.override_usd_path),
                source_usd_path=source_usd_path,
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Remove the temporary layer after the environment has closed."""
        temporary_directory = self._temporary_directory
        if temporary_directory is None:
            return
        self._temporary_directory = None
        temporary_directory.cleanup()


def apply_grocery_gripper_linkage_override(
    scene_config: Any,
) -> GripperLinkageCollisionOverride:
    """Apply a verified linkage override to one grocery embodiment instance."""
    try:
        robot = copy.deepcopy(scene_config.robot)
        source_usd_path = robot.spawn.usd_path
    except AttributeError as exc:
        raise RuntimeError(
            "CAP grocery DROID scene is missing robot.spawn.usd_path"
        ) from exc

    override = GripperLinkageCollisionOverride(source_usd_path)
    try:
        robot.spawn.usd_path = override.override_usd_path
        scene_config.robot = robot
    except BaseException:
        override.close()
        raise
    return override
