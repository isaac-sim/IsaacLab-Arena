# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Update and verify RoboLab exact scenes against their source USDA table frames."""

from __future__ import annotations

import argparse
import math
import yaml
from pathlib import Path

EXACT_ROOT = Path(__file__).parents[1]
DEFAULT_SOURCE_DIR = Path(__file__).parents[3] / "RoboLab" / "assets" / "scenes"
TABLE_WRAPPER_DIR = Path(__file__).parents[3] / "isaaclab_arena" / "assets" / "robolab"
MANIFEST_PATH = EXACT_ROOT / "SOURCE_MANIFEST.yaml"
REGULAR_SCENES_DIR = EXACT_ROOT.parent / "robolab" / "scenes"
REGULAR_TASKS_DIR = EXACT_ROOT.parent / "robolab" / "tasks"
LICENSE_HEADER = """# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
TABLE_REGISTRY_BY_FIXTURE = {
    "table_bamboo.usd": "bamboo_table_robolab",
    "table_black.usd": "black_table_robolab",
    "table_maple.usd": "maple_table_robolab",
    "table_oak.usd": "oak_table_robolab",
}
WRAPPER_SCENE_BY_FIXTURE = {
    "table_bamboo.usd": "bamboo_table.usda",
    "table_black.usd": "black_table.usda",
    "table_maple.usd": "maple_table.usda",
    "table_oak.usd": "oak_table.usda",
}
LEGACY_BACKGROUND_ID = "maple_table_robolab"


def _parser() -> argparse.ArgumentParser:
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

    parser = get_isaaclab_arena_cli_parser()
    parser.description = __doc__
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--update", action="store_true", help="Rewrite generated poses and table selections first.")
    parser.add_argument("--skip-runtime", action="store_true", help="Only validate YAML poses against source USDA.")
    parser.add_argument("--position-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=1.0e-4)
    return parser


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write(LICENSE_HEADER)
        yaml.safe_dump(data, stream, sort_keys=False)


def _source_table_prim(layer):
    for path in ("/world/table", "/World/table"):
        prim = layer.GetPrimAtPath(path)
        if prim is not None:
            return prim
    raise AssertionError(f"No /world/table or /World/table prim in {layer.identifier}")


def _table_fixture_name(table_prim) -> str:
    payload_list = table_prim.payloadList
    payloads = (
        list(payload_list.explicitItems)
        + list(payload_list.prependedItems)
        + list(payload_list.addedItems)
        + list(payload_list.appendedItems)
    )
    for payload in payloads:
        name = Path(payload.assetPath).name
        if name in TABLE_REGISTRY_BY_FIXTURE:
            return name

    # Some source scenes are flattened and no longer retain the payload arc. Their
    # oak table uses the settled table-oak location, while table-maple is near x=0.2.
    table_position, _ = _authored_pose(table_prim)
    return "table_maple.usd" if abs(table_position[0]) < 0.3 else "table_oak.usd"


def _authored_pose(prim_spec) -> tuple[list[float], object]:
    from pxr import Gf

    position_spec = prim_spec.attributes.get("xformOp:translate")
    assert position_spec is not None, f"Missing xformOp:translate on '{prim_spec.path}'"
    position = position_spec.default
    orientation_spec = prim_spec.attributes.get("xformOp:orient")
    if orientation_spec is None:
        quaternion = Gf.Quatd(1.0)
    else:
        orientation = orientation_spec.default
        quaternion = Gf.Quatd(
            float(orientation.GetReal()),
            Gf.Vec3d(*(float(value) for value in orientation.GetImaginary())),
        ).GetNormalized()
    return [float(value) for value in position], quaternion


def _pose_dict(prim_spec) -> dict[str, list[float]]:
    position, orientation = _authored_pose(prim_spec)
    imaginary = orientation.GetImaginary()
    return {
        "position_xyz": position,
        "rotation_xyzw": [float(imaginary[index]) for index in range(3)] + [float(orientation.GetReal())],
    }


def _pose_matrix(pose: dict[str, list[float]]):
    from pxr import Gf

    x, y, z, w = pose["rotation_xyzw"]
    transform = Gf.Transform()
    transform.SetTranslation(Gf.Vec3d(*pose["position_xyz"]))
    transform.SetRotation(Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z))))
    return transform.GetMatrix()


def _matrix_pose(matrix) -> dict[str, list[float]]:
    quaternion = matrix.RemoveScaleShear().ExtractRotationQuat().GetNormalized()
    imaginary = quaternion.GetImaginary()
    return {
        "position_xyz": [float(value) for value in matrix.ExtractTranslation()],
        "rotation_xyzw": [float(imaginary[index]) for index in range(3)] + [float(quaternion.GetReal())],
    }


def _wrapper_table_pose(fixture_name: str) -> dict[str, list[float]]:
    from pxr import Sdf

    wrapper_path = TABLE_WRAPPER_DIR / WRAPPER_SCENE_BY_FIXTURE[fixture_name]
    layer = Sdf.Layer.FindOrOpen(str(wrapper_path))
    assert layer is not None, f"Failed to open table wrapper '{wrapper_path}'"
    return _pose_dict(_source_table_prim(layer))


def _reframe_object_poses(
    source_table_pose: dict[str, list[float]],
    wrapper_table_pose: dict[str, list[float]],
    source_object_poses: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, list[float]]]:
    source_to_wrapper = _pose_matrix(source_table_pose).GetInverse() * _pose_matrix(wrapper_table_pose)
    return {
        object_id: _matrix_pose(_pose_matrix(object_pose) * source_to_wrapper)
        for object_id, object_pose in source_object_poses.items()
    }


def _source_scene_data(
    source_path: Path,
    object_mappings: list[dict],
) -> tuple[str, dict[str, list[float]], dict[str, dict[str, list[float]]]]:
    from pxr import Sdf

    # Read the scene layer's authored root xforms directly. Loading or unloading
    # payloads can respectively add asset-internal transforms or hide these specs.
    layer = Sdf.Layer.FindOrOpen(str(source_path))
    assert layer is not None, f"Failed to open source scene '{source_path}'"
    table_prim = _source_table_prim(layer)
    fixture_name = _table_fixture_name(table_prim)
    table_pose = _pose_dict(table_prim)
    poses = {}
    for mapping in object_mappings:
        arena_id = mapping["arena_object_id"]
        benchmark_prim = mapping["benchmark_prim"]
        parent_path = str(table_prim.path).rsplit("/", maxsplit=1)[0]
        prim = layer.GetPrimAtPath(f"{parent_path}/{benchmark_prim}")
        assert prim is not None, f"Missing source prim '{benchmark_prim}' in '{source_path}'"
        poses[arena_id] = _pose_dict(prim)
    return fixture_name, table_pose, poses


def _quaternion_angle(left_xyzw: list[float], right_xyzw: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left_xyzw))
    right_norm = math.sqrt(sum(value * value for value in right_xyzw))
    dot = abs(sum(left * right for left, right in zip(left_xyzw, right_xyzw))) / (left_norm * right_norm)
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _assert_pose_close(
    actual: dict[str, list[float]],
    expected: dict[str, list[float]],
    position_tolerance: float,
    orientation_tolerance_rad: float,
    label: str,
) -> None:
    position_error = math.dist(actual["position_xyz"], expected["position_xyz"])
    orientation_error = _quaternion_angle(actual["rotation_xyzw"], expected["rotation_xyzw"])
    assert (
        position_error <= position_tolerance
    ), f"{label} position error {position_error:.3e} m exceeds {position_tolerance:.3e} m"
    assert (
        orientation_error <= orientation_tolerance_rad
    ), f"{label} orientation error {orientation_error:.3e} rad exceeds {orientation_tolerance_rad:.3e} rad"


def _update_files(manifest: dict, source_dir: Path) -> None:
    table_registry_by_scene = {}
    for scene_name, scene_manifest in manifest["scenes"].items():
        source_path = source_dir / Path(scene_manifest["source_usda"]).name
        fixture_name, source_table_pose, source_object_poses = _source_scene_data(
            source_path, scene_manifest["object_mappings"]
        )
        table_pose = _wrapper_table_pose(fixture_name)
        expected_poses = _reframe_object_poses(source_table_pose, table_pose, source_object_poses)
        table_registry = TABLE_REGISTRY_BY_FIXTURE[fixture_name]
        table_registry_by_scene[scene_name] = table_registry

        exact_path = EXACT_ROOT / "scenes" / f"{scene_name}.yaml"
        exact = _load_yaml(exact_path)
        exact["background"]["id"] = table_registry
        exact["background"]["registry_name"] = table_registry
        exact["background"].pop("initial_pose", None)
        for object_spec in exact["objects"]:
            object_spec["initial_pose"] = expected_poses[object_spec["id"]]
        exact.pop("relations", None)
        _write_yaml(exact_path, exact)

        regular_path = REGULAR_SCENES_DIR / f"{scene_name}.yaml"
        regular = _load_yaml(regular_path)
        old_background_id = regular["background"]["id"]
        regular["background"]["id"] = table_registry
        regular["background"]["registry_name"] = table_registry
        regular["background"].pop("initial_pose", None)
        for relation in regular.get("relations", []):
            if relation["subject"] in (old_background_id, LEGACY_BACKGROUND_ID):
                relation["subject"] = table_registry
            if relation.get("reference") in (old_background_id, LEGACY_BACKGROUND_ID):
                relation["reference"] = table_registry
        _write_yaml(regular_path, regular)

        source_rotation_xyzw = source_table_pose["rotation_xyzw"]
        scene_manifest["benchmark_table_transform"] = {
            "position_xyz": source_table_pose["position_xyz"],
            "rotation_wxyz": [source_rotation_xyzw[3], *source_rotation_xyzw[:3]],
            "scale_xyz": [1.0, 1.0, 1.0],
        }
        wrapper_rotation_xyzw = table_pose["rotation_xyzw"]
        scene_manifest["wrapper_table_transform"] = {
            "position_xyz": table_pose["position_xyz"],
            "rotation_wxyz": [wrapper_rotation_xyzw[3], *wrapper_rotation_xyzw[:3]],
            "scale_xyz": [1.0, 1.0, 1.0],
        }
        scene_manifest["source_table_asset"] = fixture_name
        scene_manifest["table_wrapper_usda"] = f"isaaclab_arena/assets/robolab/{WRAPPER_SCENE_BY_FIXTURE[fixture_name]}"
        scene_manifest["table_registry_name"] = table_registry
        scene_manifest["source_usda"] = f"RoboLab/assets/scenes/{source_path.name}"

    for tasks_dir in (EXACT_ROOT / "tasks", REGULAR_TASKS_DIR):
        for task_path in sorted(tasks_dir.glob("*.yaml")):
            task = _load_yaml(task_path)
            scene_name = Path(task["external_yaml"]).stem
            table_registry = table_registry_by_scene[scene_name]
            for subtask in task["task"]["subtasks"]:
                params = subtask["params"]
                if "background_scene" in params:
                    params["background_scene"] = table_registry
            _write_yaml(task_path, task)

    manifest["source_scene_directory"] = "RoboLab/assets/scenes"
    manifest.pop("arena_table_transform", None)
    manifest["transform_formula"] = (
        "T_arena_background = I; T_arena_obj = T_benchmark_obj * inverse(T_benchmark_table) * T_wrapper_table"
    )
    _write_yaml(MANIFEST_PATH, manifest)


def _validate_yaml(
    manifest: dict,
    source_dir: Path,
    position_tolerance: float,
    orientation_tolerance_rad: float,
) -> None:
    for scene_name, scene_manifest in manifest["scenes"].items():
        source_path = source_dir / Path(scene_manifest["source_usda"]).name
        fixture_name, source_table_pose, source_object_poses = _source_scene_data(
            source_path, scene_manifest["object_mappings"]
        )
        table_pose = _wrapper_table_pose(fixture_name)
        expected_poses = _reframe_object_poses(source_table_pose, table_pose, source_object_poses)
        table_registry = TABLE_REGISTRY_BY_FIXTURE[fixture_name]
        exact = _load_yaml(EXACT_ROOT / "scenes" / f"{scene_name}.yaml")
        assert "relations" not in exact, f"{scene_name}: relations must be omitted"
        assert exact["background"]["id"] == table_registry
        assert exact["background"]["registry_name"] == table_registry
        assert "initial_pose" not in exact["background"], f"{scene_name}: exact background offset must be omitted"
        for object_spec in exact["objects"]:
            _assert_pose_close(
                object_spec["initial_pose"],
                expected_poses[object_spec["id"]],
                position_tolerance,
                orientation_tolerance_rad,
                f"{scene_name}/{object_spec['id']} YAML",
            )

        regular = _load_yaml(REGULAR_SCENES_DIR / f"{scene_name}.yaml")
        assert regular["background"]["id"] == table_registry
        assert regular["background"]["registry_name"] == table_registry
        assert "initial_pose" not in regular["background"], f"{scene_name}: regular background offset must be omitted"
        print(f"[yaml] {scene_name}: {len(expected_poses)} poses match {fixture_name}", flush=True)


def _task_by_scene() -> dict[str, Path]:
    result = {}
    for task_path in sorted((EXACT_ROOT / "tasks").glob("*.yaml")):
        task = _load_yaml(task_path)
        scene_name = Path(task["external_yaml"]).stem
        result.setdefault(scene_name, task_path)
    return result


def _runtime_quaternion_xyzw(runtime_quaternion: list[float], expected_xyzw: list[float]) -> list[float]:
    as_xyzw = runtime_quaternion
    as_wxyz = runtime_quaternion[1:] + runtime_quaternion[:1]
    if _quaternion_angle(as_xyzw, expected_xyzw) <= _quaternion_angle(as_wxyz, expected_xyzw):
        return as_xyzw
    return as_wxyz


def _validate_runtime_scene(
    scene_name: str,
    task_path: Path,
    table_pose: dict[str, list[float]],
    expected_poses: dict[str, dict[str, list[float]]],
    table_registry: str,
    args: argparse.Namespace,
) -> None:
    import omni.usd
    import warp as wp
    from pxr import Usd, UsdGeom

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.evaluation.resource_cleanup import close_environment

    arena_environment = ArenaEnvGraphSpec.from_yaml(task_path).to_arena_env()
    builder = ArenaEnvBuilder(
        arena_environment,
        ArenaEnvBuilderCfg(num_envs=1, device=args.device),
    )
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    env_cfg.recorders = {}
    env_cfg.episode_recorders = {}
    env = builder.make_registered(env_cfg, env_kwargs)
    try:
        env.reset()
        scene = env.unwrapped.scene
        env_origin = scene.env_origins[0]
        stage = omni.usd.get_context().get_stage()
        table_prim = stage.GetPrimAtPath(f"/World/envs/env_0/{table_registry}/table")
        assert table_prim.IsValid(), f"{scene_name}: missing spawned table child for {table_registry}"
        table_matrix = UsdGeom.Xformable(table_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        actual_table_pose = _matrix_pose(table_matrix)
        actual_table_pose["position_xyz"] = [
            actual - float(origin) for actual, origin in zip(actual_table_pose["position_xyz"], env_origin.tolist())
        ]
        _assert_pose_close(
            actual_table_pose,
            table_pose,
            args.position_tolerance,
            args.orientation_tolerance_rad,
            f"{scene_name} runtime background",
        )
        for object_id, expected in expected_poses.items():
            asset = scene[object_id]
            root_positions = wp.to_torch(asset.data.root_pos_w)
            position = root_positions[0] - env_origin.to(root_positions.device)
            quaternion = wp.to_torch(asset.data.root_quat_w)[0].tolist()
            actual = {
                "position_xyz": position.tolist(),
                "rotation_xyzw": _runtime_quaternion_xyzw(quaternion, expected["rotation_xyzw"]),
            }
            _assert_pose_close(
                actual,
                expected,
                args.position_tolerance,
                args.orientation_tolerance_rad,
                f"{scene_name}/{object_id} runtime",
            )
        print(f"[runtime] {scene_name}: {len(expected_poses)} spawned poses match", flush=True)
    finally:
        close_environment(env)


def _validate_runtime(manifest: dict, source_dir: Path, args: argparse.Namespace) -> None:
    tasks = _task_by_scene()
    for scene_name, scene_manifest in manifest["scenes"].items():
        assert scene_name in tasks, f"No task references exact scene '{scene_name}'"
        source_path = source_dir / Path(scene_manifest["source_usda"]).name
        fixture_name, source_table_pose, source_object_poses = _source_scene_data(
            source_path, scene_manifest["object_mappings"]
        )
        table_pose = _wrapper_table_pose(fixture_name)
        expected_poses = _reframe_object_poses(source_table_pose, table_pose, source_object_poses)
        _validate_runtime_scene(
            scene_name,
            tasks[scene_name],
            table_pose,
            expected_poses,
            TABLE_REGISTRY_BY_FIXTURE[fixture_name],
            args,
        )


def main() -> None:
    """Update generated YAMLs if requested, then verify source and runtime poses."""
    args = _parser().parse_args()
    manifest = _load_yaml(MANIFEST_PATH)
    if args.update:
        _update_files(manifest, args.source_dir)
        manifest = _load_yaml(MANIFEST_PATH)
    if args.skip_runtime:
        _validate_yaml(manifest, args.source_dir, args.position_tolerance, args.orientation_tolerance_rad)
        return
    assert not args.update, "Run --update --skip-runtime first, then run runtime verification separately"

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    with SimulationAppContext(args):
        _validate_yaml(manifest, args.source_dir, args.position_tolerance, args.orientation_tolerance_rad)
        _validate_runtime(manifest, args.source_dir, args)


if __name__ == "__main__":
    main()
