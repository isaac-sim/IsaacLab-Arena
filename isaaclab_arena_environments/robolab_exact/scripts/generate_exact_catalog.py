# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the reference-backed RoboLab exact scene and task catalog."""

from __future__ import annotations

import argparse
import json
import yaml
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOLAB_DIR = REPO_ROOT / "isaaclab_arena_environments" / "robolab"
EXACT_DIR = REPO_ROOT / "isaaclab_arena_environments" / "robolab_exact"
MANIFEST_PATH = EXACT_DIR / "SOURCE_MANIFEST.yaml"
METADATA_PATH = REPO_ROOT / "RoboLab" / "assets" / "scenes" / "_metadata" / "scene_metadata.json"
COPYRIGHT = """\
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    assert isinstance(data, dict), f"{path} must contain a YAML mapping"
    return data


def _dump_yaml(data: dict[str, Any]) -> str:
    return COPYRIGHT + yaml.safe_dump(data, sort_keys=False)


def _expected_outputs() -> dict[Path, str]:
    manifest = _load_yaml(MANIFEST_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    outputs: dict[Path, str] = {}

    regular_scenes = {path.stem: path for path in sorted((ROBOLAB_DIR / "scenes").glob("*.yaml"))}
    regular_tasks = sorted((ROBOLAB_DIR / "tasks").glob("*.yaml"))
    assert set(manifest["scenes"]) == set(regular_scenes), "Manifest scenes must exactly match regular RoboLab scenes"

    for scene_name, regular_path in regular_scenes.items():
        scene_manifest = manifest["scenes"][scene_name]
        source_name = Path(scene_manifest["source_usda"]).name
        assert source_name == f"{scene_name}.usda"
        source_records = {record["prim_path"]: record for record in metadata[source_name]}
        regular = _load_yaml(regular_path)
        regular_ids = {item["id"] for item in regular.get("objects", [])}
        mappings = scene_manifest["object_mappings"]
        mapped_ids = {item["arena_object_id"] for item in mappings}
        assert mapped_ids == regular_ids, (
            f"{scene_name}: manifest ids differ from regular scene; "
            f"missing={sorted(regular_ids - mapped_ids)}, extra={sorted(mapped_ids - regular_ids)}"
        )

        background_id = scene_manifest["background_registry_name"]
        references = []
        for mapping in mappings:
            prim_path = mapping["prim_path"]
            assert prim_path in source_records, f"{scene_name}: missing source prim {prim_path}"
            record = source_records[prim_path]
            expected_type = "rigid" if record["rigid_body"] is True else "base"
            assert (
                mapping["object_type"] == expected_type
            ), f"{scene_name}:{prim_path} declares {mapping['object_type']}, expected {expected_type}"
            references.append({
                "id": mapping["arena_object_id"],
                "parent_id": background_id,
                "prim_path": prim_path.split("/", 2)[2],
                "object_type": mapping["object_type"],
                "params": {},
            })

        exact_scene = {
            "embodiment": regular["embodiment"],
            "background": {
                "id": background_id,
                "registry_name": background_id,
                "params": {},
            },
            "objects": [],
            "object_references": references,
            "relations": [],
        }
        outputs[EXACT_DIR / "scenes" / f"{scene_name}.yaml"] = _dump_yaml(exact_scene)

    for task_path in regular_tasks:
        task = _load_yaml(task_path)
        scene_name = Path(task["external_yaml"]).stem
        scene_manifest = manifest["scenes"][scene_name]
        background_id = scene_manifest["background_registry_name"]
        for subtask in task["task"]["subtasks"]:
            if "background_scene" in subtask["params"]:
                subtask["params"]["background_scene"] = background_id
        outputs[EXACT_DIR / "tasks" / task_path.name] = _dump_yaml(task)

    return outputs


def _update(outputs: dict[Path, str]) -> None:
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _check(outputs: dict[Path, str]) -> None:
    stale = [
        str(path.relative_to(REPO_ROOT))
        for path, content in outputs.items()
        if not path.exists() or path.read_text() != content
    ]
    assert not stale, f"RoboLab exact generated files are stale: {stale}"


def main() -> None:
    """Generate or check the RoboLab exact catalog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="Write generated scene and task YAMLs.")
    args = parser.parse_args()
    outputs = _expected_outputs()
    _update(outputs) if args.update else _check(outputs)
    print(f"{'Updated' if args.update else 'Validated'} {len(outputs)} RoboLab exact YAML files.")


if __name__ == "__main__":
    main()
