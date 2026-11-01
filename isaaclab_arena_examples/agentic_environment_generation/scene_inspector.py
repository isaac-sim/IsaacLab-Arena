# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Inspect authored USD data and request scene renders from a live backend.

Usage::

    python isaaclab_arena_examples/agentic_environment_generation/scene_inspector.py \
        prim-tree --usd /path/to/scene.usd --output prim_tree.txt

    python isaaclab_arena_examples/agentic_environment_generation/scene_inspector.py \
        snapshot --env-spec scene.yaml --output snapshots

    python isaaclab_arena_examples/agentic_environment_generation/scene_inspector.py \
        preview --env-spec scene.yaml --output preview.mp4
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import time
import yaml
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.client import (
    SimAppClient,
    SimAppError,
    default_simapp_socket_path,
)


def _json_stdout(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


@contextlib.contextmanager
def _live_backend(socket_path: Path) -> Iterator[SimAppClient]:
    try:
        client = SimAppClient.connect(str(socket_path))
    except OSError as exc:
        raise SimAppError(
            f"No scene-inspector backend is listening on {socket_path}. "
            "Start scene_inspector_backend.py before running this command."
        ) from exc
    try:
        if not client.ping():
            raise SimAppError(f"Scene-inspector backend on {socket_path} did not answer ping.")
        yield client
    finally:
        client.disconnect()


def _load_env_spec(path: Path):
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    if not path.is_file():
        raise FileNotFoundError(path)
    return ArenaEnvGraphSpec.from_yaml(path)


def _spec_yaml_text(path: Path) -> str:
    """Return a self-contained graph YAML string with external includes expanded."""
    return yaml.safe_dump(_load_env_spec(path).to_dict(), sort_keys=False)


def _dump_prim_tree(args: argparse.Namespace) -> int:
    from isaaclab_arena.utils.usd.prim_tree import format_usd_prim_tree, load_usd_prim_tree

    started_at = time.monotonic()
    records = load_usd_prim_tree(args.usd)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_usd_prim_tree(records) + "\n", encoding="utf-8")
    _json_stdout({
        "command": "prim-tree",
        "status": "completed",
        "inputs": {"usd": args.usd},
        "output": str(output_path),
        "record_count": len(records),
        "timings_s": {"total": time.monotonic() - started_at},
    })
    return 0


def _backend_status(args: argparse.Namespace) -> int:
    try:
        with _live_backend(args.socket):
            pass
    except SimAppError as exc:
        _json_stdout({
            "command": "status",
            "status": "unavailable",
            "socket": str(args.socket),
            "error": str(exc),
        })
        return 1
    _json_stdout({"command": "status", "status": "ready", "socket": str(args.socket)})
    return 0


def _shutdown_backend(args: argparse.Namespace) -> int:
    try:
        client = SimAppClient.connect(str(args.socket))
    except OSError as exc:
        raise SimAppError(f"No scene-inspector backend is listening on {args.socket}.") from exc
    client.shutdown()
    _json_stdout({"command": "shutdown", "status": "completed", "socket": str(args.socket)})
    return 0


def _snapshot_scene(args: argparse.Namespace) -> int:
    from isaaclab_arena.agentic_environment_generation.spec_io import safe_filename_stem

    started_at = time.monotonic()
    spec = _load_env_spec(args.env_spec)
    with _live_backend(args.socket) as client:
        images, dimensions = client.render_spec(
            spec,
            background_panorama=args.background_panorama,
            ignore_cache=True,
        )

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    used_filenames: set[str] = set()
    for node_id, image in sorted(images.items()):
        filename = f"{safe_filename_stem(node_id)}.png"
        if filename in used_filenames:
            raise ValueError(f"Snapshot node ids map to the same output filename: {filename}")
        used_filenames.add(filename)
        output_path = output_dir / filename
        output_path.write_bytes(image)
        output_paths[node_id] = str(output_path)

    _json_stdout({
        "command": "snapshot",
        "status": "completed",
        "inputs": {"env_spec": str(args.env_spec.resolve())},
        "socket": str(args.socket),
        "outputs": output_paths,
        "aabb_dimensions_m": {node_id: list(size) for node_id, size in dimensions.items()},
        "background_panorama": args.background_panorama,
        "ignore_cache": True,
        "timings_s": {"total": time.monotonic() - started_at},
    })
    return 0


def _preview_scene(args: argparse.Namespace) -> int:
    started_at = time.monotonic()
    with _live_backend(args.socket) as client:
        response = client.run_sim_preview(
            _spec_yaml_text(args.env_spec),
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            env_spacing=args.env_spacing,
            show_placement_overlay=args.show_placement_overlay,
        )

    source_path = Path(response["video_path"])
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source_path, output_path)
    finally:
        shutil.rmtree(source_path.parent, ignore_errors=True)

    _json_stdout({
        "command": "preview",
        "status": "completed",
        "inputs": {"env_spec": str(args.env_spec.resolve())},
        "socket": str(args.socket),
        "output": str(output_path),
        "num_envs": response.get("num_envs", args.num_envs),
        "num_steps": response.get("num_steps", args.num_steps),
        "env_spacing": response.get("env_spacing", args.env_spacing),
        "show_placement_overlay": response.get("show_placement_overlay", args.show_placement_overlay),
        "timings_s": {"total": time.monotonic() - started_at},
    })
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--socket",
        type=Path,
        default=default_simapp_socket_path(),
        help="Unix socket of the persistent scene-inspector backend.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="Check whether the persistent backend is ready.")
    status.set_defaults(handler=_backend_status)
    shutdown = commands.add_parser("shutdown", help="Ask the persistent backend to exit.")
    shutdown.set_defaults(handler=_shutdown_backend)

    prim_tree = commands.add_parser("prim-tree", help="Dump the USD physics/collision prim tree.")
    prim_tree.add_argument("--usd", required=True, help="Local path or supported remote URL of the USD scene.")
    prim_tree.add_argument("--output", type=Path, required=True, help="Destination text file.")
    prim_tree.set_defaults(handler=_dump_prim_tree)

    snapshot = commands.add_parser(
        "snapshot",
        help="Render graph assets and object-reference collider visualizations.",
    )
    snapshot.add_argument("--env-spec", type=Path, required=True, help="ArenaEnvGraphSpec YAML.")
    snapshot.add_argument("--output", type=Path, required=True, help="Directory for rendered PNG files.")
    snapshot.add_argument(
        "--background-panorama",
        action="store_true",
        help="Render the background as a 360-degree panorama.",
    )
    snapshot.set_defaults(handler=_snapshot_scene)

    preview = commands.add_parser("preview", help="Run a zero-action scene preview and write viewport video.")
    preview.add_argument("--env-spec", type=Path, required=True, help="ArenaEnvGraphSpec YAML.")
    preview.add_argument("--output", type=Path, required=True, help="Destination MP4 file.")
    preview.add_argument("--num-envs", type=int, default=1)
    preview.add_argument("--num-steps", type=int, default=10)
    preview.add_argument("--env-spacing", type=float, default=30.0)
    preview.add_argument(
        "--show-placement-overlay",
        action="store_true",
        help="Overlay relation-solver placement AABBs and mesh collision geometry.",
    )
    preview.set_defaults(handler=_preview_scene)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return int(args.handler(args))
    except (AssertionError, KeyError, OSError, RuntimeError, ValueError) as exc:
        _json_stdout({
            "command": args.command,
            "status": "failed",
            "error_code": type(exc).__name__,
            "error": str(exc),
        })
        return 1


if __name__ == "__main__":
    sys.exit(main())
