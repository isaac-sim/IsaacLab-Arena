# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the scene-inspector command line tools."""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).parents[2]
_SCRIPT = _REPO_ROOT / "isaaclab_arena_examples/agentic_environment_generation/scene_inspector.py"
_BACKEND_SCRIPT = _REPO_ROOT / "isaaclab_arena_examples/agentic_environment_generation/scene_inspector_backend.py"


def test_prim_tree_command_writes_text_file(tmp_path: Path):
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text(
        """\
#usda 1.0
(
    defaultPrim = "Asset"
)

def Xform "Asset"
{
    def Cube "Counter" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
    }
}
""",
        encoding="utf-8",
    )
    output_path = tmp_path / "reports" / "prim_tree.txt"

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "prim-tree",
            "--usd",
            str(usd_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["status"] == "completed"
    assert response["record_count"] == 1
    assert response["output"] == str(output_path)
    output = output_path.read_text(encoding="utf-8")
    assert output.startswith("USD PRIM TREE:\n")
    assert "Counter (base " in output
    assert "bbox_center_m=[0,0,0]" in output
    assert "bbox_size_m=[2,2,2]" in output
    assert "collider(type=Cube,enabled=true,rigid_body_owner=world,approximation=None)" in output


def test_backend_entrypoint_help_does_not_start_simapp():
    result = subprocess.run(
        [sys.executable, str(_BACKEND_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "--socket" in result.stdout


def test_snapshot_command_writes_backend_images(tmp_path: Path, monkeypatch, capsys):
    from isaaclab_arena_examples.agentic_environment_generation import scene_inspector

    class _Client:
        def render_spec(self, _spec, *, background_panorama, ignore_cache):
            assert background_panorama
            assert ignore_cache
            return {"counter/top": b"png"}, {"counter/top": (1.0, 2.0, 3.0)}

    @contextlib.contextmanager
    def _backend(_socket):
        yield _Client()

    monkeypatch.setattr(scene_inspector, "_live_backend", _backend)
    monkeypatch.setattr(scene_inspector, "_load_env_spec", lambda _path: object())
    output_dir = tmp_path / "snapshots"
    args = argparse.Namespace(
        socket=tmp_path / "backend.sock",
        env_spec=tmp_path / "scene.yaml",
        output=output_dir,
        background_panorama=True,
    )

    assert scene_inspector._snapshot_scene(args) == 0

    assert (output_dir / "counter_top.png").read_bytes() == b"png"
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "completed"
    assert response["ignore_cache"] is True
    assert response["aabb_dimensions_m"] == {"counter/top": [1.0, 2.0, 3.0]}


def test_preview_command_copies_backend_video(tmp_path: Path, monkeypatch, capsys):
    from isaaclab_arena_examples.agentic_environment_generation import scene_inspector

    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    source_path = backend_dir / "preview.mp4"
    source_path.write_bytes(b"video")

    class _Client:
        def run_sim_preview(self, yaml_text, *, num_envs, num_steps, env_spacing, show_placement_overlay):
            assert yaml_text == "expanded"
            assert show_placement_overlay
            return {
                "video_path": str(source_path),
                "num_envs": num_envs,
                "num_steps": num_steps,
                "env_spacing": env_spacing,
                "show_placement_overlay": show_placement_overlay,
            }

    @contextlib.contextmanager
    def _backend(_socket):
        yield _Client()

    monkeypatch.setattr(scene_inspector, "_live_backend", _backend)
    monkeypatch.setattr(scene_inspector, "_spec_yaml_text", lambda _path: "expanded")
    output_path = tmp_path / "result" / "preview.mp4"
    args = SimpleNamespace(
        socket=tmp_path / "backend.sock",
        env_spec=tmp_path / "scene.yaml",
        output=output_path,
        num_envs=2,
        num_steps=3,
        env_spacing=4.0,
        show_placement_overlay=True,
    )

    assert scene_inspector._preview_scene(args) == 0

    assert output_path.read_bytes() == b"video"
    assert not backend_dir.exists()
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "completed"
    assert response["num_envs"] == 2
    assert response["show_placement_overlay"] is True


@pytest.mark.parametrize("requested_ignore_cache", [False, True])
def test_render_backend_forwards_cache_choice(requested_ignore_cache):
    from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.server import _handle_render_spec

    received_ignore_cache = None

    class _Spec:
        @staticmethod
        def from_dict(_data):
            return object()

    def _render(_app, _spec, *, background_panorama, ignore_cache):
        nonlocal received_ignore_cache
        assert not background_panorama
        received_ignore_cache = ignore_cache
        return {}, {}

    request = {"yaml_text": "{}"}
    if requested_ignore_cache:
        request["ignore_cache"] = True
    response = _handle_render_spec(object(), request, _render, _Spec)

    assert response["ok"]
    assert received_ignore_cache is requested_ignore_cache


def test_object_reference_snapshot_uses_parent_relative_prim_path():
    from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.object_reference_paths import (
        object_reference_relative_prim_path,
    )

    ref = SimpleNamespace(id="table_ref", parent_id="background")
    assets = {
        "background": SimpleNamespace(get_prim_path=lambda: "{ENV_REGEX_NS}/maple_table"),
        "table_ref": SimpleNamespace(get_prim_path=lambda: "{ENV_REGEX_NS}/maple_table/table"),
    }

    assert object_reference_relative_prim_path(ref, assets) == "table"


def test_camera_fit_tightly_frames_bounds():
    from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.camera_framing import (
        fit_camera_to_bounds,
    )

    eye, target = fit_camera_to_bounds(
        (-1.0, -0.5, 0.0),
        (1.0, 0.5, 1.0),
        (1.0, -1.0, 1.0),
    )

    assert target == pytest.approx((0.0, 0.0, 0.5))
    assert eye == pytest.approx((1.555635, -1.555635, 2.055635))
