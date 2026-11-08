# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Inspection transport, ownership, artifact, and frontend contracts without Kit."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from isaaclab_arena.agentic_tools.inspection import artifacts, backend_process, server
from isaaclab_arena.agentic_tools.inspection.artifacts import consume_result, copy_artifact, read_artifact
from isaaclab_arena.agentic_tools.inspection.client import InspectorClient
from isaaclab_arena.agentic_tools.inspection.protocol import (
    InspectionError,
    encode_record,
    read_record,
    validate_request,
)
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "isaaclab_arena/tests/test_data/pick_and_place_maple_table_env_graph.yaml"


@pytest.fixture
def service(tmp_path):
    address = tmp_path / "backend.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(address))
    listener.listen(16)
    seen = []

    def render(_app, spec, **options):
        seen.append((spec.to_dict(), options))
        return {}, {}, ["fixture warning 日本語"]

    thread = threading.Thread(
        target=server._serve_requests,
        kwargs={
            "server": listener,
            "app": SimpleNamespace(update=lambda: None),
            "render_fn": render,
            "spec_cls": ArenaEnvGraphSpec,
            "read_timeout_s": 0.1,
        },
        daemon=True,
    )
    thread.start()
    client = InspectorClient(address, request_timeout_s=2)
    try:
        yield client, seen
    finally:
        client.shutdown()
        thread.join(timeout=3)
        listener.close()
        assert not thread.is_alive()


def test_alternating_clients_and_unicode_graph_roundtrip(service):
    client, seen = service
    spec = ArenaEnvGraphSpec.from_yaml(SPEC_PATH)
    spec.env_name = "検査_é"
    for index in range(3):
        other = InspectorClient(client.socket_path)
        other.ping()
        result = client.render_spec(spec, ignore_cache=bool(index % 2), background_panorama=True)
        assert result["warnings"] == ["fixture warning 日本語"]
        assert seen[-1] == (spec.to_dict(), {"ignore_cache": bool(index % 2), "background_panorama": True})


@pytest.mark.parametrize("record", [b"[]\n", b"oops\n", b'{"cmd":"missing"}\n', b'{"cmd":"render_spec"}\n'])
def test_bad_records_do_not_stop_backend(service, record):
    client, _ = service
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(str(client.socket_path))
        sock.sendall(record)
        response = read_record(sock, timeout_s=2)
        assert response["ok"] is False
    client.ping()


def test_incomplete_request_eof_and_disconnected_reader_do_not_starve(service):
    client, _ = service
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as idle:
        idle.connect(str(client.socket_path))
        idle.sendall(b'{"cmd":')
        started = time.monotonic()
        client.ping(timeout_s=1)
        assert time.monotonic() - started < 1
    for record in (b"", encode_record({"cmd": "ping"})):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as gone:
            gone.connect(str(client.socket_path))
            if record:
                gone.sendall(record)
    client.ping()


def test_deadline_is_unknown_outcome_and_does_not_retry(tmp_path):
    address = tmp_path / "blocked.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(address))
        listener.listen(5)
        client = InspectorClient(address, request_timeout_s=0.05)
        with pytest.raises(InspectionError, match="outcome is unknown"):
            client.ping(timeout_s=0.05)
        assert address.exists()
        conn, _ = listener.accept()
        conn.close()
        listener.settimeout(0.05)
        with pytest.raises(TimeoutError):
            listener.accept()


@pytest.mark.parametrize(
    "options",
    [
        {"num_envs": True, "num_steps": 2, "env_spacing": 1},
        {"num_envs": 1, "num_steps": 1.5, "env_spacing": 1},
        {"num_envs": 1, "num_steps": 2, "env_spacing": float("inf")},
        {"num_envs": 1, "num_steps": 2, "env_spacing": 1, "show_placement_overlay": "yes"},
    ],
)
def test_preview_options_are_strict(options):
    with pytest.raises(ValueError):
        validate_request({"cmd": "run_sim_preview", "spec": {}, "options": options})


def test_expanded_includes_and_json_spec_values(tmp_path):
    source = ArenaEnvGraphSpec.from_yaml(SPEC_PATH).to_dict()
    import yaml

    (tmp_path / "objects.yaml").write_text(yaml.safe_dump({"objects": source.pop("objects")}))
    source["external_yaml"] = "objects.yaml"
    # Use the graph loader's include syntax, rather than sending file references over IPC.
    from isaaclab_arena.environment_spec.arena_env_graph_yaml_loader import load_env_graph_spec_dict

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(source))
    expanded = load_env_graph_spec_dict(spec_path)
    spec = ArenaEnvGraphSpec.from_dict(expanded)
    from isaaclab_arena.utils.physics_backend import PhysicsBackend

    spec.default_physics_backend = PhysicsBackend.PHYSX
    result = ArenaEnvGraphSpec.from_dict(json.loads(encode_record({"spec": spec.to_dict()}))["spec"])
    assert result.to_dict() == spec.to_dict()


def _result(path, directory=None):
    return {
        "artifacts": {"preview": {"path": str(path), "media_type": "video/mp4"}},
        "metadata": {},
        "warnings": [],
        "temporary_dir": str(directory) if directory else None,
    }


@pytest.mark.parametrize("fail", [False, True])
def test_result_reads_copies_and_cleans_only_owned_directory(tmp_path, monkeypatch, fail):
    root = tmp_path / "previews"
    root.mkdir()
    monkeypatch.setattr(artifacts, "sim_preview_cache_dir", lambda: root)
    owned = root / "preview-job"
    owned.mkdir()
    video = owned / "video.mp4"
    video.write_bytes(b"video")
    result = _result(video, owned)
    try:
        with consume_result(result):
            assert read_artifact(result, "preview") == b"video"
            output = copy_artifact(result, "preview", tmp_path / "export.mp4")
            if fail:
                raise RuntimeError("read failure")
    except RuntimeError:
        assert fail
    assert not owned.exists()
    assert output.read_bytes() == b"video"
    cached = tmp_path / "cached.png"
    cached.write_bytes(b"cache")
    with consume_result(_result(cached)):
        pass
    assert cached.read_bytes() == b"cache"


def test_missing_files_and_unowned_cleanup_are_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "sim_preview_cache_dir", lambda: tmp_path / "previews")
    result = _result(tmp_path / "missing")
    with pytest.raises(InspectionError, match="Missing or empty"):
        read_artifact(result, "preview")
    with pytest.raises(InspectionError, match="unowned"):
        with consume_result(_result(tmp_path / "missing", tmp_path)):
            pass
    assert tmp_path.exists()


def test_socket_selection_and_occupied_address(tmp_path, monkeypatch):
    monkeypatch.delenv(backend_process.SIMAPP_SOCKET_ENV, raising=False)
    stat = ROOT.stat()
    key = hashlib.sha256(f"{ROOT}:{stat.st_dev}:{stat.st_ino}".encode()).hexdigest()[:12]
    assert backend_process.select_socket_path().name == f"arena_scene_inspector_{os.getuid()}_{key}.sock"
    monkeypatch.setenv(backend_process.SIMAPP_SOCKET_ENV, str(tmp_path / "env.sock"))
    assert backend_process.select_socket_path() == tmp_path / "env.sock"
    address = tmp_path / "explicit.sock"
    assert backend_process.select_socket_path(address) == address
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(address))
        with pytest.raises(InspectionError, match="already exists"):
            backend_process.spawn_backend(address)
        assert address.exists()


def test_finished_child_stop_does_not_contact_another_backend(tmp_path, monkeypatch):
    client = Mock()
    monkeypatch.setattr(backend_process, "InspectorClient", client)
    backend_process.stop_owned_backend(SimpleNamespace(poll=lambda: 0), tmp_path / "other.sock")
    client.assert_not_called()


def test_core_imports_do_not_load_examples_streamlit_or_kit():
    code = """
import sys
from isaaclab_arena.agentic_tools import scene_inspector, scene_inspector_backend
from isaaclab_arena.agentic_tools.inspection import client, protocol, server, backend_process
assert not any(m.startswith(('isaaclab_arena_examples', 'streamlit', 'omni.kit', 'isaacsim')) for m in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_gui_attached_and_private_ownership(tmp_path, monkeypatch):
    from isaaclab_arena_examples.agentic_environment_generation import gui_runner

    monkeypatch.delenv(backend_process.SIMAPP_SOCKET_ENV, raising=False)
    spawn, stop, wait, ping, run = Mock(), Mock(), Mock(), Mock(), Mock()
    monkeypatch.setattr(gui_runner, "spawn_backend", spawn)
    monkeypatch.setattr(gui_runner, "stop_owned_backend", stop)
    monkeypatch.setattr(gui_runner, "wait_for_backend", wait)
    monkeypatch.setattr(gui_runner, "InspectorClient", lambda _: SimpleNamespace(ping=ping))
    monkeypatch.setattr(gui_runner.subprocess, "run", run)
    address = tmp_path / "attached.sock"
    gui_runner.serve_live_editor(None, socket_path=address)
    ping.assert_called_once()
    spawn.assert_not_called()
    stop.assert_not_called()
    assert run.call_args.kwargs["env"][backend_process.SIMAPP_SOCKET_ENV] == str(address)
    gui_runner.serve_live_editor(None)
    spawn.assert_called_once()
    wait.assert_called_once()
    stop.assert_called_once_with(spawn.return_value, spawn.call_args.args[0])


def test_gui_unavailable_attach_does_not_spawn_or_replace(tmp_path, monkeypatch):
    from isaaclab_arena_examples.agentic_environment_generation import gui_runner

    monkeypatch.delenv(backend_process.SIMAPP_SOCKET_ENV, raising=False)
    spawn = Mock()
    monkeypatch.setattr(gui_runner, "spawn_backend", spawn)
    with pytest.raises(InspectionError):
        gui_runner.serve_live_editor(None, socket_path=tmp_path / "missing.sock")
    spawn.assert_not_called()


def test_gui_private_boot_failure_keeps_editor_available(monkeypatch):
    from isaaclab_arena_examples.agentic_environment_generation import gui_runner

    monkeypatch.delenv(backend_process.SIMAPP_SOCKET_ENV, raising=False)
    spawn, stop, run = Mock(), Mock(), Mock()
    monkeypatch.setattr(gui_runner, "spawn_backend", spawn)
    monkeypatch.setattr(gui_runner, "wait_for_backend", Mock(side_effect=InspectionError("boot failed")))
    monkeypatch.setattr(gui_runner, "stop_owned_backend", stop)
    monkeypatch.setattr(gui_runner.subprocess, "run", run)
    gui_runner.serve_live_editor(None)
    stop.assert_called_once()
    run.assert_called_once()
    assert backend_process.SIMAPP_SOCKET_ENV not in run.call_args.kwargs["env"]


def test_gui_snapshot_warnings_and_placeholders(tmp_path, monkeypatch):
    from isaaclab_arena_examples.agentic_environment_generation.review_gui import visualization_service as gui

    spec = ArenaEnvGraphSpec.from_yaml(SPEC_PATH)
    monkeypatch.setattr(gui.st, "session_state", {})
    monkeypatch.setenv(backend_process.SIMAPP_SOCKET_ENV, str(tmp_path / "unused.sock"))
    monkeypatch.setattr(gui, "resolve_background_prim_tree", lambda _: [])
    warn = Mock()
    monkeypatch.setattr(gui.st, "warning", warn)
    cached = tmp_path / "background.png"
    cached.write_bytes(b"png")
    result = {
        "artifacts": {spec.background.id: {"path": str(cached), "media_type": "image/png"}},
        "metadata": {"aabb_dimensions_m": {spec.background.id: [1, 2, 3]}},
        "warnings": ["failed asset fixture"],
        "temporary_dir": None,
    }
    client = SimpleNamespace(render_spec=lambda *a, **kw: result)
    monkeypatch.setattr(gui, "get_inspector_client", lambda: client)
    cards, _ = gui.build_asset_cards_with_thumbnails(spec)
    background = next(card for card in cards if card.spec.id == spec.background.id)
    assert background.thumbnail_bytes == b"png"
    assert background.aabb_dimensions_m == (1, 2, 3)
    assert any(card.thumbnail_bytes is None for card in cards if card.spec.id != spec.background.id)
    warn.assert_called_once_with("failed asset fixture")
    cached.unlink()
    gui.clear_asset_cards_cache()
    error = Mock()
    monkeypatch.setattr(gui.st, "error", error)
    cards, _ = gui.build_asset_cards_with_thumbnails(spec)
    assert all(card.thumbnail_bytes is None for card in cards)
    error.assert_called_once()


def test_gui_preview_stores_bytes_and_releases_result(tmp_path, monkeypatch):
    from isaaclab_arena_examples.agentic_environment_generation.review_gui import simapp_connector as gui
    from isaaclab_arena_examples.agentic_environment_generation.review_gui.editor_panel import validate_yaml_text

    state = {}
    monkeypatch.setattr(gui.st, "session_state", state)
    monkeypatch.setenv(backend_process.SIMAPP_SOCKET_ENV, str(tmp_path / "unused.sock"))
    monkeypatch.setattr(artifacts, "sim_preview_cache_dir", lambda: tmp_path)
    directory = tmp_path / "preview-gui"
    directory.mkdir()
    video = directory / "video.mp4"
    video.write_bytes(b"video")
    result = _result(video, directory)
    result["metadata"] = {"num_envs": 2, "num_steps": 3, "env_spacing": 4.0}
    validation = validate_yaml_text(SPEC_PATH.read_text())

    def preview(spec, **options):
        assert spec == validation.spec
        return result

    monkeypatch.setattr(gui, "get_inspector_client", lambda: SimpleNamespace(run_sim_preview=preview))
    ok, _ = gui.run_sim_preview_pipeline(SPEC_PATH.read_text(), validation=validation)
    assert ok
    assert state["sim_preview_video"] == b"video"
    assert not directory.exists()
    monkeypatch.setattr(
        gui,
        "get_inspector_client",
        lambda: SimpleNamespace(run_sim_preview=Mock(side_effect=InspectionError("preview failure"))),
    )
    ok, message = gui.run_sim_preview_pipeline(SPEC_PATH.read_text(), validation=validation)
    assert not ok and "preview failure" in message
    assert "sim_preview_video" not in state


def test_second_backend_launcher_cannot_unlink_occupied_socket(tmp_path):
    address = tmp_path / "occupied.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(address))
        original_inode = address.stat().st_ino
        result = subprocess.run(
            [sys.executable, "-m", "isaaclab_arena.agentic_tools.scene_inspector_backend", "--socket", str(address)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert address.stat().st_ino == original_inode


def test_capture_waits_for_completion_instead_of_existing_png(tmp_path, monkeypatch):
    import asyncio
    import io

    from PIL import Image

    from isaaclab_arena.agentic_tools.inspection import kit_viewport

    path = tmp_path / "image.png"
    path.write_bytes(b"old")
    image_bytes = io.BytesIO()
    Image.new("RGB", (2, 2), color="red").save(image_bytes, format="PNG")
    fresh = image_bytes.getvalue()
    pending = []
    completion = SimpleNamespace(done=lambda: True, result=lambda: None)

    def capture(_viewport, destination):
        pending.append(Path(destination))
        return SimpleNamespace(wait_for_result=lambda: completion)

    monkeypatch.setitem(
        sys.modules,
        "omni.kit.viewport.utility",
        SimpleNamespace(capture_viewport_to_file=capture, get_active_viewport=lambda: object()),
    )
    monkeypatch.setattr(asyncio, "ensure_future", lambda future: future)

    frames = []

    def pump(_app, **kwargs):
        if pending:
            frames.append(1)
            assert path.read_bytes() == b"old"
            if len(frames) == 2:
                pending[0].write_bytes(fresh[:20])
            if len(frames) == 3:
                pending[0].write_bytes(fresh)

    monkeypatch.setattr(kit_viewport, "pump_app", pump)
    assert kit_viewport.capture_viewport_png(object(), path) == fresh
    assert pending[0] != path
    assert not pending[0].exists()


def test_graph_validation_error_and_traceback_reach_client(service):
    client, _ = service
    with pytest.raises(InspectionError, match="ValidationError"):
        client._request({"cmd": "render_spec", "spec": {}, "options": {}})
    client.ping()


def test_distinct_checkout_directories_get_distinct_default_sockets(tmp_path, monkeypatch):
    monkeypatch.delenv(backend_process.SIMAPP_SOCKET_ENV, raising=False)
    addresses = []
    for clone in ("first", "second"):
        root = tmp_path / clone
        module = root / "isaaclab_arena/agentic_tools/inspection/backend_process.py"
        module.parent.mkdir(parents=True)
        module.touch()
        monkeypatch.setattr(backend_process, "__file__", str(module))
        addresses.append(backend_process.select_socket_path())
    assert addresses[0] != addresses[1]


def test_capture_timeout_preserves_old_png_and_cleans_pending_output(tmp_path, monkeypatch):
    import asyncio

    from isaaclab_arena.agentic_tools.inspection import kit_viewport

    destination = tmp_path / "old.png"
    destination.write_bytes(b"old")
    pending = []
    completion = SimpleNamespace(done=lambda: False, cancel=Mock())

    def capture(_viewport, path):
        pending.append(Path(path))
        pending[0].write_bytes(b"partial")
        return SimpleNamespace(wait_for_result=lambda: completion)

    monkeypatch.setitem(
        sys.modules,
        "omni.kit.viewport.utility",
        SimpleNamespace(capture_viewport_to_file=capture, get_active_viewport=lambda: object()),
    )
    monkeypatch.setattr(asyncio, "ensure_future", lambda future: future)
    monkeypatch.setattr(kit_viewport, "pump_app", lambda *a, **kw: None)
    with pytest.raises(TimeoutError, match="did not complete"):
        kit_viewport.capture_viewport_png(object(), destination, max_updates=2)
    completion.cancel.assert_called_once()
    assert destination.read_bytes() == b"old"
    assert not pending[0].parent.exists()
