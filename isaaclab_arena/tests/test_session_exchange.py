# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise filesystem controller exchanges without starting a simulator."""

import hashlib
import json
import numpy as np
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from isaaclab_arena.policy.session_exchange import IDENTITY_FIELDS, SessionExchange
from isaaclab_arena_examples.robot_tool_control.session_client import inspect_session, submit_actions


@pytest.fixture
def exchange(tmp_path):
    session = SessionExchange(tmp_path, "Follow the current task only.\n", {"action_chunk_length": 2}, 3.0)
    yield session
    session.close()


def _start_episode(exchange, episode_index=0):
    exchange.start_episode(0, episode_index, "Put the cube in the bowl.", {"action_chunk_length": 2})


def _wait_for_request(exchange):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        inspection = inspect_session(exchange.directory)
        if "request" in inspection:
            return Path(inspection["request_path"]), inspection["request"]
        time.sleep(0.01)
    raise AssertionError("The policy did not publish a request")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_request_preserves_images_state_and_snapshots(exchange):
    _start_episode(exchange)
    image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    observation = {"joint_position": np.arange(7, dtype=np.float32), "gripper_position": np.float32(0.25)}
    actions = [[0.1] * 7 + [0], [0.2] * 7 + [1]]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(exchange.request_actions, observation, {"external": image})
        request_path, request = _wait_for_request(exchange)
        assert request["observation"] == {"joint_position": list(range(7)), "gripper_position": 0.25}
        assert request["task_instruction"] == "Put the cube in the bowl."
        assert request["request_index"] == 0
        image_path = exchange.directory / request["images"]["external"]
        np.testing.assert_array_equal(np.asarray(Image.open(image_path)), image)
        assert not Path(request["images"]["external"]).is_absolute()
        response_path = submit_actions(request_path, actions, "First chunk")
        response = future.result(timeout=2)

    assert response["actions"] == actions
    assert response["note"] == "First chunk"
    assert response_path == exchange.directory / request["response_path"]
    assert _read_json(request_path.parent / "request_state.json")["status"] == "answered"
    manifest = _read_json(exchange.directory / "session.json")
    assert manifest["active_request"] is None
    assert (
        manifest["controller_prompt_sha256"]
        == hashlib.sha256((exchange.directory / "controller_prompt.md").read_bytes()).hexdigest()
    )
    assert (
        manifest["policy_configuration_sha256"]
        == hashlib.sha256((exchange.directory / "policy_configuration.json").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("field_name", IDENTITY_FIELDS)
@pytest.mark.parametrize("omit_field", [False, True])
def test_mismatched_or_missing_response_identity_stops_request(exchange, field_name, omit_field):
    _start_episode(exchange)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(exchange.request_actions, {}, {})
        request_path, request = _wait_for_request(exchange)
        response = {name: request[name] for name in IDENTITY_FIELDS} | {"actions": [[0] * 8] * 2}
        if omit_field:
            del response[field_name]
        else:
            response[field_name] = -1 if isinstance(response[field_name], int) else "another-request"
        temporary_response = request_path.parent / "invalid.json"
        temporary_response.write_text(json.dumps(response), encoding="utf-8")
        temporary_response.replace(request_path.parent / "response.json")
        with pytest.raises(ValueError, match=field_name):
            future.result(timeout=2)

    error = _read_json(request_path.parent / "error.json")
    assert error["error_type"] == "ValueError"
    assert error["request_id"] == request["request_id"]
    assert _read_json(request_path.parent / "request_state.json")["status"] == "failed"
    assert _read_json(request_path.parent / "response.json") == response


def test_duplicate_submission_never_overwrites_first_response(exchange):
    _start_episode(exchange)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(exchange.request_actions, {}, {})
        request_path, _ = _wait_for_request(exchange)
        response_path = submit_actions(request_path, [[0] * 8] * 2)
        first_response = response_path.read_bytes()
        with pytest.raises((FileExistsError, ValueError)):
            submit_actions(request_path, [[1] * 8] * 2)
        assert future.result(timeout=2)["actions"] == [[0] * 8] * 2
    assert response_path.read_bytes() == first_response


def test_old_episode_response_cannot_satisfy_new_request(exchange):
    _start_episode(exchange)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first_future = executor.submit(exchange.request_actions, {}, {})
        first_request_path, first_request = _wait_for_request(exchange)
        submit_actions(first_request_path, [[0] * 8] * 2)
        first_future.result(timeout=2)
        exchange.end_episode()
        _start_episode(exchange, episode_index=1)
        second_future = executor.submit(exchange.request_actions, {}, {})
        second_request_path, second_request = _wait_for_request(exchange)
        assert first_request["request_id"] != second_request["request_id"]
        assert second_request["episode_index"] == 1
        assert second_request["request_index"] == 0
        with pytest.raises(ValueError, match="no longer awaiting"):
            submit_actions(first_request_path, [[1] * 8] * 2)
        assert not second_future.done()
        submit_actions(second_request_path, [[0.5] * 7 + [1]] * 2)
        assert second_future.result(timeout=2)["actions"] == [[0.5] * 7 + [1]] * 2


def test_request_indices_order_chunks_within_the_episode(exchange):
    _start_episode(exchange)
    with ThreadPoolExecutor(max_workers=1) as executor:
        for request_index in range(2):
            future = executor.submit(exchange.request_actions, {}, {})
            request_path, request = _wait_for_request(exchange)
            assert request["request_index"] == request_index
            submit_actions(request_path, [[0] * 8] * 2)
            future.result(timeout=2)


def test_timeout_records_error_and_refuses_late_submission(tmp_path):
    exchange = SessionExchange(tmp_path, "prompt", {}, 0.02)
    _start_episode(exchange)
    with pytest.raises(TimeoutError):
        exchange.request_actions({}, {})
    request_path = next(exchange.directory.glob("episodes/*/requests/*/request.json"))
    assert _read_json(request_path.parent / "error.json")["error_type"] == "TimeoutError"
    with pytest.raises(ValueError, match="no longer awaiting"):
        submit_actions(request_path, [[0] * 8] * 2)
    exchange.close()
    assert _read_json(exchange.directory / "episodes/env0_episode0/episode.json")["status"] == "stopped"
    assert _read_json(exchange.directory / "session.json")["status"] == "closed"


def test_policy_validation_error_keeps_received_response(exchange):
    _start_episode(exchange)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(exchange.request_actions, {}, {})
        request_path, request = _wait_for_request(exchange)
        response_path = submit_actions(request_path, [[99] * 8])
        response = future.result(timeout=2)
    exchange.record_error(ValueError("Action chunk has the wrong shape"))
    assert _read_json(response_path) == response
    assert _read_json(request_path.parent / "request_state.json")["status"] == "failed"
    error = _read_json(request_path.parent / "error.json")
    assert error["message"] == "Action chunk has the wrong shape"
    assert error["request_id"] == request["request_id"]


def test_episode_boundary_and_close_do_not_invent_success(exchange):
    _start_episode(exchange)
    exchange.end_episode()
    exchange.close()
    exchange.close()
    events = [json.loads(line) for line in (exchange.directory / "events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["episode_started", "episode_ended"]
    assert all("success" not in event for event in events)
    assert _read_json(exchange.directory / "episodes/env0_episode0/episode.json")["status"] == "ended"
    assert inspect_session(exchange.directory)["session"]["last_event"]["event"] == "episode_ended"


def test_instances_never_share_the_same_directory(tmp_path):
    first = SessionExchange(tmp_path, "first", {}, 1)
    second = SessionExchange(tmp_path, "second", {}, 1)
    assert first.directory != second.directory
    first.close()
    second.close()


def test_client_inspection_runs_without_loading_simulator(exchange):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaclab_arena_examples.robot_tool_control.session_client",
            "inspect",
            "--session",
            str(exchange.directory),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    inspection = json.loads(result.stdout)
    assert inspection["session_directory"] == str(exchange.directory.resolve())
    assert inspection["session"]["status"] == "open"
