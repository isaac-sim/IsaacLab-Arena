# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exchange observations and action responses with an external controller through files."""

import hashlib
import json
import math
import numpy as np
import os
import tempfile
import time
import uuid
from pathlib import Path

from PIL import Image

PROTOCOL_VERSION = 1
IDENTITY_FIELDS = ("protocol_version", "policy_instance_id", "env_id", "episode_index", "request_id")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def _write_json(path: Path, contents: dict) -> None:
    """Publish complete JSON after flushing it to disk."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=".", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(contents, output, indent=2, allow_nan=False, default=_json_default)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


class SessionExchange:
    """Own one policy instance's episode, request, response, and error artifacts."""

    def __init__(
        self,
        output_directory: Path,
        controller_prompt: str,
        policy_configuration: dict,
        response_timeout_s: float,
    ):
        assert (
            math.isfinite(response_timeout_s) and response_timeout_s > 0
        ), "Response timeout must be positive and finite"
        self._response_timeout_s = response_timeout_s
        self._policy_instance_id = uuid.uuid4().hex
        self.directory = Path(output_directory) / self._policy_instance_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self._episode: dict | None = None
        self._episode_directory: Path | None = None
        self._active_request: dict | None = None
        self._last_request: dict | None = None
        self._request_directory: Path | None = None
        self._request_index = 0
        prompt_path = self.directory / "controller_prompt.md"
        prompt_path.write_text(controller_prompt, encoding="utf-8")
        configuration_path = self.directory / "policy_configuration.json"
        _write_json(configuration_path, policy_configuration)
        self._manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "policy_instance_id": self._policy_instance_id,
            "status": "open",
            "active_request": None,
            "controller_prompt_path": prompt_path.name,
            "controller_prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
            "policy_configuration_path": configuration_path.name,
            "policy_configuration_sha256": hashlib.sha256(configuration_path.read_bytes()).hexdigest(),
        }
        self._write_manifest()
        print(f"Session policy directory: {self.directory.resolve()}", flush=True)

    def start_episode(self, env_id: int, episode_index: int, task_instruction: str, action_contract: dict) -> None:
        """Open an episode when its first action is requested."""
        assert self._manifest["status"] == "open", "Session exchange is closed"
        assert self._episode is None, "An episode is already active"
        episode_directory = self.directory / "episodes" / f"env{env_id}_episode{episode_index}"
        episode_directory.mkdir(parents=True, exist_ok=False)
        self._episode_directory = episode_directory
        self._last_request = None
        self._request_directory = None
        self._request_index = 0
        self._episode = {
            "protocol_version": PROTOCOL_VERSION,
            "policy_instance_id": self._policy_instance_id,
            "env_id": env_id,
            "episode_index": episode_index,
            "task_instruction": task_instruction,
            "action_contract": action_contract,
            "status": "active",
        }
        _write_json(episode_directory / "episode.json", self._episode)
        self._record_event("episode_started")

    def request_actions(self, observation: dict, images: dict[str, np.ndarray]) -> dict:
        """Publish an observation and await exactly one matching response."""
        assert self._manifest["status"] == "open", "Session exchange is closed"
        assert (
            self._episode is not None and self._episode_directory is not None
        ), "Start an episode before requesting actions"
        assert self._active_request is None, "An action request is already outstanding"
        request_id = uuid.uuid4().hex
        request_directory = self._episode_directory / "requests" / request_id
        request_directory.mkdir(parents=True, exist_ok=False)
        self._request_directory = request_directory
        identity = {name: self._episode[name] for name in IDENTITY_FIELDS if name != "request_id"} | {
            "request_id": request_id
        }
        request = {
            **identity,
            "request_index": self._request_index,
            "task_instruction": self._episode["task_instruction"],
            "action_contract": self._episode["action_contract"],
            "controller_prompt_path": "controller_prompt.md",
            "observation": observation,
            "images": {},
            "response_path": str((request_directory / "response.json").relative_to(self.directory)),
        }
        self._request_index += 1
        self._active_request = request
        self._last_request = request
        try:
            for image_index, (name, image_array) in enumerate(images.items()):
                image_path = request_directory / f"image_{image_index}.png"
                Image.fromarray(image_array).save(image_path)
                request["images"][name] = str(image_path.relative_to(self.directory))
            _write_json(request_directory / "request_state.json", {**identity, "status": "pending"})
            _write_json(request_directory / "request.json", request)
            self._manifest["active_request"] = str((request_directory / "request.json").relative_to(self.directory))
            self._write_manifest()
            response = self._wait_for_response(request_directory / "response.json", identity)
            _write_json(request_directory / "request_state.json", {**identity, "status": "answered"})
            return response
        except BaseException as error:
            self.record_error(error)
            raise
        finally:
            self._active_request = None
            self._manifest["active_request"] = None
            self._write_manifest()

    def _wait_for_response(self, response_path: Path, identity: dict) -> dict:
        deadline = time.monotonic() + self._response_timeout_s
        while True:
            if self._active_request is None or self._manifest["status"] != "open":
                raise RuntimeError("Controller request was cancelled")
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise TimeoutError(
                    f"No controller response within {self._response_timeout_s:g} seconds: {response_path}"
                )
            if response_path.is_file():
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if not isinstance(response, dict):
                    raise ValueError("Controller response must be a JSON object")
                for field_name, expected in identity.items():
                    if type(response.get(field_name)) is not type(expected) or response[field_name] != expected:
                        raise ValueError(f"Controller response has mismatched {field_name}; expected {expected!r}")
                if "actions" not in response:
                    raise ValueError("Controller response is missing actions")
                if "note" in response and not isinstance(response["note"], str):
                    raise ValueError("Controller response note must be a string")
                return response
            time.sleep(min(0.05, remaining_seconds))

    def record_error(self, error: BaseException) -> None:
        """Preserve a controller error with the most recent request and raw response path."""
        request = self._active_request or self._last_request
        error_directory = self._request_directory if request is not None else self.directory
        assert error_directory is not None
        artifact = {"error_type": type(error).__name__, "message": str(error)}
        if request is not None:
            identity = {name: request[name] for name in IDENTITY_FIELDS}
            artifact.update(identity)
            artifact["request_path"] = str((error_directory / "request.json").relative_to(self.directory))
            artifact["response_path"] = request["response_path"]
            _write_json(error_directory / "request_state.json", {**identity, "status": "failed"})
        _write_json(error_directory / "error.json", artifact)
        self._manifest["last_error"] = str((error_directory / "error.json").relative_to(self.directory))
        self._write_manifest()

    def end_episode(self) -> None:
        """Record an episode boundary without inferring a task outcome."""
        if self._episode is not None:
            self._finish_episode("ended")

    def _finish_episode(self, status: str) -> None:
        assert self._episode is not None and self._episode_directory is not None
        if self._active_request is not None:
            assert self._request_directory is not None
            identity = {name: self._active_request[name] for name in IDENTITY_FIELDS}
            _write_json(self._request_directory / "request_state.json", {**identity, "status": "cancelled"})
            self._active_request = None
            self._manifest["active_request"] = None
            self._write_manifest()
        self._episode["status"] = status
        _write_json(self._episode_directory / "episode.json", self._episode)
        self._record_event("episode_ended" if status == "ended" else "episode_stopped")
        self._episode = None
        self._episode_directory = None

    def close(self) -> None:
        """Mark unfinished work as stopped and close the session manifest."""
        if self._manifest["status"] == "closed":
            return
        try:
            if self._episode is not None:
                self._finish_episode("stopped")
        finally:
            self._manifest["status"] = "closed"
            self._manifest["active_request"] = None
            self._write_manifest()

    def _record_event(self, event: str) -> None:
        assert self._episode is not None
        record = {
            "event": event,
            "policy_instance_id": self._policy_instance_id,
            "env_id": self._episode["env_id"],
            "episode_index": self._episode["episode_index"],
        }
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as event_file:
            event_file.write(json.dumps(record, allow_nan=False) + "\n")
            event_file.flush()
            os.fsync(event_file.fileno())
        self._manifest["last_event"] = record
        self._write_manifest()

    def _write_manifest(self) -> None:
        _write_json(self.directory / "session.json", self._manifest)
