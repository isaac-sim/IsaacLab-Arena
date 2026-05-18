# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Base class for Isaac Lab Arena OSMO tasks.

Child classes define the task-specific entry script, inputs, image, and credentials; ``BaseTask``
assembles the dict that OSMO consumes under ``workflow.groups[*].tasks``.
"""

import argparse
from abc import ABC, abstractmethod
from typing import Any

from workflows.utils.workflow_types import WorkflowType
from workflows.utils.yaml_utils import block_literal_str


class BaseTask(ABC):
    """Abstract base task for an Isaac Lab Arena OSMO workflow."""

    def __init__(
        self,
        workflow_type: WorkflowType,
        workflow_args: argparse.Namespace,
        task_args: argparse.Namespace,
    ) -> None:
        self.workflow_type = workflow_type
        self.workflow_args = workflow_args
        self.task_args = task_args

    def create_task_dict(self) -> dict[str, Any]:
        """Assemble the task dict consumed by OSMO."""
        return {
            "args": ["/tmp/entry.sh"],
            "command": ["bash"],
            "credentials": self._get_credentials(),
            "downloadType": "download",
            "environment": self._get_environment(),
            "files": [{"path": "/tmp/entry.sh", "contents": block_literal_str(self._get_run_script())}],
            "image": self._get_image(),
            "inputs": self._get_inputs(),
            "lead": True,
            "name": self.get_task_name(),
        }

    def _get_environment(self) -> dict[str, str]:
        """Return environment variables for the task."""
        return {
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "ISAACLAB_PATH": "/workspaces/isaaclab_arena/submodules/IsaacLab",
            "REQUESTS_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
            "PYTHONUNBUFFERED": "1",
        }

    def _get_credentials(self) -> dict[str, dict[str, str]]:
        """Return OSMO credential mappings for the task."""
        return {
            "omni_svc": {
                "OMNI_PASS": "omni_pass",
                "OMNI_USER": "omni_user",
            }
        }

    @staticmethod
    @abstractmethod
    def get_task_name() -> str:
        """Return the task name."""

    @abstractmethod
    def _get_image(self) -> str:
        """Return the container image reference."""

    @abstractmethod
    def _get_inputs(self) -> list[dict[str, Any]]:
        """Return input dataset declarations for the task."""

    @abstractmethod
    def _get_run_script(self) -> str:
        """Return the bash entry-script body executed by the task."""
