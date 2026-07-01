# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Policy-runner task for the Isaac Lab Arena OSMO workflow.

This is the programmatic equivalent of the former ``arena_base.yaml`` template.
"""

import argparse
from abc import abstractmethod
from typing import Any

from tasks.base_task import BaseTask
from workflows.workflow_constants import DATASET_SWIFT_URL, OSMO_TASK_OUTPUT_DIR

DEFAULT_IMAGE = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:latest"
POLICY_RUNNER_COMMAND = "/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py"


def _normalize_args(args: str) -> str:
    return " ".join(args.replace("\\\n", " ").split())


class PolicyRunnerTask(BaseTask):
    """Abstract OSMO task that runs an Isaac Lab Arena policy-runner evaluation.

    Concrete subclasses add on which policy to run, by implementing ``_get_policy_args``.
    """

    def __init__(
        self,
        workflow_args: Any,
        task_args: Any,
        image: str = DEFAULT_IMAGE,
        lead: bool | None = None,
    ) -> None:
        super().__init__(workflow_args=workflow_args, task_args=task_args, lead=lead)

        self.policy_runner_args = _normalize_args(self.task_args.policy_runner_args)
        self.arena_env_args = _normalize_args(self.task_args.arena_env_args)
        self.image = image

    @staticmethod
    def add_task_arguments(parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("policy-runner task")
        group.add_argument(
            "--policy_runner_args",
            default="",
            help="Additional policy-runner arguments before the Arena environment args",
        )
        group.add_argument(
            "--arena_env_args",
            required=True,
            help="Arena environment name and env-related arguments",
        )

    @staticmethod
    def get_task_name() -> str:
        return "policy_runner"

    def _get_image(self) -> str:
        return self.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        return [{"url": DATASET_SWIFT_URL}]

    @abstractmethod
    def _get_policy_args(self) -> list[str]:
        """Return the ``policy_runner.py`` arguments selecting this task's policy and its built-ins.

        Includes the ``--policy_type`` flag plus any policy-specific arguments (e.g. remote-server
        connection flags).
        """

    def _get_run_script(self) -> str:
        policy_args_str = " ".join(self._get_policy_args())
        return (
            "set -euxo pipefail\n"
            f"{POLICY_RUNNER_COMMAND} "
            f"{policy_args_str} "
            f"--output_base_dir {OSMO_TASK_OUTPUT_DIR} "
            f"{self.policy_runner_args} "
            f"{self.arena_env_args}\n"
        )
