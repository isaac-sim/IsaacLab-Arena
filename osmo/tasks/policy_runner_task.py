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
        self.arena_env = self.task_args.arena_env
        self.arena_env_args = _normalize_args(self.task_args.arena_env_args)
        self.variations = _normalize_args(self.task_args.variations)
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
            "--arena_env",
            required=True,
            help="Graph-spec YAML path or registered example-environment name",
        )
        group.add_argument(
            "--arena_env_args",
            default="",
            help="Env-related arguments for the chosen Arena environment",
        )
        group.add_argument(
            "--variations",
            default="",
            help="Hydra-style variation overrides appended to the env, e.g. 'light.hdr_image.enabled=true'",
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
        return f"set -euxo pipefail\n{self._get_policy_runner_command()}\n"

    def _get_policy_runner_command(self) -> str:
        """Assemble the policy_runner.py command."""
        parts = [
            POLICY_RUNNER_COMMAND,
            *self._get_policy_args(),
            "--output_base_dir",
            OSMO_TASK_OUTPUT_DIR,
            self.policy_runner_args,
            self.get_arena_env_token(),
            self.arena_env_args,
            self.variations,
        ]
        return " ".join(part for part in parts if part)

    def get_arena_env_token(self) -> str:
        """Render the Arena env selector token as ``policy_runner.py`` expects it.

        The ``--arena_env`` value chooses the environment source and is resolved as follows:

        - Registered example-environment pass by name: kitchen_pick_and_place
        - A graph-spec YAML path is preceded by the flag: --env_graph_spec_yaml robolab/mustard_raisin_box.yaml``.
        """
        if self.arena_env.endswith((".yaml", ".yml")):
            return f"--env_graph_spec_yaml {self.arena_env}"
        return self.arena_env
