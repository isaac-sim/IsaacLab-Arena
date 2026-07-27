# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Policy-runner task for the Isaac Lab Arena OSMO workflow.

This is the programmatic equivalent of the former ``arena_base.yaml`` template.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from osmo.tasks.base_task import BaseTask, TaskCfg
from osmo.workflows.workflow_constants import DATASET_SWIFT_URL, OSMO_TASK_OUTPUT_DIR

POLICY_RUNNER_COMMAND = "/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py"


@dataclass
class PolicyRunnerTaskCfg(TaskCfg):
    """Config for a policy-runner evaluation task."""

    arena_env: str
    """Graph-spec YAML path or registered example-environment name."""

    policy_runner_args: list[str] = field(default_factory=list)
    """Additional policy-runner arguments before the Arena environment args."""

    arena_env_args: list[str] = field(default_factory=list)
    """Env-related arguments for the chosen Arena environment."""

    variation_args: list[str] = field(default_factory=list)
    """Hydra-style variation overrides appended to the env, e.g. 'light.hdr_image.enabled=true'."""

    image: str = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:latest"
    """Container image the policy-runner task runs in."""


def _arg_list_to_cli_string(args: list[str]) -> str:
    """Flatten argument tokens into a single whitespace-normalized command-line string."""
    return " ".join(" ".join(args).replace("\\\n", " ").split())


class PolicyRunnerTask(BaseTask):
    """Abstract OSMO task that runs an Isaac Lab Arena policy-runner evaluation.

    Concrete subclasses add on which policy to run, by implementing ``_get_policy_args``.
    """

    def __init__(
        self,
        task_cfg: PolicyRunnerTaskCfg,
        lead: bool | None = None,
        *,
        task_name: str,
    ) -> None:
        super().__init__(task_name=task_name, task_cfg=task_cfg, lead=lead)

    def _get_image(self) -> str:
        return self.task_cfg.image

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
            _arg_list_to_cli_string(self.task_cfg.policy_runner_args),
            self.get_arena_env_token(),
            _arg_list_to_cli_string(self.task_cfg.arena_env_args),
            _arg_list_to_cli_string(self.task_cfg.variation_args),
        ]
        return " ".join(part for part in parts if part)

    def get_arena_env_token(self) -> str:
        """Render the Arena env selector token as ``policy_runner.py`` expects it.

        The ``--arena_env`` value chooses the environment source and is resolved as follows:

        - Registered example-environment pass by name: kitchen_pick_and_place
        - A graph-spec YAML path is preceded by the flag: --env_graph_spec_yaml robolab/tasks/mustard_above_raisin.yaml``.
        """
        if self.task_cfg.arena_env.endswith((".yaml", ".yml")):
            return f"--env_graph_spec_yaml {self.task_cfg.arena_env}"
        return self.task_cfg.arena_env
