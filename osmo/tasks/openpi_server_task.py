# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from tasks.base_task import BaseTask
from workflows.utils.workflow_types import WorkflowType

# Openpi server image, published as a tag of the isaaclab_arena image by
# isaaclab_arena_openpi/docker/build_server_image.sh.
DEFAULT_IMAGE = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:openpi_server"


class OpenpiServerTask(BaseTask):
    """OSMO task that runs a dummy command inside the openpi server image."""

    def __init__(
        self,
        workflow_type: WorkflowType,
        workflow_args: Any,
        task_args: Any,
        image: str = DEFAULT_IMAGE,
    ) -> None:
        workflow_type = WorkflowType(workflow_type)
        assert workflow_type == WorkflowType.OPENPI_SERVER, f"Unsupported workflow type: {workflow_type.value}"
        super().__init__(workflow_type=workflow_type, workflow_args=workflow_args, task_args=task_args)
        self.image = image

    @staticmethod
    def get_task_name() -> str:
        return WorkflowType.OPENPI_SERVER.value

    def _get_image(self) -> str:
        return self.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        return []

    def _get_run_script(self) -> str:
        return 'set -euxo pipefail\necho "hello world"\n'
