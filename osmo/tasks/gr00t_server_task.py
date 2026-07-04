# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T inference-server task for OSMO eval workflows, used by the GR00T policy-runner task."""

import argparse
from typing import Any

from tasks.base_task import BaseTask

# GR00T server image (containing the droid checkpoints).
DEFAULT_IMAGE = "nvcr.io/nvstaging/isaac-amr/gr00t_1_6_droid"
# Droid checkpoint baked into the GR00T server image.
DEFAULT_MODEL_PATH = "/workspace/pretrained_ckpts/GR00T-N1.6-DROID"
# Embodiment tag for the droid manipulation config (see droid_manip_gr00t_closedloop_config.yaml).
DEFAULT_EMBODIMENT_TAG = "OXE_DROID"
DEFAULT_SERVER_PORT = 5555

# OSMO task name for the server. The eval runner reaches it via the {{host:<name>}} token,
# which OSMO resolves to the runtime IP.
GR00T_SERVER_TASK_NAME = "gr00t_server"
GR00T_SERVER_HOST_TOKEN = "{{host:" + GR00T_SERVER_TASK_NAME + "}}"

# Run in the Arena eval image (not the server image): polls the server until it accepts requests.
WAIT_FOR_SERVER_COMMAND = "/isaac-sim/python.sh -u -m isaaclab_arena_gr00t.utils.wait_for_gr00t_server"


def get_wait_for_server_script(host: str, port: int) -> str:
    """Return the bash snippet that blocks until the GR00T server accepts requests."""
    return (
        "# Wait for the GR00T server task to come up before starting the eval.\n"
        f"{WAIT_FOR_SERVER_COMMAND} \\\n"
        f"  --host {host} \\\n"
        f"  --port {port} \\\n"
        "  --timeout-sec 1200 \\\n"
        "  --poll-interval-sec 15 \\\n"
        "  --request-timeout-ms 5000\n"
    )


class Gr00tServerTask(BaseTask):
    """OSMO task that serves a GR00T policy for an eval/policy-runner task to connect to."""

    def __init__(
        self,
        workflow_args: Any,
        task_args: Any,
        image: str = DEFAULT_IMAGE,
        lead: bool | None = None,
    ) -> None:
        super().__init__(workflow_args=workflow_args, task_args=task_args, lead=lead)
        self.docker_image = getattr(task_args, "gr00t_server_image", image)
        self.model_path = task_args.gr00t_model_path
        self.embodiment_tag = task_args.gr00t_embodiment_tag
        self.server_port = task_args.server_port

    @staticmethod
    def add_task_arguments(parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("gr00t server")
        group.add_argument("--gr00t_server_image", default=DEFAULT_IMAGE, help="Override the GR00T server image")
        group.add_argument("--gr00t_model_path", default=DEFAULT_MODEL_PATH, help="Model path for the GR00T policy")
        group.add_argument(
            "--gr00t_embodiment_tag", default=DEFAULT_EMBODIMENT_TAG, help="Embodiment tag for the GR00T policy"
        )
        group.add_argument("--server_port", type=int, default=DEFAULT_SERVER_PORT, help="GR00T server port")

    @staticmethod
    def get_task_name() -> str:
        return GR00T_SERVER_TASK_NAME

    def _get_image(self) -> str:
        return self.docker_image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        return []

    def _get_run_script(self) -> str:
        return (
            "set -euxo pipefail\n"
            "nvidia-smi\n"
            "cd /workspace\n"
            "exec uv run python gr00t/eval/run_gr00t_server.py \\\n"
            f"  --model_path={self.model_path} \\\n"
            f"  --embodiment_tag={self.embodiment_tag} \\\n"
            "  --host=0.0.0.0 \\\n"
            f"  --port={self.server_port}\n"
        )
