# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import re
from typing import Any

from sphinx.application import Sphinx


def isaaclab_arena_git_clone_code_block(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces the :isaaclab_arena_git_clone_code_block: directive with a code block.
    The output git clone command depends on whether we're in release or internal mode.
    """

    def replacer(_: Any) -> str:
        release_state = app.config.isaaclab_arena_docs_config["released"]
        internal_git_url = app.config.isaaclab_arena_docs_config["internal_git_url"]
        external_git_url = app.config.isaaclab_arena_docs_config["external_git_url"]
        if release_state:
            git_clone_target = external_git_url
        else:
            git_clone_target = internal_git_url
        print(f"git_clone_target: {git_clone_target}")
        return f"""
.. code-block:: bash

    git clone {git_clone_target}

"""

    source[0] = re.sub(r":isaaclab_arena_git_clone_code_block:", replacer, source[0])


def isaaclab_arena_code_link(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces the :isaaclab_arena_code_link: directive with a code block.

    The output link is either gitlab (internal) or github (external) depending on the release state.

    """

    def replacer(match: re.Match) -> str:
        relative_path = match.group("relative_path")
        release_state = app.config.isaaclab_arena_docs_config["released"]
        internal_code_link_base_url = app.config.isaaclab_arena_docs_config["internal_code_link_base_url"]
        external_code_link_base_url = app.config.isaaclab_arena_docs_config["external_code_link_base_url"]
        # Extract the file name
        file_name = relative_path.split("/")[-1]
        if release_state:
            code_link_base_url = external_code_link_base_url
        else:
            code_link_base_url = internal_code_link_base_url
        return f"`{file_name} <{code_link_base_url}/{relative_path}>`_"

    source[0] = re.sub(r":isaaclab_arena_code_link:`<(?P<relative_path>.*)>`", replacer, source[0])


def docker_run_command_replacer(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces docker run command directives with code blocks."""

    # Default docker run command
    def default_replacer(_: Any) -> str:
        return """.. code-block:: bash

           ./docker/run_docker.sh"""

    # Docker run with GR00T dependencies
    def gr00t_replacer(_: Any) -> str:
        return """.. code-block:: bash

           ./docker/run_docker.sh -g"""

    source[0] = re.sub(r":docker_run_default:", default_replacer, source[0])
    source[0] = re.sub(r":docker_run_gr00t:", gr00t_replacer, source[0])


def setup(app: Sphinx) -> None:
    app.connect("source-read", isaaclab_arena_git_clone_code_block)
    app.connect("source-read", isaaclab_arena_code_link)
    app.connect("source-read", docker_run_command_replacer)
    app.add_config_value("isaaclab_arena_docs_config", {}, "env")
