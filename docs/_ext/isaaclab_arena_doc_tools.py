# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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
    """Replaces the :isaaclab_arena_git_clone_code_block: directive with a code block."""

    def replacer(_: Any) -> str:
        git_url = app.config.isaaclab_arena_docs_config["git_url"]
        # smv_current_version is the ref of a per-version build; empty for local `make html`.
        docs_ref = getattr(app.config, "smv_current_version", "") or "main"
        repo_dir = git_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        return f"""
.. code-block:: bash

    git clone --branch {docs_ref} --recurse-submodules {git_url}
    cd {repo_dir}

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


def uv_run_command_replacer(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces uv setup command directives with code blocks.

    ``:uv_run_source:`` may be followed by backticked extra arguments for ``uv sync``, for example
    ``:uv_run_source:`--group gr00t-client```.
    """

    def uv_source_replacer(match: re.Match) -> str:
        extra_args = match.group("args")
        sync_args = f" {extra_args}" if extra_args else ""
        return f""".. code-block:: bash

           uv sync --extra dev{sync_args}
           source .venv/bin/activate
           export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y"""

    source[0] = re.sub(r":uv_run_source:(?:`(?P<args>[^`]*)`)?", uv_source_replacer, source[0])


def docker_run_command_replacer(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces docker run command directives with code blocks."""

    # Default docker run command
    def default_replacer(_: Any) -> str:
        return """.. code-block:: bash

           ./docker/run_docker.sh"""

    source[0] = re.sub(r":docker_run_default:", default_replacer, source[0])


def runtime_setup_tabs_replacer(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces the :runtime_setup_tabs: directive with a lead-in sentence and a Native uv / Docker tab set.

    Backticked extra arguments for ``uv sync`` may follow the directive, for example
    ``:runtime_setup_tabs:`--group gr00t-client```.
    """

    template = (
        "Prepare the Arena runtime from the repository root, using either a native ``uv`` environment or the\n"
        "base Docker container (see :doc:`/pages/quickstart/installation` for the full setup):\n"
        "\n"
        ".. tab-set::\n"
        "\n"
        "   .. tab-item:: Native uv\n"
        "      :selected:\n"
        "\n"
        "      :uv_run_source:{uv_args}\n"
        "\n"
        "   .. tab-item:: Docker Container\n"
        "\n"
        "      :docker_run_default:"
    )

    def replacer(match: re.Match) -> str:
        indent = match.group("indent")
        marker = match.group("marker") or ""
        body_indent = indent + " " * len(marker)
        extra_args = match.group("args")
        uv_args = f"`{extra_args}`" if extra_args else ""
        lines = template.replace("{uv_args}", uv_args).split("\n")
        out = []
        for i, line in enumerate(lines):
            if not line:
                out.append(line)
            elif i == 0:
                out.append(indent + marker + line)
            else:
                out.append(body_indent + line)
        return "\n".join(out)

    source[0] = re.sub(
        r"^(?P<indent> *)(?P<marker>#\. )?:runtime_setup_tabs:(?:`(?P<args>[^`]*)`)?",
        replacer,
        source[0],
        flags=re.MULTILINE,
    )


def setup(app: Sphinx) -> None:
    app.connect("source-read", isaaclab_arena_git_clone_code_block)
    app.connect("source-read", isaaclab_arena_code_link)
    app.connect("source-read", runtime_setup_tabs_replacer)
    app.connect("source-read", uv_run_command_replacer)
    app.connect("source-read", docker_run_command_replacer)
    app.add_config_value("isaaclab_arena_docs_config", {}, "env")
