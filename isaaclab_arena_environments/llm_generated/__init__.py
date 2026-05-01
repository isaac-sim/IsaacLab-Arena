# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-package for LLM-generated env modules (output of
``isaaclab_arena.llm_env_gen.auto_generate_env``).

Each module here registers an environment via ``@register_environment``;
the parent package's ``__init__.py`` recursively imports submodules so
the canonical (no ``_t<N>`` suffix) generated envs land in the registry
alongside the hand-written ones.
"""
