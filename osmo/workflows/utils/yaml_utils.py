# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""YAML helpers for rendering OSMO workflow specs as human-readable YAML."""

import yaml


class block_literal_str(str):
    """String subclass rendered as a YAML block-literal scalar (``|`` style)."""


def _block_literal_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(block_literal_str, _block_literal_representer)
