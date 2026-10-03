# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export Docker requirements from the project's existing package metadata."""

import argparse
import tomllib
from pathlib import Path


def requirements(metadata: dict) -> list[str]:
    """Return project and dev-extra requirements, preserving declared constraints.

    Args:
        metadata: Parsed project metadata.

    Returns:
        Requirements for the developer image.
    """
    project = metadata["project"]
    return [*project["dependencies"], *project["optional-dependencies"]["dev"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    args = parser.parse_args()
    print("\n".join(requirements(tomllib.loads(args.metadata.read_text()))))
