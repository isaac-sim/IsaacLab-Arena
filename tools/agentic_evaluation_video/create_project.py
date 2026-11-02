# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Copy the reference video sources into a new, isolated working directory."""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    assert not destination.exists(), f"Use a new directory; preserving {destination}"
    source = Path(__file__).resolve().parent
    shutil.copytree(source / "template", destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(source / "AGENT_GUIDE.md", destination / "AGENT_GUIDE.md")
    print(f"Created {destination}. Follow AGENT_GUIDE.md before recording.")


if __name__ == "__main__":
    main()
