# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prebuild the cached workbench USD assets used by the canonical scene."""

import argparse


def main() -> None:
    """Process selected workbench assets through the normal runtime cache."""
    from isaaclab_arena_environments.workbench_cleanup import assets

    processors = {
        "workbench": assets._WORKBENCH,
        "white_bin": assets._WHITE_BIN,
        "socket_set": assets._SOCKET_SET,
        "oscilloscope": assets._OSCILLOSCOPE,
        "slotted_screwdriver": assets._SLOTTED_SCREWDRIVER,
        "small_wrench": assets._SMALL_WRENCH,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="*", choices=processors)
    args = parser.parse_args()
    for name in args.assets or processors:
        print(f"{name}: {processors[name].process()}")


if __name__ == "__main__":
    main()
