# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prebuild the cached workbench USD assets used by the canonical scene."""

import argparse


def main() -> None:
    """Cache selected workbench assets through their normal runtime paths."""
    from isaaclab_arena_environments.workbench_cleanup import assets

    cache_assets = {
        "workbench": assets._WORKBENCH.process,
        "white_bin": lambda: assets._retrieve_wrapper("white_bin.usda"),
        "socket_set": lambda: assets._retrieve_wrapper("socket_set.usda"),
        "oscilloscope": lambda: assets._retrieve_wrapper("oscilloscope.usda"),
        "slotted_screwdriver": lambda: assets._retrieve_wrapper("slotted_screwdriver.usda"),
        "small_wrench": lambda: assets._retrieve_wrapper("small_wrench.usda"),
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="*", choices=cache_assets)
    args = parser.parse_args()
    for name in args.assets or cache_assets:
        print(f"{name}: {cache_assets[name]()}")


if __name__ == "__main__":
    main()
