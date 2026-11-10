# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare the cached workbench USD assets used by the canonical scene."""

import argparse


def main() -> None:
    """Prepare selected workbench assets through the normal runtime cache."""
    from isaaclab_arena_environments.workbench_cleanup import assets

    prepare_assets = {
        "workbench": assets.WorkbenchCleanupWorkbench.processor.process,
        "white_bin": assets.WorkbenchCleanupWhiteBin.usd_source.to_local_usd_path,
        "socket_set": assets.WorkbenchCleanupSocketSet.processor.process,
        "oscilloscope": assets.WorkbenchCleanupOscilloscope.processor.process,
        "slotted_screwdriver": assets.WorkbenchCleanupSlottedScrewdriver.usd_source.to_local_usd_path,
        "small_wrench": assets.WorkbenchCleanupSmallWrench.usd_source.to_local_usd_path,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="*", choices=prepare_assets)
    args = parser.parse_args()
    for name in args.assets or prepare_assets:
        print(f"{name}: {prepare_assets[name]()}")


if __name__ == "__main__":
    main()
