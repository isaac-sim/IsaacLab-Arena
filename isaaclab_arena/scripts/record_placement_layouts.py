# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Solve placements, filter them with physics, and record complete settled poses."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab_arena.offline_placement.recording import PlacementRecordingCfg, record_placements


def main() -> None:
    """Run offline recording with Hydra settings and Isaac Lab launcher flags."""
    from hydra import compose, initialize
    from hydra.core.config_store import ConfigStore
    from isaaclab.app import AppLauncher
    from omegaconf import OmegaConf

    from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    launcher_args, overrides = parser.parse_known_args()
    assert_hydra_overrides(overrides, parser)
    ConfigStore.instance().store(name="placement_recording", node=PlacementRecordingCfg)
    with initialize(version_base=None, config_path=None):
        cfg = OmegaConf.to_object(compose(config_name="placement_recording", overrides=overrides))
    assert not Path(cfg.output).exists(), f"Output already exists: {cfg.output}"
    with SimulationAppContext(launcher_args):
        record_placements(cfg, device=launcher_args.device)


if __name__ == "__main__":
    main()
