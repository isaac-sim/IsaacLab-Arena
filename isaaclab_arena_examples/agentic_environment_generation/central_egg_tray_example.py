# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pinned SimReady Central egg-and-tray example shared by CLI and GUI."""

CENTRAL_EGG_TRAY_PROMPT = (
    "Create an environment named central_egg_tray. Franka picks up the SimReady egg from the maple table "
    "and places it into the SimReady egg tray. Include the SimReady nut and SimReady pizza as separate "
    "distractor objects on the table. Use all four supplied SimReady assets: simready_egg, simready_egg_tray, "
    "simready_nut, simready_pizza. Keep the egg outside the tray initially. "
    "Use PickAndPlaceTask for the egg and tray. Do not use object sets."
)

_DATASET_REVISION_URL = (
    "https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library/resolve/"
    "69a87329036cec0440403351c12230970c4e67d8"
)

CENTRAL_EGG_TRAY_ASSET_URLS = tuple(
    f"{_DATASET_REVISION_URL}/{asset_id}/simready_usd/{asset_id}.usd"
    for asset_id in ("D00290", "D00287", "D00554", "D00802")
)
