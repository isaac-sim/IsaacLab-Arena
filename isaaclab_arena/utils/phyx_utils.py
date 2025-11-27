# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pxr import PhysxSchema, Usd


def add_contact_report(prim: Usd.Prim) -> None:
    """Add a contact report API to a prim.

    Args:
        prim: The prim to add the contact report API to.
    """
    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
    cr_api.CreateThresholdAttr().Set(0)
