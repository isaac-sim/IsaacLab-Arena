# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.sim.utils import safe_set_attribute_on_usd_prim
from pxr import Usd


def add_contact_report(prim: Usd.Prim) -> None:
    """Add a contact report API to a prim.

    Args:
        prim: The prim to add the contact report API to.
    """
    # Only Isaac Sim ships PhysxSchema's typed bindings.
    prim.AddAppliedSchema("PhysxContactReportAPI")
    safe_set_attribute_on_usd_prim(prim, "physxContactReport:threshold", 0.0, camel_case=False)
