# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SimReady names and URLs, kept free of asset imports.

Spec inference and the asset search need these values but must not pull in the asset classes:
importing those loads pxr, and a pxr import before SimulationApp starts breaks the unit tests.
"""

from __future__ import annotations

SIMREADY_USD_OBJECT_REGISTRY_NAME = "simready_usd_object"
"""Registry name a generated spec uses to spawn a searched SimReady asset."""

ISAAC_SIMREADY_GA_S3_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/SimReady"
)

DEFAULT_SIMREADY_SERVICE_URL = "https://search.simready.omniverse.nvidia.com/"
"""Hosted SimReady search, used when the ``service`` source is selected."""

# SimReady GA props author collision/rigid APIs under the Physics=physics variant.
# Without this selection, Usd.Stage.Open sees geometry but no RigidBodyAPI, and
# PickAndPlace contact sensors fail with "No rigid body found".
SIMREADY_PHYSICS_VARIANTS: dict[str, str] = {"Physics": "physics"}
