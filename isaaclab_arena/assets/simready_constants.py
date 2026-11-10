# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SimReady names and URLs, kept free of asset imports."""

from __future__ import annotations

SIMREADY_USD_OBJECT_REGISTRY_NAME = "simready_usd_object"
"""Registry name for spawning a SimReady asset from a usd_path given in the spec's params."""

SIMREADY_SEARCH_REGISTRY_PREFIX = "simready_"
"""Prefix of the catalogue name a searched SimReady asset is registered under."""

ISAAC_SIMREADY_GA_S3_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/SimReady"
)
"""S3 root of the SimReady props that ship with Isaac Sim 6.0 GA."""

DEFAULT_SIMREADY_SERVICE_URL = "https://search.simready.omniverse.nvidia.com/"
"""Hosted SimReady search, used when the ``service`` source is selected."""

# SimReady GA props author collision/rigid APIs under the Physics=physics variant.
# Without this selection, Usd.Stage.Open sees geometry but no RigidBodyAPI, and
# PickAndPlace contact sensors fail with "No rigid body found".
SIMREADY_PHYSICS_VARIANTS: dict[str, str] = {"Physics": "physics"}

# TODO(xinjieyao, 2026-11-10): Select Newton=Enabled and PhysX=Disabled when Arena runs on the Newton backend.
SIMREADY_CENTRAL_PHYSX_VARIANTS: dict[str, str] = {"PhysX": "Enabled"}
"""Variant selections that enable the PhysX physics of SimReady Central assets.

Central packages have one variant set per simulator (PhysX, Newton and MuJoCo) and, unlike SimReady GA props,
no Physics set, so selecting Physics=physics only makes Isaac Lab warn that the set does not exist.
"""
