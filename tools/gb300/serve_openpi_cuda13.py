# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Launch OpenPI with the temporary JAX 0.7 layout compatibility shim."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, default=Path("/app"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-config", default="pi05_droid_jointpos_polaris")
    parser.add_argument("--policy-dir", default="gs://openpi-assets-simeval/pi05_droid_jointpos")
    return parser.parse_args()


def _patch_jax_layout_api() -> None:
    """Expose the JAX layout names expected by OpenPI's pinned Orbax version."""
    import jax.experimental.layout as layout

    assert hasattr(layout, "Format"), "This shim requires the JAX 0.7 layout API"
    device_local_layout = layout.Layout
    layout.DeviceLocalLayout = device_local_layout
    layout.Layout = layout.Format


def main() -> None:
    args = _parse_args()
    openpi_root = args.openpi_root.resolve()
    server_script = openpi_root / "scripts" / "serve_policy.py"
    assert server_script.is_file(), f"OpenPI server script not found: {server_script}"
    assert 1 <= args.port <= 65535, f"Invalid port: {args.port}"

    _patch_jax_layout_api()
    os.chdir(openpi_root)
    sys.argv = [
        str(server_script),
        f"--port={args.port}",
        "policy:checkpoint",
        f"--policy.config={args.policy_config}",
        f"--policy.dir={args.policy_dir}",
    ]
    runpy.run_path(str(server_script), run_name="__main__")


if __name__ == "__main__":
    main()
