# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Capture stock/proxy CAP grocery dynamics before any commanded physics step."""

from __future__ import annotations

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.integrations.cap_barrier.grocery_dynamics_calibration import (
    canonical_payload_bytes,
    capture_dynamics_payload,
    collect_runtime_versions,
    make_grocery_dynamics_calibration_environment,
    write_calibration_artifact,
)
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def _capture(device: str, *, mode: str) -> dict[str, object]:
    """Capture and tear down one calibration environment."""
    calibration = make_grocery_dynamics_calibration_environment(
        device=device,
        mode=mode,
    )
    payload = None
    try:
        payload = capture_dynamics_payload(
            calibration,
            mode=mode,
            device=device,
            runtime_versions=collect_runtime_versions(),
        )
    finally:
        calibration.close()
    assert payload is not None
    return payload


def _write_and_report(
    payload: dict[str, object],
    *,
    mode: str,
    output: str,
) -> None:
    """Write one artifact after Kit and PhysX have fully torn down."""
    destination, digest = write_calibration_artifact(output, payload)
    print(
        "CAP_GROCERY_DYNAMICS_CALIBRATION_JSON "
        + canonical_payload_bytes(payload).decode("utf-8").rstrip("\n"),
        flush=True,
    )
    print(
        "CAP_GROCERY_DYNAMICS_CALIBRATION_WRITTEN "
        f"mode={mode} bodies={len(payload['bodies'])} "
        f"sha256={digest} path={destination}",
        flush=True,
    )


def _run_cli(args_cli) -> None:
    """Capture under Kit, then write only after the app context exits."""
    args_cli.enable_cameras = False
    payload = None
    with SimulationAppContext(args_cli):
        payload = _capture(
            args_cli.device,
            mode=args_cli.mode,
        )
    assert payload is not None
    _write_and_report(
        payload,
        mode=args_cli.mode,
        output=args_cli.output,
    )


def main() -> None:
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--mode", choices=("stock", "proxy"), required=True)
    parser.add_argument("--output", required=True)
    args_cli = parser.parse_args()
    _run_cli(args_cli)


if __name__ == "__main__":
    main()
