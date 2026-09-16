# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Extract a variation condition overlay from episode JSONL results."""

from __future__ import annotations

import argparse
import yaml
from pathlib import Path

from isaaclab_arena.variations.episode_conditions import extract_overlay_from_episode_results, overlay_to_yaml_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract variation condition overlay YAML from episode JSONL.")
    parser.add_argument(
        "--episode-results",
        type=Path,
        required=True,
        help="Path to episode_results_rebuildN.jsonl from a recorded run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination YAML path for the condition overlay.",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=None,
        help="Optional experiment YAML path stored in overlay provenance.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Optional run name stored in overlay provenance.",
    )
    args = parser.parse_args()

    source: dict[str, str] = {"episode_results": str(args.episode_results)}
    if args.experiment_config is not None:
        source["experiment"] = str(args.experiment_config)
    if args.run is not None:
        source["run"] = args.run

    overlay = extract_overlay_from_episode_results(args.episode_results, source=source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(overlay_to_yaml_dict(overlay), sort_keys=False))
    print(f"Wrote {overlay.num_conditions} conditions to {args.output}")


if __name__ == "__main__":
    main()
