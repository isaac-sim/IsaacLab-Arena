# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
r"""Generate the full dynamic-scenes dataset used by the multi-scene training scripts.

Every entry in ``isaaclab_arena_datagen.environments.DATAGEN_ENVIRONMENTS``
is generated.  The on-disk layout under ``<output_root>/`` is driven by
each scene class's ``scene_metadata`` via
:func:`datagen.scene_metadata.get_dataset_subpath`:

::

    <output_root>/
    |-- one_object/
    |   |-- translation/<base_name>/dataset.h5         # 10 scenes
    |   |-- rotation/<base_name>/dataset.h5            # 10 scenes
    |   `-- translation_rotation/<base_name>/dataset.h5
    |-- two_objects/
    |   |-- no_collision/
    |   |   |-- translation/<base_name>/dataset.h5     # 10 scenes
    |   |   |-- rotation/<base_name>/dataset.h5
    |   |   `-- translation_rotation/<base_name>/dataset.h5
    |   `-- collision/
    |       |-- translation/<base_name>/dataset.h5     # 10 scenes
    |       `-- translation_rotation/<base_name>/dataset.h5
    `-- miscellaneous/                                 # SceneCategory.REFERENCE
        |-- ball_and_box/dataset.h5
        |-- single_ball/dataset.h5
        |-- single_cracker_box/dataset.h5
        `-- ball_box_robot/dataset.h5

Default frame counts:

- ``--base-num-steps`` (30): per :attr:`SceneCategory.REFERENCE` scene.
- ``--scene-num-steps`` (50): per non-reference scene across all motion
  variants.

Visualizations (per-scene PNG grid / MP4 / HTML plots) are skipped by
default to keep batch runs fast; pass ``--visualizations`` to re-enable
them for every generated scene.

Usage::

    python -m datagen.generate_all_scenes

    python -m datagen.generate_all_scenes \
        --output-root /data/my_runs \
        --base-num-steps 60 \
        --scene-num-steps 100 \
        --visualizations
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from isaaclab_arena_datagen.scene_metadata import (
    ALL_SCENE_METADATA,
    SceneCategory,
    get_dataset_subpath,
    get_registry_name,
)

_DEFAULT_OUTPUT_ROOT = Path("/datasets/dynamic_scenes")
_DEFAULT_BASE_NUM_STEPS = 30
_DEFAULT_SCENE_NUM_STEPS = 50


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate every scene in the canonical dynamic-scenes dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help=f"Dataset root directory (default: {_DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--base-num-steps",
        type=int,
        default=_DEFAULT_BASE_NUM_STEPS,
        help=f"Frames per SceneCategory.REFERENCE (miscellaneous) scene (default: {_DEFAULT_BASE_NUM_STEPS}).",
    )
    parser.add_argument(
        "--scene-num-steps",
        type=int,
        default=_DEFAULT_SCENE_NUM_STEPS,
        help=f"Frames per non-reference scene across motion variants (default: {_DEFAULT_SCENE_NUM_STEPS}).",
    )
    parser.add_argument(
        "--visualizations",
        action="store_true",
        help=(
            "Forward --visualizations to run_isaaclab_arena_datagen so it "
            "generates per-scene PNG grid, MP4, and HTML plots. Disabled by "
            "default to keep batch runs fast."
        ),
    )
    return parser


def _run_datagen(
    class_name: str,
    out_dir: Path,
    num_steps: int,
    visualizations: bool,
) -> None:
    """Invoke ``isaaclab_arena_datagen.run_datagen`` for a single scene.

    Batch generation is headless by design, so ``--headless`` is always
    forwarded; ``--visualizations`` is forwarded only when set.
    """
    print(f"=== {class_name} ({num_steps} frames) -> {out_dir} ===", flush=True)
    argv = [
        sys.executable,
        "-m",
        "isaaclab_arena_datagen.run_datagen",
        "--headless",
        *(["--visualizations"] if visualizations else []),
        "--output-dir",
        str(out_dir),
        "--num-steps",
        str(num_steps),
        class_name,
    ]
    subprocess.run(argv, check=True)


def main() -> None:
    """Generate every registered datagen scene into the canonical layout."""
    args = _build_parser().parse_args()
    output_root: Path = args.output_root

    for meta in ALL_SCENE_METADATA:
        num_steps = args.base_num_steps if meta.category is SceneCategory.REFERENCE else args.scene_num_steps
        _run_datagen(
            class_name=get_registry_name(meta),
            out_dir=output_root / get_dataset_subpath(meta),
            num_steps=num_steps,
            visualizations=args.visualizations,
        )

    print(f"All scenes generated under {output_root}")


if __name__ == "__main__":
    main()
