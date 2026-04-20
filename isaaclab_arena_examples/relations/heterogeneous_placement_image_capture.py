# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%
from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Example: heterogeneous object placement across parallel environments.

Demonstrates how different environments can contain different objects
(via ``RigidObjectSet``) with the relation solver computing a valid,
collision-free layout for each variant automatically.  Viewport images
are captured after reset and saved to disk so you can visually compare
the per-environment layouts.

Run inside the Docker container::

    /isaac-sim/python.sh isaaclab_arena_examples/relations/heterogeneous_placement_image_capture.py \\
        --num_envs 4 --enable_cameras --output_dir /tmp/placement_captures
"""

from isaaclab.app import AppLauncher

AppLauncher()

# %%

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaacsim import SimulationApp


def run_heterogeneous_placement_capture(
    num_envs: int = 4,
    output_dir: str = "/tmp/placement_captures",
    num_resets: int = 3,
    warmup_steps: int = 10,
) -> list[str]:
    """Spawn heterogeneous objects on a maple table and capture viewport images.

    Args:
        num_envs: Number of parallel environments, each potentially with
            a different object variant and layout.
        output_dir: Directory where captured PNG images are saved.
        num_resets: Number of reset cycles to capture (each reset yields
            a new random layout drawn from the placement pool).
        warmup_steps: Simulation steps to run after each reset before
            capturing, so objects settle and rendering converges.

    Returns:
        List of file paths to the saved images.
    """
    import numpy as np
    import os
    import torch

    from isaaclab.envs.common import ViewerCfg

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.scene.scene import Scene

    os.makedirs(output_dir, exist_ok=True)

    asset_registry = AssetRegistry()

    # -- Background and table anchor --
    background = asset_registry.get_asset_by_name("maple_table_robolab")()
    light = asset_registry.get_asset_by_name("light")()

    table_reference = ObjectReference(
        name="table",
        prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
        parent_asset=background,
    )
    table_reference.add_relation(IsAnchor())

    # -- Heterogeneous pick-up object (different object per env) --
    variant_names = ["cracker_box", "mug", "tomato_soup_can", "sugar_box"]
    variant_objects = [asset_registry.get_asset_by_name(n)() for n in variant_names]
    hetero_object = RigidObjectSet(name="hetero_pick", objects=variant_objects)
    hetero_object.add_relation(On(table_reference))

    # -- Extra common objects on the table --
    extra_names = ["dex_cube", "red_container"]
    extras = []
    for name in extra_names:
        obj = asset_registry.get_asset_by_name(name)()
        obj.add_relation(On(table_reference))
        extras.append(obj)

    # -- Scene and environment --
    scene = Scene(assets=[background, light, table_reference, hetero_object, *extras])

    def _set_viewer_cfg(env_cfg):
        env_cfg.viewer = ViewerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.2, 0.0, 0.0))
        return env_cfg

    arena_env = IsaacLabArenaEnvironment(
        name="heterogeneous_placement_demo",
        scene=scene,
        env_cfg_callback=_set_viewer_cfg,
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", str(num_envs), "--enable_cameras"])

    env_builder = ArenaEnvBuilder(arena_env, args_cli)
    env = env_builder.make_registered(render_mode="rgb_array")

    saved_paths: list[str] = []

    for reset_idx in range(num_resets):
        env.reset()

        for _ in range(warmup_steps):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        frame = env.render()
        if frame is not None:
            from PIL import Image

            img = Image.fromarray(np.asarray(frame))
            path = os.path.join(output_dir, f"layout_reset_{reset_idx:02d}.png")
            img.save(path)
            saved_paths.append(path)
            print(f"Saved: {path}  (shape={frame.shape})")

    env.close()
    print(f"\nCaptured {len(saved_paths)} images in {output_dir}")
    return saved_paths


def smoke_test_heterogeneous_placement_capture(simulation_app: SimulationApp) -> bool:
    """Minimal smoke test: 2 envs, 1 reset, 2 warmup steps."""
    paths = run_heterogeneous_placement_capture(num_envs=2, num_resets=1, warmup_steps=2)
    return len(paths) == 1


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Heterogeneous placement image capture")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="/tmp/placement_captures")
    parser.add_argument("--num_resets", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=10)
    args, _ = parser.parse_known_args()

    run_heterogeneous_placement_capture(
        num_envs=args.num_envs,
        output_dir=args.output_dir,
        num_resets=args.num_resets,
        warmup_steps=args.warmup_steps,
    )

# %%
