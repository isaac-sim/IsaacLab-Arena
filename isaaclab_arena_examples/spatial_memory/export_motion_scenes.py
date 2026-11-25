# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export standalone OpenUSD motion scenes and two robot-free RTX views per scene."""

import argparse
from pathlib import Path

from PIL import Image


def render_views(app, path):
    """Capture both authored camera views of an exported scene without stepping physics."""
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd

    context = omni.usd.get_context()
    assert context.open_stage(str(path)), f"Could not open {path}"
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    timeline.set_current_time(0)
    for view in ("Overview", "Approach"):
        if view == "Approach":
            stage = context.get_stage()
            timeline.set_current_time(stage.GetEndTimeCode() / 2 / stage.GetTimeCodesPerSecond())
        product = rep.create.render_product(f"/World/{view}", (1200, 800))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(product)
        try:
            for _ in range(24):
                app.update()
            pixels = rgb.get_data()
            assert pixels.shape[:2] == (800, 1200), f"Empty camera {view} in {path}"
            Image.fromarray(pixels[:, :, :3]).save(path.with_name(f"{path.stem}_{view.lower()}.png"))
        finally:
            rgb.detach(product)
            product.destroy()


def main():
    from isaaclab.app import AppLauncher

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", help="Subset of scene names from the motion scene catalog")
    parser.add_argument("--seed", type=int, default=3847)
    parser.add_argument("--usd-only", action="store_true", help="Skip the two RTX camera views")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = not args.usd_only
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with SimulationAppContext(args) as context:
        from isaaclab_arena_environments.spatial_memory.workcells.catalog import SCENES, build_stage

        names = args.scenes or SCENES
        unknown = set(names) - SCENES.keys()
        if unknown:
            raise ValueError(f"Unknown scenes: {sorted(unknown)}")
        for name in names:
            stage = build_stage(name, args.seed)
            path = output / f"{name}.usda"
            stage.GetRootLayer().Export(str(path))
            if not args.usd_only:
                render_views(context.app_launcher.app, path)
            print(f"Exported {name}", flush=True)


if __name__ == "__main__":
    main()
