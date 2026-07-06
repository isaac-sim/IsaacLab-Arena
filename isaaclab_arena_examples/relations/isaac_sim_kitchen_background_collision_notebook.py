# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Example: place objects on the robocasa kitchen counter while avoiding background fixtures.

Background fixtures whose bounding box intrudes the region above the counter are discovered with
build_placement_region + find_background_colliders (spatial culling, no name denylist) and added to
the scene as relation-free ObjectReferences. ArenaEnvBuilder's automatic relation solving then picks
them up as fixed obstacles (via Scene.get_collision_objects) and applies the solved layout -- both
position and random yaw -- through reset event terms, so the placement persists in simulation.

Run standalone from the repo root inside the container (pass --viz kit to open the Kit viewer;
omitting it runs headless, so nothing shows):

    /isaac-sim/python.sh \\
        isaaclab_arena_examples/relations/isaac_sim_kitchen_background_collision_notebook.py \\
        --viz kit --enable_cameras --view_steps 200
"""

from typing import TYPE_CHECKING

import pinocchio  # noqa: F401

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

if TYPE_CHECKING:
    from isaacsim import SimulationApp

_KITCHEN = "lightwheel_robocasa_kitchen"
_COUNTER_PRIM = "{ENV_REGEX_NS}/" + _KITCHEN + "/counter_main_main_group"


def run_kitchen_background_collision_demo(simulation_app, view_steps: int = 0, args_cli=None) -> list[str]:
    """Solve the counter layout with discovered fixtures as obstacles, then idle a viewer.

    Args:
        simulation_app: The already-launched Isaac Sim application; the viewer runs until it stops.
        view_steps: Number of steps to run (0 = until the viewer window is closed).
        args_cli: Parsed CLI namespace. When None, defaults are parsed (used by the smoke test).

    Returns:
        The names of the background fixtures discovered as collision obstacles.
    """
    import torch

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.scene.background_colliders import build_placement_region, find_background_colliders
    from isaaclab_arena.scene.scene import Scene

    if args_cli is None:
        args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.solve_relations = True  # ArenaEnvBuilder solves + applies placement via reset events
    args_cli.random_yaw_init = True  # rotate non-anchor objects about Z (persists via the reset event)

    registry = AssetRegistry()
    background = registry.get_asset_by_name(_KITCHEN)()
    light = registry.get_asset_by_name("light")()
    light.set_intensity(3000.0)  # brighten the room (default DomeLight intensity is 500)

    counter = ObjectReference(name="counter", prim_path=_COUNTER_PRIM, parent_asset=background)
    counter.add_relation(IsAnchor())

    # Placeable objects sit On the counter; the builder's relation solver spreads them. Do not set an
    # initial pose here -- that would create a per-object reset event that conflicts with relation solving.
    objects = []
    for name in ["cracker_box", "sugar_box", "tomato_soup_can", "mustard_bottle", "mug"]:
        obj = registry.get_asset_by_name(name)()
        obj.add_relation(On(counter, clearance_m=0.02))
        objects.append(obj)

    # Fixtures intruding the region above the counter become fixed obstacles: the builder's
    # automatic solve picks them up via Scene.get_collision_objects and applies the layout on reset.
    region = build_placement_region([counter], objects)
    collision_objects = find_background_colliders(background, region, anchors=[counter])
    discovered_names = [c.name for c in collision_objects]
    print(f"Discovered background obstacles: {discovered_names}", flush=True)

    scene = Scene(assets=[background, counter, *objects, *collision_objects, light])
    env = ArenaEnvBuilder(
        IsaacLabArenaEnvironment(name="kitchen_background_collision", scene=scene),
        args_cli,
    ).make_registered()

    # reset() applies the relation-solved layout (position + yaw) via the reset event terms.
    env.reset()

    # Idle with zero actions so the solved layout stays on screen.
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    step = 0
    while simulation_app.is_running():
        if view_steps and step >= view_steps:
            break
        with torch.inference_mode():
            env.step(action)
        step += 1

    return discovered_names


def smoke_test_kitchen_background_collision(simulation_app: SimulationApp) -> bool:
    """Run the demo and assert discovery works on the real asset: fixtures found, including the sink."""
    discovered = run_kitchen_background_collision_demo(simulation_app, view_steps=2)
    assert discovered, "Expected background fixtures to be discovered on the kitchen counter, got none."
    assert "sink_main_group" in discovered, f"Expected the sink among discovered fixtures, got {discovered}."
    return True


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    # Parse CLI and launch the app before importing Isaac Sim modules.
    _parser = get_isaaclab_arena_cli_parser()
    _parser.add_argument("--view_steps", type=int, default=0, help="Steps to run (0 = until the viewer is closed).")
    _args_cli = _parser.parse_args()
    _simulation_app = AppLauncher(_args_cli).app
    run_kitchen_background_collision_demo(_simulation_app, view_steps=_args_cli.view_steps, args_cli=_args_cli)
    _simulation_app.close()
