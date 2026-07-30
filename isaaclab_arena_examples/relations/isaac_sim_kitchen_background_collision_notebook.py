# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Example: place objects on the robocasa kitchen counter while avoiding the background mesh.

The kitchen is loaded as one passive background. In MESH collision mode, the builder uses one
world-frame aggregate background mesh and the relation solver keeps placed objects out of it.

Run standalone from the repo root inside the container (pass --viz kit to open the Kit viewer;
omitting it runs headless, so nothing shows):

    /isaac-sim/python.sh \\
        isaaclab_arena_examples/relations/isaac_sim_kitchen_background_collision_notebook.py \\
        --viz kit --view_steps 0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pinocchio  # noqa: F401

from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser

# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false


if TYPE_CHECKING:
    from isaacsim import SimulationApp

_KITCHEN = "lightwheel_robocasa_kitchen"
_COUNTER_PRIM = "{ENV_REGEX_NS}/" + _KITCHEN + "/counter_right_main_group/top_geometry"
_DEMO_SOLVER_MAX_ITERS = 250
_DEMO_NUM_SPHERES = 8


def run_kitchen_background_collision_demo(simulation_app, view_steps: int = 0, args_cli=None) -> list[str]:
    """Solve the counter layout against the aggregate background mesh, then idle a viewer.

    Args:
        simulation_app: The already-launched Isaac Sim application; the viewer runs until it stops.
        view_steps: Number of steps to run (0 = until the viewer window is closed).
        args_cli: Parsed CLI namespace. When None, defaults are parsed.

    Returns:
        The names of the background collision obstacles used by the builder.
    """
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.passive_collision_objects import get_passive_collision_objects
    from isaaclab_arena.relations.relation_solver_params import CollisionMode, RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.scene.scene import Scene

    if args_cli is None:
        args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    builder_cfg.solve_relations = True

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

    scene = Scene(assets=[background, counter, *objects, light])
    collision_objects = get_passive_collision_objects(scene.assets.values(), include_background=True)
    collision_names = [c.name for c in collision_objects]
    print(f"Using background collision obstacles: {collision_names}", flush=True)

    placer_params = ObjectPlacerParams(
        max_placement_attempts=10,
        min_unique_layouts_per_env=1,
        allow_best_loss_fallbacks=False,
        resolve_on_reset=False,
        random_yaw_init=True,
        solver_params=RelationSolverParams(
            collision_mode=CollisionMode.MESH,
            max_iters=_DEMO_SOLVER_MAX_ITERS,
            num_spheres=_DEMO_NUM_SPHERES,
            verbose=False,
            save_position_history=False,
        ),
    )
    env = ArenaEnvBuilder(
        IsaacLabArenaEnvironment(name="kitchen_background_collision", scene=scene, placer_params=placer_params),
        builder_cfg,
    ).make_registered()

    # reset() applies the relation-solved layout (position + yaw) via the reset event terms.
    env.reset()

    # Render without stepping physics so the solved placement stays on screen.
    step = 0
    while simulation_app.is_running():
        if view_steps and step >= view_steps:
            break
        env.unwrapped.sim.render()
        step += 1

    return collision_names


def smoke_test_kitchen_background_collision(simulation_app: SimulationApp) -> bool:
    """Run the demo and assert the aggregate background mesh is used."""
    collision_names = run_kitchen_background_collision_demo(simulation_app, view_steps=2)
    assert collision_names == ["fixed_collision_mesh"], collision_names
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
