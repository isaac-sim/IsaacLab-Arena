# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Overlay an embodiment's placement bounding box and collision mesh on the robot as spawned.

Run with the GUI and inspect by eye: the green wireframe is ``EmbodimentBase.get_bounding_box()`` and
the magenta points are ``get_collision_mesh()``, both posed at the robot's configured initial joint
positions and drawn through the spawned robot prim's transform. At ``num_steps=0`` the arm has not yet
sagged under gravity, so the overlay should hug the robot; stepping lets the joints settle a few
milliradians away and the overlay drifts by millimetres.

Passing ``spec_yaml`` builds a full env graph instead of an empty ground plane, which shows the same
geometry where the relation solver actually placed the robot — the overlay then also reveals whether a
yaw relation rotated the placement geometry along with the robot. The spec YAML may also be given as
the first command-line argument, alongside the usual Isaac Lab flags such as ``--headless``.
"""

# %%
from __future__ import annotations

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher

# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false


_args, _positional_args = get_isaaclab_arena_cli_parser().parse_known_args()
if not _args.headless:
    # A bare AppLauncher(headless=False) starts Kit without a window: Isaac Lab 3.0 only builds the
    # viewport when a visualizer is requested, so ask for "kit" the way Arena's test harness does.
    _args.visualizer = ["kit"]
print(f"Launching simulation app (headless={_args.headless})")
simulation_app = get_app_launcher(_args).app

# %%

MESH_POINT_BUDGET = 7000
"""Mesh vertices are subsampled to this many debug points to keep the viewport responsive."""

BBOX_COLOR = (0.0, 1.0, 0.2, 1.0)
MESH_COLOR = (1.0, 0.2, 0.9, 0.9)


def _build_environment(embodiment_name: str, spec_yaml: str | None):
    """Return an environment and its embodiment, either from an env graph spec or an empty scene."""
    if spec_yaml is not None:
        from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import build_arena_env_from_graph_spec
        from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

        environment = build_arena_env_from_graph_spec(ArenaEnvGraphSpec.from_yaml(spec_yaml))
        return environment, environment.embodiment

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    asset_registry = AssetRegistry()
    embodiment = asset_registry.get_asset_by_name(embodiment_name)()
    scene_assets = [asset_registry.get_asset_by_name(name)() for name in ("ground_plane", "light")]
    environment = IsaacLabArenaEnvironment(
        name="visualize_embodiment_placement_geometry",
        embodiment=embodiment,
        scene=Scene(assets=scene_assets),
    )
    return environment, embodiment


def visualize_embodiment_placement_geometry(
    embodiment_name: str = "droid_abs_joint_pos",
    spec_yaml: str | None = None,
    num_steps: int = 0,
    mesh_point_budget: int = MESH_POINT_BUDGET,
    hold_open: bool = True,
):
    """Spawn an embodiment and draw its placement bbox and collision mesh over it.

    Args:
        embodiment_name: Registry name of the embodiment to inspect. Ignored when ``spec_yaml`` is set,
            which carries its own embodiment.
        spec_yaml: Path to an env graph spec YAML to build instead of an empty ground-plane scene.
        num_steps: Simulation steps to run before drawing. Zero keeps the joints exactly at their
            configured positions, which is what the placement geometry is posed at.
        mesh_point_budget: Approximate number of mesh vertices to draw.
        hold_open: Keep rendering until the window closes. Set False to exit after drawing, which is
            what a headless check of the spec wants.
    """
    import numpy as np
    import torch

    import omni.usd
    from pxr import Usd, UsdGeom

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder, ArenaEnvBuilderCfg
    from isaaclab_arena.utils.isaac_sim_debug_draw import IsaacSimDebugDraw

    environment, embodiment = _build_environment(embodiment_name, spec_yaml)

    # Placement geometry, in the embodiment's local (default-prim) frame.
    bbox = embodiment.get_bounding_box()
    mesh = embodiment.get_collision_mesh()
    local_min = np.asarray(bbox.min_point.numpy()[0], dtype=np.float64)
    local_max = np.asarray(bbox.max_point.numpy()[0], dtype=np.float64)
    print(f"local bbox min  = {np.round(local_min, 4)}")
    print(f"local bbox max  = {np.round(local_max, 4)}")
    print(f"local bbox size = {np.round(local_max - local_min, 4)}")
    print(f"local mesh size = {np.round(mesh.extents, 4)} ({len(mesh.vertices)} verts)")
    # A positive difference is the analytic (non-mesh) geometry the bbox covers but the mesh cannot.
    print(f"bbox minus mesh = {np.round((local_max - local_min) - mesh.extents, 4)}")

    env = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg()).make_registered()
    env.reset()

    for _ in range(num_steps):
        with torch.inference_mode():
            env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))

    # The local frame is the spawned robot prim's frame, so draw through its world transform.
    stage = omni.usd.get_context().get_stage()
    robot_prim_path = env.unwrapped.scene["robot"].cfg.prim_path.replace("env_.*", "env_0")
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    assert robot_prim, f"no robot prim at {robot_prim_path}"
    robot_world = np.array(
        UsdGeom.Xformable(robot_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64
    )

    def to_world(points: np.ndarray) -> np.ndarray:
        return (np.hstack([points, np.ones((len(points), 1))]) @ robot_world)[:, :3]

    corners = to_world(
        np.array([
            [x, y, z]
            for x in (local_min[0], local_max[0])
            for y in (local_min[1], local_max[1])
            for z in (local_min[2], local_max[2])
        ])
    )
    # Corner order above is (x, y, z) bit-major, so edges join indices differing in exactly one bit.
    edges = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if i < (i ^ bit)]

    debug_draw = IsaacSimDebugDraw()
    debug_draw.clear()
    debug_draw.draw_line_segments(
        [tuple(corners[i]) for i, _ in edges],
        [tuple(corners[j]) for _, j in edges],
        color=BBOX_COLOR,
        thickness=5.0,
    )

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) > mesh_point_budget:
        vertices = vertices[:: len(vertices) // mesh_point_budget + 1]
    world_vertices = to_world(vertices)
    debug_draw.draw_points([tuple(point) for point in world_vertices], color=MESH_COLOR)
    print(f"\ndrew {len(edges)} bbox edges (green) and {len(world_vertices)} mesh points (magenta)")
    print(f"robot world position = {np.round(robot_world[3, :3], 4)}")
    if not hold_open:
        return
    print("Inspect the overlay in the viewport. Close the window to exit.")

    # Render without stepping so the viewport stays interactive while the joints stay exactly where
    # the placement geometry was posed. Debug draws persist across frames, so one draw is enough.
    while simulation_app.is_running():
        env.unwrapped.sim.render()


# %%
visualize_embodiment_placement_geometry(
    spec_yaml=_positional_args[0] if _positional_args else None,
    hold_open=not _args.headless,
)

# %%
