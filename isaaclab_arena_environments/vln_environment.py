# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""VLN benchmark environment builder.

This module defines the ``VLNBenchmarkEnvironment`` that integrates:
  - A Matterport 3D background scene.
  - The H1 humanoid embodiment configured for VLN.
  - The VLN navigation task with R2R episode management.

It follows the ``ExampleEnvironmentBase`` pattern used by all IsaacLab Arena
environments, so it plugs into the CLI and ``ArenaEnvBuilder`` seamlessly.

Usage (CLI)::

    python -m isaaclab_arena.evaluation.policy_runner \\
        --policy_type isaaclab_arena.policy.vln.vln_vlm_locomotion_policy.VlnVlmLocomotionPolicy \\
        --remote_host localhost --remote_port 5555 \\
        --num_episodes 10 \\
        VLN_Benchmark \\
        --usd_path /path/to/matterport.usd \\
        --r2r_dataset_path /path/to/vln_ce_isaac_v1.json.gz
"""

from __future__ import annotations

import argparse
import os

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


def _patch_usd_resolved_path():
    """Monkey-patch omni.usd functions that fail with ``Ar.ResolvedPath``.

    Isaac Sim 5.1.0's ``is_usd_crate_file`` and
    ``is_usd_crate_file_version_supported`` internally call
    ``Sdf.FileFormat.GetFileExtension`` and ``Usd.CrateInfo.Open`` with
    ``Ar.ResolvedPath`` objects, but those C++ APIs only accept ``str``.
    Wrapping the outer function does not help because the ``ResolvedPath``
    is created as a *local variable* inside the function body.

    The fix replaces both functions with clean re-implementations that
    convert every path to ``str`` before passing to C++ APIs.
    """
    try:
        import omni.usd
        import omni.usd._impl.utils as _usd_utils
        from pxr import Ar, Sdf, Usd

        def _fixed_is_usd_crate_file(filepath):
            ext = Sdf.FileFormat.GetFileExtension(str(filepath))
            return ext in ("usdc", "usd")

        def _fixed_is_usd_crate_file_version_supported(filepath):
            filepath = str(filepath)
            if not _fixed_is_usd_crate_file(filepath):
                return True
            try:
                resolved = Ar.GetResolver().Resolve(filepath)
                resolved_str = str(resolved) if resolved else filepath
                if not resolved_str:
                    return False
                crate_info = Usd.CrateInfo.Open(resolved_str)
                return crate_info is not None
            except Exception:
                return True

        _usd_utils.is_usd_crate_file = _fixed_is_usd_crate_file
        _usd_utils.is_usd_crate_file_version_supported = _fixed_is_usd_crate_file_version_supported
        omni.usd.is_usd_crate_file = _fixed_is_usd_crate_file
        omni.usd.is_usd_crate_file_version_supported = _fixed_is_usd_crate_file_version_supported
        print("[VLN] Patched omni.usd ResolvedPath compatibility (full replacement).")
    except Exception as e:
        print(f"[VLN] Warning: could not patch omni.usd: {e}")


class VLNBenchmarkEnvironment(ExampleEnvironmentBase):
    """IsaacLab Arena environment for VLN benchmarking."""

    name: str = "h1_vln_matterport"

    def get_env(self, args_cli: argparse.Namespace):
        """Build and return the VLN environment.

        Multi-env note:
            When ``num_envs > 1``, each env gets a full copy of the Matterport
            scene.  The ``env_spacing`` should be large enough that scenes don't
            overlap (Matterport houses are typically 20-50m wide).  The default
            is overridden to 100m if the user hasn't set it explicitly.
        """
        # Currently only num_envs=1 is supported.  Multi-env requires
        # per-env instruction tracking in the VLM server and dynamic scene
        # switching, which are not yet implemented.
        num_envs = getattr(args_cli, "num_envs", 1)
        if num_envs != 1:
            raise ValueError(
                f"VLN benchmark currently only supports num_envs=1 (got {num_envs}). "
                f"Multi-env support requires per-env VLM instruction tracking "
                f"and is planned for a future release."
            )

        # Fix Isaac Sim 5.1.0 USD ResolvedPath compatibility issue
        _patch_usd_resolved_path()

        # Matterport scenes are large — override default env_spacing if user
        # hasn't set a custom value (the Arena default is 30m, too small).
        if not hasattr(args_cli, "_env_spacing_set_by_user"):
            if getattr(args_cli, "num_envs", 1) > 1 and args_cli.env_spacing < 100.0:
                print(
                    f"[VLN] Overriding env_spacing from {args_cli.env_spacing}m to 100m "
                    f"for Matterport scenes (num_envs={args_cli.num_envs})."
                )
                args_cli.env_spacing = 100.0

        # Delayed imports — require simulation app to be running
        import isaaclab.sim as sim_utils
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        from isaaclab_arena.assets.matterport_background import MatterportBackground
        from isaaclab_arena.embodiments.h1.h1 import _DEFAULT_H1_CAMERA_OFFSET, _DEFAULT_H1_FOLLOW_CAMERA_OFFSET
        from isaaclab_arena.embodiments.h1.h1_vln import H1VlnEmbodiment
        from isaaclab_arena.tasks.vln_r2r_matterport_task import VlnR2rMatterportTask
        from isaaclab_arena.utils.pose import Pose

        # 1) Background: Matterport 3D scene
        ground_plane_z = None if getattr(args_cli, "disable_matterport_ground_plane", False) else 0.0
        background = MatterportBackground(
            usd_path=args_cli.usd_path,
            ground_plane_z=ground_plane_z,
            use_global_prim=getattr(args_cli, "use_global_matterport_prim", False),
            explicit_mesh_colliders=getattr(args_cli, "enable_matterport_child_mesh_colliders", False),
            mesh_collider_approximation=getattr(args_cli, "matterport_mesh_collider_type", "triangle"),
            collision_overlay_usd_path=getattr(args_cli, "matterport_collision_usd_path", None),
        )

        # 2) Embodiment: H1 humanoid with camera
        use_tiled = getattr(args_cli, "use_tiled_camera", False)
        enable_follow = not getattr(args_cli, "no_follow_camera", False)
        cam_res = getattr(args_cli, "camera_resolution", 512)
        head_camera_offset = _DEFAULT_H1_CAMERA_OFFSET
        follow_camera_offset = _DEFAULT_H1_FOLLOW_CAMERA_OFFSET
        if getattr(args_cli, "head_camera_offset_xyz", None) is not None:
            head_camera_offset = Pose(
                position_xyz=tuple(args_cli.head_camera_offset_xyz),
                rotation_wxyz=_DEFAULT_H1_CAMERA_OFFSET.rotation_wxyz,
            )
        if getattr(args_cli, "follow_camera_offset_xyz", None) is not None:
            follow_camera_offset = Pose(
                position_xyz=tuple(args_cli.follow_camera_offset_xyz),
                rotation_wxyz=_DEFAULT_H1_FOLLOW_CAMERA_OFFSET.rotation_wxyz,
            )
        if getattr(args_cli, "enable_height_scanner", False) and not getattr(args_cli, "use_global_matterport_prim", False):
            raise ValueError("Height scanner currently requires --use_global_matterport_prim so it can raycast /World/matterport.")
        embodiment = H1VlnEmbodiment(
            enable_cameras=True,
            camera_offset=head_camera_offset,
            use_tiled_camera=use_tiled,
            enable_follow_camera=enable_follow,
            follow_camera_offset=follow_camera_offset,
            camera_resolution=cam_res,
            enable_head_depth=getattr(args_cli, "enable_head_camera_depth", False),
            enable_height_scanner=getattr(args_cli, "enable_height_scanner", False),
            height_scanner_debug_vis=getattr(args_cli, "height_scanner_debug_vis", False),
        )

        # 3) Task: VLN navigation with R2R episodes
        # Extract scene_id from the USD path to filter episodes.
        usd_scene_id = os.path.splitext(os.path.basename(args_cli.usd_path))[0]

        episode_indices = None
        if hasattr(args_cli, "episode_start") and args_cli.episode_start is not None:
            end = getattr(args_cli, "episode_end", args_cli.episode_start + 1)
            episode_indices = list(range(args_cli.episode_start, end))

        task = VlnR2rMatterportTask(
            robot=embodiment,
            r2r_dataset_path=args_cli.r2r_dataset_path,
            episode_indices=episode_indices,
            episode_length_s=getattr(args_cli, "episode_length_s", 60.0),
            success_radius=getattr(args_cli, "success_radius", 3.0),
            scene_filter=usd_scene_id,
            robot_root_height_offset=getattr(args_cli, "robot_root_height_offset", 1.0),
        )

        # 4) Scene: Matterport background
        scene = Scene(assets=[background])

        # 5) Simulation parameters callback
        # These MUST match the low-level locomotion policy training config.
        # Default values come from NaVILA-Bench:
        #   h1_matterport_base_cfg.py -> H1MatterportBaseCfg.__post_init__()
        sim_dt = getattr(args_cli, "sim_dt", 0.005)
        decimation = getattr(args_cli, "sim_decimation", 4)

        def vln_sim_cfg_callback(env_cfg):
            env_cfg.sim.dt = sim_dt                # 200 Hz physics
            env_cfg.decimation = decimation         # 50 Hz control (200/4)
            env_cfg.sim.render_interval = decimation
            env_cfg.sim.disable_contact_processing = True
            env_cfg.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                friction_combine_mode="max",
                restitution_combine_mode="max",
            )
            return env_cfg

        # 6) Compose the Arena environment
        arena_env = IsaacLabArenaEnvironment(
            name=self.name,
            scene=scene,
            embodiment=embodiment,
            task=task,
            env_cfg_callback=vln_sim_cfg_callback,
        )
        return arena_env

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Add VLN-specific CLI arguments."""
        group = parser.add_argument_group("VLN Benchmark", "VLN benchmark environment arguments")
        group.add_argument(
            "--usd_path",
            type=str,
            required=True,
            help="Path to the Matterport USD scene file.",
        )
        group.add_argument(
            "--r2r_dataset_path",
            type=str,
            required=True,
            help="Path to the R2R VLN dataset (e.g. vln_ce_isaac_v1.json.gz).",
        )
        group.add_argument(
            "--episode_start",
            type=int,
            default=None,
            help="Starting episode index (inclusive).  If None, use all episodes.",
        )
        group.add_argument(
            "--episode_end",
            type=int,
            default=None,
            help="Ending episode index (exclusive).  Used with --episode_start.",
        )
        group.add_argument(
            "--episode_length_s",
            type=float,
            default=60.0,
            help="Maximum episode duration in seconds (default: 60).",
        )
        group.add_argument(
            "--success_radius",
            type=float,
            default=3.0,
            help="Distance threshold for goal success (default: 3.0m).",
        )
        group.add_argument(
            "--use_tiled_camera",
            action="store_true",
            default=False,
            help="Use TiledCamera for parallel evaluation (default: False).",
        )
        group.add_argument(
            "--no_follow_camera",
            action="store_true",
            default=False,
            help="Disable the third-person follow camera (default: enabled).",
        )
        group.add_argument(
            "--camera_resolution",
            type=int,
            default=512,
            help="Camera resolution in pixels (default: 512, use 1024 for high-quality demo).",
        )
        group.add_argument(
            "--head_camera_offset_xyz",
            type=float,
            nargs=3,
            default=None,
            metavar=("X", "Y", "Z"),
            help="Override head camera XYZ offset in pelvis frame.",
        )
        group.add_argument(
            "--follow_camera_offset_xyz",
            type=float,
            nargs=3,
            default=None,
            metavar=("X", "Y", "Z"),
            help="Override follow camera XYZ offset in pelvis frame.",
        )
        group.add_argument(
            "--enable_head_camera_depth",
            action="store_true",
            default=False,
            help="Expose head camera depth alongside RGB for debugging; not sent to the VLM by default.",
        )
        group.add_argument(
            "--enable_height_scanner",
            action="store_true",
            default=False,
            help="Enable a NaVILA-Bench-style height scanner sensor for debugging or future experiments.",
        )
        group.add_argument(
            "--height_scanner_debug_vis",
            action="store_true",
            default=False,
            help="Visualize height-scanner rays when the height scanner is enabled.",
        )
        group.add_argument(
            "--disable_matterport_ground_plane",
            action="store_true",
            default=False,
            help="Disable the fallback invisible ground plane and rely on Matterport mesh collision only.",
        )
        group.add_argument(
            "--use_global_matterport_prim",
            action="store_true",
            default=False,
            help="Spawn Matterport at /World/matterport with collision_group=-1 for collision experiments.",
        )
        group.add_argument(
            "--enable_matterport_child_mesh_colliders",
            action="store_true",
            default=False,
            help="Explicitly apply collision and mesh-collider schemas to descendant Matterport Mesh prims.",
        )
        group.add_argument(
            "--matterport_mesh_collider_type",
            type=str,
            default="triangle",
            choices=("triangle", "sdf", "convex_decomposition"),
            help="Approximation used when explicit Matterport child-mesh colliders are enabled.",
        )
        group.add_argument(
            "--matterport_collision_usd_path",
            type=str,
            default=None,
            help="Optional hidden collision-only USD layered under the visual Matterport scene.",
        )
        group.add_argument(
            "--robot_root_height_offset",
            type=float,
            default=1.0,
            help="Added to dataset start_position.z when resetting the robot root (default tuned for H1 pelvis root).",
        )

        # Simulation parameters — must match the low-level policy training config
        sim_group = parser.add_argument_group(
            "Simulation", "Physics simulation parameters (must match low-level policy training)"
        )
        sim_group.add_argument(
            "--sim_dt",
            type=float,
            default=0.005,
            help="Physics simulation timestep in seconds (default: 0.005 = 200Hz).",
        )
        sim_group.add_argument(
            "--sim_decimation",
            type=int,
            default=4,
            help="Number of physics steps per policy step (default: 4, giving 50Hz control).",
        )

