# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

# World-coordinate reach box for the table objects, in front of the droid base (near the origin). On()
# supplies the surface z; this only bounds x/y so the relation solver keeps every object within the arm's
# top-down reach (the stock On() alone can land an object anywhere on the table, including out of reach).
# Placement is validated in the built scene (objects land in a reachable row, well framed); the arm-IK
# reachability of the bounds is NOT yet confirmed (pending a GaP run). Only applied under --gap_profile.
_REACH_BOX = dict(x_min=0.05, x_max=0.45, y_min=-0.25, y_max=0.25)

# Opt-in dev override for the two unpromoted srl_robolab assets from PR #786 (the Maple table and the DROID
# stand): both 404 on the production Nucleus but 200 on the Isaac staging bucket (CI used staging). The
# override is a targeted production->staging HOST swap for ONLY those two files; everything else (robolab
# objects, etc.) stays on production. Off by default (production, documented promotion blocker); never a
# silent fallback. Enable with --use_staging_assets for development smoke only.
_PROD_NUCLEUS_HOST = "omniverse-content-production.s3-us-west-2.amazonaws.com"
_STAGING_NUCLEUS_HOST = "omniverse-content-staging.s3.us-west-2.amazonaws.com"


def _to_staging_url(url: str) -> str:
    """Rewrite a production Nucleus asset URL to its Isaac staging-bucket equivalent (host swap only)."""
    staged = url.replace(_PROD_NUCLEUS_HOST, _STAGING_NUCLEUS_HOST)
    assert staged != url, f"staging override did not rewrite the production host in: {url}"
    return staged


def _staging_subclass(asset_cls: type) -> tuple[type, str]:
    """Return a subclass of ``asset_cls`` whose ``usd_path`` points at staging, plus that URL.

    A dynamic subclass is used (not an in-place edit of ``asset_cls.usd_path``) because the asset registry
    hands back a SHARED class; mutating it would leak the staging URL into later non-staging jobs/rebuilds in
    the same eval_runner process. The registered class is left untouched.
    """
    staged = _to_staging_url(asset_cls.usd_path)
    return type(f"{asset_cls.__name__}Staging", (asset_cls,), {"usd_path": staged}), staged


def _apply_staging_stand_override(embodiment) -> str:
    """Point a DROID embodiment's stand at the staging asset, INSTANCE-LOCALLY, and return the staged URL.

    The stand AssetBaseCfg is a class-level configclass default shared across embodiment instances, so the
    spawn cfg is deep-copied before its usd_path is rewritten and the copy is reassigned — editing in place
    would leak the staging URL into other (stock) embodiments. Asserts the stand has a rewritable usd_path.
    """
    import copy

    stand_usd = getattr(embodiment.scene_config.stand.spawn, "usd_path", None)
    assert stand_usd, (
        "--use_staging_assets set but the DROID stand has no spawn.usd_path to rewrite "
        f"(stand spawn: {type(embodiment.scene_config.stand.spawn).__name__})."
    )
    staged = _to_staging_url(stand_usd)  # asserts the production host was actually rewritten
    stand = copy.deepcopy(embodiment.scene_config.stand)
    stand.spawn.usd_path = staged
    embodiment.scene_config.stand = stand
    return staged


@register_environment
class PickAndPlaceMapleTableEnvironment(ExampleEnvironmentBase):

    name: str = "pick_and_place_maple_table"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On, PositionLimits
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

        # Opt-in CAP/GaP evaluation profile. Off by default so stock defaults and existing VLA jobs are
        # unchanged; when on it adds the exterior agentview camera + its extrinsics variation, and the
        # reachability constraints below. (getattr keeps this safe for callers built before the flag existed.)
        gap_profile = getattr(args_cli, "gap_profile", False)
        use_staging = getattr(args_cli, "use_staging_assets", False)
        is_droid = "droid" in args_cli.embodiment

        # Fail early on a misconfigured profile rather than silently producing a scene without the GaP camera:
        # the exterior agentview camera and its variation require cameras enabled and a DROID embodiment.
        if gap_profile:
            assert args_cli.enable_cameras, "--gap_profile requires --enable_cameras (the exterior RGB-D camera)."
            assert is_droid, (
                "--gap_profile requires a DROID embodiment (the GaP adapter reads the droid exterior_cam); "
                f"got embodiment '{args_cli.embodiment}'."
            )

        # Step 1: Retrieve assets from the registry.
        # Optional, explicit, fail-closed staging override for the unpromoted Maple-table asset (see header).
        # Override via a dynamic SUBCLASS, never by mutating the registered class' usd_path — the registry hands
        # back the shared class, so an in-place edit would leak the staging URL into later non-staging jobs and
        # rebuilds in the same eval_runner process.
        background_cls = self.asset_registry.get_asset_by_name("maple_table_robolab")
        staged_table_url = None
        if use_staging:
            background_cls, staged_table_url = _staging_subclass(background_cls)
            print(
                f"[pick_and_place_maple_table] STAGING ASSET OVERRIDE (opt-in): maple_table -> {staged_table_url}",
                flush=True,
            )
        background = background_cls()

        # Pick targets. Opt-in 2-3-object sort profile: --pick_targets lists the ordered pick objects and the
        # task becomes the stock SortMultiObjectTask (all targets -> the destination). Unset (default) keeps the
        # stock single-object PickAndPlaceTask on --pick_up_object — single-object behavior is unchanged.
        # Fail closed and validate BEFORE any registry/Scene construction: --pick_targets uses nargs='+' with a
        # None default (a present flag must carry names; absent -> single-object), the count must be exactly 2-3,
        # the names must be unique, and they must not overlap the destination or the distractor objects.
        raw_pick_targets = getattr(args_cli, "pick_targets", None)
        is_multi_object = raw_pick_targets is not None
        if is_multi_object:
            pick_target_names = list(raw_pick_targets)
            assert 2 <= len(pick_target_names) <= 3, (
                f"the Maple multi-object profile supports exactly 2-3 --pick_targets; got {len(pick_target_names)}: "
                f"{pick_target_names}"
            )
            assert len(set(pick_target_names)) == len(
                pick_target_names
            ), f"--pick_targets must be unique; got {pick_target_names}"
            reserved = {args_cli.destination_location} | set(args_cli.additional_table_objects)
            overlap = reserved.intersection(pick_target_names)
            assert not overlap, (
                "--pick_targets must not overlap --destination_location/--additional_table_objects; "
                f"overlap: {sorted(overlap)}"
            )
        else:
            pick_target_names = [args_cli.pick_up_object]
        pick_objects = [self.asset_registry.get_asset_by_name(name)() for name in pick_target_names]
        destination_location = self.asset_registry.get_asset_by_name(args_cli.destination_location)()

        # Step 2: Describe spatial relationships
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        for obj in pick_objects:
            obj.add_relation(On(table_reference))
        destination_location.add_relation(On(table_reference))

        additional_table_objects = [
            self.asset_registry.get_asset_by_name(name)() for name in args_cli.additional_table_objects
        ]
        for obj in additional_table_objects:
            obj.add_relation(On(table_reference))

        # Reachability constraints are part of the GaP profile only (stock On() placement is unchanged otherwise).
        if gap_profile:
            for obj in [*pick_objects, destination_location, *additional_table_objects]:
                obj.add_relation(PositionLimits(**_REACH_BOX))

        # Step 3: Configure lighting
        light = self.asset_registry.get_asset_by_name("light")(
            spawner_cfg=sim_utils.DomeLightCfg(intensity=args_cli.light_intensity),
        )
        if args_cli.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(args_cli.hdr)())

        # Step 4: Select the embodiment
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
        )

        # Staging override for the unpromoted DROID stand asset (same PR #786 promotion gap as the table).
        # Deep-copy the stand cfg before editing so the override is instance-local: the stand AssetBaseCfg is a
        # class-level configclass default, so editing its spawn.usd_path in place could leak across embodiments.
        # Fail (not silently skip) if the stand has no rewritable usd_path — staging was explicitly requested.
        if use_staging and is_droid:
            staged_stand_url = _apply_staging_stand_override(embodiment)
            print(
                f"[pick_and_place_maple_table] STAGING ASSET OVERRIDE (opt-in): droid_stand -> {staged_stand_url}",
                flush=True,
            )

        # GaP profile (droid + cameras): attach the fixed exterior rgb+depth agentview camera the adapter reads
        # (its live cam.data.pos_w/quat_w_ros + intrinsics) and register a distinct extrinsics variation for it
        # (stock DROID only varies wrist_camera). The GaP job enables the Hydra key
        # camera_extrinsics_exterior_cam.enabled=true to activate the variation.
        if gap_profile:  # asserted above to imply enable_cameras + DROID
            from isaaclab_arena.variations.camera_extrinsics_variation import CameraExtrinsicsVariation
            from isaaclab_arena_environments.maple_cameras import MapleDroidPerceptionCameraCfg

            embodiment.camera_config = MapleDroidPerceptionCameraCfg()
            embodiment.add_variation(CameraExtrinsicsVariation(camera_name="exterior_cam"))

        # Step 5: Compose the scene
        scene = Scene(
            assets=[background, light, *pick_objects, destination_location, table_reference, *additional_table_objects]
        )

        # Step 6: Define the task (episode length configurable for the longer GaP rollouts; stock default 20 s).
        # getattr keeps legacy argparse.Namespace callers (built before the flag existed) working. Multi-target
        # uses the STOCK SortMultiObjectTask (all targets -> the destination, scored per-object under
        # SuccessMode.ALL); single-target keeps the stock PickAndPlaceTask unchanged.
        episode_length_s = getattr(args_cli, "episode_length_s", 20.0)
        if is_multi_object:
            from isaaclab_arena.tasks.sorting_task import SortMultiObjectTask

            task = SortMultiObjectTask(
                pick_up_object_list=pick_objects,
                destination_location_list=[destination_location] * len(pick_objects),
                background_scene=background,
                episode_length_s=episode_length_s,
            )
            # SortMultiObjectTask takes no task_description; set it so get_language_instruction() is populated.
            task.task_description = (
                f"Pick up the {' and the '.join(o.name for o in pick_objects)}, "
                f"and place all into the {destination_location.name}"
            )
        else:
            task = PickAndPlaceTask(
                pick_up_object=pick_objects[0],
                destination_location=destination_location,
                background_scene=background,
                episode_length_s=episode_length_s,
            )

        # Set viewport camera to match the robolab droid view
        def _set_viewer_cfg(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.2, 0.0, 0.0))
            return env_cfg

        # GaP-profile provenance + pose recording (G/H). Records the asset channel + resolved URLs + ordered
        # identities + seeds so staging-dev artifacts cannot be mistaken for production, and snapshots the
        # objects' initial (post-reset) world poses separately from their final poses. Recorder schema is
        # independent of CAP's scalar target_specs. Off unless --gap_profile (zero stock change otherwise).
        episode_recorder_terms = {}
        pose_snapshot_asset_names = []
        if gap_profile:
            from isaaclab_arena.recording.common_terms import (
                GapProvenanceEpisodeRecorderTermCfg,
                ObjectPosesEpisodeRecorderTermCfg,
            )

            distractor_names = [obj.name for obj in additional_table_objects]
            provenance = {
                "profile": "gap_profile",
                "asset_channel": "staging" if use_staging else "production",
                "table_usd": background.usd_path,
                "droid_stand_usd": getattr(embodiment.scene_config.stand.spawn, "usd_path", None),
                "task": "SortMultiObjectTask" if is_multi_object else "PickAndPlaceTask",
                "pick_targets": [obj.name for obj in pick_objects],  # ordered pick-target identities
                "destination": destination_location.name,
                "distractors": distractor_names,
                "placement_seed": getattr(args_cli, "placement_seed", None),
                "seed": getattr(args_cli, "seed", None),
            }
            episode_recorder_terms = {
                "gap_provenance": GapProvenanceEpisodeRecorderTermCfg(params={"provenance": provenance}),
                "object_poses": ObjectPosesEpisodeRecorderTermCfg(),
            }
            pose_snapshot_asset_names = [
                *[obj.name for obj in pick_objects],
                destination_location.name,
                *distractor_names,
            ]

        # Step 7: Assemble the environment
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            env_cfg_callback=_set_viewer_cfg,
            episode_recorder_terms=episode_recorder_terms,
        )
        isaaclab_arena_environment.pose_snapshot_asset_names = pose_snapshot_asset_names
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="droid_abs_joint_pos")
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--hdr", type=str, default=None)
        parser.add_argument("--light_intensity", type=float, default=500.0)
        parser.add_argument("--pick_up_object", type=str, default="rubiks_cube_hot3d_robolab")
        parser.add_argument("--destination_location", type=str, default="bowl_ycb_robolab")
        parser.add_argument(
            "--additional_table_objects",
            nargs="*",
            type=str,
            default=[],
            help="Extra (distractor) objects to place on the table alongside the pick target(s)",
        )
        parser.add_argument(
            "--pick_targets",
            nargs="+",
            type=str,
            default=None,
            help=(
                "Opt-in 2-3 object sort profile: ordered, unique pick-target assets, all placed into "
                "--destination_location via the stock SortMultiObjectTask. Unset (default) keeps the stock "
                "single-object PickAndPlaceTask on --pick_up_object (single-object behavior unchanged). When "
                "present it must carry 2-3 names not overlapping the destination/distractors. Pair with a long "
                "--episode_length_s for multi-object rollouts."
            ),
        )
        parser.add_argument(
            "--gap_profile",
            action="store_true",
            default=False,
            help=(
                "Opt-in CAP/GaP evaluation profile: add the fixed exterior rgb+depth agentview camera the GaP "
                "adapter reads, register its camera_extrinsics_exterior_cam variation, and constrain object "
                "placement to the arm reach box. Off by default — stock defaults and existing VLA jobs are unchanged."
            ),
        )
        parser.add_argument(
            "--episode_length_s",
            type=float,
            default=20.0,
            help="Task episode length in seconds (stock default 20.0; raise for longer GaP rollouts).",
        )
        parser.add_argument(
            "--use_staging_assets",
            action="store_true",
            default=False,
            help=(
                "DEV ONLY, opt-in: load the unpromoted Maple-table and DROID-stand assets (PR #786) from the Isaac "
                "staging bucket instead of production, where they currently 404. Targeted host swap for those two "
                "files only; never a silent fallback. Production (default) remains the documented promotion blocker."
            ),
        )
