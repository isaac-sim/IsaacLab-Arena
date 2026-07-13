# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

# World-coordinate reach box for the table objects, in front of the droid base (near the origin). On()
# supplies the surface z; this only bounds x/y so the relation solver keeps every object within the arm's
# top-down reach (the stock On() alone can land an object anywhere on the table, including out of reach).
# Seed 1 is validated by a successful five-object GaP rollout; full-box IK coverage has not been swept.
# Only applied under --gap_profile.
_REACH_BOX = dict(x_min=0.05, x_max=0.45, y_min=-0.25, y_max=0.25)

# Opt-in dev override for the two unpromoted srl_robolab assets from PR #786 (the Maple table and the DROID
# stand): both 404 on the production Nucleus but 200 on the Isaac staging bucket (CI used staging). The
# override is a targeted production->staging HOST swap for ONLY those two files; everything else (robolab
# objects, etc.) stays on production. Off by default (production, documented promotion blocker); never a
# silent fallback. Enable with --use_staging_assets for development/evaluation until the assets are promoted.
_PROD_NUCLEUS_HOST = "omniverse-content-production.s3-us-west-2.amazonaws.com"
_STAGING_NUCLEUS_HOST = "omniverse-content-staging.s3.us-west-2.amazonaws.com"
_LOCAL_ASSET_HOSTS = {
    _PROD_NUCLEUS_HOST,
    _STAGING_NUCLEUS_HOST,
    "omniverse-content-production.s3.us-west-2.amazonaws.com",
}
_ASSET_PROVENANCE_FILE = "CAP_ASSET_PROVENANCE.json"
# Keep exact-layout bodies just above the tabletop. A larger drop introduces avoidable lateral drift before
# the policy's first observation, which weakens millimeter-scale distance control.
_CONTROLLED_ON_CLEARANCE_M = 0.001
_CONTROLLED_TABLE_EDGE_MARGIN_M = 0.05
_CONTROLLED_PICK_WORKSPACE = dict(x_min=0.05, x_max=0.45, y_min=-0.25, y_max=0.25)
_CONTROLLED_DESTINATION_WORKSPACE = dict(x_min=0.45, x_max=0.55, y_min=-0.25, y_max=0.25)


def _compute_controlled_object_bin_layout(
    pick_bbox,
    destination_bbox,
    table_bbox,
    *,
    gap_m: float,
    side: str,
    pick_center_x: float,
    destination_center_x: float,
    pair_midpoint_y: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Return exact pick/destination poses for a controlled planar AABB gap."""
    pick_x = pick_center_x - float(pick_bbox.center[0, 0])
    destination_x = destination_center_x - float(destination_bbox.center[0, 0])
    center_separation_y = float(destination_bbox.size[0, 1]) * 0.5 + float(pick_bbox.size[0, 1]) * 0.5 + gap_m
    if side == "positive_y":
        pick_center_y = pair_midpoint_y + center_separation_y * 0.5
        destination_center_y = pair_midpoint_y - center_separation_y * 0.5
    elif side == "negative_y":
        pick_center_y = pair_midpoint_y - center_separation_y * 0.5
        destination_center_y = pair_midpoint_y + center_separation_y * 0.5
    else:
        raise ValueError(f"unsupported controlled object-bin side: {side!r}")
    pick_y = pick_center_y - float(pick_bbox.center[0, 1])
    destination_y = destination_center_y - float(destination_bbox.center[0, 1])

    table_top_z = float(table_bbox.max_point[0, 2])
    pick_position = (
        pick_x,
        pick_y,
        table_top_z + _CONTROLLED_ON_CLEARANCE_M - float(pick_bbox.min_point[0, 2]),
    )
    destination_position = (
        destination_x,
        destination_y,
        table_top_z + _CONTROLLED_ON_CLEARANCE_M - float(destination_bbox.min_point[0, 2]),
    )
    pick_world = pick_bbox.translated(pick_position)
    destination_world = destination_bbox.translated(destination_position)
    actual_gap_m = (
        float(pick_world.min_point[0, 1] - destination_world.max_point[0, 1])
        if side == "positive_y"
        else float(destination_world.min_point[0, 1] - pick_world.max_point[0, 1])
    )
    if abs(actual_gap_m - gap_m) >= 1e-6:
        raise ValueError(f"controlled layout produced {actual_gap_m} m instead of {gap_m} m")
    x_overlap_m = min(float(pick_world.max_point[0, 0]), float(destination_world.max_point[0, 0])) - max(
        float(pick_world.min_point[0, 0]), float(destination_world.min_point[0, 0])
    )
    if x_overlap_m <= 0.0:
        raise ValueError(
            "controlled layout requires overlapping pick/destination X AABBs so the requested Y gap "
            "is the closest planar gap"
        )

    for name, world_bbox, position, workspace in (
        ("pick object", pick_world, pick_position, _CONTROLLED_PICK_WORKSPACE),
        (
            "destination",
            destination_world,
            destination_position,
            _CONTROLLED_DESTINATION_WORKSPACE,
        ),
    ):
        inside_table = (
            world_bbox.min_point[0, 0] >= table_bbox.min_point[0, 0] + _CONTROLLED_TABLE_EDGE_MARGIN_M
            and world_bbox.max_point[0, 0] <= table_bbox.max_point[0, 0] - _CONTROLLED_TABLE_EDGE_MARGIN_M
            and world_bbox.min_point[0, 1] >= table_bbox.min_point[0, 1] + _CONTROLLED_TABLE_EDGE_MARGIN_M
            and world_bbox.max_point[0, 1] <= table_bbox.max_point[0, 1] - _CONTROLLED_TABLE_EDGE_MARGIN_M
        )
        if not inside_table:
            raise ValueError(f"controlled layout puts {name} outside the Maple tabletop margin")
        if not (
            workspace["x_min"] <= position[0] <= workspace["x_max"]
            and workspace["y_min"] <= position[1] <= workspace["y_max"]
        ):
            raise ValueError(f"controlled layout puts {name} origin outside the DROID workspace: {position}")

    return pick_position, destination_position, actual_gap_m


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


def _local_asset_path(source_url: str, local_root: str | Path) -> str:
    """Map one approved Isaac asset URL into the image's host-qualified local mirror."""
    parsed = urlparse(source_url)
    if parsed.scheme != "https" or parsed.netloc not in _LOCAL_ASSET_HOSTS:
        raise RuntimeError(f"unsupported CAP asset source URL: {source_url}")
    root = Path(local_root).expanduser().resolve()
    target = (root / parsed.netloc / parsed.path.lstrip("/")).resolve()
    if not target.is_relative_to(root):
        raise RuntimeError(f"CAP asset source escapes the local asset root: {source_url}")
    if not target.is_file():
        raise FileNotFoundError(f"baked CAP asset is missing: {target} (source: {source_url})")
    return str(target)


def _local_asset_subclass(asset_cls: type, local_root: str | Path) -> tuple[type, str, str]:
    """Return an instance-local asset subclass backed by the baked mirror."""
    source_url = getattr(asset_cls, "usd_path", None)
    if not source_url:
        raise RuntimeError(f"CAP asset class has no usd_path: {asset_cls.__name__}")
    local_path = _local_asset_path(source_url, local_root)
    localized = type(f"{asset_cls.__name__}Local", (asset_cls,), {"usd_path": local_path})
    return localized, source_url, local_path


def _load_local_asset_provenance(local_root: str | Path) -> dict:
    path = Path(local_root).expanduser().resolve() / _ASSET_PROVENANCE_FILE
    try:
        provenance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid baked CAP asset provenance: {path}") from exc
    tree_hash = provenance.get("tree_sha256")
    if (
        provenance.get("schema_version") != 1
        or not isinstance(tree_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", tree_hash)
    ):
        raise RuntimeError(f"invalid baked CAP asset provenance schema or tree hash: {path}")
    expected_tree_hash = os.environ.get("CAP_IMAGE_ASSET_TREE_SHA256")
    if expected_tree_hash and tree_hash != expected_tree_hash:
        raise RuntimeError(f"baked CAP asset tree hash {tree_hash} does not match image pin {expected_tree_hash}")
    return provenance


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


def _apply_local_droid_asset_override(embodiment, local_root: str | Path) -> dict[str, str]:
    """Localize DROID robot and stand spawn configs without mutating shared defaults."""
    import copy

    source_urls = {}
    local_paths = {}
    for name in ("robot", "stand"):
        asset_cfg = getattr(embodiment.scene_config, name)
        source_url = getattr(asset_cfg.spawn, "usd_path", None)
        if not source_url:
            raise RuntimeError(f"DROID {name} has no spawn.usd_path to localize")
        localized_cfg = copy.deepcopy(asset_cfg)
        localized_cfg.spawn.usd_path = _local_asset_path(source_url, local_root)
        setattr(embodiment.scene_config, name, localized_cfg)
        source_urls[f"{name}_source_usd"] = source_url
        local_paths[f"{name}_local_usd"] = localized_cfg.spawn.usd_path
    return {**source_urls, **local_paths}


@dataclass
class PickAndPlaceMapleTableEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Maple-table pick-and-place environment."""

    enable_cameras: bool = False
    embodiment: str = "droid_abs_joint_pos"
    hdr: str | None = None
    light_intensity: float = 500.0
    pick_up_object: str = "rubiks_cube_hot3d_robolab"
    destination_location: str = "bowl_ycb_robolab"
    additional_table_objects: list[str] = field(default_factory=list)
    pick_targets: list[str] | None = None
    gap_profile: bool = False
    episode_length_s: float = 20.0
    object_bin_gap_m: float | None = None
    object_bin_side: str = "positive_y"
    object_pick_center_x: float = 0.38
    object_bin_center_x: float = 0.46
    object_bin_pair_midpoint_y: float = 0.01
    placement_clearance_m: float | None = None
    use_staging_assets: bool = False

    def __post_init__(self) -> None:
        assert self.object_bin_side in {
            "positive_y",
            "negative_y",
        }, f"Unsupported object-bin side: {self.object_bin_side}"


@register_environment
class PickAndPlaceMapleTableEnvironment(ArenaEnvironmentFactory[PickAndPlaceMapleTableEnvironmentCfg]):
    """Registered provider for the Maple-table pick-and-place environment."""

    name: str = "pick_and_place_maple_table"
    _legacy_argparse_cfg_type = PickAndPlaceMapleTableEnvironmentCfg

    def build(self, cfg: PickAndPlaceMapleTableEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
        from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
        from isaaclab_arena.relations.relations import IsAnchor, On, PositionLimits
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        # Opt-in CAP/GaP evaluation profile. Off by default so stock defaults and existing VLA jobs are
        # unchanged; when on it adds the exterior agentview camera + its extrinsics variation, and the
        # reachability constraints below.
        gap_profile = cfg.gap_profile
        use_staging = cfg.use_staging_assets
        is_droid = "droid" in cfg.embodiment
        configured_local_root = os.environ.get("CAP_LOCAL_ASSET_ROOT")
        local_asset_root = Path(configured_local_root).expanduser().resolve() if configured_local_root else None
        local_asset_provenance = (
            _load_local_asset_provenance(local_asset_root) if local_asset_root is not None else None
        )

        # Materialization is independent of the GaP camera/reach profile. An explicit local mirror must never
        # silently fall back to network-backed assets, including for layout-only jobs. The baked CAP closure is
        # deliberately scoped to the DROID setup, so reject unsupported embodiments before scene construction.
        if local_asset_root is not None:
            assert is_droid, (
                "CAP_LOCAL_ASSET_ROOT contains the CAP DROID scene closure; "
                f"got unsupported embodiment '{cfg.embodiment}'."
            )

        # Fail early on a misconfigured profile rather than silently producing a scene without the GaP camera:
        # the exterior agentview camera and its variation require cameras enabled and a DROID embodiment.
        if gap_profile:
            assert cfg.enable_cameras, "--gap_profile requires --enable_cameras (the exterior RGB-D camera)."
            assert is_droid, (
                "--gap_profile requires a DROID embodiment (the GaP adapter reads the droid exterior_cam); "
                f"got embodiment '{cfg.embodiment}'."
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
        table_source_url = background_cls.usd_path
        if local_asset_root is not None:
            background_cls, table_source_url, local_table_path = _local_asset_subclass(background_cls, local_asset_root)
            print(
                f"[pick_and_place_maple_table] LOCAL ASSET OVERRIDE: maple_table -> {local_table_path}",
                flush=True,
            )
        background = background_cls()

        asset_source_urls: dict[str, str] = {}
        asset_resolved_usds: dict[str, str] = {}

        def _instantiate_asset(name: str):
            asset_cls = self.asset_registry.get_asset_by_name(name)
            if local_asset_root is not None:
                asset_cls, source_url, local_path = _local_asset_subclass(asset_cls, local_asset_root)
                asset_source_urls[name] = source_url
                asset_resolved_usds[name] = local_path
            return asset_cls()

        # Pick targets. Opt-in 2-5-object sort profile: --pick_targets lists the ordered pick objects and the
        # task becomes the stock SortMultiObjectTask (all targets -> the destination). Unset (default) keeps the
        # stock single-object PickAndPlaceTask on --pick_up_object — single-object behavior is unchanged.
        # Fail closed and validate before any registry/Scene construction: an absent value selects the
        # single-object task; a supplied list must contain 2-5 unique, non-reserved names.
        raw_pick_targets = cfg.pick_targets
        is_multi_object = raw_pick_targets is not None
        if is_multi_object:
            pick_target_names = list(raw_pick_targets)
            assert 2 <= len(pick_target_names) <= 5, (
                f"the Maple multi-object profile supports exactly 2-5 --pick_targets; got {len(pick_target_names)}: "
                f"{pick_target_names}"
            )
            assert len(set(pick_target_names)) == len(
                pick_target_names
            ), f"--pick_targets must be unique; got {pick_target_names}"
            reserved = {cfg.destination_location} | set(cfg.additional_table_objects)
            overlap = reserved.intersection(pick_target_names)
            assert not overlap, (
                "--pick_targets must not overlap --destination_location/--additional_table_objects; "
                f"overlap: {sorted(overlap)}"
            )
        else:
            pick_target_names = [cfg.pick_up_object]
        pick_objects = [_instantiate_asset(name) for name in pick_target_names]
        destination_location = _instantiate_asset(cfg.destination_location)

        object_bin_gap_m = cfg.object_bin_gap_m
        object_bin_side = cfg.object_bin_side
        object_pick_center_x = cfg.object_pick_center_x
        object_bin_center_x = cfg.object_bin_center_x
        object_bin_pair_midpoint_y = cfg.object_bin_pair_midpoint_y
        configured_placement_clearance_m = cfg.placement_clearance_m
        placement_clearance_m = (
            5e-4
            if object_bin_gap_m is not None and configured_placement_clearance_m is None
            else configured_placement_clearance_m
        )
        if object_bin_gap_m is not None:
            assert gap_profile, "--object_bin_gap_m requires --gap_profile so the realized gap is recorded"
            assert not is_multi_object, "--object_bin_gap_m currently requires the single-object task"
            assert object_bin_gap_m > 0.0, "--object_bin_gap_m must be positive"
            assert placement_clearance_m is not None
            assert placement_clearance_m >= 0.0, "--placement_clearance_m must be non-negative"
            assert object_bin_gap_m >= placement_clearance_m, (
                f"--object_bin_gap_m ({object_bin_gap_m}) must be at least the placement clearance "
                f"({placement_clearance_m})"
            )

        # Step 2: Describe spatial relationships
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        if object_bin_gap_m is not None:
            # The generic differentiable relation solver is intentionally not used for this controlled
            # sensitivity variable: its 1 cm optimizer step leaves millimeter-scale residuals. Compute the
            # exact face gap from the authored AABBs, then make both objects placement anchors. They remain
            # ordinary dynamic rigid bodies in simulation; IsAnchor only fixes their reset layout.
            pick_object = pick_objects[0]
            pick_bbox = pick_object.get_bounding_box()
            destination_bbox = destination_location.get_bounding_box()
            table_bbox = table_reference.get_world_bounding_box()
            pick_position, destination_position, _ = _compute_controlled_object_bin_layout(
                pick_bbox,
                destination_bbox,
                table_bbox,
                gap_m=object_bin_gap_m,
                side=object_bin_side,
                pick_center_x=object_pick_center_x,
                destination_center_x=object_bin_center_x,
                pair_midpoint_y=object_bin_pair_midpoint_y,
            )

            pick_object.set_initial_pose(Pose(position_xyz=pick_position))
            destination_location.set_initial_pose(Pose(position_xyz=destination_position))
            pick_object.add_relation(IsAnchor())
            destination_location.add_relation(IsAnchor())
        else:
            for obj in pick_objects:
                obj.add_relation(On(table_reference))
            destination_location.add_relation(On(table_reference))

        additional_table_objects = [_instantiate_asset(name) for name in cfg.additional_table_objects]
        for obj in additional_table_objects:
            obj.add_relation(On(table_reference))

        # Reachability constraints are part of the GaP profile only (stock On() placement is unchanged otherwise).
        if gap_profile:
            for obj in [*pick_objects, destination_location, *additional_table_objects]:
                obj.add_relation(PositionLimits(**_REACH_BOX))

        # Step 3: Configure lighting
        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(cfg.light_intensity)
        if cfg.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(cfg.hdr)())
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()

        # Step 4: Select the embodiment
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
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
        droid_asset_paths = {}
        if local_asset_root is not None:
            droid_asset_paths = _apply_local_droid_asset_override(embodiment, local_asset_root)
            print(
                "[pick_and_place_maple_table] LOCAL ASSET OVERRIDE: "
                f"droid_robot -> {droid_asset_paths['robot_local_usd']}; "
                f"droid_stand -> {droid_asset_paths['stand_local_usd']}",
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
            assets=[
                background,
                light,
                directional_light,
                *pick_objects,
                destination_location,
                table_reference,
                *additional_table_objects,
            ]
        )

        # Step 6: Define the task (episode length configurable for the longer GaP rollouts; stock default 20 s).
        episode_length_s = cfg.episode_length_s
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
            if gap_profile:
                # Reset-time placement and camera variations must be rendered before the first GaP observation.
                env_cfg.num_rerenders_on_reset = max(env_cfg.num_rerenders_on_reset, 5)
            return env_cfg

        # GaP-profile provenance + pose recording (G/H). Records the asset channel + resolved URLs + ordered
        # identities + seeds so staging-dev artifacts cannot be mistaken for production, and snapshots the
        # objects' initial (post-reset) world poses separately from their final poses. Recorder schema is
        # independent of CAP's scalar target_specs. Off unless --gap_profile (zero stock change otherwise).
        episode_recorder_terms = {}
        pose_snapshot_asset_names = []
        if gap_profile:
            from isaaclab_arena.recording.common_terms import (
                ControlledGapObservationEpisodeRecorderTermCfg,
                GapProvenanceEpisodeRecorderTermCfg,
                ObjectPosesEpisodeRecorderTermCfg,
            )

            distractor_names = [obj.name for obj in additional_table_objects]
            provenance = {
                "profile": "gap_profile",
                "asset_channel": "staging" if use_staging else "production",
                "asset_materialization": "local_baked" if local_asset_root is not None else "remote",
                "table_usd": table_source_url,
                "table_source_usd": table_source_url,
                "table_resolved_usd": background.usd_path,
                "droid_robot_usd": droid_asset_paths.get(
                    "robot_source_usd", getattr(embodiment.scene_config.robot.spawn, "usd_path", None)
                ),
                "droid_robot_source_usd": droid_asset_paths.get(
                    "robot_source_usd",
                    getattr(embodiment.scene_config.robot.spawn, "usd_path", None),
                ),
                "droid_robot_resolved_usd": getattr(embodiment.scene_config.robot.spawn, "usd_path", None),
                "droid_stand_usd": droid_asset_paths.get(
                    "stand_source_usd", getattr(embodiment.scene_config.stand.spawn, "usd_path", None)
                ),
                "droid_stand_source_usd": droid_asset_paths.get(
                    "stand_source_usd",
                    getattr(embodiment.scene_config.stand.spawn, "usd_path", None),
                ),
                "droid_stand_resolved_usd": getattr(embodiment.scene_config.stand.spawn, "usd_path", None),
                "asset_source_urls": asset_source_urls,
                "asset_resolved_usds": asset_resolved_usds,
                "baked_asset_tree_sha256": (
                    local_asset_provenance["tree_sha256"] if local_asset_provenance is not None else None
                ),
                "task": "SortMultiObjectTask" if is_multi_object else "PickAndPlaceTask",
                "pick_targets": [obj.name for obj in pick_objects],  # ordered pick-target identities
                "destination": destination_location.name,
                "distractors": distractor_names,
                "object_bin_gap_m": object_bin_gap_m,
                "object_bin_side": object_bin_side if object_bin_gap_m is not None else None,
                "object_pick_center_x": object_pick_center_x if object_bin_gap_m is not None else None,
                "object_bin_center_x": object_bin_center_x if object_bin_gap_m is not None else None,
                "object_bin_pair_midpoint_y": object_bin_pair_midpoint_y if object_bin_gap_m is not None else None,
                "object_bin_placement_mode": "exact_aabb" if object_bin_gap_m is not None else None,
                "placement_clearance_m": placement_clearance_m,
            }
            episode_recorder_terms = {
                "gap_provenance": GapProvenanceEpisodeRecorderTermCfg(params={"provenance": provenance}),
                "object_poses": ObjectPosesEpisodeRecorderTermCfg(),
            }
            if object_bin_gap_m is not None:
                episode_recorder_terms["controlled_gap_observation"] = ControlledGapObservationEpisodeRecorderTermCfg(
                    params={
                        "pick_asset_name": pick_objects[0].name,
                        "destination_asset_name": destination_location.name,
                        "side": object_bin_side,
                        "pick_local_bbox_min": pick_bbox.min_point[0].tolist(),
                        "pick_local_bbox_max": pick_bbox.max_point[0].tolist(),
                        "destination_local_bbox_min": destination_bbox.min_point[0].tolist(),
                        "destination_local_bbox_max": destination_bbox.max_point[0].tolist(),
                    }
                )
            pose_snapshot_asset_names = [
                *[obj.name for obj in pick_objects],
                destination_location.name,
                *distractor_names,
            ]

        # Step 7: Assemble the environment
        placer_params = None
        if placement_clearance_m is not None:
            assert placement_clearance_m >= 0.0, "--placement_clearance_m must be non-negative"
            placer_params = ObjectPlacerParams(
                solver_params=RelationSolverParams(
                    clearance_m=placement_clearance_m,
                    verbose=False,
                    save_position_history=False,
                )
            )

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            env_cfg_callback=_set_viewer_cfg,
            episode_recorder_terms=episode_recorder_terms,
            placer_params=placer_params,
        )
        isaaclab_arena_environment.pose_snapshot_asset_names = pose_snapshot_asset_names
        return isaaclab_arena_environment

    # TODO(cvolk, 2026-07-03): [typed-config-migration] Delete this CLI-only option when teleoperation runners
    # receive typed configuration instead of the environment subparser namespace.
    @staticmethod
    def _add_legacy_cli_only_args(parser: argparse.ArgumentParser) -> None:
        # Consumed directly by teleop.py and record_demos.py, not by build(cfg).
        parser.add_argument("--teleop_device", type=str, default=None)
