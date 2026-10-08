# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Share one fixed variant assignment between placement and native clone planning."""

from __future__ import annotations

import random
import torch
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import MultiAssetSpawnerCfg
from isaaclab.utils import configclass
from isaaclab.utils.dict import dict_to_md5_hash

if TYPE_CHECKING:
    from isaaclab_arena.assets.asset import Asset


@configclass
class ObjectVariantAssignmentCfg:
    """Variant choices and source configuration captured before placement."""

    variant_indices: tuple[int, ...] = ()
    """Variant index selected for each environment."""

    spawn_config_hashes: tuple[str, ...] = ()
    """Hashes of native spawn configurations in variant order before placement."""

    spawner_options_hash: str = ""
    """Hash of outer multi-spawner settings before placement, excluding generated spawn paths."""


def _hash_spawner_options(spawn_cfg: MultiAssetSpawnerCfg) -> str:
    """Fingerprint shared spawn settings independently of variants and clone-plan output."""
    options = spawn_cfg.to_dict()
    del options["assets_cfg"]
    del options["spawn_paths"]
    return dict_to_md5_hash(options)


def assign_object_variants(
    objects: Iterable[Asset], num_envs: int, seed: int | None = None
) -> dict[str, ObjectVariantAssignmentCfg]:
    """Retain or bind asset assignments and snapshot their configurations before placement.

    Existing assignments remain fixed; the seed only affects unassigned objects.

    Args:
        objects: Scene assets, including those without placement relations.
        num_envs: Environment count shared by placement and simulation.
        seed: Optional seed for unassigned objects; names keep sampling independent of declaration order.

    Returns:
        Serializable assignments keyed by scene configuration name.
    """
    assert num_envs > 0, "Variant assignment requires at least one environment."
    assignments = {}
    for obj in objects:
        if not getattr(obj, "has_multiple_assets", False):
            continue
        spawn_cfg = obj.spawn_cfg
        assert isinstance(spawn_cfg, MultiAssetSpawnerCfg), f"Object '{obj.name}' needs native asset variants."
        # Native settings may change after construction; the shared rigid-body view
        # still requires the current alternatives to have one common body path.
        obj.get_contact_sensor_prim_path()
        variant_count = len(spawn_cfg.assets_cfg)
        indices = obj.asset_indices_by_env
        if indices is not None:
            assert len(indices) == num_envs, (
                f"Object '{obj.name}' already has variants assigned for {len(indices)} environments; "
                f"cannot reuse it for {num_envs}."
            )
        else:
            if obj.random_choice:
                generator = random.Random(None if seed is None else f"{seed}:{obj.name}")
                indices = tuple(generator.randrange(variant_count) for _ in range(num_envs))
            else:
                indices = tuple(env_id % variant_count for env_id in range(num_envs))
            obj.bind_asset_assignment(indices)
        assignments[obj.get_scene_key()] = ObjectVariantAssignmentCfg(
            variant_indices=tuple(indices),
            spawn_config_hashes=tuple(dict_to_md5_hash(cfg) for cfg in spawn_cfg.assets_cfg),
            spawner_options_hash=_hash_spawner_options(spawn_cfg),
        )
    return assignments


def validate_object_variant_assignments(
    scene_cfg: InteractiveSceneCfg, assignments: dict[str, ObjectVariantAssignmentCfg]
) -> tuple[tuple[str, int, tuple[int, ...]], ...]:
    """Check that the final scene still matches placement's source snapshot.

    Args:
        scene_cfg: Final scene configuration, including any later overrides.
        assignments: Assignments returned before placement by assign_object_variants().

    Returns:
        Object names, variant counts and assignments in native scene declaration order.
    """
    if not assignments:
        return ()
    assert not scene_cfg.clone_cfg.clone_combinations, "Assigned variants cannot use optional clone combinations."
    ordered_assignments = []
    configured_objects = set()
    for asset_name, asset_cfg in vars(scene_cfg).items():
        if isinstance(asset_cfg, RigidObjectCollectionCfg):
            assert all(
                cloner.num_spawn_variants(member.spawn) == 1 for member in asset_cfg.rigid_objects.values()
            ), "Heterogeneous rigid-object collections cannot be combined with assigned variants."
        spawn_cfg = getattr(asset_cfg, "spawn", None)
        if spawn_cfg is None or cloner.num_spawn_variants(spawn_cfg) <= 1:
            continue
        assert asset_name in assignments, f"Multi-spawner '{asset_name}' has no placement assignment."
        assert isinstance(asset_cfg, AssetBaseCfg), f"Object '{asset_name}' must use an asset configuration."
        prim_path = cloner.expand_env_regex_ns(asset_cfg.prim_path, scene_cfg.clone_cfg.clone_template)
        assert (
            cloner.path.match(prim_path, scene_cfg.clone_cfg.clone_template) is not None
        ), f"Object '{asset_name}' variants require an environment-scoped prim path."
        assert isinstance(
            spawn_cfg, MultiAssetSpawnerCfg
        ), f"Object '{asset_name}' must retain its native MultiAssetSpawnerCfg."
        assignment = assignments[asset_name]
        variant_count = len(assignment.spawn_config_hashes)
        assert (
            len(spawn_cfg.assets_cfg) == variant_count
        ), f"Object '{asset_name}' variant count changed after placement."
        actual_hashes = tuple(dict_to_md5_hash(cfg) for cfg in spawn_cfg.assets_cfg)
        assert (
            actual_hashes == assignment.spawn_config_hashes
        ), f"Object '{asset_name}' spawn variants changed after placement."
        assert (
            _hash_spawner_options(spawn_cfg) == assignment.spawner_options_hash
        ), f"Object '{asset_name}' shared spawn settings changed after placement."
        indices = assignment.variant_indices
        assert len(indices) == scene_cfg.num_envs, (
            f"Object '{asset_name}' has variants assigned for {len(indices)} environments, "
            f"but the final scene has {scene_cfg.num_envs}."
        )
        assert all(
            type(index) is int and 0 <= index < variant_count for index in indices
        ), f"Object '{asset_name}' has invalid variant indices."
        ordered_assignments.append((asset_name, variant_count, tuple(indices)))
        configured_objects.add(asset_name)
    missing_objects = assignments.keys() - configured_objects
    assert not missing_objects, f"Object variants missing from the final scene: {sorted(missing_objects)}."
    return tuple(ordered_assignments)


@contextmanager
def object_variant_clone_strategy(
    scene_cfg: InteractiveSceneCfg, assignments: dict[str, ObjectVariantAssignmentCfg]
) -> Iterator[None]:
    """Install the fixed assignments only while Isaac Lab constructs the scene.

    Args:
        scene_cfg: Runtime scene configuration, including any Hydra overrides.
        assignments: Serializable assignments captured before placement.
    """
    ordered_assignments = validate_object_variant_assignments(scene_cfg, assignments)
    if not ordered_assignments:
        yield
        return

    def select_variants(combinations: torch.Tensor, num_clones: int, device: str) -> torch.Tensor:
        variant_counts = combinations.max(dim=0).values + 1
        variant_columns = (variant_counts > 1).nonzero(as_tuple=False).flatten().tolist()
        assert len(variant_columns) == len(ordered_assignments), "Native clone-plan variants differ from placement."
        # Isaac Lab may insert homogeneous asset or sensor columns; they all select variant zero.
        chosen = torch.zeros((num_clones, combinations.shape[1]), dtype=torch.long, device=device)
        for column, (object_name, variant_count, indices) in zip(variant_columns, ordered_assignments):
            assert (
                int(variant_counts[column]) == variant_count
            ), f"Native variant count for Object '{object_name}' differs from placement."
            chosen[:, column] = torch.tensor(indices, dtype=torch.long, device=device)
        return chosen

    # Stateful callbacks cannot roundtrip through Isaac Lab's callable-to-string serialization.
    original_strategy = scene_cfg.clone_cfg.clone_strategy
    scene_cfg.clone_cfg.clone_strategy = select_variants
    try:
        yield
    finally:
        scene_cfg.clone_cfg.clone_strategy = original_strategy
