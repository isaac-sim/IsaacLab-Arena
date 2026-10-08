# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Share object variant assignments between placement and Isaac Lab clone planning."""

from __future__ import annotations

import random
import torch
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import MultiAssetSpawnerCfg
from isaaclab.utils import configclass
from isaaclab.utils.dict import dict_to_md5_hash

if TYPE_CHECKING:
    from isaaclab_arena.assets.asset import Asset


def assign_object_variants(objects: Iterable[Asset], num_envs: int, seed: int | None = None) -> None:
    """Bind each heterogeneous object to one variant per environment before placement.

    Args:
        objects: Scene assets, including objects without placement relations.
        num_envs: Number of environments used by both placement and simulation.
        seed: Optional reproducible seed; object names keep sampling independent of asset order.
    """
    from isaaclab_arena.assets.object import Object

    assert num_envs > 0, "Variant assignment requires at least one environment."
    for obj in objects:
        if not isinstance(obj, Object) or not obj.has_variants:
            continue
        existing_assignment = obj.variant_indices_by_env
        if existing_assignment is not None:
            assert len(existing_assignment) == num_envs, (
                f"Object '{obj.name}' already has variants assigned for {len(existing_assignment)} environments; "
                f"cannot reuse it for {num_envs}."
            )
            continue
        variant_count = len(obj.variants)
        if obj.random_choice:
            generator = random.Random(None if seed is None else f"{seed}:{obj.name}")
            assignment = tuple(generator.randrange(variant_count) for _ in range(num_envs))
        else:
            assignment = tuple(env_id % variant_count for env_id in range(num_envs))
        obj.bind_variant_assignment(assignment)


@configclass
class ObjectVariantAssignmentCfg:
    """Serializable placement assignment and source identity for one scene object."""

    variant_indices: tuple[int, ...] = ()
    """Variant index selected for each environment before placement."""

    spawn_config_hashes: tuple[str, ...] = ()
    """Native spawn configurations in variant order, used to reject geometry overrides."""


def build_object_variant_assignments(
    scene_cfg: InteractiveSceneCfg, objects: Iterable[Asset]
) -> dict[str, ObjectVariantAssignmentCfg]:
    """Record placement assignments and validate the final scene without adding runtime callbacks.

    Args:
        scene_cfg: Final scene configuration after environment overrides.
        objects: Scene assets whose variant assignments were bound before placement.

    Returns:
        Serializable assignments keyed by the objects' scene configuration names.
    """
    from isaaclab_arena.assets.object import Object

    assignments = {}
    for obj in objects:
        if not isinstance(obj, Object) or not obj.has_variants:
            continue
        indices = obj.variant_indices_by_env
        assert indices is not None, f"Assign Object '{obj.name}' variants before configuring the scene."
        spawn_configs = obj.object_cfg.spawn.assets_cfg
        assignments[obj.get_scene_key()] = ObjectVariantAssignmentCfg(
            variant_indices=indices,
            spawn_config_hashes=tuple(dict_to_md5_hash(cfg) for cfg in spawn_configs),
        )
    _validate_scene_variant_assignments(scene_cfg, assignments)
    return assignments


def _validate_scene_variant_assignments(
    scene_cfg: InteractiveSceneCfg, assignments: dict[str, ObjectVariantAssignmentCfg]
) -> tuple[tuple[str, int, tuple[int, ...]], ...]:
    """Validate source geometry and return assignments in native scene declaration order."""
    if not assignments:
        return ()
    assert (
        not scene_cfg.clone_cfg.clone_combinations
    ), "Object variants with placement assignments cannot use optional clone combinations."

    ordered_assignments = []
    configured_objects = set()
    for asset_name, asset_cfg in vars(scene_cfg).items():
        if isinstance(asset_cfg, RigidObjectCollectionCfg):
            assert all(
                cloner.num_spawn_variants(member.spawn) == 1 for member in asset_cfg.rigid_objects.values()
            ), "Heterogeneous rigid-object collections cannot be combined with assigned Object variants."
        spawn_cfg = getattr(asset_cfg, "spawn", None)
        if spawn_cfg is None or cloner.num_spawn_variants(spawn_cfg) <= 1:
            continue
        assert (
            asset_name in assignments
        ), f"Multi-spawner '{asset_name}' must be declared as an Object with variants to share placement assignments."
        assignment = assignments[asset_name]
        assert isinstance(asset_cfg, AssetBaseCfg), f"Object '{asset_name}' must use an asset configuration."
        prim_path = cloner.expand_env_regex_ns(asset_cfg.prim_path, scene_cfg.clone_cfg.clone_template)
        assert (
            cloner.path.match(prim_path, scene_cfg.clone_cfg.clone_template) is not None
        ), f"Object '{asset_name}' variants require an environment-scoped prim path."
        assert isinstance(
            spawn_cfg, MultiAssetSpawnerCfg
        ), f"Object '{asset_name}' must retain its native MultiAssetSpawnerCfg."
        variant_count = len(assignment.spawn_config_hashes)
        assert (
            len(spawn_cfg.assets_cfg) == variant_count
        ), f"Object '{asset_name}' variant count changed after placement assignment."
        actual_hashes = tuple(dict_to_md5_hash(cfg) for cfg in spawn_cfg.assets_cfg)
        assert (
            actual_hashes == assignment.spawn_config_hashes
        ), f"Object '{asset_name}' spawn variants changed after placement assignment."
        indices = assignment.variant_indices
        assert len(indices) == scene_cfg.num_envs, (
            f"Object '{asset_name}' has variants assigned for {len(indices)} environments, "
            f"but the final scene has {scene_cfg.num_envs}."
        )
        assert all(
            type(index) is int and 0 <= index < variant_count for index in indices
        ), f"Object '{asset_name}' has invalid variant indices."
        ordered_assignments.append((asset_name, variant_count, indices))
        configured_objects.add(asset_name)

    missing_objects = assignments.keys() - configured_objects
    assert not missing_objects, f"Object variants missing from the final scene: {sorted(missing_objects)}."
    return tuple(ordered_assignments)


@contextmanager
def object_variant_clone_strategy(
    scene_cfg: InteractiveSceneCfg, assignments: dict[str, ObjectVariantAssignmentCfg]
) -> Iterator[None]:
    """Install the placement assignment only while Isaac Lab constructs its scene.

    Keeping the bound callback out of saved configuration preserves Isaac Lab's
    ``to_dict`` / ``from_dict`` roundtrip used by Hydra training entry points.

    Args:
        scene_cfg: Runtime scene configuration, including any Hydra overrides.
        assignments: Serializable placement assignments from ArenaEnvBuilder.
    """
    ordered_assignments = _validate_scene_variant_assignments(scene_cfg, assignments)
    if not ordered_assignments:
        yield
        return
    original_strategy = scene_cfg.clone_cfg.clone_strategy
    scene_cfg.clone_cfg.clone_strategy = _ObjectVariantCloneStrategy(ordered_assignments).select_variants
    try:
        yield
    finally:
        scene_cfg.clone_cfg.clone_strategy = original_strategy


@dataclass(frozen=True)
class _ObjectVariantCloneStrategy:
    """Supply the bound object variants to Isaac Lab's native clone planner."""

    assignments: tuple[tuple[str, int, tuple[int, ...]], ...]
    """Object name, declared variant count, and variant indices in environment order."""

    def select_variants(self, combinations: torch.Tensor, num_clones: int, device: str) -> torch.Tensor:
        """Return the same environment assignments used to compute placement bounds."""
        assert combinations.ndim == 2 and len(combinations) > 0, "Expected a nonempty clone-combination matrix."
        assert bool((combinations >= 0).all()), "Assigned Object variants require every asset in every environment."
        variant_counts = combinations.max(dim=0).values + 1
        variant_columns = (variant_counts > 1).nonzero(as_tuple=False).flatten().tolist()
        assert len(variant_columns) == len(
            self.assignments
        ), "Native clone-plan variants differ from the configured Object assignments."
        # Homogeneous columns remain zero regardless of where Isaac Lab inserts them.
        chosen = torch.zeros((num_clones, combinations.shape[1]), dtype=torch.long, device=device)
        for column, (object_name, variant_count, assignment) in zip(variant_columns, self.assignments):
            assert (
                int(variant_counts[column]) == variant_count
            ), f"Native variant count for Object '{object_name}' differs from its placement assignment."
            assert (
                len(assignment) == num_clones
            ), f"Native environment count for Object '{object_name}' differs from its placement assignment."
            chosen[:, column] = torch.tensor(assignment, dtype=torch.long, device=device)
        return chosen
