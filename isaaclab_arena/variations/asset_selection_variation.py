# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build-time variation that picks the registered asset a rigid object spawns."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.variations.choice_sampler import ChoiceSampler
from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg
from isaaclab_arena.variations.sequential_sampler import SequentialSampler
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object


@configclass
class AssetSelectionSamplerCfg(SamplerBaseCfg):
    """Config for the sampler that selects a registry name in each build."""

    random_choice: bool = False
    """False takes the names in list order, one per rebuild, starting over after the last name. True draws a name at
    random, seeded by the build seed and the variation key."""

    def build(self) -> SamplerBase:
        return ChoiceSampler() if self.random_choice else SequentialSampler()


@configclass
class AssetSelectionVariationCfg(VariationBaseCfg):
    """Configuration for AssetSelectionVariation."""

    registry_names: list[str] = field(default_factory=list)
    """Registry names of the rigid objects to choose from, in this order. Required when enabled. A name listed more
    than once is selected more often."""

    sampler_cfg: AssetSelectionSamplerCfg = field(default_factory=AssetSelectionSamplerCfg)
    """Sampler that selects the registry name in each build."""


class AssetSelectionVariation(BuildTimeVariationBase):
    """Pick the registered asset a rigid object spawns; every environment of a build spawns the same asset.

    Args:
        rigid_object: Object whose asset the selection sets.
        cfg: Selection settings. By default the selection is off.
        name: Name of the variation on its Object.
    """

    cfg: AssetSelectionVariationCfg

    changes_object_geometry = True

    def __init__(
        self,
        rigid_object: Object,
        cfg: AssetSelectionVariationCfg | None = None,
        name: str = "asset_selection",
    ):
        super().__init__(cfg=cfg if cfg is not None else AssetSelectionVariationCfg(), name=name)
        self._rigid_object = rigid_object
        self._variation_key: str | None = None
        self._rebuild_index: int = 0

    def configure_at_build_time(self, seed: int, variation_key: str, rebuild_index: int = 0) -> None:
        # VariationBase does not keep these: the checks name the key, and in-order selection needs the index.
        self._variation_key = variation_key
        self._rebuild_index = rebuild_index
        super().configure_at_build_time(seed, variation_key, rebuild_index)

    def _realize_at_build_time(self) -> None:
        registry_names = list(self.cfg.registry_names)
        assert registry_names, (
            f"'{self._variation_key}' is enabled but its registry_names is empty. List the registered rigid objects"
            f" to choose from, for example '{self._variation_key}.registry_names=[cracker_box,sugar_box]'."
        )
        # Checked at build time because bounding_box can be set after the Object is created.
        assert self._rigid_object.bounding_box is None, (
            f"'{self._rigid_object.name}' sets a bounding_box override, which placement would use for every"
            " selected asset. Set bounding_box to None so placement reads the bounds of the selected asset."
        )
        # Check every name, not only the selected one, so that an unknown name or a name declared as another object
        # type fails in the first build. The check does not build the asset, which for a wrong name can open or
        # download its USD. Only the selected name is built.
        for registry_name in dict.fromkeys(registry_names):
            _assert_registered_rigid_object(registry_name)
        # The sampler reports the name to the variation recorder and returns the recorded name during replay.
        if isinstance(self.sampler, SequentialSampler):
            # Rebuild k starts at the k-th name, so the rebuilds of a Run take the names in list order.
            selected_registry_name = self.sampler.sample(1, registry_names, start_index=self._rebuild_index)[0]
        else:
            selected_registry_name = self.sampler.sample(1, registry_names)[0]
        self._rigid_object.set_spawned_asset(selected_registry_name, _build_rigid_object(selected_registry_name))


def _assert_registered_rigid_object(registry_name: str) -> None:
    """Check, without building it, that ``registry_name`` is registered and not declared as another object type.

    Args:
        registry_name: Registry name of the asset.
    """
    asset_registry = AssetRegistry()
    assert asset_registry.is_registered(
        registry_name
    ), f"'{registry_name}' is not a registered asset. Use the registry name of a rigid object."
    asset_class_or_factory = asset_registry.get_asset_by_name(registry_name)
    # LibraryObject subclasses and register_asset_factory functions declare an object type. _build_rigid_object
    # checks the type of any other asset once a build selects it.
    declared_object_type = getattr(asset_class_or_factory, "object_type", None)
    if declared_object_type is not None:
        _assert_rigid_object_type(registry_name, asset_class_or_factory.__name__, declared_object_type)


def _build_rigid_object(registry_name: str) -> Object:
    """Build the registered rigid object ``registry_name``."""
    asset = AssetRegistry().get_asset_by_name(registry_name)()
    # Classes such as ProceduralCube set their object type only in __init__. Assets that are not objects, such as
    # embodiments, have none, so they are built before this check rejects them.
    _assert_rigid_object_type(registry_name, type(asset).__name__, getattr(asset, "object_type", None))
    return asset


def _assert_rigid_object_type(registry_name: str, asset_class_name: str, object_type: ObjectType | None) -> None:
    """Check that the registered asset ``registry_name`` is a rigid object.

    Args:
        registry_name: Registry name of the asset.
        asset_class_name: Name of the asset's class or factory, shown in the error message.
        object_type: Object type of the asset, or None when it has none.
    """
    object_type_description = f"object type: {object_type.value}" if object_type is not None else "no object type"
    assert object_type == ObjectType.RIGID, (
        f"Asset '{registry_name}' ({asset_class_name}, {object_type_description}) is not a rigid object. Use the"
        " registry name of a rigid object."
    )
