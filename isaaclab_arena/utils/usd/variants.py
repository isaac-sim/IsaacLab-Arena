# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""USD variant selection utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd


def usd_variant_selections_to_dict(variants: object | dict[str, str] | None) -> dict[str, str]:
    """Return Isaac Lab USD variant selections as a new dictionary.

    Args:
        variants: A dictionary or config class mapping variant set names to variant selections, or None.

    Returns:
        A copy of the variant selections, or an empty dictionary when unset.
    """
    if variants is None:
        values = {}
    elif isinstance(variants, dict):
        values = dict(variants)
    else:
        # Converted the way Isaac Lab's select_usd_variants does. Imported here because it loads torch.
        from isaaclab.utils.dict import class_to_dict

        values = class_to_dict(variants)
    assert all(
        isinstance(key, str) and isinstance(value, str) for key, value in values.items()
    ), "variant names and selections must be strings"
    return values


def apply_usd_variant_selections(stage: Usd.Stage, variants: object | dict[str, str] | None) -> None:
    """Select USD variants on the stage's default prim.

    Does nothing when there are no selections: None, an empty dictionary, or a config class without fields.

    Args:
        stage: Open USD stage.
        variants: A dictionary or config class mapping variant set names to variant selections, or None.
    """
    selections = usd_variant_selections_to_dict(variants)
    if not selections:
        return
    root = stage.GetDefaultPrim()
    if not root:
        return
    variant_sets = root.GetVariantSets()
    for set_name, selection in selections.items():
        if set_name in variant_sets.GetNames():
            variant_set = variant_sets.GetVariantSet(set_name)
            if selection in variant_set.GetVariantNames():
                variant_set.SetVariantSelection(selection)
