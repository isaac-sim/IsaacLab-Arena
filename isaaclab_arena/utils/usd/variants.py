# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""USD variant selection utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd


def usd_variant_selections_to_dict(variants: object | None) -> dict[str, str]:
    """Return Isaac Lab USD variant selections as a new dictionary.

    Args:
        variants: Variant selections given as a dictionary, configclass, or ``None``.

    Returns:
        A copy of the variant selections, or an empty dictionary when unset.
    """
    if variants is None:
        values = {}
    elif isinstance(variants, dict):
        values = dict(variants)
    else:
        values = variants.to_dict()
    assert isinstance(values, dict), "configclass variants must convert to a dictionary"
    assert all(
        isinstance(key, str) and isinstance(value, str) for key, value in values.items()
    ), "variant names and selections must be strings"
    return values


def apply_usd_variant_selections(stage: Usd.Stage, variants: object | None) -> None:
    """Select USD variants on the stage's default prim.

    Args:
        stage: Open USD stage.
        variants: Variant selections as Isaac Lab accepts them: a dictionary mapping each variant set
            name to the variant to select, a configclass with one field per set, or ``None``.
    """
    root = stage.GetDefaultPrim()
    if not root:
        return
    variant_sets = root.GetVariantSets()
    for set_name, selection in usd_variant_selections_to_dict(variants).items():
        if set_name in variant_sets.GetNames():
            variant_set = variant_sets.GetVariantSet(set_name)
            if selection in variant_set.GetVariantNames():
                variant_set.SetVariantSelection(selection)
