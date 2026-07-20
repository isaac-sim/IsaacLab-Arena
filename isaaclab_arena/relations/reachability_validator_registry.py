# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pull an enabled extension\'s build-time pooled-placement reachability validator.

Lets a run gate pooled placement on reachability from config, without core importing the validator\'s
dependencies. An enabled extension (e.g. ``isaaclab_arena_curobo``) exports a
``make_reachability_validator(embodiment, **kwargs)`` entry point; ``resolve_reachability_validator``
imports the extension by name and calls it into the pool\'s solve loop when a run sets
``ArenaEnvBuilderCfg.validate_reachability``. Core stays vendor-agnostic: the module names
come from config and the entry point is discovered by name, so no cuRobo import appears in core.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Well-known entry point an extension exports to provide the build-time reachability gate.
_REACHABILITY_VALIDATOR_FACTORY_ATTR = "make_reachability_validator"

if TYPE_CHECKING:
    from collections.abc import Callable

    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.relations.placement_result import PlacementResult

    ReachabilityValidator = Callable[[PlacementResult], bool]


def resolve_reachability_validator(
    embodiment: EmbodimentBase, module_names: list[str] | None, **kwargs
) -> ReachabilityValidator:
    """Build the reachability validator exported by the first enabled extension that provides one.

    Imports each enabled extension (cached if already imported) and looks for a
    ``make_reachability_validator(embodiment, **kwargs)`` entry point, calling the first one found.

    Args:
        embodiment: Embodiment the validator is built for.
        module_names: Enabled extension modules to search.
        kwargs: Extra tuning forwarded verbatim to the entry point (e.g. IK thresholds).
    """
    for module_name in module_names or []:
        factory = getattr(importlib.import_module(module_name), _REACHABILITY_VALIDATOR_FACTORY_ATTR, None)
        if factory is not None:
            return factory(embodiment, **kwargs)
    raise AssertionError(
        f"No enabled extension exports {_REACHABILITY_VALIDATOR_FACTORY_ATTR}(); "
        "enable one that provides the build-time reachability gate (e.g. 'isaaclab_arena_curobo')."
    )
