# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Toy example: Hydra schema built at runtime from a non-Hydra CLI flag.

Context
-------
Each Arena environment (see e.g. ``isaaclab_arena_environments/
pick_and_place_maple_table_environment.py``) is assembled at runtime from
argparse flags (``--pick_up_object``, ``--hdr``, ...). The variations attached
to those assets — and therefore the parameters we'd like to expose on the
command line — only become known *after* the env is built. We cannot declare
a fixed Hydra schema upfront.

This script is a minimal, Isaac-Sim-free sandbox that reproduces that shape of
problem. It demonstrates one working pattern:

1. ``argparse`` parses the non-Hydra flags (here ``--env_type``). We use
   ``parse_known_args`` so the remaining ``sys.argv`` entries — the Hydra
   overrides — are left untouched.
2. We "build" a (toy) environment that declares its variations. In the real
   code this is where we'd instantiate the
   :class:`~isaaclab_arena.environments.isaaclab_arena_environment.IsaacLabArenaEnvironment`
   and walk
   :meth:`~isaaclab_arena.scene.scene.Scene.get_variations`.
3. We build a structured Hydra schema from the env's variations
   (``dataclasses.make_dataclass``) and register it with the Hydra
   ``ConfigStore``. Defaults from each variation populate the schema so users
   only need to override what they want to change.
4. We call the Hydra compose API (``hydra.initialize`` + ``hydra.compose``) on
   the leftover argv. We use ``compose`` instead of ``@hydra.main`` because
   the latter takes over the whole CLI and won't coexist with argparse.

Run
---
.. code-block:: bash

    # Pick-and-place env: enable + narrow one color variation.
    /isaac-sim/python.sh isaaclab_arena/examples/hydra_dynamic_schema_example.py \\
        --env_type pnp \\
        pick_up_object.color.enabled=true \\
        pick_up_object.color.sampler.low=0.4 \\
        pick_up_object.color.sampler.high=0.8

    # Open-door env exposes a completely different set of overrides.
    /isaac-sim/python.sh isaaclab_arena/examples/hydra_dynamic_schema_example.py \\
        --env_type door \\
        door_object.joint_stiffness.enabled=true \\
        door_object.joint_stiffness.sampler.low=0.1 \\
        door_object.joint_stiffness.sampler.high=0.5

    # Invalid override (no door_object in pnp env) — Hydra rejects it via the
    # structured-config schema:
    /isaac-sim/python.sh isaaclab_arena/examples/hydra_dynamic_schema_example.py \\
        --env_type pnp \\
        door_object.joint_stiffness.enabled=true
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field, make_dataclass

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Variation-system stand-ins.
#
# These mirror the shape of ``UniformSampler`` and ``VariationBase`` from
# ``isaaclab_arena/variations/`` closely enough to make the plumbing below
# recognisable, without pulling in torch / Isaac Sim.
# ---------------------------------------------------------------------------


@dataclass
class UniformSamplerCfg:
    low: float = 0.0
    high: float = 1.0


@dataclass
class VariationCfg:
    enabled: bool = False
    sampler: UniformSamplerCfg = field(default_factory=UniformSamplerCfg)


# ---------------------------------------------------------------------------
# Toy environments. Each one declares a different set of per-asset variations.
# This is the asymmetry we're modelling: the Hydra schema depends on the
# environment, and we only know the environment after parsing argparse.
# ---------------------------------------------------------------------------


class ToyEnv:
    """Template: ``{asset_name: {variation_name: default VariationCfg}}``."""

    variations_by_asset: dict[str, dict[str, VariationCfg]] = {}


class PickAndPlaceToyEnv(ToyEnv):
    variations_by_asset = {
        "pick_up_object": {
            "color": VariationCfg(sampler=UniformSamplerCfg(low=0.0, high=1.0)),
        },
        "destination_object": {
            "color": VariationCfg(sampler=UniformSamplerCfg(low=0.0, high=1.0)),
        },
    }


class OpenDoorToyEnv(ToyEnv):
    variations_by_asset = {
        "door_object": {
            "joint_stiffness": VariationCfg(sampler=UniformSamplerCfg(low=0.5, high=2.0)),
        },
    }


ENV_REGISTRY: dict[str, type[ToyEnv]] = {
    "pnp": PickAndPlaceToyEnv,
    "door": OpenDoorToyEnv,
}


# ---------------------------------------------------------------------------
# Schema construction.
#
# For each asset we build a ``<Asset>Cfg`` dataclass whose fields are the
# asset's variations. We then build a top-level ``VariationsCfg`` whose fields
# are those per-asset configs. The structured schema gives us two things for
# free:
#   * typed overrides (``... .low=abc`` is rejected at parse time),
#   * rejection of unknown fields (can't set ``door_object.*`` on the pnp env).
# ---------------------------------------------------------------------------


def _asset_class_name(asset_name: str) -> str:
    # "pick_up_object" -> "PickUpObjectCfg"
    return "".join(part.capitalize() for part in asset_name.split("_")) + "Cfg"


def _build_schema_from_env(env: ToyEnv) -> type:
    """Build a top-level structured-config class from the env's variations."""
    asset_fields: list[tuple[str, type, object]] = []
    for asset_name, variations in env.variations_by_asset.items():
        per_asset_fields: list[tuple[str, type, object]] = []
        for var_name, default in variations.items():
            # default_factory must be a zero-arg callable; capture the
            # per-variation default via a default-argument and deepcopy so
            # each instantiation gets its own VariationCfg.
            per_asset_fields.append((var_name, VariationCfg, field(default_factory=(lambda d=default: deepcopy(d)))))
        asset_cls = make_dataclass(_asset_class_name(asset_name), per_asset_fields)
        asset_fields.append((asset_name, asset_cls, field(default_factory=asset_cls)))
    return make_dataclass("VariationsCfg", asset_fields)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env_type", choices=sorted(ENV_REGISTRY.keys()), required=True)
    args, hydra_overrides = parser.parse_known_args()

    # Stand-in for ``env_builder.get_env(args)`` + ``scene.get_variations()``.
    env = ENV_REGISTRY[args.env_type]()
    variations_cfg_cls = _build_schema_from_env(env)

    # Register the runtime-built schema so Hydra can validate overrides. The
    # name is arbitrary; it just has to match the ``config_name`` we pass to
    # ``compose`` below.
    ConfigStore.instance().store(name="variations_schema", node=variations_cfg_cls)

    # ``compose`` instead of ``@hydra.main``: we only want Hydra to handle the
    # leftover overrides, not take over the whole CLI. ``config_path=None``
    # plus ``version_base=None`` means "don't look on disk, use ConfigStore".
    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name="variations_schema", overrides=hydra_overrides)

    print(f"=== env_type = {args.env_type} ===")
    print(OmegaConf.to_yaml(cfg))

    # Sketch of what the next step would look like in the real pipeline: walk
    # the composed config and push each enabled entry into the underlying
    # variation object (``asset.get_variation(name).enable()`` +
    # ``.set_sampler(UniformSampler(...))``).
    print("=== Enabled variations ===")
    any_enabled = False
    for asset_name in env.variations_by_asset:
        asset_cfg = getattr(cfg, asset_name)
        for var_name in env.variations_by_asset[asset_name]:
            var_cfg = getattr(asset_cfg, var_name)
            if var_cfg.enabled:
                any_enabled = True
                print(
                    f"  {asset_name}.{var_name}: UniformSampler(low={var_cfg.sampler.low}, high={var_cfg.sampler.high})"
                )
    if not any_enabled:
        print("  (none — pass e.g. '<asset>.<variation>.enabled=true')")


if __name__ == "__main__":
    main()
