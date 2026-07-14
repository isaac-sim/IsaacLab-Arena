# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CAP-specific Maple/DROID asset routing helpers.

These helpers keep the registered Maple environment and graph-YAML environments on the same staging/local
asset path logic. They intentionally target only the CAP Maple table and DROID scene closure.
"""

from __future__ import annotations

import copy
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

_PROD_NUCLEUS_HOST = "omniverse-content-production.s3-us-west-2.amazonaws.com"
_STAGING_NUCLEUS_HOST = "omniverse-content-staging.s3.us-west-2.amazonaws.com"
_LOCAL_ASSET_HOSTS = {
    _PROD_NUCLEUS_HOST,
    _STAGING_NUCLEUS_HOST,
    "omniverse-content-production.s3.us-west-2.amazonaws.com",
}
_ASSET_PROVENANCE_FILE = "CAP_ASSET_PROVENANCE.json"


def to_staging_url(url: str) -> str:
    """Rewrite a production Nucleus asset URL to its Isaac staging-bucket equivalent."""
    staged = url.replace(_PROD_NUCLEUS_HOST, _STAGING_NUCLEUS_HOST)
    assert staged != url, f"staging override did not rewrite the production host in: {url}"
    return staged


def local_asset_path(source_url: str, local_root: str | Path) -> str:
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


def load_local_asset_provenance(local_root: str | Path) -> dict:
    """Read and validate the baked CAP asset provenance file."""
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


def staging_subclass(asset_cls: type) -> tuple[type, str]:
    """Return an instance-local staging subclass without mutating the registered asset class."""
    staged = to_staging_url(asset_cls.usd_path)
    return type(f"{asset_cls.__name__}Staging", (asset_cls,), {"usd_path": staged}), staged


def local_asset_subclass(asset_cls: type, local_root: str | Path) -> tuple[type, str, str]:
    """Return an instance-local asset subclass backed by the baked mirror."""
    source_url = getattr(asset_cls, "usd_path", None)
    if not source_url:
        raise RuntimeError(f"CAP asset class has no usd_path: {asset_cls.__name__}")
    local_path = local_asset_path(source_url, local_root)
    localized = type(f"{asset_cls.__name__}Local", (asset_cls,), {"usd_path": local_path})
    return localized, source_url, local_path


def assert_cap_local_asset_root_supports_embodiment(
    embodiment_registry_name: str,
    *,
    local_asset_root: Path | None,
) -> None:
    """Fail closed when the baked CAP asset closure is requested for a non-DROID embodiment."""
    if local_asset_root is None:
        return
    assert "droid" in embodiment_registry_name, (
        "CAP_LOCAL_ASSET_ROOT contains the CAP DROID scene closure; "
        f"got unsupported embodiment '{embodiment_registry_name}'."
    )


def _graph_spec_asset_registry_names(graph_spec) -> list[str]:
    names = [graph_spec.background.registry_name]
    names.extend(obj.registry_name for obj in graph_spec.objects)
    return list(dict.fromkeys(names))


def _routed_graph_asset_class(
    registry_name: str,
    asset_cls: type,
    *,
    use_staging_assets: bool,
    local_asset_root: Path | None,
) -> type:
    routed_cls = asset_cls
    if use_staging_assets and registry_name == "maple_table_robolab":
        routed_cls, _ = staging_subclass(routed_cls)
    if local_asset_root is not None:
        source_url = getattr(routed_cls, "usd_path", None)
        if source_url:
            routed_cls, _, _ = local_asset_subclass(routed_cls, local_asset_root)
    return routed_cls


@contextmanager
def cap_asset_registry_overrides_for_graph_spec(
    graph_spec,
    *,
    use_staging_assets: bool,
    local_asset_root: Path | None,
):
    """Temporarily route graph-spec USD asset classes before they are instantiated.

    Graph conversion constructs assets immediately, and USD-backed assets may open their USD files inside
    ``__init__``. For the unpromoted Maple/DROID CAP assets this means routing must happen before
    ``ArenaEnvGraphSpec.to_arena_env()``.
    """
    assert_cap_local_asset_root_supports_embodiment(
        graph_spec.embodiment.registry_name,
        local_asset_root=local_asset_root,
    )
    if not use_staging_assets and local_asset_root is None:
        yield
        return

    from isaaclab_arena.assets.registries import AssetRegistry

    registry = AssetRegistry()
    originals: dict[str, type] = {}
    try:
        for registry_name in _graph_spec_asset_registry_names(graph_spec):
            asset_cls = registry.get_asset_by_name(registry_name)
            routed_cls = _routed_graph_asset_class(
                registry_name,
                asset_cls,
                use_staging_assets=use_staging_assets,
                local_asset_root=local_asset_root,
            )
            if routed_cls is not asset_cls:
                originals[registry_name] = asset_cls
                registry._components[registry_name] = routed_cls
        yield
    finally:
        registry._components.update(originals)


def apply_staging_stand_override(embodiment) -> str:
    """Point a DROID embodiment's stand at staging without mutating shared defaults."""
    stand_usd = getattr(embodiment.scene_config.stand.spawn, "usd_path", None)
    assert stand_usd, (
        "--use_staging_assets set but the DROID stand has no spawn.usd_path to rewrite "
        f"(stand spawn: {type(embodiment.scene_config.stand.spawn).__name__})."
    )
    staged = to_staging_url(stand_usd)
    stand = copy.deepcopy(embodiment.scene_config.stand)
    stand.spawn.usd_path = staged
    embodiment.scene_config.stand = stand
    return staged


def apply_local_droid_asset_override(embodiment, local_root: str | Path) -> dict[str, str]:
    """Localize DROID robot and stand spawn configs without mutating shared defaults."""
    source_urls = {}
    local_paths = {}
    for name in ("robot", "stand"):
        asset_cfg = getattr(embodiment.scene_config, name)
        source_url = getattr(asset_cfg.spawn, "usd_path", None)
        if not source_url:
            raise RuntimeError(f"DROID {name} has no spawn.usd_path to localize")
        localized_cfg = copy.deepcopy(asset_cfg)
        localized_cfg.spawn.usd_path = local_asset_path(source_url, local_root)
        setattr(embodiment.scene_config, name, localized_cfg)
        source_urls[f"{name}_source_usd"] = source_url
        local_paths[f"{name}_local_usd"] = localized_cfg.spawn.usd_path
    return {**source_urls, **local_paths}


def refresh_object_asset_cfg(asset) -> None:
    """Refresh cached object/event configs after changing an already-instantiated asset's USD path."""
    if hasattr(asset, "_init_object_cfg"):
        asset.object_cfg = asset._init_object_cfg()
    if hasattr(asset, "_init_event_cfg"):
        asset.event_cfg = asset._init_event_cfg()


def rewrite_usd_backed_asset(asset, new_usd_path: str) -> str:
    """Rewrite an instantiated USD-backed asset and refresh its cached config."""
    old_usd_path = getattr(asset, "usd_path", None)
    if not old_usd_path:
        raise RuntimeError(f"asset {getattr(asset, 'name', type(asset).__name__)} has no usd_path to rewrite")
    asset.usd_path = new_usd_path
    asset.bounding_box = None
    refresh_object_asset_cfg(asset)
    return old_usd_path


def apply_cap_asset_overrides(
    arena_env,
    *,
    use_staging_assets: bool,
    local_asset_root: Path | None,
) -> dict[str, object]:
    """Apply CAP Maple/DROID staging and local asset routing to a built Arena environment."""
    provenance: dict[str, object] = {
        "asset_channel": "staging" if use_staging_assets else "production",
        "asset_materialization": "local_baked" if local_asset_root is not None else "remote",
        "asset_source_urls": {},
        "asset_resolved_usds": {},
    }
    if not use_staging_assets and local_asset_root is None:
        return provenance

    embodiment = arena_env.embodiment
    assert_cap_local_asset_root_supports_embodiment(
        getattr(embodiment, "name", ""),
        local_asset_root=local_asset_root,
    )

    for asset in arena_env.scene.assets.values():
        if getattr(asset, "name", None) == "maple_table_robolab":
            source_url = asset.usd_path
            resolved = source_url
            if source_url.startswith("https://"):
                if use_staging_assets and _PROD_NUCLEUS_HOST in source_url:
                    resolved = to_staging_url(source_url)
                if local_asset_root is not None:
                    resolved = local_asset_path(resolved, local_asset_root)
            if resolved != source_url:
                rewrite_usd_backed_asset(asset, resolved)
            provenance["table_source_usd"] = source_url
            provenance["table_resolved_usd"] = resolved

    if "droid" in getattr(embodiment, "name", ""):
        if use_staging_assets:
            provenance["droid_stand_staging_usd"] = apply_staging_stand_override(embodiment)
        if local_asset_root is not None:
            provenance["droid_asset_paths"] = apply_local_droid_asset_override(embodiment, local_asset_root)

    if local_asset_root is not None:
        provenance["baked_asset_tree_sha256"] = load_local_asset_provenance(local_asset_root)["tree_sha256"]
    return provenance
