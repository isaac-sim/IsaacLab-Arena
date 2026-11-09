# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""USD source adapter for cached, standards-based USDA overlays."""

from __future__ import annotations

from isaaclab_arena.assets.derivation.cache import cache_usd_overlay
from isaaclab_arena.assets.sources.base import UsdSource


class OverlayUsdSource(UsdSource):
    """Apply a committed USDA overlay to a lazily resolved USD source."""

    def __init__(
        self,
        source: str | UsdSource,
        overlay_spec: str,
        *,
        expected_source_sha256: str | None = None,
        source_identity: str | None = None,
    ):
        """Configure an overlay and its optional source integrity check.

        Args:
            source: Local USD path or source resolved when the overlay cache misses.
            overlay_spec: Filesystem or package-resource path to a USDA wrapper.
            expected_source_sha256: Optional SHA-256 digest of the resolved USD entry.
            source_identity: Stable cache identity. Defaults to the path or source identity.
        """
        assert isinstance(source, (str, UsdSource)), "source must be a USD path string or UsdSource"
        if isinstance(source, str):
            assert source, "source path must not be empty"
        if source_identity is None:
            if isinstance(source, str):
                source_identity = source
            else:
                identity = getattr(source, "identity", None)
                if callable(identity):
                    source_identity = identity()
        assert (
            isinstance(source_identity, str) and source_identity
        ), "source_identity must be a non-empty string; provide it for sources without identity()"
        if expected_source_sha256 is not None:
            assert len(expected_source_sha256) == 64 and all(
                character in "0123456789abcdef" for character in expected_source_sha256
            ), "expected_source_sha256 must be a lowercase SHA-256 digest"
        self.source = source
        self.source_identity = source_identity
        self.overlay_spec = overlay_spec
        self.expected_source_sha256 = expected_source_sha256
        self._resolved: str | None = None

    def resolve(self) -> str:
        """Return the cached USDA overlay, resolving its source on a cache miss."""
        if self._resolved is None:
            self._resolved = cache_usd_overlay(
                self.source,
                self.overlay_spec,
                source_identity=self.source_identity,
                expected_source_sha256=self.expected_source_sha256,
            )
        return self._resolved
