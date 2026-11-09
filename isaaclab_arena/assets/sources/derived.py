# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""USD source adapter for reusable cached asset derivation."""

from __future__ import annotations

from isaaclab_arena.assets.derivation.cache import derive_usd_package
from isaaclab_arena.assets.sources.base import UsdSource


class DerivedUsdSource(UsdSource):
    """Derive and atomically cache a complete USD package on first use."""

    def __init__(self, source: str | UsdSource, derivation_spec: str, *, source_identity: str | None = None):
        """Configure derivation with a cache identity available before acquisition.

        Args:
            source: Local USD entry path or a source that acquires its package.
            derivation_spec: Filesystem path or package resource containing the recipe.
            source_identity: Stable identity matching the recipe. Defaults to the path string
                or the source's optional ``identity()`` method. Required for resolve-only sources.
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
        self.source = source
        self.source_identity = source_identity
        self.derivation_spec = derivation_spec
        self._resolved: str | None = None

    def resolve(self) -> str:
        """Return the cached derived entry point, building its package on a miss."""
        if self._resolved is None:
            self._resolved = derive_usd_package(self.source, self.derivation_spec, source_identity=self.source_identity)
        return self._resolved
