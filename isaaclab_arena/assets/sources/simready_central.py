# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SimReady Central USD acquisition."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from isaaclab_arena.assets.sources.base import UsdSource

_ASSET_PATH_BY_URL: dict[str, str] = {}


class SimReadyCentralUsdSource(UsdSource):
    """Retrieve and reuse a pinned SimReady Central USD through Isaac Lab's asset cache."""

    def __init__(self, url: str):
        repo_id, revision, filename, _ = _parse_central_asset_url(url)
        self.url = url
        self._download_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"

    def to_local_usd_path(self) -> str:
        """Return the USD with its dependencies localized by Isaac Lab."""
        from isaaclab.utils.assets import retrieve_file_path

        if self._download_url not in _ASSET_PATH_BY_URL:
            _ASSET_PATH_BY_URL[self._download_url] = retrieve_file_path(self._download_url)
        return _ASSET_PATH_BY_URL[self._download_url]


def is_central_asset_url(url: str) -> bool:
    """Identify Hugging Face dataset URLs handled by the Central asset loader."""
    parsed = urlsplit(url)
    return (
        parsed.scheme in ("https", "hf") and parsed.netloc == "huggingface.co" and parsed.path.startswith("/datasets/")
    )


def _parse_central_asset_url(url: str) -> tuple[str, str, str, str]:
    """Parse a revision-pinned Central dataset URL and its asset package folder."""
    assert is_central_asset_url(url), "expected a Hugging Face dataset URL"
    parsed = urlsplit(url)
    assert not parsed.query and not parsed.fragment, "expected a plain asset URL"
    match = re.fullmatch(r"/datasets/([^/]+/[^/]+)/(?:resolve|blob)/([0-9a-f]{40})/(.+)", parsed.path)
    assert match, "expected a pinned Hugging Face dataset URL"
    repo_id, revision, filename = match.groups()
    path = PurePosixPath(filename)
    assert not path.is_absolute() and ".." not in path.parts, "invalid asset path"
    assert not any(char in filename for char in "*?[]\\"), "invalid asset path"
    assert path.suffix in (".usd", ".usda", ".usdc", ".usdz"), "expected USD asset"
    assert "simready_usd" in path.parts, "unsupported Central package layout"
    # Central keeps the entrypoint in <asset>/simready_usd/.
    folder = PurePosixPath(*path.parts[: path.parts.index("simready_usd")])
    assert str(folder) != ".", "asset must have a package folder"
    return repo_id, revision, filename, str(folder)
