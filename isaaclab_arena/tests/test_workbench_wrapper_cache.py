# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise wrapper revision caching through Isaac Lab's actual USD retrieval."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from isaaclab_arena_environments.workbench_cleanup.asset_processing.usd_asset_processor import (
    UsdAssetProcessor,
    retrieve_pinned_usd_path,
)

MODULE = "isaaclab_arena_environments.workbench_cleanup.asset_processing.usd_asset_processor"
SOURCE_URL = "https://example.com/revision/source.usda"


@pytest.fixture
def remote_wrapper(tmp_path, monkeypatch):
    import isaaclab.utils.assets as assets_utils
    import omni.client

    cache = tmp_path / "cache"
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: cache)
    monkeypatch.setattr(assets_utils, "_REMOTE_FINGERPRINTS", {})
    monkeypatch.setattr(assets_utils, "_ASSET_SOURCES", {})
    payloads = {
        SOURCE_URL: b'#usda 1.0\n(defaultPrim = "Source")\ndef Xform "Source" (prepend references = @body.usda@) {}\n',
        "https://example.com/revision/body.usda": b"""#usda 1.0
(defaultPrim = "Body")
def Xform "Body" {
    custom asset texture = @texture.png@
    custom int source_value = 1
}
""",
        "https://example.com/revision/texture.png": b"texture contents",
    }

    def stat(url, *args, **kwargs):
        payload = payloads.get(url)
        if payload is None:
            return omni.client.Result.ERROR_NOT_FOUND, None
        return omni.client.Result.OK, SimpleNamespace(
            hash=hashlib.sha256(payload).hexdigest(), version="revision", size=len(payload), modified_time=""
        )

    def copy(url, target, behavior):
        Path(target).write_bytes(payloads[url])
        return omni.client.Result.OK

    copies = Mock(side_effect=copy)
    monkeypatch.setattr(omni.client, "stat", stat)
    monkeypatch.setattr(omni.client, "copy", copies)
    wrapper = tmp_path / "wrapper.usda"
    wrapper.write_text(f"""#usda 1.0
(defaultPrim = "Asset")
def Xform "Asset" (prepend references = @{SOURCE_URL}@</Source>) {{
    custom int override_value = 2
}}
""")
    return wrapper, cache, copies


def _read_override(path):
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    assert stage and not stage.GetCompositionErrors()
    return stage.GetDefaultPrim().GetAttribute("override_value").Get()


def test_wrapper_edits_reuse_downloads_and_keep_prior_output(remote_wrapper):
    wrapper, cache, copies = remote_wrapper
    first = retrieve_pinned_usd_path(str(wrapper))
    assert _read_override(first) == 2
    assert copies.call_count == 3
    assert retrieve_pinned_usd_path(str(wrapper)) == first
    assert copies.call_count == 3

    wrapper.write_text(wrapper.read_text().replace("override_value = 2", "override_value = 3"))
    second = retrieve_pinned_usd_path(str(wrapper))
    assert second != first
    assert _read_override(second) == 3
    assert _read_override(first) == 2
    assert copies.call_count == 3, "Editing a wrapper must not download its unchanged dependencies again"
    assert len(list((cache / "downloads").rglob("texture.png"))) == 1

    processed = UsdAssetProcessor(SOURCE_URL, str(wrapper)).process()
    assert _read_override(processed) == 3
    assert copies.call_count == 3, "Both processing paths must share the same downloads"


def test_local_sublayer_edits_invalidate_output_without_remote_downloads(remote_wrapper):
    wrapper, _, copies = remote_wrapper
    settings = wrapper.with_name("settings.usda")
    settings.write_text('#usda 1.0\nover "Asset" {\n    custom int local_value = 1\n}\n')
    wrapper.write_text(
        wrapper.read_text().replace(
            '(defaultPrim = "Asset")',
            """(
    defaultPrim = "Asset"
    subLayers = [@settings.usda@]
)""",
        )
    )
    first = retrieve_pinned_usd_path(str(wrapper))
    settings.write_text(settings.read_text().replace("local_value = 1", "local_value = 2"))
    second = retrieve_pinned_usd_path(str(wrapper))

    assert first != second
    assert copies.call_count == 3
    from pxr import Usd

    stage = Usd.Stage.Open(second)
    assert stage.GetDefaultPrim().GetAttribute("local_value").Get() == 2


def test_identical_wrappers_in_different_directories_keep_their_local_dependencies(tmp_path, monkeypatch):
    from pxr import Usd

    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: tmp_path / "cache")
    paths = []
    for value in (1, 2):
        folder = tmp_path / str(value)
        folder.mkdir()
        (folder / "settings.usda").write_text(f'#usda 1.0\ndef Xform "Asset" {{\n    custom int value = {value}\n}}\n')
        wrapper = folder / "wrapper.usda"
        wrapper.write_text('#usda 1.0\n(defaultPrim = "Asset"\nsubLayers = [@settings.usda@])\n')
        paths.append(retrieve_pinned_usd_path(str(wrapper)))
    assert paths[0] != paths[1]
    for value, path in enumerate(paths, start=1):
        stage = Usd.Stage.Open(path)
        assert stage.GetDefaultPrim().GetAttribute("value").Get() == value


def test_corrupt_output_is_rebuilt_without_redownloading_sources(remote_wrapper):
    wrapper, _, copies = remote_wrapper
    first = retrieve_pinned_usd_path(str(wrapper))
    Path(first).write_text("corrupt")
    rebuilt = retrieve_pinned_usd_path(str(wrapper))
    assert rebuilt == first
    assert _read_override(rebuilt) == 2
    assert copies.call_count == 3


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_damaged_download_is_retrieved_without_refetching_other_dependencies(remote_wrapper, damage):
    wrapper, cache, copies = remote_wrapper
    first = retrieve_pinned_usd_path(str(wrapper))
    texture = next((cache / "downloads").rglob("texture.png"))
    if damage == "missing":
        texture.unlink()
    else:
        texture.write_bytes(b"corrupt")

    assert retrieve_pinned_usd_path(str(wrapper)) == first
    assert texture.read_bytes() == b"texture contents"
    assert copies.call_count == 4
    assert copies.call_args.args[0].endswith("/texture.png")


@pytest.mark.parametrize("problem", ["missing_default_prim", "missing_reference_prim"])
def test_invalid_composition_does_not_publish_cache_entry(remote_wrapper, problem):
    wrapper, cache, _ = remote_wrapper
    text = wrapper.read_text()
    if problem == "missing_default_prim":
        text = text.replace('(defaultPrim = "Asset")', "")
    else:
        text = text.replace("</Source>", "</Missing>")
    wrapper.write_text(text)

    with pytest.raises(AssertionError):
        retrieve_pinned_usd_path(str(wrapper))
    assert not list((cache / "pinned_usd").glob("*/manifest.json"))
    assert not [path for path in (cache / "pinned_usd").iterdir() if path.is_dir()]
    assert (cache / "downloads").is_dir()
