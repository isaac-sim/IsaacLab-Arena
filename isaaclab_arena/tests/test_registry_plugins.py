# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that the registries load the ``isaaclab_arena.plugins`` entry points of installed packages."""

import sys
from importlib.metadata import EntryPoint

import pytest

from isaaclab_arena.assets import registries
from isaaclab_arena.assets.registries import AssetRegistry

# A plugin package's module: importing it registers an asset class, and register() counts its calls.
PLUGIN_MODULE = """
from isaaclab_arena.assets.register import register_asset

calls = 0


def register():
    global calls
    calls += 1


@register_asset
class PluginAsset:
    registry_name = "plugin_asset"
"""


@pytest.fixture
def install_plugins(tmp_path, monkeypatch):
    """Return a function that installs plugins with the given entry-point values; the next registry lookup loads them."""
    # Register Arena's own components now: registered into the copy below, they would be lost when it is discarded.
    registries.ensure_assets_registered()
    registry = AssetRegistry()
    monkeypatch.setattr(registry, "_components", dict(registry._components))
    monkeypatch.setattr(registries, "_assets_registered", False)
    # Load the tests' plugins even if the shell turns plugins off; the registration above follows the shell.
    monkeypatch.delenv("ISAACLAB_ARENA_DISABLE_PLUGIN_AUTOLOAD", raising=False)
    (tmp_path / "arena_test_plugin.py").write_text(PLUGIN_MODULE)
    monkeypatch.syspath_prepend(tmp_path)

    def install(*values: str):
        plugins = [EntryPoint("test_plugin", value, "isaaclab_arena.plugins") for value in values]
        monkeypatch.setattr(
            registries, "entry_points", lambda group: plugins if group == "isaaclab_arena.plugins" else []
        )

    yield install
    sys.modules.pop("arena_test_plugin", None)


def test_module_plugin_registers_by_import(install_plugins):
    install_plugins("arena_test_plugin")
    assert AssetRegistry().is_registered("plugin_asset")


def test_function_plugin_is_called(install_plugins):
    install_plugins("arena_test_plugin:register")
    registries.ensure_assets_registered()
    assert sys.modules["arena_test_plugin"].calls == 1


def test_disabled_autoload_skips_plugins(install_plugins, monkeypatch):
    monkeypatch.setenv("ISAACLAB_ARENA_DISABLE_PLUGIN_AUTOLOAD", "1")
    install_plugins("arena_test_plugin", "arena_test_plugin:register")
    assert not AssetRegistry().is_registered("plugin_asset")
    assert "arena_test_plugin" not in sys.modules


def test_failing_plugin_raises_naming_itself(install_plugins):
    install_plugins("arena_test_plugin_missing")
    with pytest.raises(ImportError, match="'test_plugin = arena_test_plugin_missing'"):
        registries.ensure_assets_registered()


def test_class_plugin_is_rejected(install_plugins):
    install_plugins("arena_test_plugin:PluginAsset")
    with pytest.raises(ImportError, match="expected a module or a function"):
        registries.ensure_assets_registered()
