# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from isaaclab.utils import configclass

from isaaclab_arena.utils.configclass import combine_configclasses


def test_combine_configclasses_with_multiple_inheritance():

    # Side A - A class with a base class
    @configclass
    class FooCfgBase:
        a: int = 1
        b: int = 2

    @configclass
    class FooCfg(FooCfgBase):
        c: int = 3
        a: int = 4

    # Side B - A class without a base class
    @configclass
    class BarCfg(FooCfgBase):
        d: int = 4
        e: int = 5

    # Combine the two classes
    CombinedCfg = combine_configclasses("CombinedCfg", FooCfg, BarCfg, bases=(FooCfgBase,))
    assert CombinedCfg().d() == 4
    assert CombinedCfg().c() == 3
    assert CombinedCfg().b() == 2
    assert CombinedCfg().a() == 4
    assert CombinedCfg().e() == 5
    assert isinstance(CombinedCfg(), FooCfgBase)


def test_combine_configclasses_with_inheritance():

    # Side A - A class with a base class
    @configclass
    class FooCfgBase:
        a: int = 1
        b: int = 2

    @configclass
    class FooCfg(FooCfgBase):
        c: int = 3
        a: int = 4

    # Side B - A class without a base class
    @configclass
    class BarCfg:
        d: int = 4

    # Combine the two classes
    CombinedCfg = combine_configclasses("CombinedCfg", FooCfg, BarCfg, bases=(FooCfgBase,))
    assert CombinedCfg().d() == 4
    assert CombinedCfg().c() == 3
    assert CombinedCfg().b() == 2
    assert CombinedCfg().a() == 4
    assert isinstance(CombinedCfg(), FooCfgBase)


def test_combine_configclasses_with_post_init():

    # Side A - A class with a base class
    @configclass
    class FooCfg:
        a: int = 1
        b: int = 2

        def __post_init__(self):
            self.a = self.a() + 1
            self.b = self.b() + 1

    # Side B - A class without a base class
    @configclass
    class BarCfg:
        c: int = 3

        def __post_init__(self):
            self.c = self.c() + 1

    # Combine the two classes
    CombinedCfg = combine_configclasses("CombinedCfg", FooCfg, BarCfg)
    assert CombinedCfg().a == 2
    assert CombinedCfg().b == 3
    assert CombinedCfg().c == 4


def test_combine_configclass_instances_preserves_default_factory_nested_config():
    """Merged classes must not assign raw ``default_factory`` callables as field defaults (see ``get_field_info``)."""
    from isaaclab.utils import configclass

    from isaaclab_arena.utils.configclass import combine_configclass_instances

    @configclass
    class Inner:
        x: int = 0

    @configclass
    class Outer:
        inner: Inner = Inner()

    merged = combine_configclass_instances("MergedOuter", Outer())
    assert not callable(merged.inner)
    assert type(merged.inner).__name__ == "Inner"
    assert merged.inner.x == 0


if __name__ == "__main__":
    test_combine_configclasses_with_multiple_inheritance()
    test_combine_configclasses_with_inheritance()
    test_combine_configclasses_with_post_init()
    test_combine_configclass_instances_preserves_default_factory_nested_config()
