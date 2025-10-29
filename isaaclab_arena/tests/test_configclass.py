# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
