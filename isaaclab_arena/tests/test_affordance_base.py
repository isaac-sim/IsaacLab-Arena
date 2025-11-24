# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _test_affordance_base(simulation_app):

    from isaaclab_arena.affordances.openable import Openable
    from isaaclab_arena.assets.asset import Asset

    class NotAnAsset:

        def __init__(self, blah: str, **kwargs):
            super().__init__(**kwargs)
            self.blah = blah

    class OpenableAsset(Asset, Openable):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class OpenableNotAnAsset(NotAnAsset, Openable):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    _ = OpenableAsset(name="test_name", openable_joint_name="test_joint_name", openable_open_threshold=0.5)

    with pytest.raises(TypeError) as exception_info:
        _ = OpenableNotAnAsset(blah="test_name", openable_joint_name="test_joint_name", openable_open_threshold=0.5)
    assert "Can't instantiate abstract class" in str(exception_info.value)

    return True


def test_affordance_base():
    result = run_simulation_app_function(
        _test_affordance_base,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_affordance_base()
