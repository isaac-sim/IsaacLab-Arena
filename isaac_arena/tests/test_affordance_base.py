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

import pytest

from isaac_arena.tests.utils.subprocess import run_simulation_app_function_in_separate_process

HEADLESS = True


def _test_affordance_base(simulation_app):

    from isaac_arena.affordances.openable import Openable
    from isaac_arena.assets.asset import Asset

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
    result = run_simulation_app_function_in_separate_process(
        _test_affordance_base,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_affordance_base()
