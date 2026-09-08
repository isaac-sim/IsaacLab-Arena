# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import warp as wp
from isaaclab.utils.warp import ProxyArray

from isaaclab_arena.utils.isaaclab_utils.warp_patch import install_empty_cpu_warp_to_torch_patch


@pytest.mark.parametrize(
    ("dtype", "expected_shape"),
    [
        (wp.float32, (1, 0)),
        (wp.vec2f, (1, 0, 2)),
    ],
)
def test_empty_cpu_warp_array_to_torch(dtype, expected_shape):
    install_empty_cpu_warp_to_torch_patch()

    array = wp.empty((1, 0), dtype=dtype, device="cpu")

    assert wp.to_torch(array).shape == expected_shape
    assert ProxyArray(array).torch.shape == expected_shape


def test_nonempty_cpu_warp_array_to_torch_is_unchanged():
    install_empty_cpu_warp_to_torch_patch()

    array = wp.zeros(2, dtype=wp.float32, device="cpu")
    tensor = wp.to_torch(array)

    assert tensor.data_ptr() == array.ptr
