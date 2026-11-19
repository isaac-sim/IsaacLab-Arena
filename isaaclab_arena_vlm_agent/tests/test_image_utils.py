# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check VLM JPEG data URLs without a simulation runtime."""

import base64
import io
import numpy as np

import pytest
from PIL import Image

from isaaclab_arena_vlm_agent.image_utils import image_to_jpeg_data_url


@pytest.mark.parametrize("max_width_or_height,expected_size", [(4, (4, 2)), (32, (16, 8))])
def test_image_encoding_preserves_spatial_color_and_aspect_ratio(max_width_or_height, expected_size):
    pixels = np.zeros((8, 16, 4), dtype=np.uint8)
    pixels[:, :8, 0] = 255
    pixels[:, 8:, 2] = 255
    pixels[..., 3] = 255
    url, size = image_to_jpeg_data_url(pixels, max_width_or_height)
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert image.size == size == expected_size
    assert image.mode == "RGB"
    left, right = image.getpixel((0, 0)), image.getpixel((image.width - 1, 0))
    assert left[0] > left[2] + 100
    assert right[2] > right[0] + 100
