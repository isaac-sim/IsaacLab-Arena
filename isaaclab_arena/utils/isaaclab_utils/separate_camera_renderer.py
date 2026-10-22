# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Independent RGB render products for Isaac RTX cameras."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from uuid import uuid4

from isaaclab.renderers import RenderBufferKind
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.sensors import CameraCfg
from isaaclab.utils.warp import ProxyArray
from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderData, IsaacRtxRenderer
from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

if TYPE_CHECKING:
    from isaaclab.sensors.camera.camera_data import CameraData


class SeparateCameraRenderer(IsaacRtxRenderer):
    """Render RGB cameras independently and write into the usual batched buffers."""

    def supported_output_types(self):
        """Publish only the RGB/RGBA outputs written by the independent products."""
        supported = super().supported_output_types()
        return {kind: supported[kind] for kind in (RenderBufferKind.RGB, RenderBufferKind.RGBA)}

    def prepare_cameras(self, stage, spec: CameraRenderSpec) -> None:
        """Validate the camera before any renderer-specific stage changes."""
        assert_camera_cfg_compatible_with_independent_rendering(spec.cfg)

    def create_render_data(self, spec: CameraRenderSpec) -> list[IsaacRtxRenderData]:
        """Create one RGB annotator and render product per environment."""
        import omni.replicator.core as rep

        children = []
        for path in spec.camera_prim_paths:
            child_spec = replace(spec, camera_prim_paths=(path,), num_instances=1, view_count=1)
            product = rep.create.render_product(
                camera=path,
                resolution=(spec.cfg.width, spec.cfg.height),
                name=f"independent_camera_{uuid4().hex}",
            )
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device=spec.device, do_array_copy=False)
            annotator.attach([product.path])
            children.append(
                IsaacRtxRenderData(
                    annotators={"rgba": annotator},
                    render_product=product,
                    spec=child_spec,
                )
            )
        return children

    def set_outputs(self, render_data: list[IsaacRtxRenderData], output_data: dict[str, ProxyArray]) -> None:
        """Bind each product to its slice of the shared camera output buffers."""
        for i, child in enumerate(render_data):
            outputs = {name: ProxyArray(value.warp[i : i + 1]) for name, value in output_data.items()}
            super().set_outputs(child, outputs)

    def render(self, render_data: list[IsaacRtxRenderData]) -> None:
        """Write each environment's pixels into its camera output slice."""
        for child in render_data:
            super().render(child)

    def read_output(self, render_data: list[IsaacRtxRenderData], camera_data: CameraData) -> None:
        """Clear metadata for RGB/RGBA outputs, which carry no annotator metadata."""
        for name in camera_data.info:
            camera_data.info[name] = None

    def cleanup(self, render_data: list[IsaacRtxRenderData] | None) -> None:
        """Detach annotators and release every owned render product."""
        if render_data is not None:
            for child in render_data:
                if child.render_product is not None:
                    super().cleanup(child)


def assert_camera_cfg_compatible_with_independent_rendering(camera_cfg: CameraCfg) -> None:
    """Require an RTX RGB camera without ISP or a custom background."""
    assert isinstance(
        camera_cfg.renderer_cfg, IsaacRtxRendererCfg
    ), "Untiled Arena cameras require the Isaac RTX renderer"
    assert camera_cfg.renderer_cfg.class_type in (
        IsaacRtxRendererCfg().class_type,
        IsaacRtxRenderer,
        SeparateCameraRenderer,
    ), "Untiled Arena cameras cannot replace a custom renderer"
    assert camera_cfg.data_types and set(camera_cfg.data_types) <= {
        "rgb",
        "rgba",
    }, "Untiled Arena cameras currently support only RGB/RGBA outputs"
    assert camera_cfg.isp_cfg is None, "Untiled Arena cameras do not support ISP"
    assert camera_cfg.background_color is None, "Untiled Arena cameras do not support custom background colors"
