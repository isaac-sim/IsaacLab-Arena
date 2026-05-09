# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime patch for ``isaaclab_physx.renderers.isaac_rtx_renderer_utils.ensure_isaac_rtx_render_update``.

Lab's stock implementation early-returns whenever any visualizer reports
``viz.pumps_app_update() == True``. On the first call for a fresh
``SimulationContext`` the visualizer hasn't actually pumped yet (``sim.render()``
has not been called), so skipping the pump leaves annotator buffers empty and
produces a black first frame in scripts that rely on a populated frame before
the first ``env.step()`` (notably ``record_demos.py``).

Apply by calling :func:`patch_isaac_rtx_renderer` once, after ``AppLauncher``
has initialized Kit and before any camera/renderer is constructed.
"""


def patch_isaac_rtx_renderer() -> None:
    """Replace ``ensure_isaac_rtx_render_update`` with a version that performs
    the initial ``app.update()`` itself when called for the first time on a new
    ``SimulationContext``, instead of deferring to a visualizer that has not
    yet had a chance to pump.

    Patches both the canonical module (``isaac_rtx_renderer_utils``) and the
    importing module (``isaac_rtx_renderer``), since the latter does
    ``from .isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update`` at
    load time and holds its own binding.
    """
    import isaaclab.sim as sim_utils
    import isaaclab_physx.renderers.isaac_rtx_renderer as rtx_renderer
    import isaaclab_physx.renderers.isaac_rtx_renderer_utils as rtx_utils

    def patched_ensure_isaac_rtx_render_update() -> None:
        sim = sim_utils.SimulationContext.instance()
        if sim is None:
            return

        key = (id(sim), sim._physics_step_count)
        if rtx_utils._last_render_update_key == key:
            return

        if key[0] != rtx_utils._last_render_update_key[0]:
            rtx_utils._streaming_is_busy = False
            rtx_utils._streaming_subscribed = False
            rtx_utils._streaming_subscription = None

        # On the very first call for a new SimulationContext the visualizer has
        # not had a chance to pump (sim.render() was never called), so we must
        # perform the initial app.update() ourselves to populate annotator
        # buffers. Subsequent calls defer to the visualizer.
        first_call_for_sim = rtx_utils._last_render_update_key[0] != id(sim)
        if not first_call_for_sim and any(viz.pumps_app_update() for viz in sim.visualizers):
            rtx_utils._last_render_update_key = key
            return

        if not sim.is_rendering:
            return

        rtx_utils._ensure_streaming_subscription()
        sim.physics_manager.forward()

        import omni.kit.app

        sim.set_setting("/app/player/playSimulations", False)
        omni.kit.app.get_app().update()

        if rtx_utils._streaming_is_busy:
            rtx_utils._wait_for_streaming_complete()

        sim.set_setting("/app/player/playSimulations", True)

        rtx_utils._last_render_update_key = key

    rtx_utils.ensure_isaac_rtx_render_update = patched_ensure_isaac_rtx_render_update
    rtx_renderer.ensure_isaac_rtx_render_update = patched_ensure_isaac_rtx_render_update

    print("Patched isaaclab_physx.renderers.isaac_rtx_renderer_utils.ensure_isaac_rtx_render_update")
