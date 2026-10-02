# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stage isolation after authoring USD without creating a SimulationContext."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_teardown_discards_in_memory_stage(simulation_app):
    import omni.usd
    from isaaclab.sim.utils import create_new_stage, get_current_stage

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import teardown_simulation_app

    previous = create_new_stage()
    previous.DefinePrim("/LeftoverGeometry", "Xform")

    teardown_simulation_app(make_new_stage=True)

    assert get_current_stage() != previous, "Teardown must discard Isaac Lab's previous in-memory stage"
    current = omni.usd.get_context().get_stage()
    assert current is not None
    assert current != previous
    assert not current.GetPrimAtPath("/LeftoverGeometry")
    return True


def test_teardown_discards_in_memory_stage():
    assert run_function_with_persistent_simulation_app(_test_teardown_discards_in_memory_stage)
