# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Public API tests for box and cylindrical position-limit relations."""

import pytest

from isaaclab_arena.relations.relation_loss_strategies import (
    PositionLimitsBoxLossStrategy,
    PositionLimitsCylindricalLossStrategy,
)
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import PositionLimits, PositionLimitsBox, PositionLimitsCylindrical


def test_position_limit_relations_have_distinct_public_names():
    assert PositionLimitsBox(x_max=1.0).name == "position_limits_box"
    assert PositionLimitsCylindrical(center_x=0.0, center_y=0.0, radius_max=1.0).name == "position_limits_cylindrical"


def test_position_limit_relations_reject_other_geometry_parameters():
    with pytest.raises(TypeError):
        PositionLimitsBox(center_x=0.0, center_y=0.0, radius_max=1.0)
    with pytest.raises(TypeError):
        PositionLimitsCylindrical(center_x=0.0, center_y=0.0, radius_max=1.0, x_max=1.0)


def test_position_limits_legacy_python_name_aliases_box_relation():
    assert PositionLimits is PositionLimitsBox


def test_position_limit_relations_have_separate_default_loss_strategies():
    strategies = RelationSolverParams().strategies
    assert isinstance(strategies[PositionLimitsBox], PositionLimitsBoxLossStrategy)
    assert isinstance(strategies[PositionLimitsCylindrical], PositionLimitsCylindricalLossStrategy)
