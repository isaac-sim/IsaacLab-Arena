# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mimic env cfg for the static-base G1 (WBC stands the robot in place; no nav).

There is no ``StaticPickAndPlaceTask`` class: the static env reuses
``PickAndPlaceTask`` directly and injects this cfg through
``PickAndPlaceTask(mimic_env_cfg_factory=...)``. See
``galileo_g1_static_pick_and_place_environment.py`` for the closure that
constructs ``StaticPickAndPlaceMimicEnvCfg`` with the env's pickup +
destination names. Inheriting a task subclass purely to swap which Mimic cfg
class gets returned was the pattern cvolk's
``cvolk/refactor-task-mimic-channels`` branch deleted; this file follows that
convention for the static variant.

The cfg subclass ``StaticPickAndPlaceMimicEnvCfg`` survives because it still
encodes the static-specific arm and body subtask shapes (left + body collapsed
to single no-op subtasks) that ``G1PickAndPlaceMimicEnvCfg`` does not provide:
the parent encodes the locomanip 4-phase nav body sequence and a left-arm
subtask sequence, neither of which fires in the static env where the base
never moves and the left arm hangs idle.
"""

from isaaclab.envs.mimic_env_cfg import SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_arena.tasks.pick_and_place_task import G1PickAndPlaceMimicEnvCfg


@configclass
class StaticPickAndPlaceMimicEnvCfg(G1PickAndPlaceMimicEnvCfg):
    """Mimic env cfg for the static-base G1 pick-and-place task.

    Inherits the right-arm subtask sequence (``idle_right`` -> ``grasp_and_idle_right``
    -> final) and all datagen knobs from ``G1PickAndPlaceMimicEnvCfg`` so the
    generated dataset's right-arm semantics line up with the locomanip variant. Three
    overrides:

    1. ``datagen_config.name`` is rebranded ``static_pick_and_place_*`` so generated
       datasets are not confused with locomanip ones in the converter / training pipeline.
    2. ``subtask_configs["body"]`` is replaced with a single subtask spanning the whole
       episode (no ``subtask_term_signal``, so it never triggers segmentation). The
       locomanip version expects the env to emit ``navigate_to_table`` /
       ``navigate_turn_inplace`` / ``navigate_to_bin`` term signals as the robot drives
       between waypoints; in the static env the robot never moves its base, so those
       signals never fire and Mimic would deadlock waiting for them. Collapsing to a
       single no-op subtask lets Mimic treat the body channel as one homogeneous block
       (the recorded body actions are constant ``stand-in-place`` commands anyway).
    3. ``subtask_configs["left"]`` is also collapsed to a single no-op subtask. Apple-to-
       plate on a single shelf is a one-arm pinch-grasp task: the right arm reaches,
       grasps, and places while the left arm just hangs. Forcing the user to mark
       ``idle_left`` and ``grasp_and_idle_left`` boundaries during ``annotate_demos.py``
       is annotation theatre -- the left arm never actually grasps anything. The collapse
       drops the per-arm annotation count from 4 marks (2 right + 2 left) to 2 (right
       only) and encodes the task's right-arm-only nature in the cfg itself. Mimic still
       generates the left arm's trajectory verbatim from the source demo (single subtask,
       no segmentation, no nearest-neighbor switch).

    The right arm keeps the locomanip's 3-subtask sequence with ``object_ref=apple``
    everywhere; the place/final subtask's ``object_ref`` is technically the destination
    plate per the parent ``PickPlaceMimicEnvCfg`` TODO, but with a fixed-pose plate the
    distinction is degenerate and changing it here would diverge from the locomanip
    apple cfg without observable benefit.
    """

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = f"static_pick_and_place_{self.pick_up_object_name}_to_{self.destination_name}_D0"

        # Replace the locomanip's 3-step left-arm subtask sequence (``idle_left`` ->
        # ``grasp_and_idle_left`` -> final) with a single no-op. Same shape as the body
        # collapse below; ``action_noise=0`` keeps the left arm's recorded trajectory
        # bit-identical during datagen so the unused arm doesn't drift into the right
        # arm's workspace under noise injection.
        self.subtask_configs["left"] = [_no_op_subtask_config(self.pick_up_object_name)]

        # Replace the locomanip's 4-step nav body subtask sequence with a single no-op.
        # Common knobs match the locomanip body subtasks (action_noise=0, no interpolation)
        # so the body channel is never perturbed during data generation.
        #
        # ``object_ref`` is kept set to the pickup object purely to satisfy the
        # ``SubTaskConfig`` field contract -- the ``nearest_neighbor_object`` selection
        # strategy is never exercised here because there is only one body subtask, so
        # subtask selection is a no-op for the body group.
        self.subtask_configs["body"] = [_no_op_subtask_config(self.pick_up_object_name)]


def _no_op_subtask_config(object_ref: str) -> SubTaskConfig:
    """Build a single-subtask placeholder for an eef channel that never segments.

    Used for both the ``body`` and ``left`` channels in the static cfg: the channel
    runs to the end of the demo (no ``subtask_term_signal``), Mimic plays back the
    recorded trajectory verbatim (``action_noise=0``, no interpolation), and the
    ``nearest_neighbor_object`` strategy is effectively a no-op because there is only
    one subtask to select from.

    ``object_ref`` is kept set to a real scene object (the pickup object by convention)
    purely to satisfy the ``SubTaskConfig`` field contract; it has no behavioural
    effect when there is only one subtask in the channel.
    """

    return SubTaskConfig(
        object_ref=object_ref,
        # No subtask_term_signal -> Mimic treats this as the final subtask of the
        # channel (runs to end of demo), matching the "last subtask has no term signal"
        # convention used by the per-arm subtask lists.
        first_subtask_start_offset_range=(0, 0),
        subtask_term_offset_range=(0, 0),
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs={"nn_k": 3},
        action_noise=0.0,
        num_interpolation_steps=0,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
    )
