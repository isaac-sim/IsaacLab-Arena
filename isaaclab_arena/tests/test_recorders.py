# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.utils.isaaclab_utils.recorders import add_trajectory_recorder_terms

TRAJECTORY_TERM_NAMES = (
    "record_initial_state",
    "record_post_step_states",
    "record_pre_step_actions",
    "record_post_step_processed_actions",
    "record_episode_id",
    "record_end_effector_poses",
    "record_gripper_state",
)


class _MetricRecorder(RecorderTerm):
    """Stands in for a metric's recorder term, which metrics read back out of the exported dataset."""

    def record_post_step(self):
        return "success", None


@configclass
class _MetricRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = _MetricRecorder


@configclass
class _MetricsRecorderManagerCfg(RecorderManagerBaseCfg):
    record_success_rate = _MetricRecorderCfg()


def test_trajectory_terms_are_added():
    extended = add_trajectory_recorder_terms(_MetricsRecorderManagerCfg())

    for term_name in TRAJECTORY_TERM_NAMES:
        assert getattr(extended, term_name, None) is not None, f"missing trajectory term {term_name}"


def test_existing_metric_terms_survive():
    # Metrics are computed by reading their own terms back out of the exported dataset, so dropping
    # them here would silently produce a dataset the metrics cannot be derived from.
    extended = add_trajectory_recorder_terms(_MetricsRecorderManagerCfg())

    # The merged class is synthesized at runtime, so its terms are reached by name.
    metric_term = getattr(extended, "record_success_rate", None)
    assert isinstance(metric_term, RecorderTermCfg)
    assert metric_term.class_type is _MetricRecorder


def test_base_recorder_settings_survive():
    original = _MetricsRecorderManagerCfg()
    original.dataset_export_dir_path = "/custom/export/dir"
    original.dataset_filename = "custom_filename"

    extended = add_trajectory_recorder_terms(original)

    assert isinstance(extended, RecorderManagerBaseCfg)
    assert extended.dataset_export_dir_path == "/custom/export/dir"
    assert extended.dataset_filename == "custom_filename"


def test_input_config_is_left_unextended():
    original = _MetricsRecorderManagerCfg()

    add_trajectory_recorder_terms(original)

    # The caller keeps a config it can still build a non-recording environment from.
    assert not hasattr(original, "record_episode_id")
