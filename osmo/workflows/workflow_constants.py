# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# OSMO expands this in task entry.sh to the mounted task output directory.
OSMO_TASK_OUTPUT_DIR = "{{output}}"


# Swift / PDX prefix
SWIFT_URL_PREFIX = "swift://pdx.s8k.io"

# Dataset path and per-run URL.
DATASETS_PATH = "AUTH_team-isaac/isaaclab_arena/workflows"
DATASET_SWIFT_URL = f"{SWIFT_URL_PREFIX}/{DATASETS_PATH}/{{{{workflow_id}}}}"

# Evaluation output dataset, written per run under the existing isaaclab_arena container.
EVAL_OUTPUT_PATH = "AUTH_team-isaac/isaaclab_arena/datasets"
EVAL_OUTPUT_SWIFT_URL = f"{SWIFT_URL_PREFIX}/{EVAL_OUTPUT_PATH}/{{{{workflow_id}}}}"
