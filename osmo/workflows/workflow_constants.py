# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# tag denoting the output folder on OSMO
OSMO_TASK_OUTPUT_DIR = "{{output}}"


# Swift / PDX prefixes
HTTPS_URL_PREFIX = "https://pdx.s8k.io/v1"
SWIFT_URL_PREFIX = "swift://pdx.s8k.io"

# Dataset path and URLs
DATASETS_PATH = "AUTH_team-isaac/isaaclab_arena/workflows"
DATASETS_HTTPS_URL = f"{HTTPS_URL_PREFIX}/{DATASETS_PATH}"
DATASETS_SWIFT_URL = f"{SWIFT_URL_PREFIX}/{DATASETS_PATH}"

# The path for a single run.
DATASET_HTTPS_URL = f"{HTTPS_URL_PREFIX}/{DATASETS_PATH}/{{{{workflow_id}}}}"
DATASET_SWIFT_URL = f"{SWIFT_URL_PREFIX}/{DATASETS_PATH}/{{{{workflow_id}}}}"
