#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Apply the dependency versions used by the temporary GB300 OpenPI server.

set -euo pipefail

OPENPI_PYTHON="${1:-/.venv/bin/python}"

test -x "${OPENPI_PYTHON}"
command -v uv >/dev/null

# Avoid OpenPI's project-level override that pins ml-dtypes to the older CUDA 12 stack.
uv --no-config pip uninstall --python "${OPENPI_PYTHON}" \
    jax-cuda12-plugin \
    jax-cuda12-pjrt

uv --no-config pip install --python "${OPENPI_PYTHON}" --upgrade \
    'jax[cuda13]==0.7.2' \
    'numpy==2.4.6' \
    'ml-dtypes==0.6.0' \
    'flax==0.10.2' \
    'orbax-checkpoint==0.11.13' \
    'tensorstore==0.1.74'

"${OPENPI_PYTHON}" -c 'import jax; print(f"JAX {jax.__version__}: {jax.devices()}")'
