# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/pytorch:25.04-py3

ARG HF_TOKEN
ARG MODELS_DIR=/workspace/dreamzero/checkpoints

ENV MODELS_DIR=${MODELS_DIR}

# Install system deps needed by av (PyAV) and other packages
RUN apt-get update && apt-get install -y \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
    libswscale-dev libswresample-dev \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Clone dreamzero repo at a pinned commit
RUN git clone https://github.com/dreamzero0/dreamzero.git /workspace/dreamzero && \
    git -C /workspace/dreamzero checkout ab790c198fbce33503358efbbd4187ce9a89adf3

WORKDIR /workspace/dreamzero

# Install dreamzero package without re-resolving deps (container already has torch etc.)
RUN pip install -e . --no-deps

# Install all runtime deps
RUN pip install \
        tyro \
        "tianshou==0.5.1" \
        "transformers==4.51.3" \
        "python-socketio>=5.13.0" \
        websockets \
        "einops==0.8.1" \
        hydra-core \
        "openpi-client==0.1.1" \
        "imageio==2.34.2" \
        imageio-ffmpeg \
        "diffusers==0.30.2" \
        "peft==0.5.0" \
        sentencepiece \
        "albumentations==1.4.18" \
        ftfy \
        tiktoken \
        wandb \
        loguru \
        msgpack \
        msgpack-numpy \
        lark \
        termcolor \
        av

ENV NO_ALBUMENTATIONS_UPDATE=1

# Bake the DreamZero-DROID checkpoint into the image so eval jobs have no
# runtime dependency on HuggingFace.
RUN python -c "\
import os; \
from huggingface_hub import snapshot_download; \
snapshot_download( \
    'GEAR-Dreams/DreamZero-DROID', \
    repo_type='model', \
    local_dir=os.environ['MODELS_DIR'] + '/DreamZero-DROID', \
    token=os.environ.get('HF_TOKEN') or None)"
