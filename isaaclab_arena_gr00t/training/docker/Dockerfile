# Self-contained GR00T N1.6 fine-tuning image for the Alex open-microwave task.
#
# Everything is fetched from the internet at build/run time — no files from any
# particular machine are needed. On any computer with Docker + NVIDIA toolkit:
#
#   docker build -t alex-gr00t-train \
#     https://github.com/EAOZONE/IsaacLab-Arena.git#main:isaaclab_arena_gr00t/training/docker
#   docker run --gpus all --shm-size 16g -e HF_TOKEN=hf_xxx \
#     -v alex_hf_cache:/cache/huggingface -v "$PWD/checkpoints:/checkpoints" \
#     alex-gr00t-train
#
# On `docker run` the container downloads the LeRobot dataset and base model
# from HuggingFace, fine-tunes, and uploads the checkpoint back to HuggingFace.
# See README.md in this directory for details.

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv system-wide (not under /root) so the image also works on GPU
# clusters where Apptainer/Singularity runs the container as a non-root user.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Pin Isaac-GR00T to the same commit as the submodules/Isaac-GR00T checkout.
ARG GR00T_REPO=https://github.com/NVIDIA/Isaac-GR00T
ARG GR00T_COMMIT=e29d8fc50b0e4745120ae3fb72447986fe638aa6

WORKDIR /workspace
RUN git clone "${GR00T_REPO}" Isaac-GR00T \
    && cd Isaac-GR00T \
    && git checkout "${GR00T_COMMIT}"

# flash-attn compiles from source here (no GPU needed at build time, only nvcc).
# MAX_JOBS bounds the parallel nvcc jobs so the build doesn't exhaust RAM.
WORKDIR /workspace/Isaac-GR00T
ENV MAX_JOBS=4
RUN uv sync --python 3.10 && uv pip install -e .

# At runtime `uv run` must never try to re-sync the env: the filesystem is
# read-only under Apptainer/Singularity on clusters.
ENV UV_NO_SYNC=1

# Alex embodiment files (modality config, joint-space yamls) from the Arena fork.
# Pass --build-arg ARENA_REF=<commit> to pin / bust the layer cache after pushing
# config changes; the default "main" is resolved at first build and then cached.
ARG ARENA_REPO=https://github.com/EAOZONE/IsaacLab-Arena
ARG ARENA_REF=main
RUN git clone --filter=blob:none --no-checkout "${ARENA_REPO}" /workspace/IsaacLab-Arena \
    && cd /workspace/IsaacLab-Arena \
    && git sparse-checkout set isaaclab_arena_gr00t/embodiments/alex \
    && git checkout "${ARENA_REF}"

COPY upload_to_hf.py /workspace/upload_to_hf.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# wandb is a hard dependency of the trainer; disable unless WANDB_API_KEY is set.
ENV WANDB_MODE=disabled
ENV HF_HOME=/cache/huggingface

VOLUME ["/checkpoints", "/cache/huggingface"]

ENTRYPOINT ["/entrypoint.sh"]
