ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.0-dev2

FROM ${BASE_IMAGE}

USER root

# GR00T Policy Build Arguments, these are only used if INSTALL_GROOT is true
ARG INSTALL_GROOT=false

ARG WORKDIR="/workspaces/isaaclab_arena"
ENV WORKDIR=${WORKDIR}
WORKDIR "${WORKDIR}"

# Hide conflicting Vulkan files, if needed.
RUN if [ -e "/usr/share/vulkan" ] && [ -e "/etc/vulkan" ]; then \
      mv /usr/share/vulkan /usr/share/vulkan_hidden; \
    fi

# Install apt packages
RUN apt-get update && apt-get install -y \
  git \
  git-lfs \
  cmake \
  sudo \
  python3-pip

################################
# Install Isaac Lab
################################
COPY submodules/IsaacLab-Arena/submodules/IsaacLab ${WORKDIR}/submodules/IsaacLab
ENV ISAACLAB_PATH=${WORKDIR}/submodules/IsaacLab
ENV TERM=xterm
RUN ln -s /isaac-sim/ ${ISAACLAB_PATH}/_isaac_sim
RUN for DIR in ${ISAACLAB_PATH}/source/isaaclab*/; do \
      /isaac-sim/python.sh -m pip install --no-deps -e "$DIR"; \
    done
RUN chmod 777 -R /isaac-sim/kit/
RUN chmod a+x /isaac-sim
RUN /isaac-sim/python.sh -m pip install --no-deps -e ${ISAACLAB_PATH}/source/isaaclab_visualizers
RUN /isaac-sim/python.sh -m pip install --no-deps -e ${ISAACLAB_PATH}/source/isaaclab_teleop
RUN ${ISAACLAB_PATH}/isaaclab.sh -i
RUN /isaac-sim/python.sh -m pip install isaacteleop~=1.0 --extra-index-url https://pypi.nvidia.com
RUN /isaac-sim/python.sh -m pip install --upgrade pip

# Lightwheel server
ENV LW_API_ENDPOINT="https://api-dev.lightwheel.net"

# HuggingFace CLI for downloading datasets and models.
# Use pipx so the hf binary gets an isolated venv with all its deps (e.g. requests),
# without touching system Python packages.
RUN apt-get install -y pipx && \
    PIPX_HOME=/opt/pipx PIPX_BIN_DIR=/usr/local/bin \
    pipx install \
    "huggingface-hub[cli]" \
    pre-commit

###############################
# Install GR00T and CUDA 12.8 #
###############################
COPY submodules/IsaacLab-Arena/docker/setup/install_cuda.sh /tmp/install_cuda.sh
RUN chmod +x /tmp/install_cuda.sh
RUN if [ "$INSTALL_GROOT" = "true" ]; then \
    /tmp/install_cuda.sh; \
  else \
    echo "Skipping CUDA 12.8 installation"; \
  fi && \
  rm -f /tmp/install_cuda.sh

COPY submodules/IsaacLab-Arena/submodules/Isaac-GR00T ${WORKDIR}/submodules/Isaac-GR00T
COPY submodules/IsaacLab-Arena/docker/setup/install_gr00t_deps.sh /tmp/install_gr00t_deps.sh
RUN chmod +x /tmp/install_gr00t_deps.sh
RUN if [ "$INSTALL_GROOT" = "true" ]; then \
    /tmp/install_gr00t_deps.sh && echo 'export GROOT_DEPS_DIR=/opt/groot_deps' > /etc/profile.d/groot_deps.sh; \
  else \
    echo "Skipping GR00T installation" && rm -f /etc/profile.d/groot_deps.sh; \
  fi && rm -f /tmp/install_gr00t_deps.sh

################################
# Install IsaacLab Arena
################################
COPY submodules/IsaacLab-Arena ${WORKDIR}
RUN /isaac-sim/python.sh -m pip install -e "${WORKDIR}/[dev]"

# Set an alias to run the isaac lab python interpreter.
RUN echo "alias python='/isaac-sim/python.sh'" >> /etc/bash.bashrc
RUN echo "alias pip3='/isaac-sim/python.sh -m pip'" >> /etc/bash.bashrc
RUN echo "alias pytest='/isaac-sim/python.sh -m pytest'" >> /etc/bash.bashrc
RUN echo "alias debugpy='python -Xfrozen_modules=off -m debugpy --listen localhost:5678 --wait-for-client'" >> /etc/bash.bashrc
RUN echo "PS1='[IsaacLab Arena] \[\e[0;32m\]~\u \[\e[0;34m\]\w\[\e[0m\] \$ '" >> /etc/bash.bashrc
RUN echo "alias ll='ls -alF --color=auto'" >> /etc/bash.bashrc
RUN echo "alias ..='cd ..'" >> /etc/bash.bashrc
RUN cp /etc/bash.bashrc /root/.bashrc

################################
# Install <project_name>
################################
COPY pyproject.toml /workspaces/<project_name>/pyproject.toml
COPY <project_name> /workspaces/<project_name>/<project_name>
RUN /isaac-sim/python.sh -m pip install -e /workspaces/<project_name>

# Set the entrypoint
COPY submodules/IsaacLab-Arena/docker/setup/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
