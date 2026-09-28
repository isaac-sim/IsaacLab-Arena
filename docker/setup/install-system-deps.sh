#!/bin/bash
# Install common runtime dependencies and configure Isaac Sim permissions.

set -euo pipefail

# Hide conflicting Vulkan files, if needed.
if [ -e /usr/share/vulkan ] && [ -e /etc/vulkan ]; then
    mv /usr/share/vulkan /usr/share/vulkan_hidden
fi

# Preserve Kit's existing writable installation and host-user access.
chmod 777 -R /isaac-sim/kit/
chmod a+x /isaac-sim

# Install version-control, build, media, and command-line tools used in the container.
apt-get update
apt-get install -y git git-lfs cmake ffmpeg sudo jq python3-pip
# Add a lightweight image viewer without its optional recommended packages.
apt-get install -y --no-install-recommends pqiv
