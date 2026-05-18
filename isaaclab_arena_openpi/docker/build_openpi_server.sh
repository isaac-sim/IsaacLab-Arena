#!/usr/bin/env bash
# Encapsulates the openpi inference server as a self-contained Docker image.
#
# Usage:
#   ./build_openpi_server.sh                        # build at the pinned commit
#   ./build_openpi_server.sh --src-dir=<path>       # build from a local openpi checkout
#                                                   # (uses whatever commit is checked out)
#
# The pinned commit lives in the OPENPI_COMMIT file next to this script.
# To bump it, edit that file. To build at a different commit ad-hoc, use --src-dir.
#
# Env overrides:
#   IMAGE_NAME    Local image name (default: isaaclab_arena_openpi-server)
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-isaaclab_arena_openpi-server}"
OPENPI_REPO="https://github.com/Physical-Intelligence/openpi"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC_DIR=""
for arg in "$@"; do
    case "$arg" in
        --src-dir=*) SRC_DIR="${arg#*=}" ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

if [ -n "$SRC_DIR" ]; then
    OPENPI_DIR="$SRC_DIR"
    echo "Using local openpi checkout at ${OPENPI_DIR}"
else
    PINNED_COMMIT="$(tr -d '[:space:]' < "${SCRIPT_DIR}/OPENPI_COMMIT")"
    OPENPI_DIR="$TMPDIR/openpi"
    echo "Cloning openpi at ${PINNED_COMMIT} ..."
    # Partial clone: skip blob objects, git fetches them on demand at checkout.
    # Cuts the one-time clone size on a ~GB-scale repo without changing what we end up with.
    GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --filter=blob:none "$OPENPI_REPO" "$OPENPI_DIR"
    (cd "$OPENPI_DIR" && git checkout "$PINNED_COMMIT")
fi

cd "$OPENPI_DIR"

SHORT_HASH=$(git rev-parse --short HEAD)

echo "Building ${IMAGE_NAME}:${SHORT_HASH}"

# Upstream's Dockerfile installs deps but expects source to be volume-mounted.
# Append a COPY step so the image is self-contained.
cat scripts/docker/serve_policy.Dockerfile > "$TMPDIR/Dockerfile"
cat >> "$TMPDIR/Dockerfile" <<'PATCH'

# --- isaaclab_arena_openpi patches ---
COPY . /app
PATCH

docker build \
    --network=host \
    -f "$TMPDIR/Dockerfile" \
    -t "${IMAGE_NAME}:${SHORT_HASH}" \
    -t "${IMAGE_NAME}:latest" \
    .

echo "Built ${IMAGE_NAME}:${SHORT_HASH} (also tagged :latest)"
