#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Collect the Kit crash dumps and crash reports, and encrypt them to the SSH public keys of a GitHub user.
#
# Crash dumps hold process memory and, on Linux, the process environment, which contains CI secrets. The
# repository is public, so a dump must never be uploaded or printed unencrypted. This script writes only the
# encrypted archive to the output directory, and writes nothing if there are no dumps or no usable keys.
#
# Usage: encrypt_crash_dumps.sh <output_dir>
# Environment:
#   CRASH_DUMP_RECIPIENT_GITHUB_USER  GitHub user whose public SSH keys (ssh-ed25519, ssh-rsa) can decrypt.
#   CRASH_DUMP_SOURCE_DIR             Directory searched for dumps. Defaults to Isaac Sim's Kit data directory.

set -euo pipefail
OUTPUT_DIR="$1"
RECIPIENT_USER="${CRASH_DUMP_RECIPIENT_GITHUB_USER:?set CRASH_DUMP_RECIPIENT_GITHUB_USER}"
SOURCE_DIR="${CRASH_DUMP_SOURCE_DIR:-/isaac-sim/kit/data/Kit/IsaacLab}"

mapfile -t dump_files < <(find "$SOURCE_DIR" -maxdepth 2 -type f \
    \( -name '*.dmp' -o -name '*.dmp.zip' -o -name 'crash_*.txt' -o -name '*.py.txt' \) 2> /dev/null | sort)
if [ "${#dump_files[@]}" -eq 0 ]; then
    echo "No Kit crash dumps found in $SOURCE_DIR."
    exit 0
fi
echo "Found ${#dump_files[@]} crash file(s):"
ls -l "${dump_files[@]}" | awk '{print $5, $NF}'

if ! command -v age > /dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq age > /dev/null
fi

recipients_file=$(mktemp)
trap 'rm -f "$recipients_file"' EXIT
curl -fsS "https://github.com/${RECIPIENT_USER}.keys" | grep -E '^(ssh-ed25519|ssh-rsa) ' > "$recipients_file" || true
if [ ! -s "$recipients_file" ]; then
    echo "::warning::No usable public SSH keys found for ${RECIPIENT_USER}; not collecting crash dumps."
    exit 0
fi
echo "Encrypting to $(wc -l < "$recipients_file") public key(s) of ${RECIPIENT_USER}."

mkdir -p "$OUTPUT_DIR"
tar czf - "${dump_files[@]}" 2> /dev/null | age -R "$recipients_file" -o "$OUTPUT_DIR/crash_dumps.tar.gz.age"
echo "Wrote $(wc -c < "$OUTPUT_DIR/crash_dumps.tar.gz.age") encrypted bytes to $OUTPUT_DIR/crash_dumps.tar.gz.age"
