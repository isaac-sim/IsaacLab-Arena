#!/usr/bin/env bash
# Convert a GPU memory reservation into the fraction JAX reads from XLA_PYTHON_CLIENT_MEM_FRACTION.
#
# The reservation is either a fraction of the GPU's memory in (0, 1], such as 0.75, or an absolute size
# such as 12GB, 12GiB, 12G, 6000MB, 6000MiB or 6000M (GB and MB are read as GiB and MiB, the units
# nvidia-smi reports). An absolute size is divided by the total memory of the first visible GPU, which
# makes the same 12GB reserve a different fraction on a 24 GB and a 48 GB card.
#
# Usage:
#   bash gpu_memory_fraction.sh 12GB      # prints 0.2501 on a 48 GB GPU
#   source gpu_memory_fraction.sh && gpu_memory_fraction 0.75
set -euo pipefail

gpu_memory_fraction() {
    local memory="$1"
    local mebibytes total_mebibytes
    if [[ "$memory" =~ ^(0?\.[0-9]+|1(\.0+)?)$ ]]; then
        if awk -v f="$memory" 'BEGIN { exit !(f > 0) }'; then
            printf '%s\n' "$memory"
            return 0
        fi
        echo "GPU memory fraction must be above 0 and at most 1, got $memory" >&2
        return 1
    fi
    if [[ "$memory" =~ ^([0-9]+(\.[0-9]+)?)[[:space:]]*(G|GB|GiB|M|MB|MiB)$ ]]; then
        local amount="${BASH_REMATCH[1]}" unit="${BASH_REMATCH[3]}"
        case "$unit" in
            G|GB|GiB) mebibytes=$(awk -v a="$amount" 'BEGIN { printf "%.3f", a * 1024 }') ;;
            *) mebibytes="$amount" ;;
        esac
        total_mebibytes="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
        if ! [[ "$total_mebibytes" =~ ^[0-9]+$ ]]; then
            echo "could not read the GPU memory total from nvidia-smi (got '$total_mebibytes')" >&2
            return 1
        fi
        if awk -v m="$mebibytes" -v t="$total_mebibytes" 'BEGIN { exit !(m <= 0 || m > t) }'; then
            echo "GPU memory $memory must be above 0 and at most the GPU's $total_mebibytes MiB" >&2
            return 1
        fi
        awk -v m="$mebibytes" -v t="$total_mebibytes" 'BEGIN { printf "%.4f\n", m / t }'
        return 0
    fi
    echo "GPU memory must be a fraction in (0, 1] such as 0.75 or a size such as 12GB or 6000MB, got '$memory'" >&2
    return 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ $# -ne 1 ]]; then
        echo "usage: $(basename "$0") <fraction|size>, e.g. 0.75 or 12GB" >&2
        exit 1
    fi
    gpu_memory_fraction "$1"
fi
