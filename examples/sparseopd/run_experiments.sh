#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_sparseopd.sh"
SPARSEOPD_CONFIG_PATH="${SPARSEOPD_CONFIG_PATH:-${REPO_ROOT}/config/sparseopd/qwen3_1.7b_common.yaml}"
if [[ "${SPARSEOPD_CONFIG_PATH}" != /* ]]; then
    SPARSEOPD_CONFIG_PATH="${REPO_ROOT}/${SPARSEOPD_CONFIG_PATH}"
fi
if [[ ! -f "${SPARSEOPD_CONFIG_PATH}" ]]; then
    echo "SparseOPD YAML not found: ${SPARSEOPD_CONFIG_PATH}" >&2
    exit 2
fi
export SPARSEOPD_CONFIG_PATH

# Edit this list to control the default experiment suite. Comment out entries
# to skip them, or add a method handled by train_sparseopd.sh.
EXPERIMENTS=(
    # Dense OPD baseline
    dense_opd

    # Main Sparse OPD methods
    rand1tok
    mintok
    maxtok
    minmaxtok
    randmask_0p1pct
    pctltail_0p05pct

    # Reward-threshold ablations: A_t < tau
    # at_lt_m1
    # at_lt_m2
    # at_lt_m4
    # at_lt_m8
    # at_lt_m16
    # at_lt_m32

    # Reward-threshold ablations: A_t > tau
    # at_gt_0p5
    # at_gt_1
    # at_gt_2
    # at_gt_3p5
)

if [[ ! -x "${TRAIN_SCRIPT}" ]]; then
    echo "Training launcher is missing or not executable: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

declare -a METHODS=()
if (( $# == 1 )) && [[ "$1" == "--list" ]]; then
    printf '%s\n' "${EXPERIMENTS[@]}"
    exit 0
elif (( $# == 0 )); then
    METHODS=("${EXPERIMENTS[@]}")
else
    METHODS=("$@")
fi

if (( ${#METHODS[@]} == 0 )); then
    echo "No Sparse OPD methods selected" >&2
    exit 2
fi

echo "Scheduled ${#METHODS[@]} experiment(s):"
printf '  %s\n' "${METHODS[@]}"

for method in "${METHODS[@]}"; do
    echo
    echo "===== Starting ${method} at $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    bash "${TRAIN_SCRIPT}" "${method}"
    echo "===== Completed ${method} at $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
done

echo "All scheduled Sparse OPD experiments completed."
