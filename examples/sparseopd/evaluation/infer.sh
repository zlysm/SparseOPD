#!/usr/bin/env bash
# Generate benchmark responses with all visible GPUs; scoring is a separate step.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${EVAL_PYTHON:-path/to/conda/env/bin/python}"
CONFIG="${EVAL_CONFIG:-${REPO_ROOT}/config/sparseopd/evaluation.yaml}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${REPO_ROOT}/eval}"
FORCE=0

if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
    shift
fi
EXPERIMENTS=("$@")
GENERATE_COMMAND=("${PYTHON_BIN}" "${SCRIPT_DIR}/generate.py" --config "${CONFIG}" --output-dir "${OUTPUT_DIR}")
if (( FORCE )); then
    GENERATE_COMMAND+=(--force)
fi

if (( ${#EXPERIMENTS[@]} == 0 )); then
    PENDING_OUTPUT="$("${GENERATE_COMMAND[@]}" --list-needs-generation)"
    if [[ -n "${PENDING_OUTPUT}" ]]; then
        while IFS= read -r experiment; do
            [[ -n "${experiment}" ]] && EXPERIMENTS+=("${experiment}")
        done <<< "${PENDING_OUTPUT}"
    fi
fi

if (( ${#EXPERIMENTS[@]} == 0 )); then
    echo "All configured inference outputs are already complete."
    exit 0
fi

PIDS=()
stop_workers() {
    local pid
    for pid in "${PIDS[@]:-}"; do
        [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
    done
    while IFS= read -r pid; do
        kill "${pid}" 2>/dev/null || true
    done < <(jobs -pr)
}
trap stop_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

get_visible_gpus() {
    GPU_IDS=()
    if [[ "${CUDA_VISIBLE_DEVICES+x}" == x ]]; then
        if [[ -n "${CUDA_VISIBLE_DEVICES}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
            IFS=, read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
        fi
    else
        command -v nvidia-smi >/dev/null 2>&1 || {
            echo "nvidia-smi not found; set CUDA_VISIBLE_DEVICES or install the NVIDIA tools." >&2
            return 1
        }
        GPU_OUTPUT="$(nvidia-smi --query-gpu=index --format=csv,noheader)"
        if [[ -n "${GPU_OUTPUT}" ]]; then
            while IFS= read -r gpu; do
                [[ -n "${gpu}" ]] && GPU_IDS+=("${gpu}")
            done <<< "${GPU_OUTPUT}"
        fi
    fi
    if (( ${#GPU_IDS[@]} == 0 )); then
        echo "No visible GPUs found for inference." >&2
        return 1
    fi
}

for experiment in "${EXPERIMENTS[@]}"; do
    NEEDS_GENERATION_OUTPUT="$("${GENERATE_COMMAND[@]}" --list-needs-generation "${experiment}")"
    if [[ -z "${NEEDS_GENERATION_OUTPUT}" ]]; then
        echo "Inference already complete for ${experiment}; skipping."
        continue
    fi

    get_visible_gpus
    NUM_SHARDS=${#GPU_IDS[@]}
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$_inference"
    LOG_DIR="${OUTPUT_DIR}/logs/${RUN_ID}/${experiment}"
    mkdir -p "${LOG_DIR}"
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    export PYTHONUNBUFFERED=1
    PIDS=()

    echo "Generating ${experiment} across ${NUM_SHARDS} visible GPU(s)."
    for ((rank = 0; rank < NUM_SHARDS; rank++)); do
        echo "  shard ${rank}/${NUM_SHARDS} on GPU ${GPU_IDS[rank]}; log: ${LOG_DIR}/shard_${rank}.log"
        CUDA_VISIBLE_DEVICES="${GPU_IDS[rank]}" "${GENERATE_COMMAND[@]}" \
            --num-shards "${NUM_SHARDS}" --shard-id "${rank}" "${experiment}" \
            >"${LOG_DIR}/shard_${rank}.log" 2>&1 &
        PIDS+=("$!")
    done

    FAILURES=0
    for rank in "${!PIDS[@]}"; do
        if wait "${PIDS[rank]}"; then
            echo "Completed ${experiment} shard ${rank}/${NUM_SHARDS}"
        else
            status=$?
            echo "Shard ${rank} failed for ${experiment} (exit ${status}); see ${LOG_DIR}/shard_${rank}.log" >&2
            FAILURES=$((FAILURES + 1))
        fi
    done
    PIDS=()
    if (( FAILURES > 0 )); then
        echo "Some shards failed for ${experiment}; rerun to resume inference." >&2
        exit 1
    fi

    if (( NUM_SHARDS > 1 )); then
        "${GENERATE_COMMAND[@]}" --merge-shards --num-shards "${NUM_SHARDS}" "${experiment}"
    fi
    echo "Inference outputs ready for ${experiment}. Run evaluation/score.sh to score them."
done
