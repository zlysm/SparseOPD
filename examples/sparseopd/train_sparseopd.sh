#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SPARSEOPD_CONFIG_PATH="${SPARSEOPD_CONFIG_PATH:-${REPO_ROOT}/config/sparseopd/qwen3_1.7b_common.yaml}"
if [[ "${SPARSEOPD_CONFIG_PATH}" != /* ]]; then
    SPARSEOPD_CONFIG_PATH="${REPO_ROOT}/${SPARSEOPD_CONFIG_PATH}"
fi
if [[ ! -f "${SPARSEOPD_CONFIG_PATH}" ]]; then
    echo "SparseOPD YAML not found: ${SPARSEOPD_CONFIG_PATH}" >&2
    exit 2
fi
CONFIG_DIR="$(dirname -- "${SPARSEOPD_CONFIG_PATH}")"
CONFIG_NAME="$(basename -- "${SPARSEOPD_CONFIG_PATH}" .yaml)"
METHOD="${1:-minmaxtok}"
if (( $# > 0 )); then
    shift
fi
CONDA_ENV="${CONDA_ENV:-path/to/conda/env}"
export SPARSEOPD_REPO_ROOT="${REPO_ROOT}"

METHOD_OVERRIDES=()
case "${METHOD}" in
    rand1tok|maxtok|mintok|minmaxtok|dense_opd)
        MODE="${METHOD}"
        [[ "${METHOD}" == "dense_opd" ]] && MODE="dense"
        METHOD_OVERRIDES+=("distillation.distillation_loss.sparse_opd_mode=${MODE}")
        ;;
    randmask_0p1pct)
        METHOD_OVERRIDES+=(
            "distillation.distillation_loss.sparse_opd_mode=randmask"
            "distillation.distillation_loss.sparse_opd_random_fraction=0.001"
        )
        ;;
    pctltail_0p05pct)
        METHOD_OVERRIDES+=(
            "distillation.distillation_loss.sparse_opd_mode=pctltail"
            "distillation.distillation_loss.sparse_opd_tail_fraction=0.0005"
        )
        ;;
    at_lt_m1|at_lt_m2|at_lt_m4|at_lt_m8|at_lt_m16|at_lt_m32)
        threshold="-${METHOD#at_lt_m}"
        METHOD_OVERRIDES+=(
            "distillation.distillation_loss.sparse_opd_mode=at_lt"
            "distillation.distillation_loss.sparse_opd_threshold=${threshold}"
        )
        ;;
    at_gt_0p5|at_gt_1|at_gt_2|at_gt_3p5)
        threshold="${METHOD#at_gt_}"
        threshold="${threshold//p/.}"
        METHOD_OVERRIDES+=(
            "distillation.distillation_loss.sparse_opd_mode=at_gt"
            "distillation.distillation_loss.sparse_opd_threshold=${threshold}"
        )
        ;;
    --help|-h)
        echo "Usage: $0 [method] [Hydra overrides...]"
        echo "Default method: minmaxtok"
        echo "Run '$0 --list' through run_experiments.sh to list experiment methods."
        exit 0
        ;;
    *)
        echo "Unsupported Sparse OPD method: ${METHOD}" >&2
        exit 2
        ;;
esac

if [[ ! -d "${CONDA_ENV}" ]]; then
    echo "Conda environment not found: ${CONDA_ENV}" >&2
    exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not on PATH; initialize conda before launching this script" >&2
    exit 1
fi
if [[ "${CONDA_PREFIX:-}" != "${CONDA_ENV}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

EXPERIMENT_SETTINGS_TEXT="$(python - "${SPARSEOPD_CONFIG_PATH}" <<'PY'
import sys
from omegaconf import OmegaConf

config = OmegaConf.load(sys.argv[1])
experiment_prefix = config.sparseopd.get("experiment_prefix")
output_dir = config.sparseopd.get("output_dir")
if not experiment_prefix or not output_dir:
    raise SystemExit(
        "The selected YAML must define sparseopd.experiment_prefix and sparseopd.output_dir."
    )
batch_size = int(config.data.get("gen_batch_size", config.data.train_batch_size))
student_dp = int(config.trainer.n_gpus_per_node)
agent_workers = int(config.actor_rollout_ref.rollout.get("agent", {}).get("num_workers", 8))
ppo_mini_batch_size = int(config.actor_rollout_ref.actor.ppo_mini_batch_size)
checks = {
    "generation batch / student DP": (batch_size, student_dp),
    "generation batch / agent workers": (batch_size, agent_workers),
    "PPO minibatch / student DP": (ppo_mini_batch_size, student_dp),
    "generation batch / PPO minibatch": (batch_size, ppo_mini_batch_size),
}
invalid = [f"{name}: {value} % {divisor} != 0" for name, (value, divisor) in checks.items() if value % divisor]
if invalid:
    raise SystemExit("Invalid SparseOPD batch configuration:\n  " + "\n  ".join(invalid))
print(experiment_prefix)
print(output_dir)
print(int(config.trainer.n_gpus_per_node) + int(config.distillation.n_gpus_per_node))
PY
)"
mapfile -t EXPERIMENT_SETTINGS <<< "${EXPERIMENT_SETTINGS_TEXT}"
if (( ${#EXPERIMENT_SETTINGS[@]} != 3 )); then
    echo "Could not read experiment metadata from ${SPARSEOPD_CONFIG_PATH}" >&2
    exit 1
fi
EXP_NAME="${EXPERIMENT_SETTINGS[0]}_${METHOD}"
OUTPUT_ROOT="${EXPERIMENT_SETTINGS[1]}/${METHOD}"
PROJECT_NAME="${EXP_NAME}"
EXPECTED_GPU_COUNT="${EXPERIMENT_SETTINGS[2]}"
if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_COUNT="$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
else
    GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi
if (( GPU_COUNT < EXPECTED_GPU_COUNT )); then
    echo "This configuration requires ${EXPECTED_GPU_COUNT} visible GPUs; found ${GPU_COUNT}" >&2
    exit 1
fi

ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%S)_$$"
RUN_CKPT_DIR="${OUTPUT_ROOT}/ckpt/${EXP_NAME}"
LOG_DIR="${OUTPUT_ROOT}/logs/${EXP_NAME}/${ATTEMPT_ID}"
TB_DIR="${OUTPUT_ROOT}/tensorboard/${EXP_NAME}"
mkdir -p "${RUN_CKPT_DIR}" "${LOG_DIR}" "${TB_DIR}"

CONSOLE_LOG="${LOG_DIR}/console_${ATTEMPT_ID}.log"
ENVIRONMENT_LOG="${LOG_DIR}/environment_${ATTEMPT_ID}.txt"
GIT_DIFF="${LOG_DIR}/git-diff_${ATTEMPT_ID}.patch"
RESOLVED_CONFIG="${LOG_DIR}/resolved_config_${ATTEMPT_ID}.yaml"
METRICS_FILE="${LOG_DIR}/metrics_${ATTEMPT_ID}.jsonl"
MERGE_LOG="${LOG_DIR}/merge_${ATTEMPT_ID}.log"
RESULT_FILE="${LOG_DIR}/result_${ATTEMPT_ID}.txt"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TENSORBOARD_DIR="${TB_DIR}"
export TENSORBOARD_FILENAME_SUFFIX="_${ATTEMPT_ID}"
export VERL_FILE_LOGGER_PATH="${METRICS_FILE}"
export VERL_ROLLOUT_DUMP_SUFFIX="_${ATTEMPT_ID}"

OVERRIDES=(
    "sparseopd.output_dir=${OUTPUT_ROOT}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXP_NAME}_${ATTEMPT_ID}"
    "trainer.default_local_dir=${RUN_CKPT_DIR}"
    "trainer.rollout_data_dir=${LOG_DIR}/rollouts"
    "hydra.run.dir=${LOG_DIR}"
    "hydra.job.chdir=false"
    "hydra.output_subdir=null"
)

{
    printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'attempt_id=%s\ncheckpoint_dir=%s\nrepo_root=%s\nconda_env=%s\nconfig_path=%s\nmethod=%s\n' \
        "${ATTEMPT_ID}" "${RUN_CKPT_DIR}" "${REPO_ROOT}" "${CONDA_ENV}" \
        "${SPARSEOPD_CONFIG_PATH}" "${METHOD}"
    printf '\n--- git revision ---\n'
    git -C "${REPO_ROOT}" rev-parse HEAD
    printf '\n--- git status ---\n'
    git -C "${REPO_ROOT}" status --short --branch
    git -C "${REPO_ROOT}" diff --binary HEAD > "${GIT_DIFF}"
    printf '\n--- software ---\n'
    python --version
    python -m pip freeze
    printf '\n--- GPUs ---\n'
    nvidia-smi
} > "${ENVIRONMENT_LOG}" 2>&1

python -m verl.trainer.main_ppo --config-dir "${CONFIG_DIR}" --config-name "${CONFIG_NAME}" \
    --cfg job --resolve "${OVERRIDES[@]}" "${METHOD_OVERRIDES[@]}" "$@" > "${RESOLVED_CONFIG}"
python - "${RESOLVED_CONFIG}" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

config = OmegaConf.load(sys.argv[1])
teacher_model_path = config.distillation.teacher_models.teacher_model.model_path
train_files = config.data.train_files
if isinstance(train_files, str):
    train_files = [train_files]
paths = [
    config.actor_rollout_ref.model.path,
    teacher_model_path,
    *train_files,
]
for value in paths:
    path = Path(value).expanduser()
    if not path.exists():
        raise SystemExit(f"Required model/data path not found: {path}")
PY

set +e
python -m verl.trainer.main_ppo --config-dir "${CONFIG_DIR}" --config-name "${CONFIG_NAME}" \
    "${OVERRIDES[@]}" "${METHOD_OVERRIDES[@]}" "$@" 2>&1 | tee "${CONSOLE_LOG}"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

if [[ ${TRAIN_STATUS} -ne 0 ]]; then
    echo "Training failed with exit code ${TRAIN_STATUS}; see ${CONSOLE_LOG}" >&2
    exit "${TRAIN_STATUS}"
fi

LATEST_STEP="$(<"${RUN_CKPT_DIR}/latest_checkpointed_iteration.txt")"
LATEST_ACTOR_DIR="${RUN_CKPT_DIR}/global_step_${LATEST_STEP}/actor"
if [[ ! -d "${LATEST_ACTOR_DIR}" ]]; then
    echo "Latest actor checkpoint is missing: ${LATEST_ACTOR_DIR}" >&2
    exit 1
fi

MERGED_MODEL_DIR="${RUN_CKPT_DIR}/hf_model_step_${LATEST_STEP}"
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${LATEST_ACTOR_DIR}" \
    --target_dir "${MERGED_MODEL_DIR}" 2>&1 | tee "${MERGE_LOG}"

printf 'completed_at_utc=%s\nlatest_step=%s\nmerged_model=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${LATEST_STEP}" "${MERGED_MODEL_DIR}" \
    > "${RESULT_FILE}"
echo "Training and model merge completed. Run artifacts: ${OUTPUT_ROOT}"
