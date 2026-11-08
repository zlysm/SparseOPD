#!/usr/bin/env bash
# Score saved responses and update the evaluation summary.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${EVAL_PYTHON:-path/to/conda/env/bin/python}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate.py" \
    --config "${EVAL_CONFIG:-${REPO_ROOT}/config/sparseopd/evaluation.yaml}" \
    --output-dir "${EVAL_OUTPUT_DIR:-${REPO_ROOT}/eval}" "$@"
