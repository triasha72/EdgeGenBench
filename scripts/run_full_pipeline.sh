#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

cd "${PROJECT_ROOT}"

section() {
    printf '\n'
    printf '%s\n' "============================================================"
    printf '%s\n' "$1"
    printf '%s\n' "============================================================"
}

require_command() {
    local command_name="$1"

    if ! command -v "${command_name}" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "${command_name}" >&2
        exit 127
    fi
}

require_file() {
    local file_path="$1"

    if [[ ! -f "${file_path}" ]]; then
        printf 'Required file not found: %s\n' "${file_path}" >&2
        exit 1
    fi
}

section "EdgeGenBench v0.1 full pipeline"

printf 'Project root: %s\n' "${PROJECT_ROOT}"
printf 'Python: %s\n' "$(command -v python)"
python --version

require_command python
require_command edgegenbench
require_command ruff
require_command pytest

require_file "configs/v0_1.yaml"
require_file "configs/optimization_v0_1.yaml"

section "1. Repository validation"

ruff format --check .
ruff check .
pytest
bash -n scripts/run_full_pipeline.sh

section "2. Synthetic benchmark generation"

edgegenbench generate-data \
    --config configs/v0_1.yaml

require_file "data/raw/edgegenbench_v0_1.csv"

section "3. FP32 Ridge baseline"

edgegenbench train-fp32-baseline \
    --dataset data/raw/edgegenbench_v0_1.csv \
    --output-dir artifacts/fp32_baseline

require_file "artifacts/fp32_baseline/fp32_linear_model.npz"
require_file "artifacts/fp32_baseline/summary.json"

section "4. Tree-based surrogate baselines"

edgegenbench train-tree-baselines \
    --dataset data/raw/edgegenbench_v0_1.csv \
    --output-dir artifacts/tree_baselines

require_file "artifacts/tree_baselines/random_forest/model.joblib"
require_file "artifacts/tree_baselines/random_forest/summary.json"
require_file "artifacts/tree_baselines/hist_gradient_boosting/model.joblib"
require_file "artifacts/tree_baselines/hist_gradient_boosting/summary.json"

section "5. Unified surrogate comparison"

edgegenbench compare-models \
    --artifact-root artifacts \
    --output-dir reports/model_comparison

section "6. Uncertainty quantification"

edgegenbench evaluate-uncertainty \
    --dataset data/raw/edgegenbench_v0_1.csv \
    --random-forest-summary \
    artifacts/tree_baselines/random_forest/summary.json \
    --output-dir artifacts/uncertainty

section "7. Feasibility classification"

edgegenbench train-feasibility-classifier \
    --dataset data/raw/edgegenbench_v0_1.csv \
    --output-dir artifacts/feasibility_classifier \
    --max-false-safe-rate 0.05

require_file "artifacts/feasibility_classifier/model.joblib"
require_file "artifacts/feasibility_classifier/summary.json"

section "8. Constrained multi-objective optimization"

edgegenbench optimize-designs \
    --config configs/optimization_v0_1.yaml \
    --surrogate-model \
    artifacts/tree_baselines/random_forest/model.joblib \
    --feasibility-model \
    artifacts/feasibility_classifier/model.joblib \
    --output-dir artifacts/optimization

require_file "artifacts/optimization/representative_designs.csv"
require_file "artifacts/optimization/optimization_summary.json"

section "9. Physics-based optimization validation"

edgegenbench validate-optimization \
    --designs artifacts/optimization/representative_designs.csv \
    --benchmark-config configs/v0_1.yaml \
    --output-dir artifacts/optimization_validation

section "10. ONNX export"

edgegenbench export-edge-models \
    --surrogate-model \
    artifacts/tree_baselines/random_forest/model.joblib \
    --feasibility-model \
    artifacts/feasibility_classifier/model.joblib \
    --output-dir artifacts/edge_export

require_file "artifacts/edge_export/surrogate.onnx"
require_file "artifacts/edge_export/feasibility.onnx"
require_file "artifacts/edge_export/metadata.json"

section "11. ONNX equivalence and edge benchmarking"

edgegenbench benchmark-edge-models \
    --dataset data/raw/edgegenbench_v0_1.csv \
    --surrogate-model \
    artifacts/tree_baselines/random_forest/model.joblib \
    --feasibility-model \
    artifacts/feasibility_classifier/model.joblib \
    --surrogate-onnx \
    artifacts/edge_export/surrogate.onnx \
    --feasibility-onnx \
    artifacts/edge_export/feasibility.onnx \
    --metadata artifacts/edge_export/metadata.json \
    --output-dir artifacts/edge_benchmark

require_file "artifacts/edge_benchmark/equivalence.csv"
require_file "artifacts/edge_benchmark/latency.csv"
require_file "artifacts/edge_benchmark/summary.json"

section "12. Generated artifact inventory"

find artifacts reports/model_comparison \
    -type f \
    \( \
        -name "*.json" \
        -o -name "*.csv" \
        -o -name "*.png" \
        -o -name "*.joblib" \
        -o -name "*.onnx" \
    \) \
    | sort

section "EdgeGenBench v0.1 pipeline completed successfully"