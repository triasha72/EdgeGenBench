# EdgeGenBench Reproducibility Guide

## Purpose

This document describes how to reproduce the EdgeGenBench v0.1 benchmark,
including model training, uncertainty evaluation, optimization,
physics-based validation, ONNX export, and latency measurement.

The recommended reproduction path is the executable pipeline:

```bash
./scripts/run_full_pipeline.sh
```

## Reference environment

The final v0.1 release results were generated with:

| Component | Version |
|---|---:|
| Operating system | macOS 26.5.2 |
| Architecture | ARM64 |
| Python | 3.12.13 |
| NumPy | 2.5.1 |
| Pandas | 3.0.5 |
| Scikit-learn | 1.9.0 |
| ONNX | 1.22.0 |
| ONNX Runtime | 1.28.0 |

Exact latency values are system-dependent. Model accuracy, selected
hyperparameters, candidate counts, and deterministic optimization outputs
should remain stable when the same configurations, package behavior, and
random seeds are used.

## Reproducibility controls

EdgeGenBench v0.1 uses the following controls.

### Versioned configuration

```text
configs/v0_1.yaml
configs/optimization_v0_1.yaml
```

These files define the dataset, mission, design space, optimization
objectives, random seeds, and output locations.

### Fixed random seeds

The primary benchmark and optimization workflows use seed `42`.

Candidate architecture assignment uses a deterministic generator derived
from the optimization seed.

### Deterministic candidate generation

The optimization workflow uses Latin-hypercube sampling with:

```text
candidate count: 20,000
seed: 42
```

### Fixed partition sizes

The 6,000-row dataset is divided into:

| Partition | Rows |
|---|---:|
| Training | 4,200 |
| Validation | 900 |
| Test | 900 |

After validation-based model selection, models are refitted on 5,100
training-plus-validation rows.

The uncertainty workflow further separates 840 calibration rows from its
training portion.

### Held-out test evaluation

The test partition is used only after model and threshold selection.

### Artifact-backed stages

Each pipeline stage writes its outputs to disk. Downstream stages consume
those explicit artifacts.

### Stable schemas

The project fixes:

- feature-column names;
- feature order;
- target names;
- target order;
- categorical encoding;
- JSON summary structure;
- generated artifact locations.

## Environment setup

### Conda setup

```bash
conda create -n edgegenbench-py312 python=3.12
conda activate edgegenbench-py312
```

### Install EdgeGenBench

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,edge]"
```

### Confirm the environment

```bash
python --version
edgegenbench info
```

### Confirm required commands

```bash
command -v python
command -v edgegenbench
command -v ruff
command -v pytest
```

## Repository checks

Before running the benchmark:

```bash
ruff format --check .
ruff check .
pytest
bash -n scripts/run_full_pipeline.sh
git diff --check
```

A valid v0.1 source tree should pass all commands.

Warnings produced by Joblib, NumPy, or Scikit-learn do not indicate a failed
test run when Pytest reports that all tests passed.

## Complete reproduction

Run:

```bash
set -o pipefail

./scripts/run_full_pipeline.sh \
  2>&1 \
  | tee /tmp/edgegenbench_v0_1_full_run.log

pipeline_status=$?

echo "Pipeline exit status: ${pipeline_status}"
```

Required final status:

```text
Pipeline exit status: 0
```

The log should end with:

```text
EdgeGenBench v0.1 pipeline completed successfully
```

## Pipeline stages

## 1. Repository validation

Runs:

```bash
ruff format --check .
ruff check .
pytest
bash -n scripts/run_full_pipeline.sh
```

## 2. Synthetic dataset generation

Command:

```bash
edgegenbench generate-data \
  --config configs/v0_1.yaml
```

Expected primary outputs:

```text
data/raw/edgegenbench_v0_1.csv
data/raw/edgegenbench_v0_1_metadata.json
```

Expected dataset properties:

```text
rows: 6000
feasible fraction: approximately 32.6%
```

## 3. FP32 Ridge baseline

Command:

```bash
edgegenbench train-fp32-baseline \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --output-dir artifacts/fp32_baseline
```

Expected primary outputs:

```text
artifacts/fp32_baseline/fp32_linear_model.npz
artifacts/fp32_baseline/summary.json
artifacts/fp32_baseline/test_metrics.csv
artifacts/fp32_baseline/test_predictions.csv
artifacts/fp32_baseline/latency.csv
```

Reference result:

```text
best alpha: 0.0001
mean test NRMSE: 0.214590
mean test R²: 0.937690
```

## 4. Tree-based baselines

Command:

```bash
edgegenbench train-tree-baselines \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --output-dir artifacts/tree_baselines
```

Expected model outputs:

```text
artifacts/tree_baselines/random_forest/model.joblib
artifacts/tree_baselines/hist_gradient_boosting/model.joblib
```

Reference results:

| Model | Mean test NRMSE | Mean test R² |
|---|---:|---:|
| Random Forest | 0.205386 | 0.953219 |
| HistGradientBoosting | 0.062249 | 0.995171 |

## 5. Model comparison

Command:

```bash
edgegenbench compare-models \
  --artifact-root artifacts \
  --output-dir reports/model_comparison
```

Expected outputs:

```text
reports/model_comparison/aggregate_metrics.csv
reports/model_comparison/detailed_metrics.csv
reports/model_comparison/latency_comparison.csv
reports/model_comparison/comparison_summary.json
reports/model_comparison/*.png
```

Reference selections:

```text
best accuracy: hist_gradient_boosting
best mean R²: hist_gradient_boosting
lowest batch-1 latency: fp32_ridge
smallest model: fp32_ridge
```

## 6. Uncertainty evaluation

Command:

```bash
edgegenbench evaluate-uncertainty \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --random-forest-summary \
  artifacts/tree_baselines/random_forest/summary.json \
  --output-dir artifacts/uncertainty
```

Expected outputs:

```text
artifacts/uncertainty/calibration_random_forest.joblib
artifacts/uncertainty/coverage_metrics.csv
artifacts/uncertainty/conformal_intervals_80.csv
artifacts/uncertainty/conformal_intervals_90.csv
artifacts/uncertainty/conformal_intervals_95.csv
artifacts/uncertainty/uncertainty_summary.json
```

Reference row counts:

```text
proper training rows: 3360
calibration rows: 840
test rows: 900
```

## 7. Feasibility-classifier training

Command:

```bash
edgegenbench train-feasibility-classifier \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --output-dir artifacts/feasibility_classifier \
  --max-false-safe-rate 0.05
```

Expected primary outputs:

```text
artifacts/feasibility_classifier/model.joblib
artifacts/feasibility_classifier/summary.json
artifacts/feasibility_classifier/test_metrics.json
artifacts/feasibility_classifier/confusion_matrix.csv
artifacts/feasibility_classifier/threshold_search.csv
```

Reference results:

```text
selected classifier threshold: 0.30
balanced accuracy: 0.9824
false-safe rate: 0.0212
```

## 8. Multi-objective optimization

Command:

```bash
edgegenbench optimize-designs \
  --config configs/optimization_v0_1.yaml \
  --surrogate-model \
  artifacts/tree_baselines/random_forest/model.joblib \
  --feasibility-model \
  artifacts/feasibility_classifier/model.joblib \
  --output-dir artifacts/optimization
```

The classifier model stores threshold `0.30`.

The optimization configuration overrides this with:

```yaml
feasibility_threshold: 0.50
```

Expected primary outputs:

```text
artifacts/optimization/candidate_designs.csv
artifacts/optimization/feasible_candidates.csv
artifacts/optimization/pareto_front.csv
artifacts/optimization/representative_designs.csv
artifacts/optimization/optimization_summary.json
artifacts/optimization/*.png
```

Reference results:

```text
candidate count: 20000
accepted candidates: 5413
accepted fraction: 0.27065
Pareto designs: 47
representative designs: 4
applied threshold: 0.50
```

## 9. Physics-based validation

Command:

```bash
edgegenbench validate-optimization \
  --designs artifacts/optimization/representative_designs.csv \
  --benchmark-config configs/v0_1.yaml \
  --output-dir artifacts/optimization_validation
```

Expected primary outputs:

```text
artifacts/optimization_validation/physics_validation_details.csv
artifacts/optimization_validation/physics_validation_metrics.csv
artifacts/optimization_validation/physics_validation_summary.json
artifacts/optimization_validation/*.png
```

Reference results:

```text
designs validated: 4
targets validated: 6
physics-feasible designs: 4
feasibility agreement: 100%
```

## 10. ONNX export

Command:

```bash
edgegenbench export-edge-models \
  --surrogate-model \
  artifacts/tree_baselines/random_forest/model.joblib \
  --feasibility-model \
  artifacts/feasibility_classifier/model.joblib \
  --output-dir artifacts/edge_export
```

Expected outputs:

```text
artifacts/edge_export/surrogate.onnx
artifacts/edge_export/feasibility.onnx
artifacts/edge_export/metadata.json
```

The exported feasibility model preserves the classifier threshold of `0.30`.
The optimization threshold is applied by the optimization workflow and is
not embedded into the exported classifier.

## 11. ONNX equivalence and latency

Command:

```bash
edgegenbench benchmark-edge-models \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --surrogate-model \
  artifacts/tree_baselines/random_forest/model.joblib \
  --feasibility-model \
  artifacts/feasibility_classifier/model.joblib \
  --surrogate-onnx artifacts/edge_export/surrogate.onnx \
  --feasibility-onnx artifacts/edge_export/feasibility.onnx \
  --metadata artifacts/edge_export/metadata.json \
  --output-dir artifacts/edge_benchmark
```

Expected outputs:

```text
artifacts/edge_benchmark/equivalence.csv
artifacts/edge_benchmark/latency.csv
artifacts/edge_benchmark/summary.json
```

Reference results:

```text
test rows: 900
classifier agreement: 100%
maximum classifier probability error: approximately 1.82e-7
maximum surrogate absolute difference: approximately 0.01059
```

## Optimization-threshold reproduction

The final optimization threshold was selected through a deterministic
sensitivity study.

### Broad sweep

Tested thresholds:

```text
0.30
0.50
0.70
0.90
```

Results:

| Threshold | Accepted | Pareto designs | Pareto agreement | Representative agreement |
|---:|---:|---:|---:|---:|
| 0.30 | 5,687 | 52 | 84.62% | 25% |
| 0.50 | 5,413 | 47 | 100% | 100% |
| 0.70 | 5,333 | 46 | 100% | 100% |
| 0.90 | 5,094 | 43 | 100% | 100% |

### Refined sweep

Tested thresholds:

```text
0.35
0.40
0.45
0.50
```

Results:

| Threshold | Accepted | Pareto designs | Pareto agreement | Representative agreement |
|---:|---:|---:|---:|---:|
| 0.35 | 5,562 | 46 | 93.48% | 25% |
| 0.40 | 5,487 | 44 | 100% | 75% |
| 0.45 | 5,437 | 50 | 98% | 100% |
| 0.50 | 5,413 | 47 | 100% | 100% |

Threshold `0.50` was selected because it was the smallest tested value
achieving complete agreement for both the full Pareto front and the four
representative designs.

Threshold-study directories are experimental outputs and should not be
committed:

```text
artifacts/threshold_sweep/
artifacts/threshold_refinement/
```

## Validate generated artifacts

Run:

```bash
for path in \
  data/raw/edgegenbench_v0_1.csv \
  artifacts/fp32_baseline \
  artifacts/tree_baselines/random_forest \
  artifacts/tree_baselines/hist_gradient_boosting \
  reports/model_comparison \
  artifacts/uncertainty \
  artifacts/feasibility_classifier \
  artifacts/optimization \
  artifacts/optimization_validation \
  artifacts/edge_export \
  artifacts/edge_benchmark
do
  if [ -e "$path" ]; then
    echo "OK      $path"
  else
    echo "MISSING $path"
  fi
done
```

Every line should begin with:

```text
OK
```

## Verify the final thresholds

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

classifier = json.loads(
    Path(
        "artifacts/feasibility_classifier/summary.json"
    ).read_text(encoding="utf-8")
)

optimization = json.loads(
    Path(
        "artifacts/optimization/optimization_summary.json"
    ).read_text(encoding="utf-8")
)

validation = json.loads(
    Path(
        "artifacts/optimization_validation/"
        "physics_validation_summary.json"
    ).read_text(encoding="utf-8")
)

assert classifier["selected_threshold"] == 0.30
assert optimization["feasibility_threshold"] == 0.50
assert optimization["feasible_count"] == 5413
assert optimization["pareto_count"] == 47
assert validation["physics_feasible_count"] == 4
assert validation["feasibility_agreement_rate"] == 1.0

print("Classifier threshold: 0.30")
print("Optimization threshold: 0.50")
print("Accepted candidates: 5413")
print("Pareto designs: 47")
print("Representative feasibility agreement: 100%")
PY
```

## Latency reproducibility

Latency measurements depend on:

- CPU model;
- operating system;
- background processes;
- power-management state;
- package versions;
- thread scheduling;
- ONNX Runtime execution provider.

For meaningful comparisons:

1. benchmark all candidate models in the same environment;
2. avoid comparing measurements from different machines as though they were
   directly equivalent;
3. record both mean and P95 latency;
4. report batch size and repeat count;
5. preserve package-version metadata.

Accuracy and prediction-equivalence results should be more stable than
latency measurements.

## Generated files and Git

The following are generated and should not be staged:

```text
artifacts/
data/raw/
reports/model_comparison/
```

Check:

```bash
git status --short \
  | grep -E \
  'artifacts/|data/raw/|reports/model_comparison/' \
  || echo "No generated artifacts are staged or tracked."
```

## Final release validation

Before creating a release commit:

```bash
ruff format --check .
ruff check .
pytest
bash -n scripts/run_full_pipeline.sh
git diff --check
```

Search documentation for incomplete placeholders:

```bash
grep -RniE \
  'TODO|TBD|placeholder|replace later|insert value|actual value' \
  README.md \
  CHANGELOG.md \
  docs
```

Expected: no output.

Check Markdown fences:

```bash
for file in \
  README.md \
  CHANGELOG.md \
  docs/architecture.md \
  docs/reproducibility.md \
  docs/results.md
do
  count=$(grep -c '^```' "$file" || true)
  echo "$file: $count code fences"
done
```

Each count must be even.

## Known reproducibility limits

- The synthetic physics model is not a high-fidelity aircraft simulator.
- Package changes may affect serialized model representations.
- Latency is hardware-dependent.
- Threshold `0.50` is validated for the current mission and design space.
- Changed missions or design-space limits require renewed threshold
  validation.
- Conformal coverage is not guaranteed under arbitrary distribution shift.
- Results do not constitute aircraft certification evidence.