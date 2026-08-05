# EdgeGenBench

[![CI](https://github.com/triasha72/EdgeGenBench/actions/workflows/ci.yml/badge.svg)](https://github.com/triasha72/EdgeGenBench/actions/workflows/ci.yml)

**Uncertainty-aware surrogate modeling, constrained aircraft-design
optimization, and edge-ready inference for hybrid-electric and hydrogen
regional-aircraft studies.**

## Overview

EdgeGenBench is an independent, reproducible research benchmark for studying
how machine-learning surrogates behave inside early aircraft-design workflows.

The project connects:

- synthetic physics-based data generation;
- classical multi-output surrogate modeling;
- uncertainty quantification;
- safety-conscious feasibility classification;
- constrained multi-objective optimization;
- physics-based validation;
- ONNX export;
- edge-inference benchmarking.

The benchmark is designed to examine more than predictive accuracy. It also
asks whether a model:

- identifies feasible designs safely;
- expresses useful uncertainty;
- remains reliable when used by an optimizer;
- reproduces its predictions after deployment conversion;
- meets model-size and latency constraints.

EdgeGenBench uses synthetic or public information only. It does not contain
proprietary aircraft-manufacturer data, software, or design information.

## EdgeGenBench v0.1

Version 0.1 provides a complete reproducible workflow from synthetic design
generation through edge deployment.

### Current capabilities

- Versioned YAML benchmark configurations
- Deterministic synthetic aircraft-design generation
- Reproducible train, validation, calibration, and test partitions
- FP32 multi-output Ridge regression
- Random Forest multi-output regression
- HistGradientBoosting multi-output regression
- Validation-based hyperparameter selection
- Held-out test evaluation
- MAE, RMSE, normalized RMSE, and R² reporting
- Model-size and inference-latency measurement
- Random Forest tree-quantile uncertainty intervals
- Split-conformal prediction intervals
- Safety-conscious feasibility classification
- False-safe-rate-aware threshold selection
- Latin-hypercube candidate generation
- Constrained multi-objective optimization
- Pareto-front extraction
- Representative-design selection
- Physics-based optimization validation
- Separate classifier and optimization feasibility thresholds
- Deterministic categorical feature encoding
- ONNX surrogate and classifier export
- ONNX Runtime inference
- Scikit-learn-to-ONNX equivalence validation
- Batch latency benchmarking
- Automated tests and GitHub Actions CI
- Executable end-to-end release pipeline

## Benchmark problem

The v0.1 benchmark represents an early regional-aircraft design study.

### Design variables

- Passenger capacity
- Design range
- Cruise speed
- Battery specific energy
- Hydrogen storage efficiency
- Hybridization ratio
- Propulsion architecture

Supported propulsion architectures include:

- conventional turboprop;
- parallel hybrid;
- series hybrid;
- fuel-cell electric.

### Predicted targets

The surrogate models predict:

- estimated takeoff mass;
- mission energy;
- energy per passenger-kilometre;
- lifecycle-emissions proxy;
- operating-cost proxy;
- noise proxy.

The synthetic physics model also generates feasibility labels and supporting
engineering quantities used by the benchmark workflow.

## v0.1 result highlights

### Surrogate comparison

| Model | Mean test NRMSE | Mean test R² | Model size | Batch-1 latency |
|---|---:|---:|---:|---:|
| HistGradientBoosting | **0.062249** | **0.995171** | 7.216 MiB | 76.339 ms |
| Random Forest | 0.205386 | 0.953219 | 172.296 MiB | 14.652 ms |
| FP32 Ridge | 0.214590 | 0.937690 | **0.002473 MiB** | **0.142 ms** |

HistGradientBoosting produced the strongest classical predictive accuracy.
FP32 Ridge produced the smallest model and lowest classical-model latency.

The Random Forest surrogate is used for uncertainty estimation, optimization,
and ONNX export in v0.1.

### Feasibility classification

| Metric | Value |
|---|---:|
| Balanced accuracy | 98.24% |
| Precision | 95.62% |
| Recall | 98.61% |
| F1 score | 97.09% |
| ROC AUC | 99.89% |
| False-safe rate | 2.12% |
| Classifier threshold | 0.30 |

### Optimization validation

The classifier threshold selected from ordinary validation data was not
sufficiently conservative when applied directly to optimizer-selected
designs. EdgeGenBench therefore separates two decision thresholds:

| Threshold | Purpose |
|---:|---|
| 0.30 | Validation-selected classifier operating threshold |
| **0.50** | Physics-validated optimization acceptance threshold |

The optimization threshold was selected through Pareto-front and
representative-design physics validation.

| Optimization result | Value |
|---|---:|
| Generated candidates | 20,000 |
| Accepted candidates | 5,413 |
| Accepted fraction | 27.065% |
| Pareto designs | 47 |
| Representative designs | 4 |
| Pareto-front feasibility agreement | 100% |
| Representative-design feasibility agreement | 100% |

### ONNX deployment

The Random Forest surrogate and feasibility classifier were exported to ONNX.

| Result | Value |
|---|---:|
| Classifier decision agreement | 100% |
| Maximum classifier probability error | 1.82 × 10⁻⁷ |
| Maximum surrogate absolute difference | 0.01059 |
| ONNX surrogate size | 108.956 MiB |
| ONNX classifier size | 2.728 MiB |

At batch size 1:

| Model | Scikit-learn | ONNX Runtime |
|---|---:|---:|
| Surrogate | 13.795 ms | **0.305 ms** |
| Feasibility classifier | 14.787 ms | **0.311 ms** |

Detailed results are available in
[`docs/results.md`](docs/results.md).

## System workflow

```text
Versioned configuration
        |
        v
Synthetic physics model
        |
        v
Dataset generation and deterministic partitioning
        |
        v
Surrogate training and comparison
        |
        +--> FP32 Ridge
        +--> Random Forest
        +--> HistGradientBoosting
        |
        v
Uncertainty quantification
        |
        v
Feasibility classifier
        |
        +--> classifier threshold: 0.30
        |
        v
Conservative optimization gate
        |
        +--> optimization threshold: 0.50
        |
        v
Multi-objective optimization and Pareto extraction
        |
        v
Physics-based validation
        |
        v
Deterministic feature encoding
        |
        v
ONNX export and equivalence testing
        |
        v
Accuracy-size-latency benchmark
```

See [`docs/architecture.md`](docs/architecture.md) for the detailed software
architecture.

## Installation

### 1. Create a Python environment

EdgeGenBench v0.1 was validated with Python 3.12.

Using Conda:

```bash
conda create -n edgegenbench-py312 python=3.12
conda activate edgegenbench-py312
```

### 2. Install the project

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,edge]"
```

The `dev` extra installs development and testing tools. The `edge` extra
installs ONNX export and runtime dependencies.

### 3. Confirm the installation

```bash
edgegenbench info
```

## Run the complete v0.1 workflow

The recommended execution path is:

```bash
./scripts/run_full_pipeline.sh
```

The script runs:

1. formatting, linting, and tests;
2. deterministic dataset generation;
3. FP32 Ridge training;
4. Random Forest and HistGradientBoosting training;
5. unified surrogate comparison;
6. uncertainty evaluation;
7. feasibility-classifier training;
8. constrained multi-objective optimization;
9. physics-based optimization validation;
10. ONNX export;
11. ONNX equivalence and latency benchmarking;
12. artifact inventory generation.

A successful run ends with:

```text
EdgeGenBench v0.1 pipeline completed successfully
```

## Manual CLI workflow

### Generate data

```bash
edgegenbench generate-data \
  --config configs/v0_1.yaml
```

### Train the FP32 Ridge baseline

```bash
edgegenbench train-fp32-baseline \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --output-dir artifacts/fp32_baseline
```

### Train nonlinear baselines

```bash
edgegenbench train-tree-baselines \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --output-dir artifacts/tree_baselines
```

### Compare surrogate models

```bash
edgegenbench compare-models \
  --artifact-root artifacts \
  --output-dir reports/model_comparison
```

### Evaluate uncertainty

```bash
edgegenbench evaluate-uncertainty \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --random-forest-summary \
  artifacts/tree_baselines/random_forest/summary.json \
  --output-dir artifacts/uncertainty
```

### Train the feasibility classifier

```bash
edgegenbench train-feasibility-classifier \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --output-dir artifacts/feasibility_classifier \
  --max-false-safe-rate 0.05
```

### Run optimization

```bash
edgegenbench optimize-designs \
  --config configs/optimization_v0_1.yaml \
  --surrogate-model \
  artifacts/tree_baselines/random_forest/model.joblib \
  --feasibility-model \
  artifacts/feasibility_classifier/model.joblib \
  --output-dir artifacts/optimization
```

The optimization configuration applies the separate conservative threshold
of `0.50`.

### Validate optimized designs

```bash
edgegenbench validate-optimization \
  --designs artifacts/optimization/representative_designs.csv \
  --benchmark-config configs/v0_1.yaml \
  --output-dir artifacts/optimization_validation
```

### Export models to ONNX

```bash
edgegenbench export-edge-models \
  --surrogate-model \
  artifacts/tree_baselines/random_forest/model.joblib \
  --feasibility-model \
  artifacts/feasibility_classifier/model.joblib \
  --output-dir artifacts/edge_export
```

### Benchmark edge models

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

## Generated artifacts

The workflow creates outputs under:

```text
data/raw/
artifacts/fp32_baseline/
artifacts/tree_baselines/
artifacts/uncertainty/
artifacts/feasibility_classifier/
artifacts/optimization/
artifacts/optimization_validation/
artifacts/edge_export/
artifacts/edge_benchmark/
reports/model_comparison/
```

Generated data, models, reports, and experimental threshold-sweep directories
are intentionally excluded from Git.

## Repository structure

```text
EdgeGenBench/
├── .github/workflows/
├── configs/
│   ├── v0_1.yaml
│   └── optimization_v0_1.yaml
├── data/
├── docs/
│   ├── architecture.md
│   ├── DESIGN_CONTRACT.md
│   ├── reproducibility.md
│   └── results.md
├── notebooks/
├── reports/
├── scripts/
│   └── run_full_pipeline.sh
├── src/edgegenbench/
│   ├── data/
│   ├── deployment/
│   ├── evaluation/
│   ├── models/
│   ├── optimization/
│   ├── physics/
│   ├── training/
│   └── uncertainty/
├── tests/
├── CHANGELOG.md
├── pyproject.toml
└── README.md
```

## Development checks

```bash
ruff format --check .
ruff check .
pytest
bash -n scripts/run_full_pipeline.sh
git diff --check
```

## Reproducibility

The v0.1 workflow uses:

- versioned configuration files;
- deterministic random seeds;
- fixed train, validation, calibration, and test partitions;
- deterministic candidate generation;
- recorded package versions;
- artifact-backed reports;
- a complete executable pipeline.

See [`docs/reproducibility.md`](docs/reproducibility.md).

## Roadmap

### v0.2 — Compact neural surrogate and quantization

- Compact multi-output PyTorch surrogate
- Training-only feature and target normalization
- Validation-based early stopping
- PyTorch-to-ONNX export
- FP16 conversion
- INT8 quantization
- Accuracy, size, latency, and throughput comparisons

### v0.3 — Robustness and distribution shift

- Out-of-distribution evaluation
- Extrapolation tests
- Shift-aware uncertainty evaluation
- Failure-region analysis
- Optimization robustness under changed missions

### v0.4 — Hardware-aware deployment

- Mobile and embedded CPU benchmarks
- GPU and NPU execution providers
- Memory and energy measurements
- Hardware-aware model selection

### v0.5 — Expanded design studies

- Additional missions
- Additional design-space configurations
- Multiple aircraft classes
- Expanded physics-model fidelity

### v1.0 — Stable public benchmark

- Stable data and artifact schemas
- Versioned benchmark suites
- Reproducible public baselines
- Documented extension interfaces

## Limitations

EdgeGenBench v0.1 has important limitations:

- The benchmark dataset is synthetic.
- The physics model is an analytical research proxy.
- No proprietary, certification, or flight-test data are included.
- Results currently represent one mission and design-space configuration.
- Threshold selection must be reevaluated for changed design spaces.
- Conformal coverage is demonstrated on the current held-out split.
- Latency was measured on one ARM64 macOS system.
- No mobile GPU, NPU, or microcontroller benchmark is included.
- FP16 and INT8 deployment are not included in v0.1.
- The Random Forest surrogate remains large for constrained edge devices.

EdgeGenBench is not:

- a certified aircraft-sizing tool;
- an operational aircraft-performance model;
- a manufacturer design prediction;
- a substitute for validated engineering analysis;
- a safety-critical decision system.

## Author

Triasha Sarkar