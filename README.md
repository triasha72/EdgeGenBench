# EdgeGenBench
[![CI](https://github.com/triasha72/EdgeGenBench/actions/workflows/ci.yml/badge.svg)](https://github.com/triasha72/EdgeGenBench/actions/workflows/ci.yml)
**Uncertainty-aware surrogate modeling and edge-ready inference for hybrid-electric and hydrogen regional-aircraft design.**

## Overview

EdgeGenBench is an independent and reproducible benchmark for comparing surrogate models used in early aircraft-design trade studies.

The project evaluates:

- predictive accuracy;
- uncertainty behavior;
- optimization usefulness;
- model size;
- inference latency;
- readiness for edge deployment.

The project uses synthetic or public data only. It does not use proprietary aircraft-manufacturer data, software, or design information.

## Current capabilities

EdgeGenBench currently supports:

- physics-informed synthetic aircraft-design data generation;
- deterministic train, validation, and test splits;
- a multi-output FP32 ridge-regression surrogate;
- validation-based hyperparameter selection;
- held-out test evaluation;
- MAE, RMSE, normalized RMSE, and R² reporting;
- model serialization;
- batch-inference latency measurement;
- reproducibility metadata.

## Design variables

The synthetic benchmark uses:

- passenger capacity;
- design range;
- cruise speed;
- battery specific energy;
- hydrogen-storage efficiency;
- hybridization ratio;
- propulsion architecture.

## Synthetic outputs

The generated targets include:

- estimated takeoff mass;
- mission energy demand;
- energy per passenger-kilometre;
- lifecycle-emissions proxy;
- operating-cost proxy;
- noise proxy;
- battery and hydrogen-system quantities;
- feasibility margins.

## Roadmap

1. Build a reproducible synthetic aircraft-design dataset.
2. Train and validate linear and nonlinear surrogate baselines.
3. Compare uncertainty and failure behavior.
4. Run constrained multi-objective optimization.
5. Export suitable models to ONNX.
6. Benchmark model size, quantization, and edge inference.

## Installation

Create and activate a Python 3.12 environment, then install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Confirm the CLI installation:

```bash
edgegenbench info
```

## Generate the benchmark dataset

```bash
edgegenbench generate-data \
  --config configs/v0_1.yaml
```

Generated data are written to:

```text
data/raw/
```

The generated dataset is intentionally excluded from Git.

## Train the FP32 baseline

```bash
edgegenbench train-fp32-baseline \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --output-dir artifacts/fp32_baseline
```

The pipeline:

- tunes ridge regularization using the validation split;
- refits the selected model using training and validation data;
- evaluates once on the held-out test split;
- saves metrics, predictions, latency measurements, and model metadata.

Generated model artifacts are written to:

```text
artifacts/fp32_baseline/
```

## Run project checks

```bash
ruff format .
ruff check .
pytest
```

## Run the complete workflow

```bash
ruff format .
ruff check .
pytest

edgegenbench info

edgegenbench generate-data \
  --config configs/v0_1.yaml

edgegenbench train-fp32-baseline \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --output-dir artifacts/fp32_baseline
```

## Repository structure

```text
EdgeGenBench/
├── configs/
├── data/
├── docs/
├── notebooks/
├── reports/
├── scripts/
├── src/edgegenbench/
│   ├── data/
│   ├── deployment/
│   ├── evaluation/
│   ├── models/
│   ├── optimization/
│   ├── physics/
│   └── training/
├── tests/
├── pyproject.toml
└── README.md
```

## Limitations

The synthetic physics model is intended for machine-learning benchmarking and software-development experiments.

It is not:

- a certified aircraft-sizing tool;
- an operational performance model;
- a manufacturer design prediction;
- a substitute for validated engineering analysis.

## Author

Triasha Sarkar