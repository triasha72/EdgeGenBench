# EdgeGenBench

[![CI](https://github.com/triasha72/EdgeGenBench/actions/workflows/ci.yml/badge.svg)](https://github.com/triasha72/EdgeGenBench/actions/workflows/ci.yml)

**Scientific machine learning, uncertainty-aware aircraft-design optimization,
and hardware-aware edge inference for hybrid-electric and hydrogen regional-aircraft studies.**

## Overview

EdgeGenBench is an independent, reproducible benchmark for studying how
machine-learning surrogates behave inside early aircraft-design workflows and
how those models translate into efficient edge-inference systems.

The project connects:

- synthetic physics-based data generation;
- classical multi-output surrogate modeling;
- compact PyTorch neural surrogate modeling;
- leakage-safe feature and target normalization;
- validation-based early stopping;
- uncertainty quantification;
- safety-conscious feasibility classification;
- constrained multi-objective optimization;
- physics-based validation;
- ONNX deployment;
- CPU and accelerator inference benchmarking;
- reproducible testing and continuous integration.

The benchmark evaluates more than predictive accuracy. It also asks whether a
model:

- identifies feasible designs safely;
- expresses useful uncertainty;
- remains reliable when used by an optimizer;
- preserves predictions across runtime conversion;
- satisfies model-size and latency constraints;
- remains reproducible across training and deployment workflows.

EdgeGenBench uses synthetic or public information only. It does not contain
proprietary aircraft-manufacturer data, software, or design information.

## Current release

### EdgeGenBench v0.2

Version 0.2 adds a compact PyTorch multi-output neural surrogate to the complete
v0.1 scientific-ML and optimization benchmark.

The current v0.2 neural model uses:

```text
10 encoded inputs
      |
      v
Linear(10, 64)
      |
     ReLU
      |
      v
Linear(64, 32)
      |
     ReLU
      |
      v
Linear(32, 16)
      |
     ReLU
      |
      v
Linear(16, 6)
      |
      v
6 aircraft-design targets
```

The architecture contains only **3,414 trainable parameters**.

### v0.2 capabilities

- Compact multi-output PyTorch surrogate
- Training-only feature normalization
- Training-only target normalization
- Deterministic categorical encoding
- Reproducible train, validation, and test partitions
- AdamW optimization
- Validation-based early stopping
- Best-checkpoint restoration
- Held-out test evaluation
- Persisted preprocessing statistics
- Model and preprocessing serialization
- CPU and Apple MPS execution
- Batch-1, batch-32, and batch-256 inference benchmarking
- Mean and P95 latency measurement
- Public `train-neural-surrogate` CLI command
- CPU-compatible automated neural tests
- GitHub Actions neural dependency support

The next v0.2 deployment milestones are:

- PyTorch-to-ONNX export;
- ONNX Runtime numerical-equivalence validation;
- FP16 conversion;
- INT8 quantization;
- accuracy-size-latency comparison;
- Qualcomm QNN and Snapdragon NPU profiling.

## Benchmark problem

The benchmark represents an early regional-aircraft design study.

### Design variables

- Passenger capacity
- Design range
- Cruise speed
- Battery specific energy
- Hydrogen storage efficiency
- Hybridization ratio
- Propulsion architecture

Supported propulsion architectures are:

- conventional turboprop;
- parallel hybrid;
- series hybrid;
- fuel-cell electric.

The six numerical features plus four one-hot propulsion categories produce a
ten-dimensional encoded input for the neural surrogate.

### Predicted targets

All surrogate models predict:

- estimated takeoff mass;
- mission energy;
- energy per passenger-kilometre;
- lifecycle-emissions proxy;
- operating-cost proxy;
- noise proxy.

The synthetic physics model also produces feasibility labels and supporting
engineering quantities used by the optimization and validation workflows.

## Dataset

The versioned v0.1 benchmark generator creates 6,000 synthetic aircraft-design
cases.

| Partition | Rows |
|---|---:|
| Training | 4,200 |
| Validation | 900 |
| Test | 900 |
| Total | 6,000 |

The generated dataset has an overall physics-labeled feasible fraction of
approximately 32.6%.

Neural preprocessing statistics are fitted **only on the training partition**.
Validation and test rows are transformed using the frozen training statistics.

## v0.2 neural-surrogate results

### Aggregate performance

The compact PyTorch model achieved:

| Metric | Result |
|---|---:|
| Architecture | 10 → 64 → 32 → 16 → 6 |
| Trainable parameters | 3,414 |
| Best epoch | 141 |
| Training epochs completed | 171 |
| Mean test NRMSE | **0.050425** |
| Mean test R² | **0.996956** |
| Serialized model size | **16,881 bytes / 16.49 KiB** |

The reported aggregate accuracy above uses the CPU validation run.

CPU and MPS training produced effectively equivalent held-out accuracy:

| Device | Mean test NRMSE | Mean test R² |
|---|---:|---:|
| CPU | **0.050425** | **0.996956** |
| Apple MPS | 0.050433 | 0.996955 |

### Target-level test results

Reference MPS test metrics were:

| Target | NRMSE | R² |
|---|---:|---:|
| Estimated takeoff mass | 0.073473 | 0.994602 |
| Mission energy | 0.048346 | 0.997663 |
| Energy per passenger-km | 0.083380 | 0.993048 |
| Lifecycle-emissions proxy | 0.035690 | 0.998726 |
| Operating-cost proxy | 0.045074 | 0.997968 |
| Noise proxy | 0.016633 | 0.999723 |

All six targets achieved R² above 0.993 on the held-out test partition.

### Classical versus neural accuracy

| Model | Mean test NRMSE | Mean test R² | Model size |
|---|---:|---:|---:|
| Compact PyTorch MLP | **0.050425** | **0.996956** | **16.49 KiB** |
| HistGradientBoosting | 0.062249 | 0.995171 | 7.216 MiB |
| Random Forest | 0.205386 | 0.953219 | 172.296 MiB |
| FP32 Ridge | 0.214590 | 0.937690 | 2.53 KiB |

The compact neural surrogate reduces mean NRMSE by approximately 19% relative
to the strongest classical model while using a much smaller serialized model
artifact.

Serialized sizes depend on runtime and file format, so size comparisons should
be interpreted as deployment-oriented measurements rather than pure
architecture complexity comparisons.

## v0.2 PyTorch inference benchmark

### CPU

The compact network is small enough that CPU execution is especially effective
for low-latency inference on the development machine.

| Batch size | Mean batch latency | P95 batch latency | Mean sample latency |
|---:|---:|---:|---:|
| 1 | **0.0220 ms** | 0.0226 ms | 22.02 µs |
| 32 | **0.0307 ms** | 0.0312 ms | 0.958 µs |
| 256 | **0.0484 ms** | 0.0507 ms | 0.189 µs |

These results are specific to the local ARM64 macOS development environment.

### Apple MPS observation

Apple MPS execution was successfully validated for both training and
inference, and predictive accuracy matched CPU execution closely.

However, MPS latency showed substantial run-to-run variability for this very
small 3,414-parameter network. Accelerator-dispatch overhead dominated the
tiny amount of neural computation.

For that reason:

- CPU is used as the primary v0.2 PyTorch local-latency baseline;
- MPS is treated as a functional accelerator-validation path;
- no general CPU-versus-GPU performance claim is made;
- later ONNX, QNN, and NPU measurements will be reported separately by runtime
  and hardware target.

## v0.1 scientific-ML results

Version 0.1 established the complete classical surrogate, uncertainty,
optimization, physics-validation, and ONNX workflow that v0.2 builds upon.

### Classical surrogate comparison

| Model | Mean test NRMSE | Mean test R² | Model size | Batch-1 latency |
|---|---:|---:|---:|---:|
| HistGradientBoosting | **0.062249** | **0.995171** | 7.216 MiB | 76.339 ms |
| Random Forest | 0.205386 | 0.953219 | 172.296 MiB | 14.652 ms |
| FP32 Ridge | 0.214590 | 0.937690 | **0.002473 MiB** | **0.142 ms** |

HistGradientBoosting produced the strongest classical predictive accuracy.

The Random Forest surrogate remains the v0.1 model used for uncertainty
estimation, constrained optimization, and the original Scikit-learn-to-ONNX
deployment workflow.

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

EdgeGenBench explicitly separates ordinary classification and
optimization-specific feasibility thresholds.

| Threshold | Purpose |
|---:|---|
| 0.30 | Validation-selected classifier operating threshold |
| **0.50** | Physics-validated optimization acceptance threshold |

Final optimization results:

| Optimization result | Value |
|---|---:|
| Generated candidates | 20,000 |
| Accepted candidates | 5,413 |
| Accepted fraction | 27.065% |
| Pareto designs | 47 |
| Representative designs | 4 |
| Pareto-front feasibility agreement | 100% |
| Representative-design feasibility agreement | 100% |

### v0.1 ONNX deployment

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

Detailed v0.1 results are available in
[`docs/results.md`](docs/results.md).

Detailed v0.2 neural results are available in
[`docs/v0_2_results.md`](docs/v0_2_results.md).

## System workflow

```text
Versioned configuration
        |
        v
Synthetic physics model
        |
        v
Dataset generation
        |
        v
Deterministic train / validation / test partitions
        |
        +--------------------------------------+
        |                                      |
        v                                      v
Classical surrogate branch               Neural surrogate branch
        |                                      |
        +--> FP32 Ridge                         +--> train-only normalization
        +--> Random Forest                      +--> compact PyTorch MLP
        +--> HistGradientBoosting               +--> AdamW
        |                                      +--> validation early stopping
        |                                      +--> best checkpoint
        |                                      +--> CPU / MPS benchmark
        |                                      |
        +------------------+-------------------+
                           |
                           v
                 Held-out test evaluation
                           |
                           v
                Uncertainty quantification
                           |
                           v
                 Feasibility classifier
                           |
                           v
              Conservative optimization gate
                           |
                           v
             Multi-objective optimization
                           |
                           v
                 Physics validation
                           |
                           v
                 Deployment evaluation
                           |
             +-------------+-------------+
             |                           |
             v                           v
       v0.1 classical ONNX       v0.2 neural deployment
                                      |
                                      +--> ONNX next
                                      +--> FP16 next
                                      +--> INT8 next
                                      +--> QNN/NPU next
```

## Installation

### 1. Create a Python 3.12 environment

Using Conda:

```bash
conda create -n edgegenbench-py312 python=3.12
conda activate edgegenbench-py312
```

### 2. Install EdgeGenBench

For development, classical edge deployment, and neural modeling:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,edge,neural]"
```

Extras:

- `dev`: testing and linting;
- `edge`: ONNX export and ONNX Runtime;
- `neural`: PyTorch neural-surrogate support.

### 3. Confirm the installation

```bash
edgegenbench info
```

Expected:

```text
EdgeGenBench 0.2.0
Status: compact neural edge-inference benchmark.
```

## Neural-surrogate workflow

### Generate the dataset

```bash
edgegenbench generate-data \
  --config configs/v0_1.yaml
```

### Train the compact PyTorch surrogate

```bash
edgegenbench train-neural-surrogate \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --config configs/neural_v0_2.yaml \
  --output-dir artifacts/neural_surrogate
```

The training command produces:

```text
artifacts/neural_surrogate/
├── model.pt
├── preprocessing.npz
├── training_history.csv
├── test_metrics.csv
├── test_predictions.csv
├── latency.csv
└── summary.json
```

## v0.1 classical workflow

The complete v0.1 scientific-ML and optimization workflow remains executable:

```bash
./scripts/run_full_pipeline.sh
```

The classical pipeline covers:

1. formatting, linting, and tests;
2. deterministic dataset generation;
3. FP32 Ridge training;
4. Random Forest and HistGradientBoosting training;
5. surrogate comparison;
6. uncertainty evaluation;
7. feasibility classification;
8. constrained multi-objective optimization;
9. physics-based validation;
10. classical ONNX export;
11. ONNX equivalence validation;
12. latency benchmarking.

## Manual classical CLI commands

### Train the FP32 baseline

```bash
edgegenbench train-fp32-baseline \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --output-dir artifacts/fp32_baseline
```

### Train tree baselines

```bash
edgegenbench train-tree-baselines \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --output-dir artifacts/tree_baselines
```

### Compare models

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
  --surrogate-model artifacts/tree_baselines/random_forest/model.joblib \
  --feasibility-model artifacts/feasibility_classifier/model.joblib \
  --output-dir artifacts/optimization
```

### Validate optimized designs

```bash
edgegenbench validate-optimization \
  --designs artifacts/optimization/representative_designs.csv \
  --benchmark-config configs/v0_1.yaml \
  --output-dir artifacts/optimization_validation
```

### Export classical models to ONNX

```bash
edgegenbench export-edge-models \
  --surrogate-model artifacts/tree_baselines/random_forest/model.joblib \
  --feasibility-model artifacts/feasibility_classifier/model.joblib \
  --output-dir artifacts/edge_export
```

### Benchmark classical edge models

```bash
edgegenbench benchmark-edge-models \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --surrogate-model artifacts/tree_baselines/random_forest/model.joblib \
  --feasibility-model artifacts/feasibility_classifier/model.joblib \
  --surrogate-onnx artifacts/edge_export/surrogate.onnx \
  --feasibility-onnx artifacts/edge_export/feasibility.onnx \
  --metadata artifacts/edge_export/metadata.json \
  --output-dir artifacts/edge_benchmark
```

## Generated artifacts

Generated data, models, benchmark reports, and temporary experimental
directories are intentionally excluded from Git.

Typical output directories are:

```text
data/raw/
artifacts/fp32_baseline/
artifacts/tree_baselines/
artifacts/neural_surrogate/
artifacts/uncertainty/
artifacts/feasibility_classifier/
artifacts/optimization/
artifacts/optimization_validation/
artifacts/edge_export/
artifacts/edge_benchmark/
reports/model_comparison/
```

## Repository structure

```text
EdgeGenBench/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   ├── v0_1.yaml
│   ├── optimization_v0_1.yaml
│   └── neural_v0_2.yaml
├── data/
├── docs/
│   ├── architecture.md
│   ├── DESIGN_CONTRACT.md
│   ├── reproducibility.md
│   ├── results.md
│   └── v0_2_results.md
├── notebooks/
├── reports/
├── scripts/
│   └── run_full_pipeline.sh
├── src/
│   └── edgegenbench/
│       ├── data/
│       ├── deployment/
│       ├── evaluation/
│       ├── models/
│       │   ├── neural_preprocessing.py
│       │   └── neural_surrogate.py
│       ├── optimization/
│       ├── physics/
│       ├── training/
│       │   └── neural_surrogate.py
│       └── uncertainty/
├── tests/
│   └── neural/
├── CHANGELOG.md
├── pyproject.toml
└── README.md
```

## Development checks

Run:

```bash
ruff format --check .
ruff check .
pytest -q
git diff --check
```

The GitHub Actions workflow runs the test suite under Python 3.12 on CPU and
installs the development, edge, and neural dependency groups.

## Reproducibility

EdgeGenBench uses:

- versioned configuration files;
- fixed random seeds;
- deterministic dataset generation;
- fixed train, validation, calibration, and test partitions;
- train-only neural normalization;
- validation-based model selection;
- best-checkpoint restoration;
- held-out test evaluation;
- serialized preprocessing state;
- artifact-backed experiment summaries;
- explicit runtime and device reporting.

The neural configuration uses seed `42`.

Accelerator execution can introduce small floating-point differences, so
hardware-specific results are recorded separately rather than assumed to be
bitwise identical.

## Roadmap

### v0.2 — Compact neural surrogate and efficient inference

Completed:

- [x] Compact multi-output PyTorch surrogate
- [x] Training-only feature normalization
- [x] Training-only target normalization
- [x] Validation-based early stopping
- [x] Best-checkpoint restoration
- [x] Persisted preprocessing state
- [x] CPU and MPS execution
- [x] PyTorch latency benchmarking
- [x] Public neural-training CLI
- [x] Neural CI coverage

Next:

- [ ] PyTorch-to-ONNX export
- [ ] PyTorch-to-ONNX numerical-equivalence validation
- [ ] ONNX Runtime neural benchmark
- [ ] FP16 conversion
- [ ] INT8 quantization
- [ ] Accuracy-size-latency-throughput comparison
- [ ] Qualcomm AI Hub integration
- [ ] QNN compilation
- [ ] Snapdragon NPU profiling

### v0.3 — Robustness and distribution shift

- Out-of-distribution evaluation
- Extrapolation tests
- Shift-aware uncertainty evaluation
- Failure-region analysis
- Optimization robustness under changed missions

### v0.4 — Hardware-aware deployment

- Additional mobile and embedded CPU benchmarks
- GPU and NPU execution providers
- Memory measurements
- Energy measurements
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

EdgeGenBench has important limitations:

- The benchmark dataset is synthetic.
- The physics model is an analytical research proxy.
- No proprietary, certification, or flight-test data are included.
- Current results represent one mission and design-space configuration.
- Optimization thresholds must be reevaluated for changed design spaces.
- Conformal coverage is demonstrated only on the current benchmark split.
- Latency results are hardware-, runtime-, and environment-dependent.
- CPU and MPS measurements are not interchangeable hardware comparisons.
- MPS latency is inefficient and variable for the current tiny neural model.
- The neural surrogate has not yet been exported to ONNX.
- FP16 and INT8 neural deployment are not yet included.
- Qualcomm QNN and Snapdragon NPU measurements are not yet included.
- No microcontroller deployment benchmark is currently included.

EdgeGenBench is not:

- a certified aircraft-sizing tool;
- an operational aircraft-performance model;
- a manufacturer design prediction;
- a substitute for validated engineering analysis;
- a safety-critical decision system.

## Author

Triasha Sarkar
