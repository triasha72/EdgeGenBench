# EdgeGenBench

[![CI](https://github.com/triasha72/EdgeGenBench/actions/workflows/ci.yml/badge.svg)](https://github.com/triasha72/EdgeGenBench/actions/workflows/ci.yml)

**Scientific machine learning, uncertainty-aware aircraft-design optimization,
and hardware-aware edge inference for hybrid-electric and hydrogen regional-aircraft studies.**

## Overview

EdgeGenBench is an independent, reproducible benchmark for studying how
machine-learning surrogates behave inside early aircraft-design workflows and
how trained models translate into deployable edge-inference systems.

The project connects:

- synthetic physics-based aircraft-design data generation;
- classical multi-output surrogate modeling;
- compact PyTorch neural surrogate modeling;
- leakage-safe feature and target normalization;
- validation-based early stopping;
- uncertainty quantification;
- safety-conscious feasibility classification;
- constrained multi-objective optimization;
- physics-based validation;
- ONNX deployment;
- numerical-equivalence testing;
- CPU and accelerator inference benchmarking;
- reproducible testing and continuous integration.

EdgeGenBench uses synthetic or public information only. It does not contain
proprietary aircraft-manufacturer data, software, or design information.

## Current status

### Released: EdgeGenBench v0.2.0

Version 0.2.0 added the compact PyTorch multi-output surrogate while retaining
the complete v0.1 scientific-ML, optimization, and classical ONNX workflow.

### Unreleased deployment work

The current development branch adds the first neural deployment milestone:

- checkpoint reconstruction from stored architecture metadata;
- PyTorch-to-ONNX FP32 export;
- dynamic ONNX batch dimensions;
- ONNX graph validation;
- ONNX Runtime CPU inference;
- frozen-preprocessor reuse;
- held-out PyTorch-to-ONNX numerical-equivalence validation;
- physical-unit equivalence validation;
- paired PyTorch CPU versus ONNX Runtime CPU benchmarking;
- repeated local runtime measurements;
- public `export-neural-onnx` CLI support;
- public `benchmark-neural-onnx` CLI support.

FP16, INT8, Qualcomm QNN, and Snapdragon NPU deployment remain future work.

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

Supported propulsion architectures:

- conventional turboprop;
- parallel hybrid;
- series hybrid;
- fuel-cell electric.

The six numerical variables plus four one-hot propulsion categories produce a
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

The versioned benchmark generator creates 6,000 synthetic aircraft-design
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
Validation and test rows use the frozen training statistics.

## Compact neural surrogate

The v0.2 neural architecture is:

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

The model contains **3,414 trainable parameters**.

### Held-out predictive performance

| Metric | Result |
|---|---:|
| Architecture | 10 → 64 → 32 → 16 → 6 |
| Trainable parameters | 3,414 |
| Best epoch | 141 |
| Training completion epoch | 171 |
| CPU mean test NRMSE | **0.050425** |
| CPU mean test R² | **0.996956** |
| MPS mean test NRMSE | 0.050433 |
| MPS mean test R² | 0.996955 |
| PyTorch checkpoint size | **16,881 bytes** |

Reference target-level results:

| Target | NRMSE | R² |
|---|---:|---:|
| Estimated takeoff mass | 0.073473 | 0.994602 |
| Mission energy | 0.048346 | 0.997663 |
| Energy per passenger-km | 0.083380 | 0.993048 |
| Lifecycle-emissions proxy | 0.035690 | 0.998726 |
| Operating-cost proxy | 0.045074 | 0.997968 |
| Noise proxy | 0.016633 | 0.999723 |

Every target achieved held-out R² above 0.993.

### Classical versus neural accuracy

| Model | Mean test NRMSE | Mean test R² | Serialized size |
|---|---:|---:|---:|
| Compact PyTorch MLP | **0.050425** | **0.996956** | **16.49 KiB** |
| HistGradientBoosting | 0.062249 | 0.995171 | 7.216 MiB |
| Random Forest | 0.205386 | 0.953219 | 172.296 MiB |
| FP32 Ridge | 0.214590 | 0.937690 | 2.53 KiB |

The compact neural surrogate reduces mean NRMSE by approximately 19% relative
to the strongest classical predictive baseline while using a much smaller
serialized artifact.

Serialized formats differ, so file-size comparisons should be interpreted as
deployment-oriented measurements rather than pure architecture-memory
comparisons.

## PyTorch CPU baseline

Reference PyTorch CPU latency:

| Batch | Mean batch latency | P95 batch latency | Mean sample latency |
|---:|---:|---:|---:|
| 1 | 0.022017 ms | 0.022585 ms | 22.017 µs |
| 32 | 0.030655 ms | 0.031171 ms | 0.958 µs |
| 256 | 0.048388 ms | 0.050673 ms | 0.189 µs |

These values are specific to the local ARM64 macOS development environment.

Apple MPS execution was also validated, but latency for this very small network
showed substantial run-to-run variability. CPU therefore remains the primary
local PyTorch reference runtime.

## Neural ONNX deployment

The compact PyTorch surrogate now exports to a dynamic-batch FP32 ONNX graph.

### Exported graph

```text
features [batch, 10]
        |
        v
  FP32 ONNX MLP
        |
        v
predictions [batch, 6]
```

| Property | Result |
|---|---:|
| ONNX checker | PASS |
| Opset | 18 |
| Dynamic batch | Yes |
| Input width | 10 |
| Output width | 6 |
| PyTorch checkpoint size | 16,881 bytes |
| ONNX graph size | 25,420 bytes |
| ONNX/PyTorch serialized-size ratio | 1.506× |

The two serialized formats contain different metadata and should not be treated
as direct parameter-memory equivalents.

### Held-out PyTorch ↔ ONNX Runtime equivalence

All **900 held-out test rows** were evaluated through both runtimes.

| Metric | Result |
|---|---:|
| Mean normalized absolute difference | **1.3064681070e-07** |
| Maximum normalized absolute difference | **9.5367431641e-07** |
| `rtol` | 1e-5 |
| `atol` | 1e-5 |
| Numerical equivalence | **PASS** |

Physical-unit maximum differences remained negligible:

| Target | Maximum absolute difference | Maximum reference-relative difference |
|---|---:|---:|
| Estimated takeoff mass | 0.00781250 kg | 1.151e-07 |
| Mission energy | 0.00390625 kWh | 1.347e-07 |
| Energy per passenger-km | 5.960e-08 | 1.697e-07 |
| Lifecycle-emissions proxy | 0.00146484 | 2.068e-07 |
| Operating-cost proxy | 0.00048828 USD | 1.170e-07 |
| Noise proxy | 7.629e-06 dB | 8.260e-08 |

## Repeated PyTorch CPU versus ONNX Runtime CPU benchmark

The paired runtime benchmark compares the same trained network and identical
preprocessed FP32 inputs. Preprocessing and CSV/Pandas work are intentionally
outside the timed region.

Three independent local benchmark repetitions produced:

| Run | Batch 1 ratio | Batch 32 ratio | Batch 256 ratio |
|---:|---:|---:|---:|
| 1 | 2.868× | 2.958× | 1.659× |
| 2 | 3.768× | 2.818× | 1.788× |
| 3 | 3.481× | 2.872× | 1.303× |

A ratio greater than 1 means lower ONNX Runtime mean latency.

Median results:

| Batch | Median PyTorch latency | Median ORT latency | Median PyTorch/ORT ratio |
|---:|---:|---:|---:|
| 1 | 0.051814 ms | **0.014886 ms** | **3.481×** |
| 32 | 0.065456 ms | **0.022665 ms** | **2.872×** |
| 256 | 0.103179 ms | **0.060500 ms** | **1.659×** |

ONNX Runtime produced lower mean latency in every repeat at all three tested
batch sizes. Absolute microsecond-scale timings varied between runs, so these
results are workload- and machine-specific rather than universal runtime
claims.

Detailed deployment results are recorded in
[`docs/neural_onnx_results.md`](docs/neural_onnx_results.md).

## v0.1 scientific-ML results

Version 0.1 established the complete classical surrogate, uncertainty,
optimization, physics-validation, and classical ONNX workflow.

### Classical surrogate comparison

| Model | Mean test NRMSE | Mean test R² | Model size | Batch-1 latency |
|---|---:|---:|---:|---:|
| HistGradientBoosting | **0.062249** | **0.995171** | 7.216 MiB | 76.339 ms |
| Random Forest | 0.205386 | 0.953219 | 172.296 MiB | 14.652 ms |
| FP32 Ridge | 0.214590 | 0.937690 | **0.002473 MiB** | **0.142 ms** |

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

| Result | Value |
|---|---:|
| Generated candidates | 20,000 |
| Accepted candidates | 5,413 |
| Accepted fraction | 27.065% |
| Pareto designs | 47 |
| Representative designs | 4 |
| Pareto-front feasibility agreement | 100% |
| Representative-design feasibility agreement | 100% |

The optimization path uses a conservative, physics-validated feasibility
threshold of 0.50, separate from the ordinary classifier threshold of 0.30.

### Classical ONNX deployment

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
        +--> Ridge                              +--> train-only normalization
        +--> Random Forest                      +--> compact PyTorch MLP
        +--> HistGradientBoosting               +--> AdamW + early stopping
        |                                      +--> checkpoint reconstruction
        |                                      +--> PyTorch CPU / MPS
        |                                      +--> FP32 ONNX export
        |                                      +--> ORT equivalence
        |                                      +--> repeated CPU benchmark
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
```

## Installation

### 1. Create a Python 3.12 environment

```bash
conda create -n edgegenbench-py312 python=3.12
conda activate edgegenbench-py312
```

### 2. Install development, edge, and neural dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,edge,neural]"
```

The `edge` extra includes:

- ONNX;
- ONNX Runtime;
- ONNX Script;
- skl2onnx.

The `neural` extra provides PyTorch.

### 3. Confirm installation

```bash
edgegenbench info
```

## Neural workflow

### Generate data

```bash
edgegenbench generate-data \
  --config configs/v0_1.yaml
```

### Train the compact neural surrogate

```bash
edgegenbench train-neural-surrogate \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --config configs/neural_v0_2.yaml \
  --output-dir artifacts/neural_surrogate
```

### Export the trained neural surrogate to ONNX

```bash
edgegenbench export-neural-onnx \
  --model artifacts/neural_surrogate/model.pt \
  --preprocessing artifacts/neural_surrogate/preprocessing.npz \
  --output-dir artifacts/neural_onnx \
  --opset 18
```

### Benchmark PyTorch CPU against ONNX Runtime CPU

```bash
edgegenbench benchmark-neural-onnx \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --model artifacts/neural_surrogate/model.pt \
  --preprocessing artifacts/neural_surrogate/preprocessing.npz \
  --onnx-model artifacts/neural_onnx/neural_surrogate.onnx \
  --metadata artifacts/neural_onnx/metadata.json \
  --output-dir artifacts/neural_onnx_benchmark \
  --repeats 500 \
  --warmups 50
```

## Generated neural artifacts

Training:

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

ONNX export:

```text
artifacts/neural_onnx/
├── metadata.json
└── neural_surrogate.onnx
```

Deployment benchmark:

```text
artifacts/neural_onnx_benchmark/
├── equivalence.csv
├── latency.csv
└── summary.json
```

Generated artifacts are intentionally ignored by Git and are reproducible from
the source workflow.

## Validation

Current neural deployment validation includes:

- checkpoint reconstruction tests;
- preprocessing serialization tests;
- neural training tests;
- ONNX export tests;
- dynamic batch tests;
- ONNX Runtime inference tests;
- normalized parity tests;
- physical-unit parity tests;
- benchmark artifact tests;
- CLI registration tests.

The neural suite contains **27 passing tests** at the FP32 ONNX milestone.
The complete repository formatting, lint, and test suite also pass locally.

## Documentation

- [`docs/results.md`](docs/results.md): v0.1 scientific-ML results
- [`docs/v0_2_results.md`](docs/v0_2_results.md): v0.2 neural-training results
- [`docs/neural_onnx_results.md`](docs/neural_onnx_results.md): FP32 neural ONNX deployment results
- [`ROADMAP.md`](ROADMAP.md): deployment and evaluation roadmap
- [`CHANGELOG.md`](CHANGELOG.md): release history and unreleased work

## Roadmap summary

Completed:

```text
v0.1 scientific-ML pipeline
        ↓
v0.2 compact PyTorch surrogate
        ↓
FP32 PyTorch → ONNX
        ↓
900-row equivalence validation
        ↓
repeated PyTorch CPU ↔ ORT CPU benchmark
```

Next:

```text
FP16 feasibility + equivalence
        ↓
INT8 quantization + calibration
        ↓
accuracy / size / latency comparison
        ↓
Qualcomm QNN integration
        ↓
Snapdragon NPU profiling
        ↓
distribution-shift / extrapolation evaluation
```

See [`ROADMAP.md`](ROADMAP.md) for the detailed plan.

## Reproducibility

The project emphasizes:

- deterministic dataset generation;
- fixed train/validation/test partitions;
- training-only preprocessing fits;
- explicit checkpoint metadata;
- versioned configuration;
- reproducible CLI workflows;
- generated deployment metadata;
- automated test coverage;
- CI validation;
- explicit separation of measured results from planned work.

## Limitations

- The benchmark uses synthetic regional-aircraft design data.
- Results represent one design-space configuration.
- Local latency measurements are machine-specific.
- Microsecond-scale timings are sensitive to operating-system scheduling and runtime state.
- PyTorch and ONNX serialized sizes are not directly equivalent formats.
- MPS timing is not used as a universal CPU-versus-GPU comparison.
- FP16 and INT8 deployment are not yet validated.
- Qualcomm QNN and Snapdragon NPU profiling are not yet validated.
- EdgeGenBench is not a certified aircraft-design or safety-critical system.
