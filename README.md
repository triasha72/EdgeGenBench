# EdgeGenBench

[![CI](https://github.com/triasha72/EdgeGenBench/actions/workflows/ci.yml/badge.svg)](https://github.com/triasha72/EdgeGenBench/actions/workflows/ci.yml)

**Scientific machine learning, uncertainty-aware aircraft-design optimization,
and hardware-aware edge inference for hybrid-electric and hydrogen
regional-aircraft studies.**

## Overview

EdgeGenBench is an independent, reproducible benchmark for studying how
machine-learning surrogates behave inside early aircraft-design workflows and
how trained models translate into deployable edge-inference systems.

The project combines:

- synthetic physics-based aircraft-design data generation;
- classical multi-output surrogate modeling;
- compact PyTorch neural surrogate modeling;
- leakage-safe preprocessing;
- validation-based model selection and early stopping;
- uncertainty quantification;
- feasibility classification;
- constrained multi-objective optimization;
- physics-based validation;
- classical and neural ONNX deployment;
- numerical-equivalence testing;
- FP16 reduced-precision evaluation;
- CPU and CoreML execution-provider benchmarking;
- reproducible testing and continuous integration.

EdgeGenBench uses synthetic or public information only. It does not contain
proprietary aircraft-manufacturer data, software, or design information.

## Current status

### Released: EdgeGenBench v0.2.0

Version 0.2.0 added the compact PyTorch multi-output surrogate while retaining
the complete v0.1 scientific-ML, optimization, uncertainty, and classical ONNX
workflow.

### Unreleased neural-deployment work

The current development work adds:

- checkpoint reconstruction from stored neural architecture metadata;
- PyTorch-to-ONNX FP32 export;
- dynamic ONNX batch dimensions;
- ONNX graph validation;
- ONNX Runtime CPU inference;
- held-out PyTorch-to-ONNX numerical-equivalence validation;
- corrected paired PyTorch CPU versus ONNX Runtime CPU benchmarking;
- FP32-to-FP16 ONNX conversion;
- FP32 external I/O with internal FP16 weights;
- FP16 graph validation;
- static batch specialization for CoreML;
- held-out provider-drift and precision-drift analysis;
- FP16 predictive-quality regression guards;
- repeated FP32 versus FP16 CoreML benchmarking;
- public FP32 and FP16 deployment CLI commands.

INT8, Qualcomm QNN, Snapdragon NPU profiling, and distribution-shift studies
remain future work.

## Benchmark problem

The benchmark represents an early regional-aircraft design study.

### Design variables

The design space contains:

- passenger capacity;
- design range;
- cruise speed;
- battery specific energy;
- hydrogen storage efficiency;
- hybridization ratio;
- propulsion architecture.

Supported propulsion architectures are:

- conventional turboprop;
- parallel hybrid;
- series hybrid;
- fuel-cell electric.

The six numerical variables plus four one-hot propulsion categories produce a
ten-dimensional encoded neural input.

### Predicted targets

All regression surrogates predict:

- estimated takeoff mass;
- mission energy;
- energy per passenger-kilometre;
- lifecycle-emissions proxy;
- operating-cost proxy;
- noise proxy.

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
Validation and test rows reuse the frozen training statistics.

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
| CPU mean test NRMSE | **0.050425** |
| CPU mean test R² | **0.996956** |
| MPS reference mean test NRMSE | 0.050433 |
| MPS reference mean test R² | 0.996955 |
| PyTorch checkpoint size | **16,881 bytes** |

Every target achieved held-out R² above 0.993.

### Classical versus neural accuracy

| Model | Mean test NRMSE | Mean test R² | Serialized size |
|---|---:|---:|---:|
| Compact PyTorch MLP | **0.050425** | **0.996956** | **16.49 KiB** |
| HistGradientBoosting | 0.062249 | 0.995171 | 7.216 MiB |
| Random Forest | 0.205386 | 0.953219 | 172.296 MiB |
| FP32 Ridge | 0.214590 | 0.937690 | 2.53 KiB |

The compact neural surrogate reduces mean NRMSE by approximately 19% relative
to the strongest classical predictive baseline.

Serialized formats differ, so file-size comparisons should be interpreted as
deployment-oriented measurements rather than direct parameter-memory
comparisons.

## FP32 neural ONNX deployment

The trained neural surrogate exports to an ONNX opset-18 graph with a dynamic
batch dimension.

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
| FP32 ONNX graph size | 25,420 bytes |

### Held-out PyTorch ↔ ONNX Runtime equivalence

All 900 held-out test rows were evaluated through both runtimes.

| Metric | Result |
|---|---:|
| Mean normalized absolute difference | **1.3064681070e-07** |
| Maximum normalized absolute difference | **9.5367431641e-07** |
| `rtol` | 1e-5 |
| `atol` | 1e-5 |
| Numerical equivalence | **PASS** |

The conversion differences are negligible relative to the predictive error of
the trained surrogate.

## Corrected PyTorch CPU versus ONNX Runtime CPU benchmark

The runtime comparison uses:

- the same trained network;
- the same preprocessed FP32 inputs;
- preprocessing outside the timed region;
- model/session construction outside the timed region;
- 50 warmup iterations;
- 500 measured iterations per run;
- three independent runs;
- one outer `torch.inference_mode()` context around the PyTorch timing loop.

The inference context is not entered and exited inside each timed PyTorch
forward pass.

Three-run aggregate results:

| Batch | Median PyTorch latency | Median ORT latency | Median PyTorch/ORT ratio | Ratio range |
|---:|---:|---:|---:|---:|
| 1 | 0.018341 ms | **0.006157 ms** | **2.979×** | 2.715×–3.967× |
| 32 | 0.018427 ms | **0.007984 ms** | **2.385×** | 2.078×–2.404× |
| 256 | **0.030841 ms** | 0.033574 ms | 0.919× | 0.860×–1.079× |

A ratio greater than 1 means lower ONNX Runtime mean latency.

The defensible conclusion is:

- ONNX Runtime shows a clear local latency advantage at batch sizes 1 and 32;
- batch 256 is near parity and changes direction across repeated runs;
- microsecond-scale results are workload- and machine-specific.

Detailed FP32 deployment results are recorded in
[`docs/neural_onnx_results.md`](docs/neural_onnx_results.md).

## FP16 neural ONNX deployment study

The validated FP32 ONNX graph is also converted to FP16 using
`onnxconverter-common`.

The conversion keeps external input/output tensors in FP32 while converting
eligible internal parameters to FP16.

### FP16 artifact

| Property | FP32 | FP16 |
|---|---:|---:|
| External input width | 10 | 10 |
| External output width | 6 | 6 |
| External I/O precision | FP32 | FP32 |
| Internal precision | FP32 | FP16 |
| FP16 initializers | — | 8 |
| Serialized ONNX size | 25,420 B | 19,221 B |

The FP16 graph reduces serialized ONNX size by **24.39%**.

### Held-out FP16 precision drift

All 900 held-out test rows were evaluated.

| Metric | Result | Regression ceiling |
|---|---:|---:|
| Mean normalized FP32-CoreML ↔ FP16-CoreML difference | **9.7869e-04** | 0.002 |
| Maximum normalized FP32-CoreML ↔ FP16-CoreML difference | **9.1944e-03** | 0.012 |
| Mean-drift guard | **PASS** | — |
| Maximum-drift guard | **PASS** | — |

FP32 CPU ↔ FP32 CoreML provider drift was substantially smaller:

| Metric | Result |
|---|---:|
| Mean normalized provider difference | 1.4848e-07 |
| Maximum normalized provider difference | 1.4305e-06 |

This separates ordinary provider-level numerical variation from the much larger,
but still bounded, FP16 precision effect.

### FP16 predictive quality

| Metric | FP32 reference | FP16 |
|---|---:|---:|
| Mean test NRMSE | 0.050433 | **0.050473** |
| Mean test R² | 0.996955 | **0.996954** |

FP16 therefore preserves essentially the same held-out predictive quality while
reducing serialized model size.

Target-level FP16 results:

| Target | NRMSE | R² |
|---|---:|---:|
| Estimated takeoff mass | 0.073412 | 0.994611 |
| Mission energy | 0.048342 | 0.997663 |
| Energy per passenger-km | 0.083341 | 0.993054 |
| Lifecycle-emissions proxy | 0.035779 | 0.998720 |
| Operating-cost proxy | 0.045180 | 0.997959 |
| Noise proxy | 0.016787 | 0.999718 |

### FP32 versus FP16 CoreML benchmark

For CoreML evaluation, dynamic models are reproducibly specialized to static
batch sizes 1, 32, and 256.

The ONNX Runtime provider configuration is:

```text
CoreMLExecutionProvider
ModelFormat=MLProgram
MLComputeUnits=ALL
RequireStaticInputShapes=1
EnableOnSubgraphs=0

CPUExecutionProvider fallback
```

Production benchmark settings:

```text
runs    = 5
repeats = 500
warmups = 50
```

Median results:

| Batch | FP32 median | FP16 median | Median FP32/FP16 ratio | FP16 faster runs |
|---:|---:|---:|---:|---:|
| 1 | 0.038480 ms | 0.038685 ms | 0.995× | 2/5 |
| 32 | 0.040879 ms | 0.041161 ms | 1.009× | 4/5 |
| 256 | **0.051657 ms** | 0.059952 ms | 0.862× | 0/5 |

Interpretation:

- batch 1 is effectively at parity;
- batch 32 is effectively at parity;
- FP16 is consistently slower at batch 256 in this experiment;
- FP16 should therefore not be described as a universal latency optimization.

`MLComputeUnits=ALL` allows CoreML to select supported compute units. These
measurements do **not** independently prove exclusive Apple Neural Engine
execution.

Detailed FP16 results are recorded in
[`docs/neural_fp16_results.md`](docs/neural_fp16_results.md).

## v0.1 scientific-ML results

Version 0.1 established the classical surrogate, uncertainty, optimization,
physics-validation, and classical ONNX workflow.

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

The optimization path uses a conservative physics-validated feasibility
threshold of 0.50, separate from the ordinary classifier threshold of 0.30.

## System workflow

```text
Versioned configuration
        |
        v
Synthetic physics model
        |
        v
6,000-case benchmark dataset
        |
        v
deterministic train / validation / test split
        |
        +-----------------------------------+
        |                                   |
        v                                   v
Classical branch                       Neural branch
        |                                   |
Ridge / RF / HGB                      compact PyTorch MLP
        |                                   |
uncertainty / feasibility             FP32 ONNX
        |                                   |
optimization                          ORT CPU evaluation
        |                                   |
physics validation                    FP16 ONNX
                                            |
                                      static CoreML models
                                            |
                                precision + latency evaluation
```

## Installation

### Create a Python 3.12 environment

```bash
conda create -n edgegenbench-py312 python=3.12
conda activate edgegenbench-py312
```

### Install development, edge, and neural dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,edge,neural]"
```

The `edge` extra includes ONNX, ONNX Runtime, ONNX Script,
`onnxconverter-common`, and `skl2onnx`.

The `neural` extra provides PyTorch.

### Confirm installation

```bash
edgegenbench info
```

## Neural deployment workflow

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

### Export FP32 neural ONNX

```bash
edgegenbench export-neural-onnx \
  --model artifacts/neural_surrogate/model.pt \
  --preprocessing artifacts/neural_surrogate/preprocessing.npz \
  --output-dir artifacts/neural_onnx \
  --opset 18
```

### Benchmark PyTorch CPU versus ONNX Runtime CPU

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

### Export FP16 neural ONNX

```bash
edgegenbench export-neural-fp16 \
  --fp32-model artifacts/neural_onnx/neural_surrogate.onnx \
  --fp32-metadata artifacts/neural_onnx/metadata.json \
  --output-dir artifacts/neural_fp16
```

### Benchmark FP32 versus FP16 with CoreML

```bash
edgegenbench benchmark-neural-fp16 \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --preprocessing artifacts/neural_surrogate/preprocessing.npz \
  --fp32-model artifacts/neural_onnx/neural_surrogate.onnx \
  --fp16-model artifacts/neural_fp16/neural_surrogate_fp16.onnx \
  --output-dir artifacts/neural_fp16_benchmark \
  --runs 5 \
  --repeats 500 \
  --warmups 50 \
  --max-mean-normalized-drift 0.002 \
  --max-normalized-drift 0.012
```

The CoreML benchmark requires an ONNX Runtime build exposing
`CoreMLExecutionProvider`.

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

FP32 ONNX:

```text
artifacts/neural_onnx/
├── metadata.json
└── neural_surrogate.onnx
```

FP16 ONNX:

```text
artifacts/neural_fp16/
├── metadata.json
└── neural_surrogate_fp16.onnx
```

FP16 production benchmark:

```text
artifacts/neural_fp16_benchmark/
├── equivalence.csv
├── task_metrics.csv
├── latency_runs.csv
├── latency_summary.csv
├── summary.json
└── runtime_models/
    ├── fp32_batch1.onnx
    ├── fp32_batch32.onnx
    ├── fp32_batch256.onnx
    ├── fp16_batch1.onnx
    ├── fp16_batch32.onnx
    └── fp16_batch256.onnx
```

Generated benchmark artifacts are intentionally ignored by Git and are
reproducible from source.

## Validation

The neural deployment workflow is covered by:

- checkpoint reconstruction tests;
- preprocessing serialization tests;
- training tests;
- FP32 ONNX export tests;
- dynamic-batch inference tests;
- PyTorch/ORT equivalence tests;
- corrected runtime benchmark tests;
- FP16 conversion tests;
- static-batch specialization tests;
- FP16 drift-regression tests;
- CoreML integration tests when the provider is available;
- CLI registration and parser-level option tests.

Local validation includes:

```bash
ruff format --check .
ruff check .
pytest -q tests/neural
pytest -q
python -m pip check
git diff --check
```

## Limitations

- The aircraft-design data are synthetic.
- This project is not a certified aircraft-design or safety-critical system.
- Runtime measurements are hardware- and environment-specific.
- Microsecond-scale benchmarks are sensitive to runtime and operating-system
  state.
- PyTorch and ONNX serialized files are different formats.
- CoreML `MLComputeUnits=ALL` does not prove exclusive ANE execution.
- FP16 reduced serialized size but did not provide a universal latency
  improvement in the measured workload.
- INT8 deployment has not yet been validated.
- Qualcomm QNN and Snapdragon NPU execution have not yet been validated.

## Roadmap

The next reduced-precision milestone is INT8 quantization, followed by unified
precision/runtime comparison and hardware-specific Qualcomm QNN / Snapdragon
profiling.

See [`ROADMAP.md`](ROADMAP.md) for the detailed progression.
