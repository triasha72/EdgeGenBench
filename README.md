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
- mixed-precision INT8/FP32 static-QDQ deployment;
- quantization calibration and drift analysis;
- CPU and CoreML execution-provider benchmarking;
- repeated latency benchmarking;
- reproducible testing, type checking, and continuous integration.

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
- static-QDQ INT8 quantization;
- per-channel QInt8 weight quantization;
- training-only MinMax calibration;
- validation-selected FP32 output-head retention;
- mixed INT8/FP32 graph validation;
- held-out INT8 drift and predictive-quality evaluation;
- repeated FP32 versus mixed-INT8 ONNX Runtime CPU benchmarking;
- public FP32, FP16, and INT8 deployment CLI commands;
- targeted static type checking for deployment modules.

Qualcomm QNN, Snapdragon NPU profiling, unified deployment-policy tooling, and
distribution-shift studies remain future work.

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

The production INT8 calibration path also uses **only the 4,200 training
rows**. Validation and held-out test rows are never used to fit quantization
calibration ranges.

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

A ratio greater than 1 means lower ONNX Runtime latency.

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

This separates ordinary provider-level numerical variation from the larger,
but still bounded, FP16 precision effect.

### FP16 predictive quality

| Metric | FP32 reference | FP16 |
|---|---:|---:|
| Mean test NRMSE | 0.050433 | **0.050473** |
| Mean test R² | 0.996955 | **0.996954** |

FP16 preserves essentially the same held-out predictive quality while reducing
serialized model size.

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

## Mixed-precision INT8/FP32 neural ONNX deployment

The FP32 neural ONNX graph was also evaluated using static QDQ quantization.

Initial experiments showed that a smaller calibration subset could miss
activation tails and clip the estimated-takeoff-mass output. The final
configuration therefore uses all 4,200 training rows for calibration.

Validation-based experiments also showed that retaining the final output
`Gemm` in FP32 improved overall drift and predictive quality relative to
quantizing that layer.

The production configuration is:

```text
Quantization format      Static QDQ
Activation type          QInt8
Weight type              QInt8
Weight granularity       Per-channel
Calibration method       MinMax
Calibration split        Train only
Calibration rows         4,200
Excluded node            node_linear_3
Output head precision    FP32
External input precision FP32
External output precision FP32
Reference provider       CPUExecutionProvider
```

This is therefore a **mixed INT8/FP32 model**, not a fully INT8 graph.

### INT8 production artifact

| Property | FP32 | Mixed INT8/FP32 |
|---|---:|---:|
| Input width | 10 | 10 |
| Output width | 6 | 6 |
| Dynamic batch | Yes | Yes |
| External input precision | FP32 | FP32 |
| External output precision | FP32 | FP32 |
| INT8 initializers | — | 10 |
| INT32 initializers | — | 6 |
| Serialized ONNX size | 25,420 B | **16,977 B** |

The selected deployment reduces serialized ONNX size by **33.21%** relative to
the FP32 graph.

It is also approximately 11.7% smaller than the 19,221-byte FP16 ONNX
artifact.

### Production graph structure

The validated production graph contains:

```text
DequantizeLinear: 10
Gemm:              4
QuantizeLinear:    4

FLOAT initializers: 12
INT8 initializers:  10
INT32 initializers: 6
```

The output head remains:

```text
node_linear_3       Gemm
network.6.weight    FLOAT
network.6.bias      FLOAT
```

### Production/probe reproducibility

The production exporter reproduced the validation-selected experimental graph
exactly across all 900 held-out test rows.

| Metric | Result |
|---|---:|
| Test rows | 900 |
| Mean production/probe absolute difference | **0.0** |
| Maximum production/probe absolute difference | **0.0** |
| `allclose` | **True** |

### Held-out quantization drift

All 900 test rows were evaluated using the same frozen preprocessing and
ONNX Runtime CPU execution.

| Metric | Result | Regression ceiling |
|---|---:|---:|
| Mean normalized drift | **0.008028** | 0.015 |
| P95 normalized drift | **0.020127** | — |
| P99 normalized drift | **0.027546** | 0.040 |
| P99.9 normalized drift | **0.041373** | 0.060 |
| Maximum normalized drift | **0.058695** | 0.080 |
| Mean guard | **PASS** | — |
| P99 guard | **PASS** | — |
| P99.9 guard | **PASS** | — |
| Maximum guard | **PASS** | — |

### Target-level mixed-precision INT8 drift

| Target | Mean normalized drift | P99 normalized drift | Maximum normalized drift |
|---|---:|---:|---:|
| Estimated takeoff mass | 0.010334 | 0.036256 | 0.058695 |
| Mission energy | 0.006909 | 0.021749 | 0.030003 |
| Energy per passenger-km | 0.008195 | 0.027552 | 0.039312 |
| Lifecycle-emissions proxy | 0.007653 | 0.026186 | 0.033258 |
| Operating-cost proxy | 0.007199 | 0.023455 | 0.032798 |
| Noise proxy | 0.007877 | 0.025629 | 0.030175 |

### Predictive quality

| Metric | FP32 ORT reference | Mixed INT8/FP32 |
|---|---:|---:|
| Mean test NRMSE | 0.050433 | **0.051566** |
| Mean test R² | 0.996955 | **0.996855** |

The mixed-precision model therefore retains strong held-out predictive quality
while reducing serialized model size.

Target-level metrics:

| Target | FP32 NRMSE | INT8 NRMSE | FP32 R² | INT8 R² |
|---|---:|---:|---:|---:|
| Estimated takeoff mass | 0.073473 | 0.074670 | 0.994602 | 0.994424 |
| Mission energy | 0.048346 | 0.048787 | 0.997663 | 0.997620 |
| Energy per passenger-km | 0.083380 | 0.084665 | 0.993048 | 0.992832 |
| Lifecycle-emissions proxy | 0.035690 | 0.036422 | 0.998726 | 0.998673 |
| Operating-cost proxy | 0.045074 | 0.044985 | 0.997968 | 0.997976 |
| Noise proxy | 0.016633 | 0.019869 | 0.999723 | 0.999605 |

### FP32 versus mixed-INT8 ONNX Runtime CPU benchmark

The production latency benchmark uses:

```text
Provider CPUExecutionProvider
Runs     5
Repeats  500
Warmups  50
Batches  1, 32, 256
```

The same preprocessed FP32 inputs are used for both models and preprocessing is
outside the timed region.

Median results:

| Batch | FP32 median | Mixed INT8 median | Median FP32/INT8 ratio | INT8 faster runs |
|---:|---:|---:|---:|---:|
| 1 | **0.004613 ms** | 0.005370 ms | 0.859× | 0/5 |
| 32 | **0.008414 ms** | 0.008715 ms | 0.973× | 0/5 |
| 256 | 0.037539 ms | **0.030682 ms** | **1.232×** | **5/5** |

A ratio above 1 indicates lower INT8 latency.

Interpretation:

- batch 1 is slower under mixed INT8/FP32 inference;
- batch 32 is close to parity but remains slightly slower under INT8;
- batch 256 consistently benefits from INT8 in this benchmark;
- batch-256 median latency is approximately **18% lower** under the
  mixed-precision model;
- INT8 was faster in all five batch-256 runs;
- the largest individual speedup should not be treated as representative
  because one FP32 run contained a larger latency excursion;
- quantization is therefore not presented as a universal latency optimization.

Detailed INT8 results are recorded in
[`docs/neural_int8_results.md`](docs/neural_int8_results.md).

## Neural deployment comparison

The three validated neural ONNX deployment configurations expose different
size, precision, provider, and latency tradeoffs.

| Deployment | Execution path | Mean NRMSE | Mean R² | ONNX size | Observed latency behavior |
|---|---|---:|---:|---:|---|
| FP32 ONNX | ORT CPU | 0.050433 | 0.996955 | 25,420 B | Reference |
| FP16 ONNX | ORT CoreML path | 0.050473 | 0.996954 | 19,221 B | B1/B32 near parity; B256 slower |
| Mixed INT8/FP32 QDQ | ORT CPU | 0.051566 | 0.996855 | **16,977 B** | B1 slower; B32 near parity; B256 ~18% lower median latency |

These are not direct cross-provider speed rankings. FP16 was evaluated through
the tested CoreML execution path, while the INT8 comparison is an ONNX Runtime
CPU benchmark.

The appropriate deployment choice depends on workload and hardware rather than
a single universal precision winner.

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
        +------------------------------------------+
        |                                          |
        v                                          v
Classical branch                              Neural branch
        |                                          |
Ridge / RF / HGB                         compact PyTorch MLP
        |                                          |
uncertainty / feasibility                       FP32 ONNX
        |                                          |
optimization                         +-------------+-------------+
        |                            |                           |
physics validation                   v                           v
                                  ORT CPU                 reduced precision
                                                               |
                                              +----------------+----------------+
                                              |                                 |
                                              v                                 v
                                           FP16 ONNX                    mixed INT8/FP32
                                              |                                 |
                                        CoreML study                       ORT CPU
                                              |                                 |
                                      precision + latency             drift + latency
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

The `dev` extra provides pytest, Ruff, and mypy.

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

### Export mixed-precision INT8/FP32 ONNX

```bash
edgegenbench export-neural-int8 \
  --fp32-model artifacts/neural_onnx/neural_surrogate.onnx \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --preprocessing artifacts/neural_surrogate/preprocessing.npz \
  --output-dir artifacts/neural_int8
```

The exporter uses all 4,200 training rows for MinMax calibration and retains
the validation-selected final `Gemm` output head in FP32.

### Benchmark FP32 versus mixed INT8/FP32 on CPU

```bash
edgegenbench benchmark-neural-int8 \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --preprocessing artifacts/neural_surrogate/preprocessing.npz \
  --fp32-model artifacts/neural_onnx/neural_surrogate.onnx \
  --int8-model artifacts/neural_int8/neural_surrogate_int8.onnx \
  --output-dir artifacts/neural_int8_benchmark \
  --runs 5 \
  --repeats 500 \
  --warmups 50 \
  --max-mean-normalized-drift 0.015 \
  --max-p99-normalized-drift 0.040 \
  --max-p999-normalized-drift 0.060 \
  --max-normalized-drift 0.080
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

Mixed INT8/FP32 ONNX:

```text
artifacts/neural_int8/
├── metadata.json
└── neural_surrogate_int8.onnx
```

INT8 production benchmark:

```text
artifacts/neural_int8_benchmark/
├── equivalence.csv
├── task_metrics.csv
├── latency_runs.csv
├── latency_summary.csv
└── summary.json
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
- mixed-precision INT8 export tests;
- training-only INT8 calibration checks;
- QDQ graph-structure tests;
- FP32 output-head retention tests;
- dynamic-batch INT8 runtime tests;
- INT8 drift-regression tests;
- repeated FP32/INT8 CPU benchmark tests;
- CLI registration and parser-level option tests;
- targeted mypy checks for the deployment implementation.

Local validation includes:

```bash
ruff format --check .
ruff check .

mypy src/edgegenbench/deployment/neural_int8.py
mypy src/edgegenbench/deployment/neural_int8_benchmark.py
mypy src/edgegenbench/cli.py

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
- Mixed INT8/FP32 reduced serialized size and improved batch-256 CPU latency,
  but was slower at batch 1 and slightly slower at batch 32.
- The INT8 benchmark validates ONNX Runtime CPU behavior only; it does not
  establish Qualcomm QNN or NPU performance.
- Qualcomm QNN and Snapdragon NPU execution have not yet been validated.
- Distribution-shift and extrapolation robustness remain future work.

## Roadmap

The next deployment milestone is a unified precision/runtime decision layer,
followed by Qualcomm AI Hub / QNN integration and supported-device Snapdragon
NPU profiling.

See [`ROADMAP.md`](ROADMAP.md) for the detailed progression.
