# EdgeGenBench Roadmap

This roadmap separates **completed, validated work** from **planned deployment
and research milestones**.

## Guiding principles

EdgeGenBench should remain:

- reproducible;
- scientifically interpretable;
- explicit about train/validation/test boundaries;
- conservative about safety and feasibility claims;
- transparent about runtime/hardware-specific measurements;
- modular across training, evaluation, deployment, and hardware backends;
- careful not to present planned work as completed work.

## Milestone map

```text
v0.1 Scientific-ML foundation                    COMPLETE
        |
        v
v0.2 Compact PyTorch neural surrogate            COMPLETE
        |
        v
FP32 neural ONNX deployment                      COMPLETE / UNRELEASED
        |
        v
FP16 deployment study                            NEXT
        |
        v
INT8 quantization study                          PLANNED
        |
        v
Reduced-precision model selection                PLANNED
        |
        v
Qualcomm AI Hub / QNN                            PLANNED
        |
        v
Snapdragon NPU profiling                         PLANNED
        |
        v
Robustness / distribution-shift evaluation       PLANNED
```

---

## Milestone 1 — v0.1 scientific-ML foundation

**Status: COMPLETE**

Validated capabilities:

- deterministic synthetic aircraft-design dataset generation;
- FP32 Ridge baseline;
- Random Forest baseline;
- HistGradientBoosting baseline;
- held-out regression metrics;
- model-size and latency comparison;
- uncertainty estimation;
- conformal intervals;
- feasibility classification;
- false-safe evaluation;
- constrained multi-objective optimization;
- Pareto-front extraction;
- physics-based optimization validation;
- classical ONNX export;
- Scikit-learn ↔ ONNX Runtime numerical equivalence;
- classical ONNX latency benchmarking.

Primary result:

- HistGradientBoosting achieved the strongest classical held-out accuracy;
- Random Forest remains the uncertainty/optimization path;
- classical ONNX Runtime substantially reduced low-batch runtime overhead for
  the deployed tree models.

---

## Milestone 2 — v0.2 compact neural surrogate

**Status: COMPLETE**

Validated capabilities:

- leakage-safe training-only feature normalization;
- training-only target normalization;
- deterministic categorical encoding;
- 10 → 64 → 32 → 16 → 6 PyTorch MLP;
- 3,414 trainable parameters;
- AdamW optimization;
- validation-based early stopping;
- best-checkpoint restoration;
- CPU and Apple MPS execution;
- held-out test evaluation;
- persisted preprocessing state;
- public neural-training CLI;
- CPU-compatible automated tests.

Reference accuracy:

| Metric | Value |
|---|---:|
| Mean test NRMSE | 0.050425 |
| Mean test R² | 0.996956 |
| Parameters | 3,414 |
| PyTorch checkpoint | 16,881 bytes |

---

## Milestone 3 — FP32 neural ONNX deployment

**Status: COMPLETE / UNRELEASED**

Implemented:

- neural checkpoint reconstruction;
- architecture validation;
- PyTorch-to-ONNX export;
- ONNX opset 18;
- dynamic batch dimension;
- ONNX checker validation;
- ONNX Runtime CPU inference wrapper;
- frozen-preprocessor reuse;
- normalized prediction equivalence;
- physical-unit prediction equivalence;
- paired PyTorch CPU ↔ ORT CPU benchmark;
- repeated runtime benchmark;
- `export-neural-onnx` CLI;
- `benchmark-neural-onnx` CLI;
- export/inference/benchmark/CLI tests.

Validated parity on 900 held-out rows:

| Metric | Value |
|---|---:|
| Mean normalized abs difference | 1.306e-07 |
| Max normalized abs difference | 9.537e-07 |
| Equivalence | PASS |

Repeated local CPU benchmark:

| Batch | Median PyTorch/ORT ratio |
|---:|---:|
| 1 | 3.481× |
| 32 | 2.872× |
| 256 | 1.659× |

ONNX Runtime produced lower mean latency in every one of the three repeat runs
at all three batch sizes. Absolute microsecond-scale timings remain
machine-specific.

### Release gate

Before merging/releasing this milestone:

- full Ruff format check;
- full Ruff lint check;
- full pytest suite;
- `pip check`;
- staged diff audit;
- GitHub Actions CI;
- documentation review.

---

## Milestone 4 — FP16 deployment study

**Status: NEXT**

### Goal

Determine whether reduced FP16 representation provides a meaningful size or
runtime advantage without unacceptable predictive drift.

### Work items

1. Research the correct FP16 conversion path for the selected runtime.
2. Keep the FP32 ONNX graph as the reference artifact.
3. Produce an FP16 deployment artifact.
4. Validate ONNX graph integrity.
5. Compare FP16 outputs against the frozen FP32 reference on all 900 held-out rows.
6. Record:
   - mean normalized absolute difference;
   - maximum normalized absolute difference;
   - per-target physical-unit error;
   - relative error.
7. Measure serialized artifact size.
8. Benchmark batch sizes 1, 32, and 256.
9. Repeat latency measurements rather than relying on one run.
10. Document hardware/runtime limitations explicitly.

### Acceptance criteria

- conversion is reproducible;
- graph passes validation;
- output drift is quantified;
- no unsupported "FP16 is faster" claim without measured evidence;
- any speed/size claim identifies hardware and runtime.

---

## Milestone 5 — INT8 quantization

**Status: PLANNED**

### Goal

Evaluate the accuracy-size-latency tradeoff of INT8 deployment.

### Work items

- choose dynamic versus static quantization based on operator/runtime support;
- define a calibration split that does not leak held-out test information;
- freeze calibration configuration;
- quantize model;
- validate graph/operator support;
- run 900-row held-out equivalence/error analysis;
- compare FP32, FP16, and INT8 target-level performance;
- compare artifact sizes;
- run repeated batch-1/32/256 latency benchmarks;
- record unsupported operators or fallback behavior.

### Scientific guardrails

- never calibrate on the test set;
- report target-level drift, not only one aggregate metric;
- separate quantization error from original surrogate predictive error;
- report runtime execution provider and hardware.

---

## Milestone 6 — reduced-precision model selection

**Status: PLANNED**

Create a unified comparison table:

| Runtime / precision | Accuracy drift | Artifact size | Batch-1 latency | Batch-32 latency | Batch-256 latency |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | reference | measured | measured | measured | measured |
| ORT FP32 | measured | measured | measured | measured | measured |
| ORT FP16 | planned | planned | planned | planned | planned |
| ORT INT8 | planned | planned | planned | planned | planned |

Selection should depend on deployment constraints rather than one universal
"best" model.

---

## Milestone 7 — Qualcomm AI Hub / QNN

**Status: PLANNED**

### Goals

- establish a reproducible path from validated ONNX to Qualcomm tooling;
- document model/operator compatibility;
- compile or convert using supported QNN tooling;
- record target SoC/runtime details;
- separate host-side preprocessing from accelerator graph execution;
- compare numerical parity against the FP32 reference.

### Deliverables

- QNN conversion script or documented reproducible command;
- target-device configuration;
- conversion logs;
- compatibility report;
- parity report;
- hardware-specific latency report.

---

## Milestone 8 — Snapdragon NPU profiling

**Status: PLANNED**

Measure on supported Snapdragon hardware:

- warm-start latency;
- repeated batch latency;
- throughput;
- memory footprint where available;
- power/energy proxy where tooling supports it;
- fallback operators;
- sustained versus burst behavior.

No NPU performance claim should be made before a real supported-device run.

---

## Milestone 9 — robustness and extrapolation

**Status: PLANNED**

Extend beyond in-distribution test performance.

Candidate studies:

- design-range extrapolation;
- passenger-capacity extrapolation;
- unseen combinations near feasibility boundaries;
- propulsion-architecture subgroup analysis;
- noisy input perturbation;
- missing or degraded input scenarios;
- uncertainty under distribution shift;
- optimizer-induced out-of-distribution queries.

Deliverables:

- shift definitions;
- reproducible stress datasets;
- target-level error analysis;
- calibration/equivalence reports;
- failure-mode documentation.

---

## Milestone 10 — deployment policy layer

**Status: PLANNED**

Longer-term objective:

Select a model/runtime based on requirements such as:

- maximum allowable accuracy drift;
- batch size;
- memory budget;
- latency target;
- hardware availability;
- safety/feasibility constraints.

Example:

```text
deployment requirements
        |
        v
accuracy / latency / size constraints
        |
        v
validated model-runtime candidates
        |
        v
hardware-aware selection policy
        |
        v
deployment decision + provenance
```

---

## Immediate next branch after FP32 ONNX merge

Recommended branch:

```text
feat/neural-fp16-evaluation
```

Recommended first tasks:

1. start from clean `main`;
2. reproduce FP32 ONNX reference artifacts;
3. investigate supported FP16 conversion behavior;
4. implement a separate conversion module;
5. add unit tests before benchmarking;
6. validate on the full 900-row held-out test set;
7. only then add latency and size claims.

## Release strategy

Suggested progression:

```text
0.2.0  compact PyTorch surrogate
  |
  +-- unreleased FP32 neural ONNX deployment
  |
  +-- FP16 / INT8 deployment studies
  |
  v
0.3.0  deployment-runtime milestone
```

Do not bump the public version until the intended release boundary is decided.
