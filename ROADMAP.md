# EdgeGenBench Roadmap

This roadmap separates completed, validated work from planned deployment and
research milestones.

## Guiding principles

EdgeGenBench should remain:

- reproducible;
- scientifically interpretable;
- explicit about train/validation/test boundaries;
- conservative about safety and feasibility claims;
- transparent about runtime- and hardware-specific measurements;
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
FP16 deployment study                            COMPLETE / UNRELEASED
        |
        v
INT8 quantization study                          NEXT
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

Validated capabilities include deterministic synthetic data generation,
classical surrogate modeling, uncertainty quantification, feasibility
classification, constrained optimization, physics validation, and classical
ONNX deployment.

Reference outcomes include:

- HistGradientBoosting mean test NRMSE: 0.062249;
- HistGradientBoosting mean test R²: 0.995171;
- feasibility-classifier balanced accuracy: 98.24%;
- false-safe rate: 2.12%;
- 20,000 optimization candidates;
- 47 Pareto designs;
- 100% feasibility agreement on the validated Pareto front.

---

## Milestone 2 — v0.2 compact neural surrogate

**Status: COMPLETE**

Validated capabilities:

- train-only feature normalization;
- train-only target normalization;
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
| CPU mean test NRMSE | 0.050425 |
| CPU mean test R² | 0.996956 |
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
- ONNX Runtime CPU inference;
- frozen preprocessing reuse;
- normalized prediction equivalence;
- physical-unit prediction equivalence;
- paired PyTorch CPU ↔ ORT CPU benchmarking;
- corrected inference timing methodology;
- repeated runtime benchmarking;
- `export-neural-onnx` CLI;
- `benchmark-neural-onnx` CLI;
- export, inference, benchmark, and CLI tests.

Validated parity on 900 held-out rows:

| Metric | Value |
|---|---:|
| Mean normalized absolute difference | 1.306e-07 |
| Maximum normalized absolute difference | 9.537e-07 |
| Equivalence | PASS |

Corrected three-run CPU benchmark:

| Batch | Median PyTorch | Median ORT | Median PyTorch/ORT ratio | Ratio range |
|---:|---:|---:|---:|---:|
| 1 | 0.018341 ms | 0.006157 ms | 2.979× | 2.715×–3.967× |
| 32 | 0.018427 ms | 0.007984 ms | 2.385× | 2.078×–2.404× |
| 256 | 0.030841 ms | 0.033574 ms | 0.919× | 0.860×–1.079× |

Interpretation:

- ORT shows a clear local advantage for batches 1 and 32;
- batch 256 is approximately parity and changes direction across runs;
- no universal ONNX Runtime speedup is claimed.

---

## Milestone 4 — FP16 deployment study

**Status: COMPLETE / UNRELEASED**

### Goal

Determine whether FP16 representation reduces deployment size while retaining
acceptable predictive quality, and measure whether it changes runtime behavior
on the tested CoreML execution path.

### Implemented

- reproducible FP32-to-FP16 ONNX conversion;
- external FP32 I/O retained;
- eligible internal initializers converted to FP16;
- ONNX graph validation;
- dynamic-batch FP16 graph;
- static batch specialization;
- batch-1, batch-32, and batch-256 CoreML variants;
- FP32 CPU versus FP32 CoreML provider-drift measurement;
- FP32 CoreML versus FP16 CoreML precision-drift measurement;
- per-target physical-unit drift;
- held-out FP16 regression metrics;
- project-specific FP16 drift guardrails;
- five-run paired CoreML latency benchmark;
- public `export-neural-fp16` CLI;
- public `benchmark-neural-fp16` CLI;
- portable and CoreML-aware tests;
- parser-level tests for long CLI options.

### Validated results

| Metric | Value |
|---|---:|
| Test rows | 900 |
| FP32 ONNX size | 25,420 bytes |
| FP16 ONNX size | 19,221 bytes |
| Size reduction | 24.39% |
| FP16 initializers | 8 |
| Mean normalized precision drift | 9.7869e-04 |
| Maximum normalized precision drift | 9.1944e-03 |
| Mean drift limit | 0.002 |
| Maximum drift limit | 0.012 |
| Mean drift guard | PASS |
| Maximum drift guard | PASS |
| FP16 mean NRMSE | 0.050473 |
| FP16 mean R² | 0.996954 |

CoreML latency:

| Batch | FP32 median | FP16 median | Median FP32/FP16 ratio | FP16 faster runs |
|---:|---:|---:|---:|---:|
| 1 | 0.038480 ms | 0.038685 ms | 0.995× | 2/5 |
| 32 | 0.040879 ms | 0.041161 ms | 1.009× | 4/5 |
| 256 | 0.051657 ms | 0.059952 ms | 0.862× | 0/5 |

### Conclusion

FP16 preserved predictive quality and reduced serialized model size by
approximately 24.4%.

No robust FP16 latency improvement was observed:

- batches 1 and 32 were effectively at parity;
- FP16 was consistently slower at batch 256.

The CoreML benchmark uses `MLComputeUnits=ALL` and therefore does not establish
exclusive Apple Neural Engine execution.

---

## Milestone 5 — INT8 quantization

**Status: NEXT**

### Goal

Evaluate the accuracy-size-latency tradeoff of INT8 deployment.

### Work items

1. Determine supported quantization behavior for the validated neural graph.
2. Select dynamic or static quantization based on runtime/operator support.
3. Define a calibration subset without using held-out test targets.
4. Freeze calibration configuration and seed.
5. Produce a versioned INT8 artifact.
6. Validate the quantized graph.
7. Run all 900 held-out test rows.
8. Separate provider drift, quantization drift, and predictive error.
9. Record per-target physical-unit drift.
10. Compare serialized FP32, FP16, and INT8 sizes.
11. Benchmark batches 1, 32, and 256.
12. Repeat latency measurements.
13. Record fallback or unsupported operators.
14. Add CLI, tests, and reproducible result artifacts.

### Scientific guardrails

- never calibrate on the test set;
- report target-level drift;
- distinguish quantization error from original model error;
- report provider and hardware;
- do not assume INT8 is faster before measuring it.

Recommended branch after the current FP16 work is merged:

```text
feat/neural-int8-evaluation
```

---

## Milestone 6 — unified reduced-precision model selection

**Status: PLANNED**

Create a unified comparison such as:

| Runtime / precision | Accuracy drift | Artifact size | Batch-1 latency | Batch-32 latency | Batch-256 latency |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 CPU | reference | measured | measured | measured | measured |
| ORT FP32 CPU | measured | 25,420 B | measured | measured | measured |
| ORT/CoreML FP32 | measured | 25,420 B | measured | measured | measured |
| ORT/CoreML FP16 | measured | 19,221 B | measured | measured | measured |
| INT8 | planned | planned | planned | planned | planned |

Model/runtime selection should depend on deployment constraints rather than a
single universal "best" model.

---

## Milestone 7 — Qualcomm AI Hub / QNN

**Status: PLANNED**

Goals:

- establish a reproducible path from validated ONNX to Qualcomm tooling;
- document model and operator compatibility;
- compile or convert using supported QNN tooling;
- record target SoC and runtime;
- separate host-side preprocessing from accelerator execution;
- compare numerical parity against the validated FP32 reference.

Deliverables:

- conversion script or documented reproducible command;
- target-device configuration;
- compatibility report;
- conversion logs;
- parity report;
- device-specific latency report.

---

## Milestone 8 — Snapdragon NPU profiling

**Status: PLANNED**

Measure on supported Snapdragon hardware:

- warm-start latency;
- repeated batch latency;
- throughput;
- memory footprint when available;
- power or energy proxy where tooling supports it;
- fallback operators;
- sustained versus burst behavior.

No NPU performance claim should be made before a real supported-device run.

---

## Milestone 9 — robustness and extrapolation

**Status: PLANNED**

Candidate studies:

- design-range extrapolation;
- passenger-capacity extrapolation;
- unseen combinations near feasibility boundaries;
- propulsion-architecture subgroup analysis;
- input perturbation;
- missing or degraded inputs;
- uncertainty under distribution shift;
- optimizer-induced out-of-distribution queries.

Deliverables should include reproducible shift definitions, stress datasets,
target-level metrics, calibration results, and documented failure modes.

---

## Milestone 10 — deployment policy layer

**Status: PLANNED**

Longer-term objective:

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

Candidate requirements include:

- maximum allowable accuracy drift;
- batch size;
- memory budget;
- latency target;
- hardware availability;
- feasibility constraints.

---

## Release strategy

Current progression:

```text
0.1.0  scientific-ML foundation
  |
0.2.0  compact PyTorch neural surrogate
  |
  +-- FP32 neural ONNX deployment        complete / unreleased
  |
  +-- FP16 deployment study              complete / unreleased
  |
  +-- INT8 study                         next
  |
  v
future deployment-runtime release boundary
```

Do not bump the public version until the intended release boundary is
explicitly selected.
