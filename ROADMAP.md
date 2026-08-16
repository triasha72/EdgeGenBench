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
Mixed INT8/FP32 quantization study               COMPLETE / UNRELEASED
        |
        v
Unified reduced-precision model selection        NEXT
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

## Milestone 5 — mixed INT8/FP32 quantization

**Status: COMPLETE / UNRELEASED**

### Goal

Evaluate the accuracy-size-latency tradeoff of quantized neural ONNX deployment
without using held-out validation or test data to fit calibration statistics.

### Quantization investigation

The validated FP32 neural graph contains:

```text
Gemm: 4
Relu: 3
```

ONNX Runtime operator support showed that dynamic IntegerOps quantization does
not directly support the existing `Gemm` structure in the required way, while
static QDQ supports the graph.

The INT8 investigation therefore selected static QDQ rather than manually
rewriting `Gemm` operators into `MatMul`.

### Calibration investigation

An initial deterministic 512-row training calibration subset produced a large
estimated-takeoff-mass drift outlier.

The root cause was traced to activation-range saturation:

- the 512-row calibration subset did not observe the full output range;
- a held-out test prediction exceeded the calibrated representable range;
- the quantized output clipped at the maximum representable value.

Calibration was therefore expanded to **all 4,200 training rows**.

No validation or test rows are used for calibration.

### Candidate selection

Three final candidates were compared:

1. per-channel QInt8 using a 512-row training calibration subset;
2. per-channel QInt8 using all 4,200 training rows;
3. per-channel QInt8 using all 4,200 training rows while retaining the final
   `node_linear_3` output `Gemm` in FP32.

Selection was based on validation results.

The retained-FP32-output-head configuration achieved the strongest overall
validation tradeoff across drift, NRMSE, R², and serialized size.

### Frozen production configuration

```text
Quantization format      Static QDQ
Activations              QInt8
Weights                  QInt8
Weight quantization      Per-channel
Calibration              MinMax
Calibration population   All 4,200 training rows
Calibration split        Train only
Excluded node            node_linear_3
Final output head        FP32
External input            FP32
External output           FP32
Reference execution      ONNX Runtime CPU
Selection basis          Validation
```

The correct deployment label is:

> Mixed-precision INT8/FP32 static QDQ with an FP32 output head.

The model must not be described as fully INT8.

### Production export

| Property | Value |
|---|---:|
| Test input width | 10 |
| Output width | 6 |
| Dynamic batch | Yes |
| INT8 initializers | 10 |
| INT32 initializers | 6 |
| FP32 ONNX size | 25,420 bytes |
| Mixed INT8/FP32 size | 16,977 bytes |
| Size reduction versus FP32 | 33.21% |

The final output parameters remain FP32:

```text
node_linear_3       Gemm
network.6.weight    FLOAT
network.6.bias      FLOAT
```

### Production/probe parity

The production exporter was compared directly with the validation-selected
experimental artifact over all 900 held-out test rows.

| Metric | Value |
|---|---:|
| Mean absolute difference | 0.0 |
| Maximum absolute difference | 0.0 |
| `allclose` | True |

This verifies that the production implementation reproduces the selected
experimental configuration exactly.

### Held-out drift

| Metric | Result | Guard |
|---|---:|---:|
| Mean normalized drift | 0.008028 | ≤ 0.015 |
| P95 normalized drift | 0.020127 | — |
| P99 normalized drift | 0.027546 | ≤ 0.040 |
| P99.9 normalized drift | 0.041373 | ≤ 0.060 |
| Maximum normalized drift | 0.058695 | ≤ 0.080 |
| Mean guard | PASS | — |
| P99 guard | PASS | — |
| P99.9 guard | PASS | — |
| Maximum guard | PASS | — |

### Held-out predictive quality

| Metric | FP32 reference | Mixed INT8/FP32 |
|---|---:|---:|
| Mean NRMSE | 0.050433 | 0.051566 |
| Mean R² | 0.996955 | 0.996855 |

Predictive quality remains strong after quantization.

### Repeated ONNX Runtime CPU latency

Benchmark configuration:

```text
Provider CPUExecutionProvider
Runs     5
Repeats  500
Warmups  50
```

| Batch | FP32 median | INT8 median | Median FP32/INT8 ratio | INT8 faster runs |
|---:|---:|---:|---:|---:|
| 1 | 0.004613 ms | 0.005370 ms | 0.859× | 0/5 |
| 32 | 0.008414 ms | 0.008715 ms | 0.973× | 0/5 |
| 256 | 0.037539 ms | 0.030682 ms | 1.232× | 5/5 |

### Conclusion

Mixed INT8/FP32 deployment reduced the serialized neural ONNX graph by
approximately 33.2% while preserving strong held-out predictive quality.

Latency was batch dependent:

- batch 1 was slower under INT8;
- batch 32 was close to parity but slightly slower;
- batch 256 was faster under INT8 in all five runs;
- median batch-256 latency was approximately 18% lower under INT8.

No universal INT8 latency improvement is claimed.

---

## Milestone 6 — unified reduced-precision model selection

**Status: NEXT**

### Goal

Create a unified deployment comparison that makes precision/runtime selection
dependent on explicit deployment constraints rather than a single universal
"best" model.

Current validated evidence:

| Runtime / precision | Mean NRMSE | Mean R² | Artifact size | Batch-1 behavior | Batch-32 behavior | Batch-256 behavior |
|---|---:|---:|---:|---|---|---|
| ORT FP32 CPU | 0.050433 | 0.996955 | 25,420 B | reference | reference | reference |
| ORT/CoreML FP16 | 0.050473 | 0.996954 | 19,221 B | near parity | near parity | slower |
| ORT mixed INT8/FP32 CPU | 0.051566 | 0.996855 | 16,977 B | slower | near parity/slightly slower | ~18% lower median latency |

Cross-provider timings should not be interpreted as a direct hardware ranking.

### Planned work

1. Define a machine-readable deployment-candidate schema.
2. Normalize size, quality-drift, provider, and latency metadata.
3. Define explicit accuracy, size, provider, and latency constraints.
4. Implement candidate filtering.
5. Implement deterministic deployment recommendation logic.
6. Preserve provenance for every recommendation.
7. Add CLI support.
8. Add policy tests.
9. Document examples for latency-sensitive, memory-sensitive, and
   accuracy-sensitive deployment scenarios.

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

No Qualcomm accelerator claim should be made before supported tooling and
hardware are used.

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
  +-- mixed INT8/FP32 deployment         complete / unreleased
  |
  +-- unified deployment policy          next
  |
  v
future deployment-runtime release boundary
```

Do not bump the public version until the intended release boundary is
explicitly selected.
