# EdgeGenBench Roadmap

[Project overview and measured results](README.md)

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
FP32 neural ONNX deployment                      COMPLETE ON MAIN
        |
        v
FP16 deployment study                            COMPLETE ON MAIN
        |
        v
Mixed INT8/FP32 quantization study               COMPLETE ON MAIN
        |
        v
Deployment-aware neural model selection         COMPLETE ON MAIN
        |
        v
Qualcomm AI Hub / QNN                            COMPLETE ON MAIN
        |
        v
Snapdragon NPU profiling                         COMPLETE ON MAIN
        |
        v
Qualcomm-native INT8/QDQ study                   COMPLETE ON MAIN
        |
        v
Deployment policy layer                          COMPLETE ON MAIN
        |
        v
Native iOS inference path                        COMPLETE ON MAIN
        |
        v
Installable iPhone browser deployment            COMPLETE ON MAIN
        |
        v
Robustness / distribution-shift evaluation       NEXT
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

## Milestone 6 — deployment-aware neural model selection

**Status: COMPLETE / UNRELEASED**

### Goal

Select among measured FP32, FP16, and mixed INT8/FP32 deployment candidates
using explicit deployment requirements instead of assuming one precision or
runtime is universally optimal.

### Implemented

- machine-readable provider-aware deployment candidates;
- measured FP32 CPU benchmark ingestion;
- measured mixed INT8/FP32 CPU benchmark ingestion;
- measured FP32 CoreML benchmark ingestion;
- measured FP16 CoreML benchmark ingestion;
- batch-size constraints;
- execution-provider constraints;
- maximum-latency constraints;
- maximum serialized-model-size constraints;
- minimum R² constraints;
- maximum NRMSE constraints;
- maximum normalized-drift constraints;
- explicit rejection reasons for infeasible candidates;
- deterministic `lowest_latency` policy;
- deterministic `smallest_model` policy;
- deterministic `highest_accuracy` policy;
- transparent weighted `balanced` policy;
- deterministic tie breaking;
- JSON deployment-decision reports;
- Markdown deployment-decision reports;
- public `select-neural-deployment` CLI;
- deployment-selection unit tests;
- CLI regression tests.

### Validated deployment decisions

| Scenario | Selected candidate | Provider | Median latency | Model size | Mean R² |
|---|---|---|---:|---:|---:|
| Batch 1, lowest latency | `fp32_cpu` | CPUExecutionProvider | 0.004613 ms | 25,420 B | 0.996955 |
| Batch 32, balanced | `fp32_cpu` | CPUExecutionProvider | 0.008414 ms | 25,420 B | 0.996955 |
| Batch 256, lowest latency | `mixed_int8_fp32_cpu` | CPUExecutionProvider | 0.030682 ms | 16,977 B | 0.996855 |
| Batch 256, CPU only | `mixed_int8_fp32_cpu` | CPUExecutionProvider | 0.030682 ms | 16,977 B | 0.996855 |
| Batch 32, CoreML only, balanced | `fp32_coreml` | CoreMLExecutionProvider | 0.040879 ms | 25,420 B | 0.996955 |

### Conclusion

No tested precision/runtime configuration is universally optimal.

FP32 remains preferable for the measured small-batch CPU workloads, while
mixed INT8/FP32 becomes the lowest-latency measured CPU deployment at batch
256 and also provides the smallest validated ONNX artifact.

FP16 reduces serialized model size but did not produce a robust universal
latency improvement on the tested CoreML execution path.

Cross-provider timings represent distinct measured deployment configurations
and should not be interpreted as a direct hardware or precision-only ranking.

---

## Milestone 7 — Qualcomm AI Hub / QNN

**Status: COMPLETE / UNRELEASED**

### Goal

Establish a reproducible path from the validated neural ONNX artifact to
Qualcomm QNN deployment on supported Snapdragon hardware.

### Implemented

- Qualcomm AI Hub dependency isolation through the `qualcomm` package extra;
- authenticated hosted-device discovery;
- Snapdragon 8 Elite QRD target selection;
- source ONNX validation and SHA-256 provenance;
- static batch-1, batch-32, and batch-256 QNN compilation;
- HTP backend validation;
- Hexagon v79 metadata validation;
- QAIRT version provenance;
- QNN Context Binary provenance;
- hosted Snapdragon profile jobs;
- compute-unit inspection;
- all-nine-layer NPU placement;
- numerical runtime-drift evaluation;
- 900-row held-out inference validation;
- modern compile-and-link deployment workflow;
- one linked QNN Context Binary containing batch-1, batch-32, and batch-256
  graphs;
- graph-selective profile and inference validation;
- offline profile/parity helpers and unit tests;
- canonical machine-readable Qualcomm evidence report.

### Linked deployment results

| Batch | AI Hub profile latency | Peak inference memory | Compute units |
|---:|---:|---:|---|
| 1 | 38 us | 122,937,344 B | NPU: 9 |
| 32 | 34 us | 122,888,192 B | NPU: 9 |
| 256 | 57 us | 123,211,776 B | NPU: 9 |

### Held-out validation

The linked batch-1 graph was evaluated on all 900 held-out rows.

| Metric | Local FP32 ONNX | Snapdragon QNN |
|---|---:|---:|
| Mean R2 | 0.996955004 | 0.996953249 |
| Mean NRMSE | 0.050432628 | 0.050444571 |

Maximum normalized deployment drift was 0.003636.

### Conclusion

The compact neural surrogate has been compiled, linked, profiled, and
executed through QNN HTP on supported Snapdragon hardware with all profiled
layers placed on the NPU and essentially unchanged held-out predictive
quality.

AI Hub profile measurements are hardware-contextual and are not treated as
end-to-end Android latency or same-hardware comparisons with Mac CPU/CoreML
measurements.

---

## Milestone 8 — Snapdragon NPU optimization

**Status: COMPLETE ON MAIN**

### Goal

Evaluate whether Qualcomm-native reduced-precision deployment improves the
Snapdragon NPU accuracy-size-latency tradeoff relative to the validated
FP32-I/O / HTP-FP16-relaxed baseline.

### Completed

- Qualcomm-native INT8/QDQ post-training quantization;
- training-only calibration;
- batch-1, batch-32, and batch-256 INT8 graph variants;
- linked multi-graph INT8 QNN deployment;
- NPU placement verification;
- held-out predictive-quality evaluation;
- normalized runtime-drift evaluation;
- serialized QNN artifact-size comparison;
- profile-latency comparison;
- runtime-memory comparison;
- derived model-throughput comparison;
- repeated-profile variability study where practical.

The measured INT8/QDQ candidate ran on the Snapdragon 8 Elite NPU for batch
sizes 1, 32, and 256. Its maximum normalized deployment drift (`0.036602`)
exceeded the frozen `0.01` acceptance limit, so the fail-closed decision kept
the FP32-I/O / HTP-FP16-relaxed baseline. The rejected candidate, device
profiles, hashes, and decision provenance are recorded in
`reports/qualcomm_int8_qnn_v0_1.json`.

---

## Milestone 9 — robustness and extrapolation

**Status: NEXT**

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

**Status: COMPLETE ON MAIN**

Implemented flow:

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

Supported requirements include:

- maximum allowable accuracy drift;
- batch size;
- memory budget;
- latency target;
- hardware availability;
- feasibility constraints.

The selector evaluates only measured provider-specific candidates, rejects
infeasible choices with explicit reasons, and writes JSON and Markdown
decision records. The CLI and validation examples are documented in
`docs/deployment_model_selection.md`.

---

## Milestone 11 — installable iPhone browser deployment

**Status: COMPLETE ON MAIN**

The FP32 ONNX surrogate is now packaged as a mobile-first progressive web app.
It preserves the frozen feature and target scaling contract, runs inference in
Safari with ONNX Runtime Web, caches the app and model, and deploys through
GitHub Pages. This closes browser-based iPhone delivery without presenting it
as a signed native build or as physical-device performance evidence.

Completed evidence:

- deployable static application and versioned ONNX model;
- install manifest, standalone display mode, and service-worker cache;
- local preprocessing and inverse target scaling;
- contract tests against the ONNX deployment metadata;
- GitHub Pages deployment workflow;
- iPhone installation and evidence-boundary documentation.

---

## Release strategy

Current progression:

```text
0.1.0  scientific-ML foundation
  |
0.2.0  compact PyTorch neural surrogate
  |
  +-- FP32 neural ONNX deployment        complete on main
  |
  +-- FP16 deployment study              complete on main
  |
  +-- mixed INT8/FP32 deployment         complete on main
  |
  +-- unified deployment policy          complete on main
  |
  +-- Qualcomm QNN / Snapdragon profile  complete on main
  |
  +-- Qualcomm-native INT8/QDQ            complete on main
  |
  +-- installable iPhone browser app      complete on main
  |
  v
robustness / distribution-shift study     next
```

Do not bump the public version until the intended release boundary is
explicitly selected.
