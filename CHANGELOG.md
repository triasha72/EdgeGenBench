# Changelog

All notable changes to EdgeGenBench are documented in this file.

The project uses semantic versioning for public releases.

## Unreleased

### Added

- Neural checkpoint reconstruction from stored architecture metadata
- PyTorch-to-ONNX FP32 neural-surrogate export
- Dynamic-batch neural ONNX graph support
- ONNX graph validation
- Frozen-preprocessor compatibility validation
- ONNX Runtime CPU neural-surrogate inference
- Held-out PyTorch-to-ONNX numerical-equivalence evaluation
- Physical-unit conversion-equivalence reporting
- Paired PyTorch CPU versus ONNX Runtime CPU benchmarking
- Corrected PyTorch inference timing using an outer `torch.inference_mode()`
  context
- Three-run neural-runtime repeatability evaluation
- Batch-1, batch-32, and batch-256 FP32 deployment benchmarks
- Reproducible FP32-to-FP16 ONNX conversion
- FP32 external I/O with FP16 internal initializers
- `onnxconverter-common` edge dependency
- Dynamic-batch FP16 ONNX artifacts
- Static batch specialization for CoreML
- FP32 CPU versus FP32 CoreML provider-drift analysis
- FP32 CoreML versus FP16 CoreML precision-drift analysis
- Per-target physical-unit FP16 drift reporting
- FP16 held-out task metrics
- FP16 numerical-drift regression guardrails
- Five-run paired FP32-versus-FP16 CoreML benchmarking
- Batch-1, batch-32, and batch-256 FP16 deployment benchmarks
- Static-QDQ mixed INT8/FP32 neural ONNX export
- QInt8 activation quantization
- Per-channel QInt8 weight quantization
- Training-only MinMax INT8 calibration
- Full 4,200-row training calibration
- Validation-based mixed-precision INT8 candidate selection
- FP32 output-head retention for `node_linear_3`
- INT8/FP32 ONNX graph-structure validation
- INT8 and INT32 initializer validation
- FP32 external I/O validation for the mixed-precision model
- Dynamic-batch validation for the mixed-precision model
- Production-versus-probe INT8 reproducibility validation
- Held-out INT8 drift reporting
- P95, P99, P99.9, and maximum quantization-drift reporting
- INT8 quantization-drift regression guardrails
- Per-target physical-unit INT8 drift reporting
- Mixed INT8/FP32 held-out task metrics
- Five-run paired FP32-versus-INT8 ONNX Runtime CPU benchmarking
- Batch-1, batch-32, and batch-256 INT8 CPU deployment benchmarks
- Public `export-neural-onnx` CLI command
- Public `benchmark-neural-onnx` CLI command
- Public `export-neural-fp16` CLI command
- Public `benchmark-neural-fp16` CLI command
- Public `export-neural-int8` CLI command
- Public `benchmark-neural-int8` CLI command
- Parser-level tests for long FP16 and INT8 CLI options
- Neural ONNX export, inference, benchmark, FP16, INT8, and CLI tests
- Neural FP32 ONNX deployment-results documentation
- Neural FP16 deployment-results documentation
- Neural mixed-precision INT8 deployment-results documentation
- `mypy` development dependency
- Targeted mypy configuration for third-party runtime libraries
- Targeted static type checking for the INT8 exporter, INT8 benchmark, and CLI
- Updated deployment roadmap
- Updated FP32 / FP16 / mixed-INT8 deployment comparison

### Corrected

- Removed per-inference `torch.no_grad()` context-manager overhead from the
  timed PyTorch CPU microbenchmark.
- PyTorch timing now executes under one outer `torch.inference_mode()` context.
- Repeated PyTorch CPU versus ONNX Runtime CPU performance claims were
  regenerated using the corrected methodology.
- Replaced the initial 512-row INT8 calibration subset after identifying
  activation-range saturation in held-out estimated-takeoff-mass predictions.
- INT8 calibration now uses all 4,200 training rows and does not use validation
  or test rows.
- Rejected fully quantizing the final output `Gemm` after validation showed a
  stronger overall drift/quality tradeoff when the output head remained FP32.
- INT8 performance language now reports batch-dependent behavior rather than a
  universal quantization speedup.

### Validated — FP32 neural ONNX

- Test rows: 900
- Mean normalized PyTorch-to-ONNX difference: 1.3064681070e-07
- Maximum normalized PyTorch-to-ONNX difference: 9.5367431641e-07
- Dynamic input shape: `[batch, 10]`
- Dynamic output shape: `[batch, 6]`
- FP32 ONNX graph size: 25,420 bytes
- Corrected three-run PyTorch/ORT median mean-latency ratios:
  - batch 1: 2.979×
  - batch 32: 2.385×
  - batch 256: 0.919×
- Batch-256 PyTorch/ORT ratio range crossed parity:
  - 0.860× to 1.079×
- The corrected benchmark supports a clear ORT advantage at batches 1 and 32,
  but not a universal advantage at batch 256.

### Validated — FP16 neural ONNX

- Test rows: 900
- FP16 ONNX graph size: 19,221 bytes
- FP32 ONNX graph size: 25,420 bytes
- Serialized-size reduction: 24.39%
- FP16 initializers: 8
- Mean FP32-CoreML versus FP16-CoreML normalized difference: 9.7869e-04
- Maximum FP32-CoreML versus FP16-CoreML normalized difference: 9.1944e-03
- Mean normalized-drift ceiling: 0.002 — PASS
- Maximum normalized-drift ceiling: 0.012 — PASS
- FP16 mean test NRMSE: 0.050473
- FP16 mean test R²: 0.996954
- FP32 CPU versus FP32 CoreML mean normalized provider drift: 1.4848e-07
- FP32 CPU versus FP32 CoreML maximum normalized provider drift: 1.4305e-06
- CoreML batch-1 median:
  - FP32: 0.038480 ms
  - FP16: 0.038685 ms
  - paired ratio: 0.995×
- CoreML batch-32 median:
  - FP32: 0.040879 ms
  - FP16: 0.041161 ms
  - paired ratio: 1.009×
- CoreML batch-256 median:
  - FP32: 0.051657 ms
  - FP16: 0.059952 ms
  - paired ratio: 0.862×
- FP16 was faster in 0 of 5 batch-256 runs.
- FP16 preserved predictive quality while reducing serialized size, but did not
  provide a universal latency improvement on the tested CoreML configuration.

### Validated — mixed INT8/FP32 neural ONNX

- Quantization format: static QDQ
- Activation type: QInt8
- Weight type: QInt8
- Weight granularity: per-channel
- Calibration method: MinMax
- Calibration split: training only
- Calibration rows: 4,200
- Validation-selected excluded node: `node_linear_3`
- Final output-head precision: FP32
- External input precision: FP32
- External output precision: FP32
- Dynamic batch preserved
- INT8 initializers: 10
- INT32 initializers: 6
- FP32 ONNX graph size: 25,420 bytes
- Mixed INT8/FP32 ONNX graph size: 16,977 bytes
- Serialized-size reduction versus FP32: 33.21%
- Production/probe mean absolute difference: 0.0
- Production/probe maximum absolute difference: 0.0
- Production/probe `allclose`: True
- Held-out test rows: 900
- Mean normalized quantization drift: 0.008028
- P95 normalized quantization drift: 0.020127
- P99 normalized quantization drift: 0.027546
- P99.9 normalized quantization drift: 0.041373
- Maximum normalized quantization drift: 0.058695
- Mean normalized-drift ceiling: 0.015 — PASS
- P99 normalized-drift ceiling: 0.040 — PASS
- P99.9 normalized-drift ceiling: 0.060 — PASS
- Maximum normalized-drift ceiling: 0.080 — PASS
- Mixed INT8/FP32 mean test NRMSE: 0.051566
- Mixed INT8/FP32 mean test R²: 0.996855
- ONNX Runtime CPU batch-1 median:
  - FP32: 0.004613 ms
  - mixed INT8/FP32: 0.005370 ms
  - paired FP32/INT8 ratio: 0.859×
  - INT8 faster runs: 0/5
- ONNX Runtime CPU batch-32 median:
  - FP32: 0.008414 ms
  - mixed INT8/FP32: 0.008715 ms
  - paired FP32/INT8 ratio: 0.973×
  - INT8 faster runs: 0/5
- ONNX Runtime CPU batch-256 median:
  - FP32: 0.037539 ms
  - mixed INT8/FP32: 0.030682 ms
  - paired FP32/INT8 ratio: 1.232×
  - INT8 faster runs: 5/5
- Batch-256 mixed INT8/FP32 median latency was approximately 18% lower than
  FP32.
- Mixed INT8/FP32 therefore reduced serialized size and preserved strong
  predictive quality while producing a workload-dependent latency tradeoff.
- No universal INT8 speedup is claimed.
- Complete neural and repository test suites passed locally.
- Ruff formatting and lint checks passed locally.
- Targeted mypy checks passed locally.
- `pip check` and `git diff --check` passed locally.

### Planned

- Unified FP32 / FP16 / mixed-INT8 deployment decision tooling
- Machine-readable deployment-candidate metadata
- Constraint-based precision/runtime selection
- Qualcomm AI Hub integration
- Qualcomm QNN compilation
- Snapdragon NPU profiling
- Distribution-shift and extrapolation evaluation
- Additional missions and aircraft-design spaces
- Hardware-aware model-selection policies

## 0.2.0 — 2026-08-15

### Added

- Compact multi-output PyTorch neural surrogate
- 10 → 64 → 32 → 16 → 6 feed-forward architecture
- 3,414-parameter compact neural model
- Training-only numerical-feature normalization
- Training-only target normalization
- Deterministic propulsion-architecture encoding
- Frozen neural preprocessing state
- Preprocessor save/load support
- Validation-based neural early stopping
- Best-checkpoint restoration
- AdamW neural optimization
- CPU-compatible deterministic test configuration
- Apple MPS execution support
- CPU execution support
- Batch-1, batch-32, and batch-256 PyTorch latency benchmarking
- Mean and P95 PyTorch latency reporting
- Per-target held-out neural regression metrics
- Neural training-history artifacts
- Neural prediction artifacts
- Neural experiment summary artifacts
- Public `train-neural-surrogate` CLI command
- Neural CLI registration test
- Neural unit and integration test suite
- PyTorch dependency group through the `neural` package extra
- Neural dependencies in GitHub Actions CI
- v0.2 neural-results documentation

### Validated neural results

- Total rows: 6,000
- Training rows: 4,200
- Validation rows: 900
- Test rows: 900
- Encoded inputs: 10
- Hidden dimensions: 64, 32, 16
- Outputs: 6
- Trainable parameters: 3,414
- Serialized model size: 16,881 bytes
- Best epoch: 141
- CPU mean test NRMSE: 0.050425
- CPU mean test R²: 0.996956
- MPS mean test NRMSE: 0.050433
- MPS mean test R²: 0.996955

### Notes

- CPU is the primary local PyTorch reference runtime.
- Apple MPS training and inference were functionally validated.
- MPS latency for the tiny model showed run-to-run variability.
- Neural ONNX deployment was intentionally left for subsequent development.

## 0.1.0 — 2026-08-05

### Added

- Versioned synthetic regional-aircraft benchmark configuration
- Reproducible physics-based dataset generation
- Deterministic training, validation, calibration, and test partitions
- Six-target aircraft-design surrogate benchmark
- FP32 multi-output Ridge regression baseline
- Random Forest multi-output regression baseline
- HistGradientBoosting multi-output regression baseline
- Validation-based surrogate hyperparameter selection
- Held-out test evaluation
- MAE, RMSE, normalized RMSE, and R² reporting
- Unified model-comparison reports and plots
- Classical-model latency benchmarking
- Serialized model-size comparison
- Random Forest tree-quantile uncertainty intervals
- Split-conformal prediction intervals
- Empirical uncertainty-coverage evaluation
- Random Forest feasibility classifier
- False-safe and false-reject evaluation
- Validation-selected classifier decision threshold
- Latin-hypercube optimization candidate generation
- Constrained three-objective optimization
- Pareto-front extraction
- Representative-design selection
- Physics-based optimization validation
- Deterministic edge feature encoder
- ONNX surrogate export
- ONNX feasibility-classifier export
- ONNX metadata schema
- ONNX Runtime inference wrappers
- Scikit-learn-to-ONNX numerical-equivalence validation
- Classifier-decision agreement evaluation
- Batch-one, batch-32, and batch-256 latency benchmarks

### Validated results

- HistGradientBoosting mean test NRMSE: 0.062249
- HistGradientBoosting mean test R²: 0.995171
- Feasibility classifier balanced accuracy: 98.24%
- Feasibility classifier false-safe rate: 2.12%
- Generated optimization candidates: 20,000
- Accepted optimization candidates: 5,413
- Pareto designs: 47
- Representative designs: 4
- Pareto-front feasibility agreement: 100%
- Representative-design feasibility agreement: 100%
- Classical ONNX classifier decision agreement: 100%
