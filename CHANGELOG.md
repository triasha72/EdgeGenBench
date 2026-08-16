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
- ONNX Runtime CPU neural-surrogate inference wrapper
- Held-out PyTorch-to-ONNX numerical-equivalence evaluation
- Physical-unit conversion-equivalence reporting
- Paired PyTorch CPU versus ONNX Runtime CPU latency benchmarking
- Three-run neural-runtime repeatability evaluation
- Batch-1, batch-32, and batch-256 neural deployment benchmarks
- Neural ONNX export and benchmark metadata artifacts
- Public `export-neural-onnx` CLI command
- Public `benchmark-neural-onnx` CLI command
- Neural ONNX export, inference, benchmark, and CLI tests
- ONNX Script dependency for modern PyTorch ONNX export
- Neural ONNX deployment-results documentation
- Project deployment roadmap

### Validated

- 900 held-out rows preserved across PyTorch and ONNX Runtime
- Mean normalized absolute conversion difference: 1.306e-07
- Maximum normalized absolute conversion difference: 9.537e-07
- Dynamic ONNX input shape: `[batch, 10]`
- Dynamic ONNX output shape: `[batch, 6]`
- ONNX graph size: 25,420 bytes
- PyTorch checkpoint size: 16,881 bytes
- Three repeated local CPU benchmarks with lower ONNX Runtime mean latency at
  batch sizes 1, 32, and 256
- Median PyTorch/ORT mean-latency ratios:
  - batch 1: approximately 3.481×
  - batch 32: approximately 2.872×
  - batch 256: approximately 1.659×
- Complete neural suite with 27 passing tests
- Full local formatting, lint, and repository test suite

### Planned

- FP16 neural-model conversion and equivalence evaluation
- INT8 neural-model quantization
- Accuracy, model-size, latency, and throughput comparison across precision modes
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

- CPU is the primary local PyTorch latency baseline.
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
