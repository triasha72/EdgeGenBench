# Changelog

All notable changes to EdgeGenBench are documented in this file.

The project uses semantic versioning for public releases.

## Unreleased

### Planned

- Compact multi-output PyTorch surrogate
- Training-only feature and target normalization
- Validation-based neural-network early stopping
- PyTorch-to-ONNX export
- FP16 conversion
- INT8 quantization
- Accuracy, model-size, latency, and throughput comparison
- Distribution-shift and extrapolation evaluation
- Additional missions and aircraft-design spaces
- Hardware-aware edge benchmarks

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
- Target-level surrogate-versus-physics metrics
- Optimization validation plots
- Separate conservative optimization feasibility threshold
- Broad optimization-threshold sensitivity study
- Refined optimization-threshold study
- Physics-validated optimization threshold of 0.50
- Deterministic edge feature encoder
- ONNX surrogate export
- ONNX feasibility-classifier export
- ONNX metadata schema
- ONNX Runtime inference wrappers
- Scikit-learn-to-ONNX numerical-equivalence validation
- Classifier-decision agreement evaluation
- Batch-one, batch-32, and batch-256 latency benchmarks
- Mean and P95 batch-latency reporting
- Unit and integration tests
- GitHub Actions continuous integration
- Executable end-to-end release pipeline
- Complete architecture documentation
- Complete reproducibility documentation
- Artifact-backed results documentation

### Changed

- Separated the validation-selected classifier threshold from the
  optimization-specific acceptance threshold
- Retained classifier threshold 0.30 for ordinary classification
- Applied threshold 0.50 during optimization
- Updated optimization configuration to store the conservative threshold
- Updated candidate search to support threshold overrides without retraining
- Updated optimization summaries to record the applied threshold
- Updated the official optimization result from 5,687 to 5,413 accepted
  candidates
- Updated the official Pareto front from 52 to 47 designs
- Improved representative-design physics-feasibility agreement from 25% to
  100%
- Improved final Pareto-front physics-feasibility agreement to 100%
- Replaced the descriptive pipeline outline with an executable workflow
- Expanded README documentation to cover the complete v0.1 system
- Added explicit architecture and artifact contracts
- Added reproducible release-validation instructions

### Fixed

- Restored the missing physics-validation implementation
- Restored physics-validation test coverage
- Corrected the FP32 model artifact path in the full pipeline
- Corrected the optimization summary artifact path in the full pipeline
- Removed tracked local backup files
- Prevented the optimizer from using the more permissive classifier threshold
  directly
- Added validation for optimization-threshold configuration values
- Added test coverage confirming that the optimization threshold overrides
  the stored classifier threshold
- Added test coverage confirming that the applied threshold is recorded in
  optimization artifacts

### Validated results

#### Surrogate models

- HistGradientBoosting mean test R²: 0.995171
- HistGradientBoosting mean test NRMSE: 0.062249
- Random Forest mean test R²: 0.953219
- FP32 Ridge mean test R²: 0.937690
- FP32 Ridge serialized size: 2,593 bytes

#### Feasibility classifier

- Balanced accuracy: 98.24%
- Precision: 95.62%
- Recall: 98.61%
- F1 score: 97.09%
- False-safe rate: 2.12%
- Classifier threshold: 0.30

#### Optimization

- Candidate designs: 20,000
- Optimization threshold: 0.50
- Accepted candidates: 5,413
- Accepted fraction: 27.065%
- Pareto designs: 47
- Representative designs: 4
- Pareto-front physics-feasibility agreement: 100%
- Representative-design physics-feasibility agreement: 100%

#### ONNX deployment

- Classifier agreement: 100%
- Maximum feasibility-probability error: approximately 1.82 × 10⁻⁷
- Maximum surrogate absolute difference: approximately 0.01059
- ONNX surrogate size: approximately 108.96 MiB
- ONNX feasibility-classifier size: approximately 2.73 MiB

### Limitations

- The benchmark dataset is synthetic
- The physics model is an analytical research proxy
- The benchmark does not use flight-test or certification data
- Results represent one mission and design-space configuration
- Optimization threshold selection must be repeated for changed missions
- Conformal coverage is evaluated only on the current held-out split
- Latency measurements are hardware- and environment-dependent
- The v0.1 deployment benchmark uses CPU execution
- FP16 and INT8 quantization are not included
- EdgeGenBench is not a certified aircraft-design or safety-critical tool