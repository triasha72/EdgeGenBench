# Changelog

All notable changes to EdgeGenBench are documented in this file.

The project uses semantic versioning for public releases.

## Unreleased

### Planned

- PyTorch-to-ONNX neural-surrogate export
- ONNX Runtime numerical-equivalence validation
- FP16 neural-model conversion
- INT8 neural-model quantization
- Accuracy, model-size, latency, and throughput comparison
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
- Preprocessor serialization parity tests
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

### Changed

- Updated project version to 0.2.0
- Updated project status to compact neural edge-inference benchmark
- Extended EdgeGenBench from classical surrogate benchmarking to include
  compact neural scientific-ML surrogates
- Extended the public CLI with neural-surrogate training
- Extended CI to execute neural tests on CPU
- Updated installation instructions to use `.[dev,edge,neural]`
- Updated README documentation for the v0.2 neural workflow
- Added explicit CPU-versus-MPS runtime reporting
- Retained the complete v0.1 classical scientific-ML workflow
- Retained the v0.1 Random Forest uncertainty and optimization path
- Retained the existing classical ONNX workflow while preparing a separate
  neural ONNX deployment track

### Validated neural results

#### Dataset

- Total rows: 6,000
- Training rows: 4,200
- Validation rows: 900
- Test rows: 900

#### Neural architecture

- Encoded inputs: 10
- Hidden dimensions: 64, 32, 16
- Outputs: 6
- Trainable parameters: 3,414
- Serialized model size: 16,881 bytes
- Random seed: 42

#### Training

- Best epoch: 141
- Early-stopping completion epoch: 171
- CPU best validation loss: approximately 0.003442
- MPS best validation loss: approximately 0.003429

#### Held-out accuracy

CPU:

- Mean test NRMSE: 0.050425
- Mean test R²: 0.996956

Apple MPS:

- Mean test NRMSE: 0.050433
- Mean test R²: 0.996955

Reference MPS target-level R²:

- Estimated takeoff mass: 0.994602
- Mission energy: 0.997663
- Energy per passenger-km: 0.993048
- Lifecycle-emissions proxy: 0.998726
- Operating-cost proxy: 0.997968
- Noise proxy: 0.999723

#### CPU PyTorch latency

- Batch 1 mean latency: approximately 0.0220 ms
- Batch 1 P95 latency: approximately 0.0226 ms
- Batch 32 mean latency: approximately 0.0307 ms
- Batch 256 mean latency: approximately 0.0484 ms

#### Accelerator observation

- Apple MPS training and inference were functionally validated.
- CPU and MPS produced effectively equivalent held-out accuracy.
- MPS latency showed substantial run-to-run variability for the compact model.
- CPU is therefore used as the primary local v0.2 PyTorch latency baseline.

### Limitations

- Neural ONNX export is not included in v0.2.0.
- FP16 and INT8 neural deployment are not yet included.
- Qualcomm QNN deployment is not yet included.
- Snapdragon NPU profiling is not yet included.
- CPU and MPS latency values are hardware- and environment-specific.
- MPS dispatch overhead dominates execution for the current compact network.
- The benchmark remains synthetic and is not a certified aircraft-design tool.

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
- Architecture documentation
- Reproducibility documentation
- Artifact-backed results documentation

### Changed

- Separated the validation-selected classifier threshold from the
  optimization-specific acceptance threshold
- Retained classifier threshold 0.30 for ordinary classification
- Applied threshold 0.50 during optimization
- Updated optimization configuration to store the conservative threshold
- Updated optimization summaries to record the applied threshold
- Updated the official optimization result to 5,413 accepted candidates
- Updated the official Pareto front to 47 designs
- Improved representative-design physics-feasibility agreement to 100%
- Improved Pareto-front physics-feasibility agreement to 100%
- Replaced the descriptive pipeline outline with an executable workflow

### Fixed

- Restored the missing physics-validation implementation
- Restored physics-validation test coverage
- Corrected FP32 model artifact paths
- Corrected optimization summary artifact paths
- Removed tracked local backup files
- Prevented the optimizer from using the permissive classifier threshold
  directly
- Added optimization-threshold validation
- Added tests for optimization-threshold overrides

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
