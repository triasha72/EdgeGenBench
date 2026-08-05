# EdgeGenBench Architecture

## Purpose

EdgeGenBench is a modular research benchmark for studying the interaction
between surrogate models, uncertainty estimation, feasibility decisions,
optimization, physics validation, and edge deployment.

The architecture is intentionally separated into stages so that each model
or decision can be evaluated independently before it influences a later
engineering workflow.

## End-to-end architecture

```text
Versioned configuration
        |
        v
Synthetic aircraft physics model
        |
        v
Dataset generation
        |
        +--> raw features
        +--> surrogate targets
        +--> physics feasibility labels
        +--> reproducibility metadata
        |
        v
Deterministic data partitioning
        |
        +--> training
        +--> validation
        +--> calibration
        +--> test
        |
        v
Surrogate model training
        |
        +--> FP32 Ridge
        +--> Random Forest
        +--> HistGradientBoosting
        |
        v
Model evaluation and comparison
        |
        +--> MAE and RMSE
        +--> normalized RMSE
        +--> R²
        +--> model size
        +--> inference latency
        |
        v
Uncertainty workflow
        |
        +--> Random Forest tree quantiles
        +--> split-conformal intervals
        +--> empirical coverage evaluation
        |
        v
Feasibility classifier
        |
        +--> validation-selected threshold: 0.30
        +--> false-safe-rate control
        |
        v
Optimization-specific safety gate
        |
        +--> physics-validated threshold: 0.50
        |
        v
Constrained multi-objective optimization
        |
        +--> Latin-hypercube candidates
        +--> surrogate objective predictions
        +--> feasibility filtering
        +--> Pareto-front extraction
        +--> representative-design selection
        |
        v
Physics-based validation
        |
        +--> feasibility agreement
        +--> target-level error
        +--> surrogate-versus-physics plots
        |
        v
Deterministic edge feature encoder
        |
        v
ONNX export
        |
        +--> surrogate.onnx
        +--> feasibility.onnx
        +--> metadata.json
        |
        v
ONNX Runtime benchmark
        |
        +--> numerical equivalence
        +--> classifier agreement
        +--> model size
        +--> batch latency
        +--> per-sample latency
```

## Design principles

### Separation of model selection and final evaluation

Hyperparameters are selected using the validation partition. The selected
models are then refitted using the combined training and validation data and
evaluated once on the held-out test partition.

This prevents test metrics from influencing model selection.

### Separation of classification and optimization thresholds

EdgeGenBench uses two different feasibility thresholds.

| Threshold | Role |
|---:|---|
| 0.30 | Classifier operating threshold selected from ordinary validation data |
| 0.50 | Conservative optimization threshold selected through physics validation |

The classifier threshold is optimized for held-out classification under a
maximum false-safe-rate constraint.

The optimization threshold addresses a different problem. Optimization tends
to select unusual, high-performing, boundary-adjacent designs. These samples
do not necessarily follow the same distribution as randomly selected test
cases.

The optimization workflow therefore overrides the classifier's stored
threshold without retraining the underlying Random Forest.

```text
Feasibility classifier
        |
        +--> stored classifier threshold: 0.30
        |
        v
Optimization configuration
        |
        +--> threshold override: 0.50
        |
        v
Candidate filtering
```

The v0.1 threshold study found that `0.50` was the smallest tested value that
produced:

- 100% feasibility agreement across the complete Pareto front;
- 100% feasibility agreement across the four representative designs.

### Artifact-backed execution

Each stage writes explicit machine-readable artifacts. Later stages consume
those artifacts rather than relying on hidden in-memory state.

Examples include:

- JSON summaries;
- CSV predictions and metrics;
- serialized Scikit-learn models;
- NPZ Ridge parameters;
- ONNX models;
- PNG evaluation figures;
- versioned YAML configurations.

### Deterministic interfaces

The project uses stable feature names, target names, artifact locations, and
configuration schemas.

This makes each stage replaceable without requiring the entire system to be
rewritten.

## Major subsystems

## 1. Configuration layer

Configuration files are stored under `configs/`.

### `configs/v0_1.yaml`

Defines the synthetic dataset and physics benchmark, including:

- design-variable ranges;
- random seed;
- dataset size;
- output locations;
- split behavior.

### `configs/optimization_v0_1.yaml`

Defines:

- optimization name;
- candidate count;
- optimization seed;
- mission requirements;
- design-space limits;
- propulsion architectures;
- objective names;
- objective directions;
- optimization-specific feasibility threshold;
- output directory.

The v0.1 optimization threshold is:

```yaml
feasibility_threshold: 0.50
```

## 2. Synthetic physics and data generation

Primary package areas:

```text
src/edgegenbench/physics/
src/edgegenbench/data/
```

Responsibilities include:

- sampling aircraft-design variables;
- evaluating analytical physics proxies;
- generating target quantities;
- assigning feasibility labels;
- storing reproducibility metadata;
- producing deterministic data partitions.

The synthetic physics layer acts as the benchmark source of truth for both
training labels and post-optimization validation.

## 3. Surrogate-model layer

Primary package areas:

```text
src/edgegenbench/models/
src/edgegenbench/training/
```

Supported v0.1 surrogate families:

### FP32 Ridge

Purpose:

- compact linear baseline;
- low inference overhead;
- small serialization footprint;
- interpretable regularization study.

Artifact:

```text
artifacts/fp32_baseline/fp32_linear_model.npz
```

### Random Forest

Purpose:

- nonlinear multi-output baseline;
- ensemble uncertainty estimation;
- optimization surrogate;
- ONNX deployment baseline.

Artifact:

```text
artifacts/tree_baselines/random_forest/model.joblib
```

### HistGradientBoosting

Purpose:

- high-accuracy nonlinear baseline;
- comparison against Ridge and Random Forest;
- evaluation of accuracy-versus-latency trade-offs.

Artifact:

```text
artifacts/tree_baselines/hist_gradient_boosting/model.joblib
```

## 4. Evaluation layer

Primary package area:

```text
src/edgegenbench/evaluation/
```

Responsibilities include:

- regression metrics;
- classifier metrics;
- model comparison;
- latency aggregation;
- physics-validation metrics;
- Pareto and representative-design plots;
- reporting utilities.

The model-comparison layer consolidates independent training outputs into a
shared schema.

## 5. Uncertainty layer

Primary package area:

```text
src/edgegenbench/uncertainty/
```

Two approaches are implemented.

### Random Forest tree quantiles

Prediction intervals are derived from the distribution of predictions across
individual trees.

These intervals are data-adaptive but were conservative in the v0.1
benchmark.

### Split conformal prediction

The workflow separates:

- proper training data;
- calibration data;
- held-out test data.

Calibration residuals are used to produce prediction intervals at requested
nominal coverage levels.

The v0.1 workflow evaluates 80%, 90%, and 95% conformal intervals.

## 6. Feasibility-classification layer

Primary package areas:

```text
src/edgegenbench/models/feasibility.py
src/edgegenbench/training/feasibility.py
```

The feasibility classifier:

- uses a Random Forest;
- produces feasibility probabilities;
- stores its selected threshold with the serialized model;
- supports threshold overrides without retraining;
- reports false-safe and false-reject behavior.

A false-safe prediction occurs when an infeasible design is accepted as
feasible. This error is given special attention because it can allow unsafe
or invalid designs into optimization.

## 7. Optimization layer

Primary package area:

```text
src/edgegenbench/optimization/
```

### Candidate generation

The candidate generator uses deterministic Latin-hypercube sampling for
continuous variables and reproducible assignment of propulsion
architectures.

The official v0.1 study generates:

```text
20,000 candidates
```

### Candidate scoring

Every candidate receives:

- six surrogate predictions;
- feasibility probability;
- infeasibility probability;
- applied feasibility threshold;
- predicted feasibility decision.

### Conservative feasibility filtering

The optimizer applies the configuration threshold of `0.50`, not the
classifier's stored threshold of `0.30`.

Only accepted candidates proceed to Pareto extraction.

### Pareto extraction

The v0.1 study minimizes:

- lifecycle-emissions proxy;
- operating-cost proxy;
- noise proxy.

Dominated candidates are removed to create the Pareto front.

### Representative selection

Four representative roles are selected:

- low emissions;
- low cost;
- low noise;
- balanced.

## 8. Physics-validation layer

Primary file:

```text
src/edgegenbench/evaluation/physics_validation.py
```

The validation layer reevaluates optimized designs using the source synthetic
physics model.

It reports:

- design count;
- physics-feasible count;
- classifier-feasible count;
- feasibility agreement;
- target-level MAE;
- target-level RMSE;
- signed error;
- relative error;
- maximum relative error;
- correlation.

This stage detects cases where optimization exploits classifier or surrogate
errors.

## 9. Deployment layer

Primary package area:

```text
src/edgegenbench/deployment/
```

### Feature encoder

The deterministic encoder transforms:

- six numeric inputs;
- one categorical propulsion architecture;

into a ten-feature numeric tensor.

The architecture categories are encoded in a fixed order to preserve
deployment equivalence.

### ONNX export

The deployment workflow exports:

```text
artifacts/edge_export/surrogate.onnx
artifacts/edge_export/feasibility.onnx
artifacts/edge_export/metadata.json
```

The metadata records:

- feature order;
- encoded feature names;
- target order;
- class labels;
- feasible-class index;
- classifier threshold;
- ONNX paths;
- target opset.

### ONNX inference

ONNX Runtime wrappers reproduce:

- surrogate target predictions;
- classifier probabilities;
- classifier decisions.

### Equivalence validation

The edge benchmark compares Scikit-learn and ONNX outputs for all 900 test
rows.

The v0.1 results showed:

- 100% classifier decision agreement;
- maximum classifier probability error of approximately `1.82 × 10⁻⁷`;
- maximum surrogate absolute difference of approximately `0.01059`.

## Module map

```text
src/edgegenbench/
├── cli.py
├── data/
│   └── generate.py
├── deployment/
│   ├── benchmark.py
│   ├── feature_encoder.py
│   ├── onnx_export.py
│   └── onnx_inference.py
├── evaluation/
│   ├── calibration.py
│   ├── classification.py
│   ├── model_comparison.py
│   ├── optimization.py
│   ├── physics_validation.py
│   └── plots.py
├── models/
│   ├── feasibility.py
│   ├── preprocessing.py
│   └── tree_surrogate.py
├── optimization/
│   ├── design_space.py
│   ├── pareto.py
│   ├── pipeline.py
│   └── search.py
├── physics/
├── training/
│   ├── feasibility.py
│   ├── fp32_baseline.py
│   └── tree_baselines.py
└── uncertainty/
```

## Artifact contracts

## Dataset contract

The generated dataset must contain all required feature, target, split, and
feasibility columns.

## Surrogate contract

A surrogate must:

- validate feature columns;
- preserve target ordering;
- return finite numeric predictions;
- support reproducible serialization;
- publish a summary artifact.

## Feasibility contract

A feasibility model must:

- return a probability in `[0, 1]`;
- identify the feasible class unambiguously;
- store or accept a decision threshold;
- expose reproducible prediction behavior.

## Optimization contract

Optimization must:

- consume versioned configuration;
- use unique candidate identifiers;
- record the applied threshold;
- reject nonfinite predictions;
- retain traceable output paths;
- validate final selected designs against physics.

## Deployment contract

Deployment must:

- use the same feature order as training;
- preserve categorical encoding;
- preserve target order;
- report equivalence errors;
- report model sizes and latency.

## Generated artifact structure

```text
artifacts/
├── fp32_baseline/
├── tree_baselines/
│   ├── random_forest/
│   └── hist_gradient_boosting/
├── uncertainty/
├── feasibility_classifier/
├── optimization/
├── optimization_validation/
├── edge_export/
└── edge_benchmark/

reports/
└── model_comparison/
```

These directories are generated and are not intended to be committed.

## Extension points

Future versions can extend the architecture by adding:

- neural surrogates;
- Gaussian-process surrogates;
- alternate conformal methods;
- out-of-distribution detectors;
- new physics backends;
- additional missions;
- hardware-specific deployment runtimes;
- quantized model formats.

New implementations should preserve the existing artifact and feature
contracts whenever possible.

## Safety and scope boundary

EdgeGenBench is a research benchmark, not a certified aircraft-design system.

The architecture demonstrates how validation and conservative decision gates
can be incorporated into a machine-learning workflow. It does not establish
airworthiness, certification compliance, or operational safety.