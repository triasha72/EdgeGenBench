# EdgeGenBench v0.2 Neural-Surrogate Results

This document records the validated compact PyTorch surrogate results for
EdgeGenBench v0.2.

The v0.2 milestone extends the existing classical scientific-ML benchmark with
a compact neural surrogate designed for subsequent ONNX, quantization, and
hardware-accelerator deployment studies.

## Evaluation scope

The v0.2 neural milestone evaluates:

- leakage-safe preprocessing;
- compact neural architecture design;
- deterministic train/validation/test partition reuse;
- validation-based early stopping;
- best-checkpoint restoration;
- held-out test accuracy;
- model size;
- CPU inference latency;
- Apple MPS execution;
- cross-device predictive consistency;
- reproducible CLI execution.

PyTorch-to-ONNX export, FP16, INT8, QNN, and Snapdragon NPU results are not part
of this milestone.

## Environment

The validated local environment used:

| Component | Value |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.13.0 |
| Architecture | ARM64 macOS |
| CPU runtime | PyTorch CPU |
| Accelerator runtime | Apple MPS |

All latency measurements are hardware- and environment-specific.

## Dataset

The existing deterministic EdgeGenBench dataset contains 6,000 synthetic
regional-aircraft design cases.

| Partition | Rows |
|---|---:|
| Training | 4,200 |
| Validation | 900 |
| Test | 900 |
| Total | 6,000 |

The six numerical design variables are:

- passenger capacity;
- design range;
- cruise speed;
- battery specific energy;
- hydrogen storage efficiency;
- hybridization ratio.

Propulsion architecture contains four categories:

- conventional turboprop;
- fuel-cell electric;
- parallel hybrid;
- series hybrid.

After one-hot encoding, the neural surrogate receives ten input features.

## Leakage-safe preprocessing

Feature and target normalization statistics are fitted exclusively on the
training partition.

The fitted state contains:

- propulsion categories;
- numerical feature means;
- numerical feature scales;
- target means;
- target scales;
- output target ordering.

Validation and test data use the frozen training statistics.

The preprocessing state is serialized independently of the neural weights and
can be loaded for later inference or deployment conversion.

Save/load parity is covered by automated tests.

## Neural architecture

The compact neural surrogate uses:

```text
10 → 64 → 32 → 16 → 6
```

with ReLU activations between hidden layers.

| Property | Value |
|---|---:|
| Encoded inputs | 10 |
| Hidden layer 1 | 64 |
| Hidden layer 2 | 32 |
| Hidden layer 3 | 16 |
| Outputs | 6 |
| Trainable parameters | 3,414 |
| Serialized model size | 16,881 bytes |

The six outputs are:

- estimated takeoff mass;
- mission energy;
- energy per passenger-kilometre;
- lifecycle-emissions proxy;
- operating-cost proxy;
- noise proxy.

## Training configuration

The reference configuration uses:

| Parameter | Value |
|---|---:|
| Random seed | 42 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Maximum epochs | 500 |
| Early-stopping patience | 30 |
| Minimum improvement | 0.000001 |

The optimizer is AdamW and the normalized training objective is mean squared
error.

## Early stopping

The best reference model occurred at epoch 141.

Training terminated at epoch 171 after 30 consecutive epochs without
sufficient validation improvement.

This confirms that the configured early-stopping patience was applied as
intended.

## CPU held-out results

The CPU validation run produced:

| Metric | Value |
|---|---:|
| Best epoch | 141 |
| Best validation loss | 0.003442 |
| Mean test NRMSE | **0.050425** |
| Mean test R² | **0.996956** |
| Model size | 16,881 bytes |
| Parameters | 3,414 |

## MPS held-out results

The reference Apple MPS run produced:

| Metric | Value |
|---|---:|
| Best epoch | 141 |
| Best validation loss | 0.003429 |
| Mean test NRMSE | **0.050433** |
| Mean test R² | **0.996955** |
| Model size | 16,881 bytes |
| Parameters | 3,414 |

CPU and MPS therefore produced effectively equivalent aggregate predictive
accuracy.

## Target-level reference results

The reference MPS evaluation produced:

| Target | MAE | RMSE | NRMSE | R² |
|---|---:|---:|---:|---:|
| Estimated takeoff mass | 394.558 | 563.268 | 0.073473 | 0.994602 |
| Mission energy | 179.615 | 246.530 | 0.048346 | 0.997663 |
| Energy per passenger-km | 0.002528 | 0.003301 | 0.083380 | 0.993048 |
| Lifecycle emissions | 36.840 | 50.022 | 0.035690 | 0.998726 |
| Operating cost | 22.496 | 31.385 | 0.045074 | 0.997968 |
| Noise | 0.041856 | 0.055205 | 0.016633 | 0.999723 |

Every target achieved held-out R² above 0.993.

## Comparison with classical surrogates

| Model | Mean NRMSE | Mean R² | Serialized size |
|---|---:|---:|---:|
| Compact PyTorch MLP | **0.050425** | **0.996956** | **16.49 KiB** |
| HistGradientBoosting | 0.062249 | 0.995171 | 7.216 MiB |
| Random Forest | 0.205386 | 0.953219 | 172.296 MiB |
| FP32 Ridge | 0.214590 | 0.937690 | 2.53 KiB |

The neural surrogate reduces mean NRMSE by approximately 19% relative to the
strongest classical predictive model.

Serialized model sizes use different runtime/file formats and should therefore
be treated as deployment artifacts rather than direct parameter-count
comparisons.

## CPU inference latency

The CPU benchmark produced:

| Batch | Mean batch latency | P95 batch latency | Mean sample latency |
|---:|---:|---:|---:|
| 1 | **0.022017 ms** | 0.022585 ms | 22.017 µs |
| 32 | **0.030655 ms** | 0.031171 ms | 0.958 µs |
| 256 | **0.048388 ms** | 0.050673 ms | 0.189 µs |

The compact model is sufficiently small that CPU execution is very efficient
for low-batch inference on the reference development machine.

## Apple MPS behavior

Apple MPS execution was functionally successful.

A reference MPS latency run produced:

| Batch | Mean batch latency | P95 batch latency |
|---:|---:|---:|
| 1 | 2.880 ms | 6.999 ms |
| 32 | 0.502 ms | 0.685 ms |
| 256 | 0.547 ms | 0.653 ms |

A later CLI validation run showed substantially higher and more variable MPS
latency, including approximately 8.20 ms mean batch-1 latency and approximately
68 ms P95 batch-1 latency.

This indicates that MPS latency for the current tiny model is dominated by
accelerator-dispatch and runtime overhead rather than neural computation.

Therefore:

- CPU is the official local PyTorch latency baseline for v0.2;
- MPS remains a supported functional execution path;
- no universal CPU-versus-GPU speed claim is made.

## CLI validation

The public neural workflow is:

```bash
edgegenbench train-neural-surrogate \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --config configs/neural_v0_2.yaml \
  --output-dir artifacts/neural_surrogate
```

A successful run produces:

```text
model.pt
preprocessing.npz
training_history.csv
test_metrics.csv
test_predictions.csv
latency.csv
summary.json
```

The public CLI reproduced the same reference MPS accuracy metrics as the direct
Python training invocation.

## Test coverage

The neural test suite covers:

- model output shape;
- batch-one inference;
- parameter-count validation;
- invalid dimension handling;
- feature preprocessing;
- target normalization;
- train-only preprocessing behavior;
- unknown categorical values;
- preprocessing save/load parity;
- full training execution;
- generated artifacts;
- expected held-out rows;
- CLI command registration.

The current neural suite contains 14 passing tests.

## Conclusions

The v0.2 neural milestone establishes a compact, high-accuracy PyTorch
surrogate that is suitable for deployment-oriented optimization work.

The validated model:

- improves predictive accuracy over the strongest v0.1 classical model;
- uses only 3,414 trainable parameters;
- produces a roughly 16.5 KiB serialized PyTorch artifact;
- supports reproducible preprocessing and checkpointing;
- executes efficiently on CPU;
- supports MPS as an alternate execution path;
- is exposed through the public EdgeGenBench CLI.

The next milestone is neural deployment conversion:

```text
PyTorch FP32
    |
    v
ONNX FP32
    |
    v
Numerical-equivalence validation
    |
    v
FP16
    |
    v
INT8
    |
    v
Qualcomm QNN
    |
    v
Snapdragon NPU
```

## Limitations

- Results use a synthetic aircraft-design benchmark.
- Results represent one design-space configuration.
- Latency measurements are machine-specific.
- CPU and MPS latency are not universal hardware comparisons.
- Neural ONNX export is not yet implemented.
- FP16 and INT8 results are not yet available.
- Qualcomm QNN and Snapdragon NPU results are not yet available.
- The benchmark is not a certified aircraft-design or safety-critical system.
