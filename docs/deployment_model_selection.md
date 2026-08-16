# Deployment-Aware Neural Model Selection

## Objective

This milestone turns EdgeGenBench's validated neural deployment experiments into an explicit model-selection layer.

The selector does **not** assume that lower precision is automatically faster or better. It consumes the measured FP32, FP16, and mixed INT8/FP32 benchmark artifacts already produced by the repository, applies hard deployment constraints, and then ranks only the feasible candidates.

The current candidate set is provider-aware:

- FP32 with `CPUExecutionProvider`
- mixed INT8/FP32 with `CPUExecutionProvider`
- FP32 with `CoreMLExecutionProvider`
- FP16 with `CoreMLExecutionProvider`

Each batch size is treated as a separate measured deployment candidate.

## Why provider-aware selection matters

The existing FP16 and INT8 studies use different execution paths:

- FP16 latency is measured through CoreML-backed ONNX Runtime execution.
- mixed INT8/FP32 latency is measured through ONNX Runtime CPU execution.

Therefore, the selection layer preserves the execution provider as part of candidate identity. A cross-provider ranking is a comparison of measured deployment candidates on the tested machine; it is **not** a precision-only FP32-versus-FP16-versus-INT8 benchmark.

## Input artifacts

The default loader consumes:

```text
artifacts/neural_surrogate/summary.json
artifacts/neural_fp16_benchmark/summary.json
artifacts/neural_int8_benchmark/summary.json
artifacts/neural_onnx/neural_surrogate.onnx
artifacts/neural_fp16/neural_surrogate_fp16.onnx
artifacts/neural_int8/neural_surrogate_int8.onnx
```

The neural-training summary supplies the FP32 held-out quality metrics. The FP16 and INT8 benchmark summaries supply provider-specific latency, compressed-model quality, drift, and serialized-size metadata.

## Candidate fields

Every candidate records:

```text
name
precision
provider
model_path
model_size_bytes
batch_size
median_latency_ms
mean_r2
mean_nrmse_std
max_normalized_drift
benchmark_source
measurement_context
```

The model-size field is the serialized canonical ONNX artifact size. It should not be interpreted as provider-compiled memory footprint, peak RAM, accelerator SRAM use, or total device memory consumption.

## Hard constraints

Hard constraints are evaluated before ranking:

```text
batch_size
max_latency_ms
max_model_size_mb
min_r2
max_nrmse_std
max_normalized_drift
required_provider
```

A candidate that violates any supplied hard constraint is rejected. Rejection reasons are written to the selection report instead of being hidden inside a weighted score.

Example:

```text
candidate: mixed_int8_fp32_cpu
status: rejected
reason: median latency 0.005370 ms exceeds 0.005000 ms
```

## Ranking policies

After hard filtering, one of four policies ranks the feasible candidates.

### `lowest_latency`

Prioritizes measured median batch latency. Ties are resolved by higher R², smaller serialized model size, and stable candidate name ordering.

### `smallest_model`

Prioritizes serialized model size. Ties are resolved by latency, R², and candidate name.

### `highest_accuracy`

Prioritizes mean held-out R², followed by lower NRMSE, lower latency, and smaller size.

### `balanced`

Uses an explicit normalized score over the feasible set:

```text
0.40 * normalized latency
+ 0.25 * normalized model size
+ 0.25 * normalized NRMSE
+ 0.10 * normalized maximum drift
```

Lower score is better.

The balanced score is intentionally simple and transparent. It is not a learned utility function and should not be presented as universally optimal.

## Output artifacts

The selector writes:

```text
artifacts/deployment_selection/selection.json
artifacts/deployment_selection/selection.md
```

The JSON report contains:

- requested constraints
- selection policy
- selected candidate, when one exists
- feasible-candidate count
- every evaluated candidate
- rejection reasons
- balanced score when applicable
- benchmark provenance and measurement context

The Markdown report provides the same decision in a human-readable format.

## CLI examples

Lowest measured latency for batch 256:

```bash
edgegenbench select-neural-deployment \
  --batch-size 256 \
  --policy lowest_latency
```

CPU-only selection with quality and drift constraints:

```bash
edgegenbench select-neural-deployment \
  --batch-size 256 \
  --provider CPUExecutionProvider \
  --min-r2 0.99 \
  --max-nrmse-std 0.06 \
  --max-normalized-drift 0.08 \
  --policy lowest_latency
```

CoreML-only balanced selection:

```bash
edgegenbench select-neural-deployment \
  --batch-size 32 \
  --provider CoreMLExecutionProvider \
  --min-r2 0.99 \
  --policy balanced \
  --output-dir artifacts/deployment_selection/coreml_batch32
```

Impossible constraints are reported explicitly and return a non-zero CLI exit status after the report is written.

## Recommended validation scenarios

Run at least the following profiles after implementation.

### Batch 1: latency-sensitive online inference

```bash
edgegenbench select-neural-deployment \
  --batch-size 1 \
  --min-r2 0.99 \
  --policy lowest_latency \
  --output-dir artifacts/deployment_selection/batch1_latency
```

### Batch 32: balanced deployment

```bash
edgegenbench select-neural-deployment \
  --batch-size 32 \
  --min-r2 0.99 \
  --policy balanced \
  --output-dir artifacts/deployment_selection/batch32_balanced
```

### Batch 256: throughput-oriented inference

```bash
edgegenbench select-neural-deployment \
  --batch-size 256 \
  --min-r2 0.99 \
  --policy lowest_latency \
  --output-dir artifacts/deployment_selection/batch256_latency
```

## Validated deployment-selection results

The selector was validated against measured FP32, FP16, and mixed INT8/FP32
deployment artifacts generated by EdgeGenBench.

| Scenario | Selected candidate | Provider | Median latency | Model size | Mean R² |
|---|---|---|---:|---:|---:|
| Batch 1, lowest latency | `fp32_cpu` | CPUExecutionProvider | 0.004613 ms | 25,420 B | 0.996955 |
| Batch 32, balanced | `fp32_cpu` | CPUExecutionProvider | 0.008414 ms | 25,420 B | 0.996955 |
| Batch 256, lowest latency | `mixed_int8_fp32_cpu` | CPUExecutionProvider | 0.030682 ms | 16,977 B | 0.996855 |
| Batch 256, CPU only | `mixed_int8_fp32_cpu` | CPUExecutionProvider | 0.030682 ms | 16,977 B | 0.996855 |
| Batch 32, CoreML only, balanced | `fp32_coreml` | CoreMLExecutionProvider | 0.040879 ms | 25,420 B | 0.996955 |

### Interpretation

No precision/runtime configuration is universally optimal.

For ONNX Runtime CPU execution, FP32 is faster at batch 1 and slightly faster
at batch 32, while mixed INT8/FP32 is faster in all five paired batch-256 runs.

The mixed INT8/FP32 artifact also reduces serialized ONNX size from 25,420
bytes to 16,977 bytes, a 33.21% reduction, while retaining mean test R² of
0.996855.

For the tested CoreML path, FP32 and FP16 are effectively near parity at
batches 1 and 32, while FP16 is slower at batch 256.

Cross-provider timings represent distinct measured deployment configurations
and should not be interpreted as a precision-only hardware ranking.

## Validation requirements

Before merging this milestone, run:

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy src/edgegenbench/deployment/model_selection.py
python -m mypy src/edgegenbench/cli.py
python -m pytest -q tests/neural/test_deployment_model_selection.py
python -m pytest -q tests/neural/test_neural_cli.py
python -m pytest -q tests/neural
python -m pytest -q
python -m pip check
git diff --check
```

## Scope boundary

This milestone selects among deployment candidates already measured on the local development machine. It does not yet claim:

- Qualcomm QNN compatibility
- Snapdragon NPU execution
- Android deployment
- accelerator memory measurements
- energy measurements
- device-side thermal behavior
- universal FP16 or INT8 speedup

Those remain hardware-specific follow-on milestones.

## Next milestone

After the selector is validated and merged:

1. add a hardware-capability profile abstraction;
2. integrate Qualcomm AI Hub where supported;
3. convert or validate models for QNN;
4. profile supported Snapdragon targets;
5. record device/provider/precision provenance;
6. feed measured on-target latency and size information back into the same deployment-selection interface.
