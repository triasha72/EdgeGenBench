# EdgeGenBench Neural FP16 Deployment Results

This document records the reduced-precision FP16 neural deployment study built
from the validated EdgeGenBench FP32 ONNX reference.

## Scope

The FP16 milestone evaluates whether a compact neural aircraft-design surrogate
can reduce serialized deployment size without materially degrading held-out
predictive quality.

The study includes:

- reproducible FP32-to-FP16 ONNX conversion;
- FP32 external input/output tensors;
- FP16 internal initializers;
- ONNX graph validation;
- dynamic batch preservation;
- static batch specialization for CoreML;
- provider-drift measurement;
- precision-drift measurement;
- per-target physical-unit drift;
- held-out FP16 task metrics;
- project-specific drift regression guardrails;
- repeated FP32 versus FP16 CoreML benchmarking;
- public FP16 export and benchmark CLI commands.

## Reference model

Architecture:

```text
10 → 64 → 32 → 16 → 6
```

Reference deployment artifact:

```text
artifacts/neural_onnx/neural_surrogate.onnx
```

FP16 artifact:

```text
artifacts/neural_fp16/neural_surrogate_fp16.onnx
```

## Conversion design

The dynamic FP32 ONNX graph is converted using `onnxconverter-common`.

External I/O remains FP32:

```text
FP32 features [batch, 10]
        |
        v
      Cast
        |
        v
FP16 internal neural graph
        |
        v
      Cast
        |
        v
FP32 predictions [batch, 6]
```

This allows the same frozen preprocessing pipeline to provide FP32 tensors to
both deployment variants.

## Artifact validation

| Property | FP32 | FP16 |
|---|---:|---:|
| ONNX checker | PASS | PASS |
| Opset | 18 | 18 |
| Dynamic batch | Yes | Yes |
| Input shape | `[batch, 10]` | `[batch, 10]` |
| Output shape | `[batch, 6]` | `[batch, 6]` |
| External input precision | FP32 | FP32 |
| External output precision | FP32 | FP32 |
| FP16 initializers | 0 | 8 |
| Serialized size | 25,420 bytes | 19,221 bytes |

Serialized size ratio:

```text
FP16 / FP32 = 0.7561369
```

Serialized size reduction:

```text
24.3863%
```

## Static CoreML specialization

Dynamic CoreML smoke tests were functional, but the CoreML backend reported an
unbounded-dimension warning for the dynamic feature dimension.

For the production latency benchmark, the canonical dynamic models are
therefore specialized reproducibly into exact static batch variants:

```text
runtime_models/
├── fp32_batch1.onnx
├── fp32_batch32.onnx
├── fp32_batch256.onnx
├── fp16_batch1.onnx
├── fp16_batch32.onnx
└── fp16_batch256.onnx
```

Static models were validated with ONNX checker and exact shapes:

```text
batch 1:
[1, 10] → [1, 6]

batch 32:
[32, 10] → [32, 6]

batch 256:
[256, 10] → [256, 6]
```

## CoreML runtime configuration

The local ONNX Runtime environment exposed:

```text
CoreMLExecutionProvider
AzureExecutionProvider
CPUExecutionProvider
```

The CoreML benchmark uses:

```text
ModelFormat=MLProgram
MLComputeUnits=ALL
RequireStaticInputShapes=1
EnableOnSubgraphs=0
```

with `CPUExecutionProvider` available as fallback.

`MLComputeUnits=ALL` does not establish exclusive Apple Neural Engine execution.
The results should therefore be described as CoreML execution-provider
measurements, not ANE-only benchmarks.

## Held-out evaluation

All 900 held-out test rows were evaluated.

The benchmark separates three effects:

```text
FP32 CPU
   |
   | provider difference
   v
FP32 CoreML
   |
   | precision difference
   v
FP16 CoreML
```

This helps distinguish normal provider-level numerical variation from FP16
precision drift.

## FP32 CPU versus FP32 CoreML provider drift

| Metric | Result |
|---|---:|
| Mean normalized absolute difference | 1.4848178864e-07 |
| Maximum normalized absolute difference | 1.4305114746e-06 |

Provider drift is extremely small.

## FP32 CoreML versus FP16 CoreML precision drift

| Metric | Result |
|---|---:|
| Mean normalized absolute difference | **9.7868568264e-04** |
| Maximum normalized absolute difference | **9.1943740845e-03** |

Project regression guardrails:

| Guard | Limit | Result | Status |
|---|---:|---:|---:|
| Mean normalized drift | 0.002 | 9.7869e-04 | **PASS** |
| Maximum normalized drift | 0.012 | 9.1944e-03 | **PASS** |

These values are EdgeGenBench regression guardrails derived for this deployment
study. They are not general FP16 standards.

## Target-level physical precision drift

| Target | Mean absolute drift | Maximum absolute drift | Maximum reference-relative drift |
|---|---:|---:|---:|
| Estimated takeoff mass | 8.549401 kg | 68.648438 kg | 0.1011% |
| Mission energy | 4.792264 kWh | 45.472656 kWh | 0.1568% |
| Energy per passenger-km | 3.8499e-05 | 2.3261e-04 | 0.0662% |
| Lifecycle-emissions proxy | 1.362976 | 7.480469 | 0.1056% |
| Operating-cost proxy | 0.644470 | 4.438721 | 0.1063% |
| Noise proxy | 0.003210 dB | 0.015823 dB | 0.0171% |

The largest reference-relative drift occurs for mission energy and remains below
approximately 0.157%.

## FP16 held-out predictive performance

| Target | MAE | RMSE | NRMSE | R² |
|---|---:|---:|---:|---:|
| Estimated takeoff mass | 394.384796 | 562.801841 | 0.073412 | 0.994611 |
| Mission energy | 179.564513 | 246.512638 | 0.048342 | 0.997663 |
| Energy per passenger-km | 0.002527 | 0.003299 | 0.083341 | 0.993054 |
| Lifecycle-emissions proxy | 36.924714 | 50.146407 | 0.035779 | 0.998720 |
| Operating-cost proxy | 22.525814 | 31.458905 | 0.045180 | 0.997959 |
| Noise proxy | 0.042478 | 0.055716 | 0.016787 | 0.999718 |

Aggregate:

| Metric | FP32 reference | FP16 | Change |
|---|---:|---:|---:|
| Mean NRMSE | 0.050433 | 0.050473 | +0.000041 |
| Mean R² | 0.996955 | 0.996954 | -0.0000009 |

The mean NRMSE increase is approximately 0.081% relative to the FP32 reference.

The result supports the conclusion that FP16 preserves essentially the same
predictive quality for this model and dataset.

## Production CoreML latency methodology

Production benchmark settings:

```text
batch sizes = 1, 32, 256
runs        = 5
repeats     = 500
warmups     = 50
```

Methodology:

- same frozen neural model;
- same transformed held-out inputs;
- static FP32 and FP16 models;
- same CoreML provider configuration;
- preprocessing outside the timed region;
- session creation outside the timed region;
- alternating FP32-first and FP16-first ordering across runs;
- garbage collection disabled during timed measurements;
- median and range reported across independent paired runs.

The benchmark does not enforce that FP16 must be faster.

## Individual latency runs

| Run | Batch | FP32 mean ms | FP16 mean ms | FP32/FP16 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.042298 | 0.038387 | 1.101878× |
| 2 | 1 | 0.039467 | 0.038158 | 1.034296× |
| 3 | 1 | 0.037398 | 0.039090 | 0.956704× |
| 4 | 1 | 0.038480 | 0.038685 | 0.994704× |
| 5 | 1 | 0.037483 | 0.039743 | 0.943150× |
| 1 | 32 | 0.042127 | 0.041161 | 1.023473× |
| 2 | 32 | 0.040386 | 0.041185 | 0.980607× |
| 3 | 32 | 0.040879 | 0.040393 | 1.012054× |
| 4 | 32 | 0.042150 | 0.041787 | 1.008690× |
| 5 | 32 | 0.040770 | 0.040686 | 1.002065× |
| 1 | 256 | 0.051429 | 0.070811 | 0.726296× |
| 2 | 256 | 0.051324 | 0.061661 | 0.832352× |
| 3 | 256 | 0.052998 | 0.059067 | 0.897264× |
| 4 | 256 | 0.051657 | 0.059952 | 0.861629× |
| 5 | 256 | 0.053251 | 0.057543 | 0.925417× |

A ratio greater than 1 means FP16 had lower mean latency.

## Aggregate CoreML latency

| Batch | FP32 median | FP16 median | Median FP32/FP16 ratio | Ratio range | FP16 faster |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.038480 ms | 0.038685 ms | 0.994704× | 0.943150×–1.101878× | 2/5 |
| 32 | 0.040879 ms | 0.041161 ms | 1.008690× | 0.980607×–1.023473× | 4/5 |
| 256 | 0.051657 ms | 0.059952 ms | 0.861629× | 0.726296×–0.925417× | 0/5 |

## Latency interpretation

### Batch 1

FP32 and FP16 are effectively at parity.

The median paired ratio is approximately 0.995× and the repeated-run range
crosses 1.0.

### Batch 32

FP32 and FP16 are also effectively at parity.

FP16 was faster in four of five paired runs, but the median difference is less
than 1% and the repeated-run range crosses 1.0.

### Batch 256

FP16 is consistently slower.

FP16 was slower in all five paired runs. Median FP16 latency is approximately
16% higher than the FP32 median for this batch size.

## Overall conclusion

For this compact surrogate and local CoreML configuration, FP16 provides a
clear **serialized-size benefit** without materially degrading predictive
quality.

It does **not** provide a universal latency benefit.

The measured tradeoff is:

```text
FP32 ONNX
  |
  | 25,420 bytes
  | NRMSE ≈ 0.050433
  |
  v
FP16 ONNX
  |
  | 19,221 bytes
  | -24.39% serialized size
  | NRMSE ≈ 0.050473
  |
  +--> batch 1:   latency parity
  +--> batch 32:  latency parity
  +--> batch 256: FP16 slower
```

This is a more useful deployment result than assuming reduced numerical
precision must always improve runtime.

## CLI workflow

Export FP16:

```bash
edgegenbench export-neural-fp16 \
  --fp32-model artifacts/neural_onnx/neural_surrogate.onnx \
  --fp32-metadata artifacts/neural_onnx/metadata.json \
  --output-dir artifacts/neural_fp16
```

Run the production benchmark:

```bash
edgegenbench benchmark-neural-fp16 \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --preprocessing artifacts/neural_surrogate/preprocessing.npz \
  --fp32-model artifacts/neural_onnx/neural_surrogate.onnx \
  --fp16-model artifacts/neural_fp16/neural_surrogate_fp16.onnx \
  --output-dir artifacts/neural_fp16_benchmark \
  --runs 5 \
  --repeats 500 \
  --warmups 50 \
  --max-mean-normalized-drift 0.002 \
  --max-normalized-drift 0.012
```

The CLI parser accepts the complete drift-option names even when Rich visually
truncates them with an ellipsis in narrow terminal help output.

## Generated artifacts

```text
artifacts/neural_fp16/
├── metadata.json
└── neural_surrogate_fp16.onnx

artifacts/neural_fp16_benchmark/
├── equivalence.csv
├── task_metrics.csv
├── latency_runs.csv
├── latency_summary.csv
├── summary.json
└── runtime_models/
    ├── fp32_batch1.onnx
    ├── fp32_batch32.onnx
    ├── fp32_batch256.onnx
    ├── fp16_batch1.onnx
    ├── fp16_batch32.onnx
    └── fp16_batch256.onnx
```

Generated runtime artifacts are intentionally ignored by Git.

## Automated validation

The FP16 milestone is covered by:

- conversion tests;
- FP16 initializer tests;
- dynamic-shape preservation tests;
- ONNX checker validation;
- static batch specialization tests;
- invalid-batch validation;
- provider availability checks;
- drift-limit evaluation tests;
- latency aggregation tests;
- CoreML integration tests when available;
- FP16 CLI registration tests;
- parser-level long-option tests.

Local release-gate validation includes:

```bash
ruff format --check .
ruff check .
pytest -q tests/neural
pytest -q
python -m pip check
git diff --check
```

## Limitations

- Results use a synthetic aircraft-design benchmark.
- CoreML measurements are specific to the tested ARM64 macOS environment.
- `MLComputeUnits=ALL` does not prove exclusive Apple Neural Engine execution.
- CPU fallback remains configured in the ONNX Runtime provider stack.
- Microsecond-scale latency is sensitive to operating-system and runtime state.
- FP16 size reduction does not imply latency reduction.
- No INT8 result has yet been validated.
- No Qualcomm QNN or Snapdragon NPU result has yet been validated.
- EdgeGenBench is not a certified aircraft-design or safety-critical system.

## Next milestone

The next reduced-precision study is INT8 quantization.

That work should maintain the same separation between:

```text
provider drift
precision / quantization drift
predictive error
serialized model size
runtime performance
```

and should not use the held-out test set for calibration.
