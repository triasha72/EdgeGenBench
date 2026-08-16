# Mixed-Precision INT8 Neural ONNX Evaluation

## Objective

This study evaluates whether the compact EdgeGenBench neural surrogate can be
quantized for smaller ONNX deployment while preserving held-out predictive
quality and providing useful CPU inference behavior.

The study compares the validated FP32 neural ONNX graph with a production
mixed-precision INT8/FP32 graph using ONNX Runtime CPU execution.

The final model is not fully INT8. Hidden-layer computation is quantized using
static QDQ, while the final output `Gemm` remains FP32.

## Reference model

The reference model is the compact EdgeGenBench neural surrogate:

```text
10 encoded inputs
      |
Linear(10, 64)
      |
     ReLU
      |
Linear(64, 32)
      |
     ReLU
      |
Linear(32, 16)
      |
     ReLU
      |
Linear(16, 6)
      |
6 outputs
```

Reference properties:

| Property | Value |
|---|---:|
| Trainable parameters | 3,414 |
| FP32 ONNX size | 25,420 bytes |
| Input shape | `[batch, 10]` |
| Output shape | `[batch, 6]` |
| Dynamic batch | Yes |
| Reference provider | `CPUExecutionProvider` |

## Quantization investigation

The validated FP32 ONNX graph contains four `Gemm` operators and three
activation layers.

Dynamic IntegerOps quantization was not selected because the existing `Gemm`
structure did not provide the desired supported deployment path.

Static QDQ was selected because it preserves the validated graph structure and
is supported by ONNX Runtime for this model.

## Calibration investigation

An initial deterministic 512-row training calibration subset was evaluated.

That configuration produced a large estimated-takeoff-mass drift outlier. The
cause was traced to calibration-range saturation: the smaller subset did not
observe the full activation/output range later encountered by a held-out test
sample.

The final calibration population therefore uses all 4,200 training rows.

No validation or test rows are used to fit quantization calibration ranges.

## Candidate selection

The final candidate study compared:

1. per-channel QInt8 using a 512-row training calibration subset;
2. per-channel QInt8 using all 4,200 training rows;
3. per-channel QInt8 using all 4,200 training rows while retaining the final
   `node_linear_3` output `Gemm` in FP32.

Candidate selection was based on validation results rather than held-out test
performance.

Retaining the final output head in FP32 produced the strongest overall
validation tradeoff across quantization drift, predictive quality, and
serialized size.

## Frozen production configuration

```text
Quantization format      Static QDQ
Activation type          QInt8
Weight type              QInt8
Weight quantization      Per-channel
Calibration method       MinMax
Calibration split        Training only
Calibration rows         4,200
Calibration batch size   64
Excluded node            node_linear_3
Output head precision    FP32
External input precision FP32
External output precision FP32
Dynamic batch            Yes
Reference provider       CPUExecutionProvider
Selection basis          Validation
```

The correct model description is:

> Mixed-precision INT8/FP32 static-QDQ neural ONNX model with an FP32 output
> head.

It should not be described as a fully INT8 model.

## Production artifact

The production exporter generated:

```text
artifacts/neural_int8/
├── metadata.json
└── neural_surrogate_int8.onnx
```

Measured artifact properties:

| Property | FP32 | Mixed INT8/FP32 |
|---|---:|---:|
| Serialized ONNX size | 25,420 B | 16,977 B |
| Size reduction | — | 33.214% |
| Dynamic batch | Yes | Yes |
| External input precision | FP32 | FP32 |
| External output precision | FP32 | FP32 |
| INT8 initializers | — | 10 |
| INT32 initializers | — | 6 |

The mixed-precision graph is approximately 33.2% smaller than the FP32 ONNX
graph.

It is also smaller than the previously evaluated 19,221-byte FP16 artifact,
although FP16 and INT8 were evaluated through different runtime/provider paths
and should not be interpreted as a direct latency comparison.

## Production graph validation

The production ONNX graph contains:

```text
DequantizeLinear: 10
Gemm:              4
QuantizeLinear:    4

FLOAT initializers: 12
INT8 initializers:  10
INT32 initializers: 6
```

The final output head remains FP32:

```text
node_linear_3       Gemm
network.6.weight    FLOAT
network.6.bias      FLOAT
```

This confirms that the final artifact matches the intended mixed-precision
deployment configuration.

## Production/probe reproducibility

The production exporter was compared directly with the validation-selected
experimental artifact over all 900 held-out test rows.

| Metric | Result |
|---|---:|
| Test rows | 900 |
| Mean absolute difference | 0.0 |
| Maximum absolute difference | 0.0 |
| `allclose(rtol=1e-6, atol=1e-7)` | True |

The production exporter therefore reproduces the selected experimental
candidate exactly for the held-out test inputs.

## Held-out quantization drift

All 900 test rows were evaluated through both FP32 and mixed INT8/FP32 ONNX
Runtime CPU inference using the same frozen preprocessing.

Project-specific regression ceilings were defined before final reporting.

| Metric | Result | Ceiling | Status |
|---|---:|---:|---|
| Mean normalized absolute drift | 0.008028 | 0.015 | PASS |
| P95 normalized absolute drift | 0.020127 | — | — |
| P99 normalized absolute drift | 0.027546 | 0.040 | PASS |
| P99.9 normalized absolute drift | 0.041373 | 0.060 | PASS |
| Maximum normalized absolute drift | 0.058695 | 0.080 | PASS |

All configured quantization-drift guards passed.

## Per-target drift

| Target | Mean normalized drift | P95 | P99 | P99.9 | Maximum |
|---|---:|---:|---:|---:|---:|
| Estimated takeoff mass | 0.010334 | 0.024061 | 0.036256 | 0.054880 | 0.058695 |
| Mission energy | 0.006909 | 0.016401 | 0.021749 | 0.028976 | 0.030003 |
| Energy per passenger-km | 0.008195 | 0.019944 | 0.027552 | 0.034810 | 0.039312 |
| Lifecycle-emissions proxy | 0.007653 | 0.019332 | 0.026186 | 0.032593 | 0.033258 |
| Operating-cost proxy | 0.007199 | 0.018210 | 0.023455 | 0.030821 | 0.032798 |
| Noise proxy | 0.007877 | 0.019796 | 0.025629 | 0.028227 | 0.030175 |

Estimated takeoff mass remains the target with the largest observed maximum
normalized drift, but it stays below the configured maximum regression ceiling.

## Physical-unit drift

Maximum observed physical-unit differences include:

| Target | Mean absolute difference | Maximum absolute difference |
|---|---:|---:|
| Estimated takeoff mass | 77.159 kg | 438.250 kg |
| Mission energy | 36.245 kWh | 157.387 kWh |
| Energy per passenger-km | 0.000331 kWh | 0.001588 kWh |
| Lifecycle-emissions proxy | 11.151 kgCO2e | 48.463 kgCO2e |
| Operating-cost proxy | 5.018 USD | 22.861 USD |
| Noise proxy | 0.0253 dB | 0.0968 dB |

These values describe quantization-induced prediction differences between the
FP32 and mixed-precision models rather than total surrogate prediction error.

## Predictive quality

The FP32 and mixed-precision models were both evaluated against held-out test
targets.

| Metric | FP32 reference | Mixed INT8/FP32 |
|---|---:|---:|
| Mean NRMSE | 0.050433 | 0.051566 |
| Mean R² | 0.996955 | 0.996855 |

The mixed-precision model increases mean NRMSE by approximately 2.25% relative
to FP32 while retaining mean R² above 0.9968.

It therefore preserves strong predictive quality while reducing serialized
model size.

## Per-target predictive metrics

| Target | FP32 NRMSE | INT8 NRMSE | FP32 R² | INT8 R² |
|---|---:|---:|---:|---:|
| Estimated takeoff mass | 0.073473 | 0.074670 | 0.994602 | 0.994424 |
| Mission energy | 0.048346 | 0.048787 | 0.997663 | 0.997620 |
| Energy per passenger-km | 0.083380 | 0.084665 | 0.993048 | 0.992832 |
| Lifecycle-emissions proxy | 0.035690 | 0.036422 | 0.998726 | 0.998673 |
| Operating-cost proxy | 0.045074 | 0.044985 | 0.997968 | 0.997976 |
| Noise proxy | 0.016633 | 0.019869 | 0.999723 | 0.999605 |

## CPU latency methodology

The production runtime benchmark compares FP32 ONNX Runtime CPU execution with
mixed INT8/FP32 ONNX Runtime CPU execution.

The methodology uses:

```text
Provider     CPUExecutionProvider
Batch sizes  1, 32, 256
Runs         5
Repeats      500
Warmups      50
Test rows    900
```

Additional controls:

- the same frozen model inputs are used for both precision paths;
- preprocessing occurs outside the timed region;
- session construction occurs outside the timed region;
- each precision path is repeatedly measured;
- the paired ratio is reported as FP32 latency divided by INT8 latency.

A ratio greater than 1 therefore means INT8 is faster.

## Production latency results

| Batch | FP32 median | INT8 median | Median FP32/INT8 ratio | Ratio range | INT8 faster runs |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.004613 ms | 0.005370 ms | 0.859× | 0.543×–0.875× | 0/5 |
| 32 | 0.008414 ms | 0.008715 ms | 0.973× | 0.900×–0.991× | 0/5 |
| 256 | 0.037539 ms | 0.030682 ms | 1.232× | 1.035×–1.752× | 5/5 |

## Latency interpretation

### Batch 1

Mixed INT8/FP32 inference is slower than FP32.

The median INT8 latency is approximately 16.4% higher than the FP32 median.

For this small batch, quantization/dequantization overhead outweighs the
reduced-precision compute benefit.

### Batch 32

The two paths are close, but FP32 remains slightly faster.

The median INT8 latency is approximately 3.6% higher than the FP32 median.

This should be described as near parity with a small FP32 advantage on the
tested machine.

### Batch 256

Mixed INT8/FP32 inference is consistently faster.

The mixed-precision model is faster in all five repeated runs.

The median latency decreases from approximately 0.037539 ms to 0.030682 ms,
corresponding to approximately 18.3% lower median latency.

One FP32 run showed a larger latency excursion and produced the maximum
1.752× paired ratio. The median result is therefore the more defensible summary
than the maximum observed speedup.

## Deployment conclusion

The production mixed INT8/FP32 configuration provides a clear model-size
benefit while preserving strong predictive performance.

The measured tradeoff is workload dependent:

- serialized ONNX size decreases by approximately 33.2%;
- mean NRMSE changes from approximately 0.05043 to 0.05157;
- mean R² remains approximately 0.99686;
- batch-1 CPU latency is worse;
- batch-32 CPU latency is close to parity but slightly worse;
- batch-256 CPU latency improves consistently, with approximately 18% lower
  median latency.

INT8 should therefore not be described as a universal latency optimization.

The strongest deployment case in the current experiment is a
memory-constrained or larger-batch ONNX Runtime CPU workload.

## CLI usage

### Export

```bash
edgegenbench export-neural-int8   --fp32-model artifacts/neural_onnx/neural_surrogate.onnx   --dataset data/raw/edgegenbench_v0_1.csv   --preprocessing artifacts/neural_surrogate/preprocessing.npz   --output-dir artifacts/neural_int8
```

### Benchmark

```bash
edgegenbench benchmark-neural-int8   --dataset data/raw/edgegenbench_v0_1.csv   --preprocessing artifacts/neural_surrogate/preprocessing.npz   --fp32-model artifacts/neural_onnx/neural_surrogate.onnx   --int8-model artifacts/neural_int8/neural_surrogate_int8.onnx   --output-dir artifacts/neural_int8_benchmark   --runs 5   --repeats 500   --warmups 50   --max-mean-normalized-drift 0.015   --max-p99-normalized-drift 0.040   --max-p999-normalized-drift 0.060   --max-normalized-drift 0.080
```

## Generated benchmark artifacts

```text
artifacts/neural_int8_benchmark/
├── equivalence.csv
├── task_metrics.csv
├── latency_runs.csv
├── latency_summary.csv
└── summary.json
```

Generated model and benchmark artifacts are reproducible and intentionally
excluded from source control.

## Automated validation

The INT8 implementation is covered by:

- calibration-reader tests;
- training-only calibration tests;
- static-QDQ export tests;
- graph-structure validation;
- integer-initializer validation;
- FP32 output-head retention checks;
- dynamic-batch execution tests;
- drift-statistic tests;
- predictive-quality tests;
- paired latency aggregation tests;
- benchmark-output tests;
- CLI command registration tests;
- parser-level CLI option tests;
- targeted mypy checks.

Local validation includes:

```bash
ruff format --check .
ruff check .

mypy src/edgegenbench/deployment/neural_int8.py
mypy src/edgegenbench/deployment/neural_int8_benchmark.py
mypy src/edgegenbench/cli.py

pytest -q tests/neural
pytest -q

python -m pip check
git diff --check
```

## Limitations

- The aircraft-design dataset is synthetic.
- The model is very small, so microsecond-scale latency measurements are
  sensitive to operating-system and runtime noise.
- CPU results are specific to the tested ARM64 macOS environment.
- The INT8 graph retains an FP32 final output head and should not be described
  as fully INT8.
- The current benchmark does not measure energy consumption.
- The current benchmark does not validate Qualcomm QNN execution.
- The current benchmark does not validate Snapdragon NPU execution.
- FP16 and INT8 latency results use different execution-provider paths and
  should not be treated as a direct precision-only comparison.
- Distribution-shift and extrapolation behavior remain outside the current
  deployment study.

## Next milestone

The next logical deployment milestone is a unified FP32 / FP16 / mixed-INT8
selection layer that consumes measured accuracy, size, provider, and latency
metadata and selects candidates according to explicit deployment constraints.

Qualcomm AI Hub / QNN integration and supported-device Snapdragon profiling
remain subsequent hardware-specific milestones.
