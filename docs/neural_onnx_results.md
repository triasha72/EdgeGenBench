# EdgeGenBench Neural FP32 ONNX Deployment Results

This document records the validated FP32 neural deployment milestone following
the EdgeGenBench v0.2 compact PyTorch surrogate release.

## Scope

The FP32 deployment milestone adds:

- trained-checkpoint reconstruction;
- PyTorch-to-ONNX export;
- dynamic ONNX batch dimensions;
- graph validation;
- ONNX Runtime CPU execution;
- frozen preprocessing reuse;
- normalized numerical-equivalence testing;
- physical-unit numerical-equivalence testing;
- paired PyTorch CPU versus ONNX Runtime CPU benchmarking;
- repeated runtime measurements;
- corrected microbenchmark methodology;
- public FP32 neural ONNX export and benchmark CLI commands.

## Environment

Reference local deployment environment:

| Component | Value |
|---|---|
| Python | 3.12 |
| PyTorch | 2.13.0 |
| ONNX Runtime | 1.28.0 |
| ONNX provider | CPUExecutionProvider |
| PyTorch CPU threads | 4 |
| Development platform | ARM64 macOS |

Runtime measurements are hardware- and environment-specific.

## Model

```text
10 → 64 → 32 → 16 → 6
```

The network contains 3,414 trainable parameters.

The exported graph interface is:

```text
features    [batch, 10]
     |
     v
FP32 ONNX neural surrogate
     |
     v
predictions [batch, 6]
```

The batch dimension is dynamic.

## ONNX export validation

| Property | Result |
|---|---:|
| ONNX checker | PASS |
| Opset | 18 |
| Input dimension | 10 |
| Output dimension | 6 |
| Dynamic batch | Yes |
| PyTorch checkpoint size | 16,881 bytes |
| ONNX graph size | 25,420 bytes |

The PyTorch checkpoint and ONNX graph use different serialization formats, so
their file sizes are not direct parameter-memory equivalents.

## Held-out numerical equivalence

Equivalence was evaluated on all 900 held-out test rows.

| Metric | Result |
|---|---:|
| Test rows | 900 |
| Mean normalized absolute difference | 1.3064681070e-07 |
| Maximum normalized absolute difference | 9.5367431641e-07 |
| `rtol` | 1e-5 |
| `atol` | 1e-5 |
| Equivalent | **True** |

### Physical-unit differences

| Target | Mean absolute difference | Maximum absolute difference | Maximum reference-relative difference |
|---|---:|---:|---:|
| Estimated takeoff mass | 0.00110460 kg | 0.00781250 kg | 1.151e-07 |
| Mission energy | 0.00060981 kWh | 0.00390625 kWh | 1.347e-07 |
| Energy per passenger-km | 5.133e-09 | 5.960e-08 | 1.697e-07 |
| Lifecycle-emissions proxy | 0.00018677 | 0.00146484 | 2.068e-07 |
| Operating-cost proxy | 0.00008070 USD | 0.00048828 USD | 1.170e-07 |
| Noise proxy | 3.137e-07 dB | 7.629e-06 dB | 8.260e-08 |

Runtime-conversion differences are negligible relative to the trained
surrogate's predictive error.

## Corrected CPU benchmark methodology

A timing-methodology audit found that an earlier implementation entered and
exited `torch.no_grad()` inside each individually timed PyTorch inference.

For a network whose inference latency is measured in tens of microseconds, that
per-call Python/context-manager overhead can materially influence the reported
runtime ratio.

The benchmark was corrected so that:

```python
def pytorch_operation():
    return pytorch_model(batch)


with torch.inference_mode():
    measure_latency(pytorch_operation)
```

The inference context is therefore created once around the warmup and timed
loop rather than inside every measured forward pass.

Other methodology remains unchanged:

- identical trained weights;
- identical transformed FP32 inputs;
- PyTorch CPU;
- ONNX Runtime `CPUExecutionProvider`;
- preprocessing outside the timed region;
- runtime/model construction outside the timed region;
- 50 warmups;
- 500 measured repetitions;
- three independent benchmark runs.

## Corrected three-run results

Aggregate results:

| Batch | Median PyTorch latency | Median ORT latency | Median PyTorch/ORT ratio | Ratio range |
|---:|---:|---:|---:|---:|
| 1 | 0.018341172 ms | 0.006156959 ms | **2.978933×** | 2.715243×–3.966734× |
| 32 | 0.018427053 ms | 0.007984216 ms | **2.384702×** | 2.077892×–2.403927× |
| 256 | 0.030840994 ms | 0.033573637 ms | **0.918607×** | 0.860143×–1.079221× |

A PyTorch/ORT ratio greater than 1 means ONNX Runtime had lower mean latency.

## Interpretation

The corrected experiment establishes:

1. the compact FP32 PyTorch surrogate converts to a valid dynamic-batch ONNX
   graph;
2. predictions are preserved on all 900 held-out test rows;
3. frozen training preprocessing can be reused for deployment;
4. physical-unit conversion differences remain negligible;
5. ONNX Runtime has a clear local latency advantage at batch size 1;
6. ONNX Runtime has a clear local latency advantage at batch size 32;
7. batch 256 is approximately parity and changes direction across repeated
   runs;
8. no universal ONNX Runtime latency advantage should be claimed from these
   measurements.

This correction is important because it removes Python inference-context
overhead from the timed PyTorch operation.

## CLI workflow

Export:

```bash
edgegenbench export-neural-onnx \
  --model artifacts/neural_surrogate/model.pt \
  --preprocessing artifacts/neural_surrogate/preprocessing.npz \
  --output-dir artifacts/neural_onnx \
  --opset 18
```

Benchmark:

```bash
edgegenbench benchmark-neural-onnx \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --model artifacts/neural_surrogate/model.pt \
  --preprocessing artifacts/neural_surrogate/preprocessing.npz \
  --onnx-model artifacts/neural_onnx/neural_surrogate.onnx \
  --metadata artifacts/neural_onnx/metadata.json \
  --output-dir artifacts/neural_onnx_benchmark \
  --repeats 500 \
  --warmups 50
```

For repeatability-sensitive performance reporting, execute the benchmark
multiple times and report median/range rather than one invocation.

## Generated artifacts

```text
artifacts/neural_onnx/
├── metadata.json
└── neural_surrogate.onnx

artifacts/neural_onnx_benchmark/
├── equivalence.csv
├── latency.csv
└── summary.json
```

Corrected multi-run aggregate results may be stored separately under ignored
local benchmark-artifact directories.

## Automated validation

The FP32 deployment implementation is covered by:

- checkpoint reconstruction tests;
- ONNX export tests;
- dynamic batch tests;
- graph validation;
- ONNX Runtime inference tests;
- normalized parity tests;
- physical-unit parity tests;
- benchmark-generation tests;
- benchmark-schema tests;
- CLI registration tests.

## Relationship to FP16 work

The validated FP32 graph is the reference artifact for the subsequent FP16
deployment study.

See:

[`docs/neural_fp16_results.md`](neural_fp16_results.md)

for the reduced-precision evaluation.

## Limitations

- Results use a synthetic aircraft-design benchmark.
- Runtime measurements are machine-specific.
- Microsecond-scale latency is sensitive to operating-system scheduling and
  runtime state.
- PyTorch and ONNX serialized sizes are not directly equivalent.
- CPU benchmark results should not be generalized to other hardware or batch
  regimes.
- EdgeGenBench is not a certified aircraft-design or safety-critical system.
