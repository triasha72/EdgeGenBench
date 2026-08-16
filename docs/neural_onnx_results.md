# EdgeGenBench Neural ONNX Deployment Results

This document records the FP32 neural deployment milestone that follows the
EdgeGenBench v0.2 compact PyTorch surrogate release.

## Scope

The deployment milestone adds:

- reconstruction of the trained neural surrogate from checkpoint metadata;
- PyTorch-to-ONNX export;
- dynamic ONNX batch dimensions;
- ONNX graph validation;
- ONNX Runtime CPU execution;
- frozen preprocessing reuse;
- normalized numerical-equivalence testing;
- physical-unit numerical-equivalence testing;
- paired PyTorch CPU versus ONNX Runtime CPU benchmarking;
- repeated runtime measurements;
- public neural ONNX export and benchmark CLI commands.

FP16, INT8, Qualcomm QNN, and Snapdragon NPU deployment remain future work.

## Environment

| Component | Value |
|---|---|
| Python | 3.12 |
| PyTorch | 2.13.0 |
| ONNX | 1.22.0 |
| ONNX Script | 0.7.1 |
| ONNX Runtime | 1.28.0 |
| ONNX provider | CPUExecutionProvider |
| PyTorch CPU threads | 4 |
| Development platform | ARM64 macOS |

All runtime measurements are hardware- and environment-specific.

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
| ONNX/PyTorch serialized-size ratio | 1.506× |

The PyTorch checkpoint and ONNX graph are different serialization formats, so
the file-size ratio is not a direct parameter-memory comparison.

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

## Reference paired CPU benchmark

The first reference measurement used:

- identical trained weights;
- identical transformed FP32 inputs;
- PyTorch CPU;
- ONNX Runtime `CPUExecutionProvider`;
- 500 measured repetitions;
- 50 warmup iterations;
- preprocessing outside the timed region.

Reference measurement:

| Batch | PyTorch mean | PyTorch P95 | ORT mean | ORT P95 | PyTorch / ORT |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.026470 ms | 0.036541 ms | 0.006706 ms | 0.006833 ms | 3.947× |
| 32 | 0.031845 ms | 0.042877 ms | 0.012230 ms | 0.012666 ms | 2.604× |
| 256 | 0.047511 ms | 0.058212 ms | 0.049818 ms | 0.068117 ms | 0.954× |

Because absolute timings are in the microsecond regime, a repeatability study
was then performed before making a final runtime claim.

## Three-run repeatability experiment

| Run | Batch 1 PyTorch/ORT | Batch 32 PyTorch/ORT | Batch 256 PyTorch/ORT |
|---:|---:|---:|---:|
| 1 | 2.868× | 2.958× | 1.659× |
| 2 | 3.768× | 2.818× | 1.788× |
| 3 | 3.481× | 2.872× | 1.303× |

Median absolute timings:

| Batch | Median PyTorch latency | Median ORT latency | Median ratio |
|---:|---:|---:|---:|
| 1 | 0.051814 ms | 0.014886 ms | **3.481×** |
| 32 | 0.065456 ms | 0.022665 ms | **2.872×** |
| 256 | 0.103179 ms | 0.060500 ms | **1.659×** |

ONNX Runtime produced lower mean latency than PyTorch in every one of the three
repeat runs at all tested batch sizes.

Absolute timing varied across runs, which is expected for very small
microsecond-scale workloads. Therefore, the defensible result is the repeated
directional advantage and batch-dependent ratio, not a universal latency value.

## Interpretation

The experiment establishes that:

1. the compact FP32 PyTorch surrogate converts to a dynamic-batch ONNX graph;
2. the exported graph preserves predictions on all 900 held-out rows;
3. frozen training preprocessing can be reused safely for deployment;
4. physical-unit output differences remain negligible;
5. ONNX Runtime reduced local CPU mean latency in all three repeated runs for
   batch sizes 1, 32, and 256;
6. runtime advantages are workload- and machine-dependent.

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

Generated artifacts are intentionally ignored by Git.

## Automated validation

The deployment implementation is covered by:

- checkpoint reconstruction tests;
- ONNX export tests;
- dynamic batch tests;
- graph validation;
- ONNX Runtime inference tests;
- normalized parity tests;
- physical-unit parity tests;
- benchmark generation tests;
- benchmark schema tests;
- CLI registration tests.

The complete neural suite contains **27 passing tests** at this milestone.

## Next deployment milestones

```text
PyTorch FP32
    |
    v
ONNX FP32                     COMPLETE
    |
    v
FP16                          NEXT
    |
    v
INT8
    |
    v
reduced-precision comparison
    |
    v
Qualcomm QNN
    |
    v
Snapdragon NPU
```

## Limitations

- Results use a synthetic aircraft-design benchmark.
- Runtime measurements are machine-specific.
- Microsecond-scale latency is sensitive to OS scheduling and runtime state.
- PyTorch and ONNX serialized sizes are not directly equivalent.
- FP16 and INT8 conversion have not yet been validated.
- Qualcomm QNN and Snapdragon NPU deployment have not yet been validated.
- EdgeGenBench is not a certified aircraft-design or safety-critical system.
