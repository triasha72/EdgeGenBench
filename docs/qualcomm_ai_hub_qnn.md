# Qualcomm AI Hub / QNN Deployment

## Status

COMPLETE ON MAIN

## Objective

EdgeGenBench extends its local ONNX Runtime and CoreML deployment studies to a
reproducible Qualcomm QNN deployment workflow on supported Snapdragon
hardware.

The workflow validates:

- device-specific QNN compilation;
- HTP target metadata;
- accelerator placement;
- profile latency and memory;
- serialized QNN artifact provenance;
- numerical deployment drift;
- held-out predictive quality;
- multi-batch deployment through one linked QNN Context Binary.

## Hardware target

| Property | Value |
|---|---|
| Device | Snapdragon 8 Elite QRD |
| OS | Android 15 |
| Chipset | Snapdragon 8 Elite |
| Chipset alias | sm8750 |
| SoC model | 69 |
| Backend | HTP |
| Hexagon | v79 |
| QAIRT | 2.45.0.260326154327 |

## Source model

The source artifact is:

`artifacts/neural_onnx/neural_surrogate.onnx`

Interface:

| Property | Value |
|---|---|
| Input | `features` |
| Input shape | `[batch, 10]` |
| Output | `predictions` |
| Output shape | `[batch, 6]` |
| External precision | float32 |
| Dynamic batch | yes |

Source SHA-256:

`40f588b329b98fdaa38a7eda202fc89573fce5e91a4193f0a0a93b3142c0382f`

## Precision terminology

The QNN deployment preserves float32 external model I/O while the HTP graph
uses relaxed FP16 precision internally.

The validated baseline is therefore described as:

`FP32 I/O / QNN HTP FP16-relaxed`

It should not be described as a pure FP32 NPU model.

## Initial batch-specific QNN baseline

Before linking, independent QNN Context Binaries were validated for the three
deployment batch sizes.

| Batch | AI Hub profile latency | QNN binary size | Peak memory | Compute units |
|---:|---:|---:|---:|---|
| 1 | 32 us | 53,248 B | 123,199,488 B | NPU: 9 |
| 32 | 35 us | 53,248 B | 122,978,304 B | NPU: 9 |
| 256 | 54 us | 57,344 B | 122,892,288 B | NPU: 9 |

These experiments established the initial Snapdragon deployment baseline.

## Linked multi-graph QNN deployment

The permanent deployment path uses a single linked QNN Context Binary
containing three statically specialized graph variants:

- `edgegenbench_batch1`
- `edgegenbench_batch32`
- `edgegenbench_batch256`

Linked target model:

`mnjjex79n`

Link job:

`jgk4d8lvp`

The linked artifact retains:

- SoC model 69;
- HTP backend;
- Hexagon v79;
- QAIRT 2.45.0.260326154327.

### Linked graph profiles

| Graph | Batch | AI Hub profile latency | Derived model throughput | Peak memory | Compute units |
|---|---:|---:|---:|---:|---|
| `edgegenbench_batch1` | 1 | 38 us | 26,315.8 samples/s | 122,855,424 B | NPU: 9 |
| `edgegenbench_batch32` | 32 | 40 us | 800,000.0 samples/s | 122,880,000 B | NPU: 9 |
| `edgegenbench_batch256` | 256 | 47 us | 5,446,808.5 samples/s | 122,896,384 B | NPU: 9 |

The throughput values are derived from the configured batch size divided by
AI Hub's estimated model-inference time. They are not end-to-end Android
application throughput measurements.

All three linked graphs placed all nine profiled layers on the NPU.

## Held-out numerical validation

### Batch 1

All 900 held-out test rows were evaluated through the linked batch-1 graph.

| Metric | Local FP32 ONNX | Linked Snapdragon QNN |
|---|---:|---:|
| Mean R2 | 0.996955989 | 0.996954462 |
| Mean NRMSE | 0.050425306 | 0.050449587 |

Deployment drift:

| Metric | Value |
|---|---:|
| MAE | 0.000409681 |
| RMSE | 0.000530486 |
| Maximum absolute error | 0.003739834 |
| Mean normalized drift | 0.000411944 |
| Maximum normalized drift | 0.003635877 |

### Batch 32

The first 256 held-out rows were evaluated as eight batch-32 inference
entries.

| Metric | Local FP32 ONNX | Linked Snapdragon QNN |
|---|---:|---:|
| Mean R2 | 0.997124157 | 0.997120062 |
| Mean NRMSE | 0.048953505 | 0.048983762 |

Mean normalized deployment drift was approximately 0.000407800 and maximum
normalized drift was approximately 0.003222448.

### Batch 256

The same first 256 held-out rows were evaluated as one batch-256 inference
entry.

| Metric | Local FP32 ONNX | Linked Snapdragon QNN |
|---|---:|---:|
| Mean R2 | 0.997124157 | 0.997120062 |
| Mean NRMSE | 0.048953505 | 0.048983762 |

The batch-256 graph reproduced the batch-32 linked predictions for the same
256-row evaluation subset.

## Acceptance semantics

Strict elementwise `numpy.allclose` with `rtol=1e-3` and `atol=1e-3` is
retained as a diagnostic.

It is not used as the sole deployment acceptance condition.

EdgeGenBench instead reports:

- held-out R2;
- held-out NRMSE;
- R2 delta;
- NRMSE delta;
- MAE;
- RMSE;
- mean normalized drift;
- maximum normalized drift.

This prevents one isolated tensor element from overriding model-level
predictive-quality evidence.

## Reproducibility

Install Qualcomm-specific dependencies with:

```bash
pip install -e '.[qualcomm]'
```

Compile, link, profile, and validate the current ONNX model:

```bash
PYTHONPATH=src python scripts/rerun_qualcomm_qnn_current_model.py
```

Build the canonical evidence report:

```bash
python scripts/build_portfolio_acceptance.py
```

The complete machine-readable result is stored in:

`reports/qualcomm_qnn_v0_1.json`

## Credential boundary

The Qualcomm AI Hub token is configured outside the repository.

No token is stored in:

- source files;
- tests;
- documentation;
- reports;
- GitHub Actions configuration.

Ordinary CI does not make live Qualcomm AI Hub calls.

## Measurement boundaries

The project does not claim that:

- AI Hub profile latency is end-to-end Android application latency;
- derived model throughput is end-to-end application throughput;
- runtime peak memory equals serialized model size;
- Snapdragon NPU timings are directly comparable with Mac CPU/CoreML timings
  as a same-hardware benchmark;
- current measurements establish power or energy efficiency;
- current results generalize to every Snapdragon SoC;
- the current deployment is Qualcomm-native INT8.

## Qualcomm-native INT8/QDQ follow-up

The planned INT8 study was completed with 4,200 training-only calibration rows
and the frozen 900-row test partition. The mixed INT8/FP32 ONNX artifact was
33.56% smaller than its regenerated FP32 source and passed every local drift
guard before submission.

Separate QNN Context Binaries were compiled and measured on the Snapdragon 8
Elite QRD:

| Batch | Compile job | Profile job | Inference job | Latency | Peak memory | Placement |
|---:|---|---|---|---:|---:|---|
| 1 | `jp8x7v28g` | `jgk4ymvop` | `j5671l27p` | 43 us | 122,986,496 B | NPU: 9 |
| 32 | `jgj7l4d8g` | `jp1j8x2lp` | `jgd3vlnlp` | 37 us | 123,002,880 B | NPU: 9 |
| 256 | `j5w79nwjg` | `j574d30r5` | `jp41w0klp` | 42 us | 123,101,184 B | NPU: 9 |

Batch 1 evaluated all 900 held-out rows. Batches 32 and 256 evaluated the same
first 256 held-out rows. Batch-1 remote quality was mean R² `0.996786` and mean
NRMSE `0.052115`. Mean normalized deployment drift was `0.004516`, while the
maximum was `0.036602`.

The candidate therefore failed the preregistered maximum normalized drift
limit of `0.01`. The `assess_qualcomm_int8_candidate` gate rejected it without
changing the limit after observing results. The FP32-I/O / QNN HTP
FP16-relaxed deployment remains selected.

The complete machine-readable evidence, including source and quantized hashes,
target model IDs, profiles, parity, predictive quality, and rejection reason,
is stored in `reports/qualcomm_int8_qnn_v0_1.json`.
