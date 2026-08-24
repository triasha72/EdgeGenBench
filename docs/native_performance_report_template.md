# Native/device performance report

| Field | Value |
|---|---|
| Git commit | TODO |
| Model SHA-256 | TODO |
| Device / SoC / Android build | TODO |
| ORT / QAIRT / QNN versions | TODO |
| Backend and provider options | TODO |
| CPU fallback disabled | TODO (attach placement log) |
| Context binary SHA-256 | TODO |

| Path | Cold ms | Warm p50 ms | Warm p95 ms | Throughput/s | Peak RSS MiB | Accuracy/drift | Thermal start/end °C |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline FP32 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Fused FP32 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| QNN | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

Power: **not measured** until the report names the external or calibrated device
measurement tool, sample rate, rail scope, idle subtraction, and test duration.

Interpretation must distinguish preprocessing time, inference time, end-to-end
time, context creation, and thermal steady state. Report failed/fallback runs.
