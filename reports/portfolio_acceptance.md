# EdgeGenBench portfolio acceptance

| Evidence lane | Status | Claim boundary |
|---|---|---|
| `native_cpp` | `validated_in_ci` | C++17 reference runtime, tests, CLI, and fused preprocessing acceptance. |
| `android_reference` | `validated_physical_device` | Reference JNI/application measurements; not QNN. |
| `qualcomm_ai_hub_qnn` | `tracked_ai_hub_report_model_provenance_mismatch` | Physical AI Hub model profiling; not Android APK end-to-end latency. |
| `android_qnn_apk` | `implementation_complete_evidence_pending` | Build/JNI/capture paths exist; requires a supported Snapdragon APK run. |
| `android_16kb_runtime` | `packaging_validated_runtime_pending` | ELF/APK alignment passes; runtime PAGE_SIZE=16384 evidence is pending. |
| `power` | `not_measured` | No power-savings claim is made without a named calibrated tool. |

## Validated Qualcomm QNN results

Device: **Snapdragon 8 Elite QRD**; backend: **QNN HTP**; QAIRT: `2.45.0.260326154327`.
Source-model provenance match: **False**.

| Batch | AI Hub latency (ms) | Throughput (samples/s) | Peak memory (bytes) | Placement | Max normalized drift |
|---:|---:|---:|---:|---|---:|
| 1 | 0.038000 | 26315.789 | 122822656 | NPU × 9 | 0.003635877 |
| 32 | 0.034000 | 941176.471 | 122888192 | NPU × 9 | 0.003222448 |
| 256 | 0.057000 | 4491228.070 | 123211776 | NPU × 9 | 0.003222448 |

AI Hub measurements are physical-device model profiles, not Android application end-to-end timings. They cannot be promoted as current-model acceptance until the reported model artifact is restored or the current model is re-profiled.
Power remains unmeasured. The two pending proof items are a supported-device QNN APK run and a runtime page size of 16384 bytes.
