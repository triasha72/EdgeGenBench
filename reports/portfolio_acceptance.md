# EdgeGenBench portfolio acceptance

| Evidence lane | Status | Claim boundary |
|---|---|---|
| `native_cpp` | `validated_in_ci` | C++17 reference runtime, tests, CLI, and fused preprocessing acceptance. |
| `android_reference` | `validated_physical_device` | Reference JNI/application measurements; not QNN. |
| `qualcomm_ai_hub_qnn` | `validated_ai_hub_physical_qnn` | Physical AI Hub model profiling; not Android APK end-to-end latency. |
| `android_qnn_apk` | `implementation_complete_evidence_pending` | Build/JNI/capture paths exist; requires a supported Snapdragon APK run. |
| `android_16kb_runtime` | `packaging_validated_runtime_pending` | ELF/APK alignment passes; runtime PAGE_SIZE=16384 evidence is pending. |
| `power` | `not_measured` | No power-savings claim is made without a named calibrated tool. |

## Validated Qualcomm QNN results

Device: **Snapdragon 8 Elite QRD**; backend: **QNN HTP**; QAIRT: `2.45.0.260326154327`.
Source-model provenance match: **True**.

| Batch | AI Hub latency (ms) | Throughput (samples/s) | Peak memory (bytes) | Placement | Max normalized drift |
|---:|---:|---:|---:|---|---:|
| 1 | 0.038000 | 26315.789 | 122855424 | NPU × 9 | 0.003233890 |
| 32 | 0.040000 | 800000.000 | 122880000 | NPU × 9 | 0.002865936 |
| 256 | 0.047000 | 5446808.511 | 122896384 | NPU × 9 | 0.002865936 |

AI Hub measurements are physical-device model profiles, not Android application end-to-end timings. Current-model acceptance requires source-model provenance to match the repository, as reported above.
Power remains unmeasured. The two pending proof items are a supported-device QNN APK run and a runtime page size of 16384 bytes.
