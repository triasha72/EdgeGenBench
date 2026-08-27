# Android 16 KB runtime evidence: API 35 ARM64 emulator

Captured on 2026-08-27 from EdgeGenBench Android v0.1.7 (version code 8)
using `scripts/profile_android_reference.sh` for 10 automated cold launches.

## Provenance and scope

| Field | Observed value |
|---|---|
| Environment | Google Android Emulator `sdk_gphone16k_arm64` |
| Android / SDK | Android 15 / SDK 35 |
| ABI | `arm64-v8a` |
| Runtime page size | **16,384 bytes** |
| App version | 0.1.7 (code 8) |
| Backend | Deterministic reference backend, **not QNN** |
| QNN/NPU placement | Not tested |
| Power | Not measured |

The runtime page size was read from the booted environment with
`adb shell getconf PAGE_SIZE`. Every retained machine-readable benchmark result
also records `runtime_page_size_bytes=16384`. Before installation, Android SDK
`zipalign -c -P 16 -v 4` passed and all `PT_LOAD` segments in both packaged
native libraries reported `0x4000` alignment.

This closes the Android 16 KB **runtime compatibility** gap for the reference
APK/JNI path. An emulator is not physical-device performance evidence, and
these timings must not be compared with the Samsung or Qualcomm measurements.

## Ten-run results

Each outer run force-stopped and cold-launched the activity. The native
benchmark then performed 100 measured iterations.

| Metric | Mean | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Native cold latency (ms) | 0.022621 | 0.014333 | 0.011500 | 0.060667 |
| Native warm mean (ms) | 0.002608 | 0.002440 | 0.002259 | 0.003924 |
| Native warm p95 (ms) | 0.003333 | 0.002667 | 0.002458 | 0.006250 |
| Baseline preprocessing (ms) | 26.361593 | 23.912975 | 21.260857 | 38.929105 |
| Fused preprocessing (ms) | 12.570465 | 12.233062 | 11.625028 | 14.820786 |
| Preprocessing speedup | 2.092613x | 1.951007x | 1.772588x | 2.852528x |
| Android launch total (ms) | 4032.9 | 3948.0 | 3190 | 5258 |
| Post-run PSS (KiB) | 60,623 | 60,698 | 60,233 | 60,853 |
| Post-run RSS (KiB) | 157,038 | 157,096 | 156,608 | 157,232 |

All ten runs reported zero preprocessing drift and zero downstream output
drift. PSS and RSS are post-run process snapshots, not sampled peaks.

## Thermal and power boundaries

The emulator reported thermal status 0 and a constant synthetic test sensor.
Those values are emulator diagnostics, not physical thermal measurements.
Power remains `not measured`; no energy or power-savings claim is made.

## Retained evidence

The complete capture is stored under
`reports/device/android-16kb-api35-reference-10-runs/` and includes:

- device/app identity and the observed runtime page size;
- ten activity-start, logcat, memory, and thermal records;
- the aggregate latency/memory CSV;
- pre/post thermal snapshots and the final screenshot;
- explicit reference-backend, QNN, memory, thermal, and power claim boundaries.

The remaining hardware-only gaps are an end-to-end QNN APK run on a supported
Snapdragon device and a calibrated physical power measurement.
