# Android repeated-device evidence: Samsung SM-A356E

Captured on 2026-08-25 from EdgeGenBench Android v0.1.4 (version code 5)
using `scripts/profile_android_reference.sh` for 10 automated cold launches.

## Scope

| Field | Observed value |
|---|---|
| Device | Samsung SM-A356E |
| Android / SDK | Android 16 / SDK 36 |
| ABI | `arm64-v8a` |
| Runtime page size | 4,096 bytes |
| Backend | Deterministic reference backend, **not QNN** |
| QNN/NPU placement | Not tested |
| Power | Not measured |

The device uses 4 KiB runtime pages. CI verifies that the APK ZIP entries and
native ELF load segments are compatible with 16 KiB pages; this capture is not
a run on a 16 KiB-page device.

## Ten-run results

Each outer run force-stopped and cold-launched the activity. The in-app native
benchmark then performed 10 warmups and 100 measured reference iterations.

| Metric | Mean | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Native cold latency (ms) | 0.012573 | 0.009481 | 0.008923 | 0.033885 |
| Native warm mean (ms) | 0.006064 | 0.003201 | 0.003172 | 0.018636 |
| Native warm p95 (ms) | 0.004815 | 0.003520 | 0.003308 | 0.011616 |
| Android launch total (ms) | 809.3 | 819.0 | 595 | 967 |
| Post-run PSS (KiB) | 106,064 | 105,283 | 102,518 | 111,129 |
| Post-run RSS (KiB) | 196,890 | 195,989 | 192,265 | 203,022 |

All 10 retained logs name `backend=reference (NOT QNN)`, disclose
`power=not measured`, and produce the same output value, `0.156160`. The stable
output verifies deterministic behavior for this small synthetic reference
workload; it is not an accuracy comparison against an ONNX/QNN result.

The latency distribution includes visible outliers (for example, run 3 cold
latency and runs 3/9 warm means). With only 10 outer repetitions, the report
retains rather than removes them and presents both mean and median.

## Memory interpretation

PSS and RSS were collected with `dumpsys meminfo` after each completed run.
They are process snapshots, not sampled peak-memory measurements. The mean
post-run PSS was 103.58 MiB and mean post-run RSS was 192.28 MiB. These totals
include the Android UI, runtime, mapped code, graphics, and native workload;
they must not be presented as model-only memory.

## Thermal interpretation

Android thermal status remained 0, so the platform did not report throttling.
Snapshot changes across the sequence were:

| Sensor | Before | After | Change |
|---|---:|---:|---:|
| AP | 45.1 °C | 50.4 °C | +5.3 °C |
| PA | 37.8 °C | 41.0 °C | +3.2 °C |
| Skin | 37.1 °C | 38.9 °C | +1.8 °C |
| Battery | 32.4 °C | 32.4 °C | 0.0 °C |
| USB | 32.6 °C | 32.7 °C | +0.1 °C |

The sequence was not run in a controlled chamber, does not include an idle
control, and does not measure energy. These readings show a thermal change
during the capture but do not attribute that change solely to EdgeGenBench or
support a power-efficiency claim.

## Remaining proof gaps

- Execute the canonical ONNX model through QNN on supported Snapdragon
  hardware, with CPU fallback disabled and full HTP/NPU placement retained.
- Retain the QNN context binary, provider options, SDK/runtime versions, model
  hash, and placement/profile report.
- Collect sampled peak memory rather than post-run snapshots.
- Use a named, calibrated power tool before making energy or power claims.
- Run the APK on a device configured for a 16 KiB runtime page size.

The source CSV and capture metadata are retained under
`reports/device/sm-a356e-v0.1.4-reference-10-run/`. The checksum manifest also
indexes the external per-run logs, memory reports, thermal snapshots, and final
screenshot supplied with this capture.
