# Android device evidence: Samsung SM-A356E, reference backend

Captured on 2026-08-25 at 16:52:08 UTC from EdgeGenBench Android v0.1.3
(version code 4).

## Provenance and scope

| Field | Observed value |
|---|---|
| Manufacturer / model | Samsung SM-A356E (`a35x`) |
| Android / SDK | Android 16 / SDK 36 |
| ABI | `arm64-v8a` |
| Runtime page size | 4,096 bytes |
| App version | 0.1.3 (code 4) |
| Backend | Deterministic reference backend, **not QNN** |
| QNN/NPU placement | Not tested or established |
| Power | Not measured |

The APK passed CI checks for 16 KiB ZIP alignment and ELF `PT_LOAD` alignment,
but this phone runs with 4 KiB pages. This device capture therefore does not
claim execution on a 16 KiB-page device.

## Benchmark result

The user launched the Kotlin activity, crossed the JNI boundary, and ran the
native reference benchmark with 10 warmup iterations followed by 100 measured
iterations.

| Metric | Result |
|---|---:|
| Cold latency | 0.013923 ms |
| Warm mean latency | 0.003465 ms |
| Warm p95 latency | 0.003654 ms |
| Measured iterations | 100 |
| Output value | 0.156160 |

These are timings for the repository's small deterministic reference workload,
not an ONNX model, end-to-end application workload, or NPU performance result.
`CPU fallback=false` in the log means that this explicit reference session did
not fall back from another provider; it is not evidence that CPU execution was
disabled for a QNN graph.

## Memory snapshot

`dumpsys meminfo dev.edgegenbench` was captured after the run.

| Metric | Result |
|---|---:|
| Total PSS | 76,631 KiB (74.84 MiB) |
| Total RSS | 146,858 KiB (143.42 MiB) |
| Total swap PSS | 13,985 KiB (13.66 MiB) |
| Java heap PSS | 5,924 KiB |
| Native heap PSS | 9,740 KiB |
| Graphics PSS | 18,454 KiB |

This is a single post-run process snapshot, not a sampled peak-memory trace.

## Thermal snapshot

Android reported thermal status 0 (no throttling) at capture time. Reported
temperatures included AP 39.6 °C, PA 38.9 °C, skin 37.3 °C, battery 33.4 °C,
and USB 33.2 °C. These are one-time platform sensor readings; without pre-run
and repeated-run samples they do not establish thermal behavior or savings.

## Evidence assessment

Verified by the retained capture:

- Android 16 installation and launch on a physical arm64 Samsung device;
- visible Kotlin UI with correct system-bar inset handling;
- JNI/native benchmark completion and consistent displayed/logged values;
- one post-run memory snapshot and one thermal-status snapshot;
- explicit disclosure that power was not measured.

Still required for direct QNN/runtime proof:

- run the canonical ONNX model through `QNNExecutionProvider` or QAIRT/QNN;
- set `session.disable_cpu_ep_fallback=1` and retain provider/session options;
- retain an operator-placement/profile report proving full HTP/NPU assignment;
- retain the QNN context binary and its SHA-256;
- repeat device runs for latency distribution, sampled peak memory, and thermal
  trend; use a real measurement tool before making any power claim;
- repeat compatibility validation on a device configured with a 16 KiB runtime
  page size.

The received evidence bundle is integrity-indexed in
[`reports/device/sm-a356e-v0.1.3/source-evidence-sha256.txt`](device/sm-a356e-v0.1.3/source-evidence-sha256.txt).
The device properties and EdgeGenBench-tagged logcat are retained alongside
that index. Memory and thermal values above were transcribed from the indexed
source captures.
