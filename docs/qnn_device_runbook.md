# QNN device verification runbook

This workflow is evidence collection, not a claim that QNN has already run.
Use the exact ONNX artifact for both CPU/reference validation and QNN.

## Preconditions

- Qualcomm device supported by the QAIRT/QNN SDK
- ONNX Runtime Android build with the QNN execution provider
- `libQnnHtp.so`, matching stub/system libraries, and device permissions
- `adb`, Android NDK, and a pinned model checksum

## Fail-closed rules

1. Create the session with only `QNNExecutionProvider`; do not append CPU EP.
2. Set `session.disable_cpu_ep_fallback=1` and `session.disable_cpu_ep_fallback=true`
   where supported by the pinned ORT release.
3. Enable verbose ORT/QNN profiling and save the complete logcat output.
4. Treat any unassigned node, CPU provider line, or context-cache failure as a
   failed run—not as a partially accelerated result.

## Evidence bundle

Record the model SHA-256, ORT/QAIRT versions, SoC/device build fingerprint,
provider options, context binary SHA-256, per-node placement, cold first-run
latency, warm p50/p95, peak RSS, throughput, battery temperature at start/end,
and output drift versus the FP32 host reference. A context binary proves cached
QNN compilation, not NPU placement by itself; retain the placement log too.

```bash
adb shell getprop ro.build.fingerprint > reports/device/build-fingerprint.txt
adb shell getprop ro.soc.model > reports/device/soc.txt
adb logcat -c
adb shell am force-stop dev.edgegenbench
adb shell am start -W dev.edgegenbench/.MainActivity
adb logcat -d -v threadtime > reports/device/logcat.txt
adb shell dumpsys meminfo dev.edgegenbench > reports/device/meminfo.txt
sha256sum models/model.onnx reports/device/qnn_context.bin > reports/device/checksums.txt
```

Power savings must remain `not measured` unless a Monsoon, Otii, Trepn-supported
rail, or another named calibrated measurement path is used. Battery percentage
is not a power measurement.
