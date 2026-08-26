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

## Evidence manifest contract

After collecting the files, create `reports/device/qnn-evidence.json` using the
schema exercised by `tests/test_release_evidence.py`. Paths are relative to the
JSON file. Then validate before publishing:

```bash
python scripts/validate_qnn_evidence.py reports/device/qnn-evidence.json \
  --output reports/device/qnn-summary.json
```

Validation is intentionally fail-closed: all four retained artifacts must
match their SHA-256 digests, placement must report at least one QNN node and
zero CPU/unassigned nodes, CPU fallback must be false, and output drift must
not exceed the predeclared limit.

The repository can assemble that manifest without manually calculating hashes:

```bash
python scripts/capture_qnn_evidence.py \
  --benchmark reports/device/qnn-benchmark.json \
  --context-binary reports/device/qnn_context.bin \
  --placement-report reports/device/placement.json \
  --profile reports/device/qnn-profile.json \
  --logcat reports/device/logcat.txt \
  --model models/model.qdq.onnx \
  --input reports/device/input.bin \
  --output-dir reports/device/validated-qnn \
  --ort-version "$ORT_VERSION" \
  --qairt-version "$QAIRT_VERSION" \
  --device-fingerprint "$(adb shell getprop ro.build.fingerprint)" \
  --soc-model "$(adb shell getprop ro.soc.model)" \
  --cold-ms "$COLD_MS" \
  --peak-rss-mb "$PEAK_RSS_MB" \
  --max-abs-drift-vs-fp32 "$OUTPUT_DRIFT" \
  --max-allowed-abs-drift 0.0001
```

`placement.json` is deliberately separate from the native benchmark output.
Populate its node counts from the retained ORT/QNN placement log; do not infer
exclusive placement merely because session creation succeeded:

```json
{
  "provider": "QNNExecutionProvider",
  "qnn_node_count": 9,
  "cpu_node_count": 0,
  "unassigned_node_count": 0
}
```

The collector copies all four artifacts into one portable directory, computes
their hashes plus the model/input hashes, calculates throughput from batch size
and warm p50 latency, writes `evidence.json`, and runs the validator before
reporting success.
