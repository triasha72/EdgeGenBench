# Android device validation

1. Open the successful GitHub Actions run and download the versioned Android
   artifact (currently `EdgeGenBench-0.1.7-device-evidence-apk`).
2. Unzip it to obtain the correspondingly versioned debug APK.
3. Enable Android developer options and USB debugging, connect exactly one device,
   and confirm that `adb devices` reports it as `device`.
4. Capture the reproducible evidence bundle:

```bash
scripts/capture_android_device_evidence.sh path/to/app-debug.apk
```

For v0.1.7 and later, tap **Export evidence bundle** after at least three runs,
save the shared JSON file, and validate it on the host:

```bash
python scripts/validate_android_evidence.py \
  path/to/android-evidence.json \
  --output-dir reports/android-device-export
```

This fails closed on missing device/app identity, fewer than three retained
runs, invalid reference placement metadata, non-finite measurements, drift over
`1e-6`, missing runtime page size, or unsupported QNN/power claims. It produces
`summary.json` and `report.md` for review.

To include the validated export in the cross-runtime release bundle, pass the
JSON file to `scripts/build_release_evidence.py --device-evidence`. The release
manifest will mark it `validated_reference` and retain the raw JSON, computed
summary, and Markdown report with checksums.

5. Open the app, tap **Run cold + warm benchmark**, and run the capture command
   again so logcat contains the displayed result.

For repeated reference-backend runs, install the APK once and use the app's
explicit `auto_run` launch extra through the profiling script:

```bash
scripts/profile_android_reference.sh 10 reports/device/reference-10-runs
```

The script force-stops and cold-launches the activity for each repetition,
waits for the JNI benchmark to complete, and retains per-run latency, launch,
post-run memory, and thermal snapshots. It fails if the log does not explicitly
name the reference backend and `power=not measured`.

The current APK reports the reference backend and is not QNN evidence. A future
QNN run is valid only when logcat names `QNNExecutionProvider`, session creation
uses `session.disable_cpu_ep_fallback=1`, and the retained QNN profile shows
exclusive NPU/HTP placement. `dumpsys battery` temperature is a coarse thermal
snapshot; it is not a calibrated power measurement.

The first retained physical-device reference capture is summarized in
[`reports/android_sm_a356e_reference_v0_1_3.md`](../reports/android_sm_a356e_reference_v0_1_3.md).
That phone reports a 4 KiB runtime page size, so CI alignment checks—not this
device run—are the current evidence for 16 KiB APK/ELF compatibility.
