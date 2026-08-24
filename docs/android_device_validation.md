# Android device validation

1. Open the successful GitHub Actions run and download `EdgeGenBench-debug-apk`.
2. Unzip the artifact to obtain `app-debug.apk`.
3. Enable Android developer options and USB debugging, connect exactly one device,
   and confirm that `adb devices` reports it as `device`.
4. Capture the reproducible evidence bundle:

```bash
scripts/capture_android_device_evidence.sh path/to/app-debug.apk
```

5. Open the app, tap **Run cold + warm benchmark**, and run the capture command
   again so logcat contains the displayed result.

The current APK reports the reference backend and is not QNN evidence. A future
QNN run is valid only when logcat names `QNNExecutionProvider`, session creation
uses `session.disable_cpu_ep_fallback=1`, and the retained QNN profile shows
exclusive NPU/HTP placement. `dumpsys battery` temperature is a coarse thermal
snapshot; it is not a calibrated power measurement.
