# EdgeGenBench Android harness

This is a thin UI/JNI shell around `native/`; it is not a separate product.
Open `android/` in Android Studio or use a local Gradle installation:

```bash
cd android
gradle :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb logcat -s EdgeGenBench
```

The checked-in build uses the deterministic reference backend and says so in
the UI and logs. QNN results require wiring a pinned QNN-enabled ONNX Runtime
package, following `../docs/qnn_device_runbook.md`, and retaining placement
evidence. No certificate or signing-key collection is needed for a debug APK.
