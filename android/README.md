# EdgeGenBench Android harness

This is a thin UI/JNI shell around `native/`; it is not a separate product.
The build pins Gradle 8.9, Android API 35, NDK 27.0.12077973, and CMake 3.22.1.
Native libraries are linked and packaged for Android's 16 KiB page-size
requirement, and CI rejects APKs that fail ZIP or ELF alignment checks.
Open `android/` in Android Studio or use the pinned, checksum-verified Gradle wrapper:

```bash
cd android
./gradlew lintDebug testDebugUnitTest assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb logcat -s EdgeGenBench
```

GitHub Actions publishes the verified build as
`EdgeGenBench-0.1.2-16kb-visible-ui-apk.zip`. Extract it and install
`EdgeGenBench-0.1.2-16kb-visible-ui-debug.apk`; do not reuse an older
`app-debug.apk` from Downloads. CI checks both APK ZIP alignment and every
packaged native library's ELF load-segment alignment before publishing it.

The checked-in build uses the deterministic reference backend and says so in
the UI and logs. QNN results require wiring a pinned QNN-enabled ONNX Runtime
package, following `../docs/qnn_device_runbook.md`, and retaining placement
evidence. No certificate or signing-key collection is needed for a debug APK.
