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
`EdgeGenBench-0.1.7-device-evidence-apk.zip`. Extract it and install
`EdgeGenBench-0.1.7-device-evidence-debug.apk`; do not reuse an older
`app-debug.apk` from Downloads. CI checks both APK ZIP alignment and every
packaged native library's ELF load-segment alignment before publishing it.

The native result also reports baseline and fused preprocessing means, their
speedup ratio, preprocessing maximum absolute drift, and downstream output
maximum absolute drift. These are measured separately from the fused
preprocess-plus-reference-inference cold/warm timings.

JNI returns a versioned JSON result contract. The Kotlin layer validates the
backend, run counts, latency values, and baseline/fused drift before displaying
or retaining a run. The latest 20 valid runs are stored locally. **Export latest
result** opens Android's share sheet with the machine-readable JSON evidence;
sharing happens only after the user chooses a destination. The exported bundle
includes all retained runs plus the app version, CI Git revision, device model,
Android version, supported ABIs, and observed runtime page size.

The checked-in build uses the deterministic reference backend and says so in
the UI and logs. QNN results require wiring a pinned QNN-enabled ONNX Runtime
package, following `../docs/qnn_device_runbook.md`, and retaining placement
evidence. No certificate or signing-key collection is needed for a debug APK.

## Opt-in QNN dependency build

Keep proprietary/prebuilt runtime files outside Git. Prepare one absolute
directory with this minimum layout:

```text
qnn-android-root/
├── include/onnxruntime_cxx_api.h
└── lib/arm64-v8a/
    ├── libonnxruntime.so
    ├── libQnnHtp.so
    └── libQnnSystem.so
```

First validate and checksum that bundle, then compile the arm64-only APK:

```bash
python ../scripts/verify_android_qnn_bundle.py "$QNN_ANDROID_ROOT" \
  --output ../reports/device/qnn-android-dependencies.json

./gradlew assembleDebug \
  -PedgegenbenchEnableQnn=true \
  -PedgegenbenchQnnRoot="$QNN_ANDROID_ROOT"
```

The Gradle switch enables the native ONNX Runtime implementation, passes the
pinned root into CMake, packages its shared libraries, and removes the x86_64
ABI because the supplied QNN runtime is arm64 device software. A missing root
fails during Gradle configuration. The default command remains the reference
build used by CI.

This build switch proves dependency and linker integration only. Until the JNI
entry point is switched to the QNN session and its physical-device evidence
passes the validator, the UI continues to identify and run the reference
backend.
