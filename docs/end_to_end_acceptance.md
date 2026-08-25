# End-to-end release acceptance

EdgeGenBench releases are accepted as a connected runtime product, not as a
collection of smoke tests. CI preserves one evidence bundle that joins the
native runtime and Android application at the same Git revision.

## Acceptance chain

```text
C++17 runtime tests
        |
        +--> baseline preprocessing benchmark JSON
        +--> fused preprocessing benchmark JSON
        |        |
        |        +--> numerical output parity gate (max drift <= 1e-6)
        |        +--> structured placement and latency schema gate
        |
Android NDK build
        |
        +--> lint + JVM tests + APK build
        +--> 16 KiB ZIP and ELF alignment verification
        |
        v
versioned release-evidence bundle
        |
        +--> manifest.json with Git revision and SHA-256 for every file
        +--> raw native benchmark results
        +--> APK and alignment report
        +--> explicit device/QNN/power measurement status
```

The release bundle is available from the successful GitHub Actions run as
`EdgeGenBench-<version>-release-evidence`. `manifest.json` is the entry point
for reviewers.

## What CI establishes

- the C++17 runtime builds and its unit tests pass;
- its public benchmark CLI runs through preprocessing, session execution,
  postprocessing output, placement reporting, and structured serialization;
- baseline and fused preprocessing produce equivalent downstream output;
- the Kotlin/JNI/NDK Android application builds and is unit-tested;
- the packaged native libraries and APK meet Android 16 KiB compatibility
  checks;
- all retained files are bound to a Git revision with SHA-256 digests.

## What requires a physical device

CI does not claim NPU placement, thermals, or power. Attach physical-device
evidence with `--device-evidence` when building a local bundle. QNN evidence is
accepted only when it contains a QNN context binary, a placement report naming
`QNNExecutionProvider`, disabled CPU fallback, and output drift against the
same model/input contract. Power remains `not measured` unless a real power
measurement tool is used.

When `--device-evidence` points to a v0.1.7 Android JSON export, the builder
validates every retained result and generates `device/summary.json` plus
`device/report.md`. Valid reference evidence is labeled `validated_reference`;
arbitrary evidence directories remain `supplied_unverified`.

Physical QNN evidence uses a separate fail-closed contract. Create an evidence
JSON alongside the retained context binary, placement report, profile, and
logcat files, then run:

```bash
python scripts/validate_qnn_evidence.py reports/qnn/evidence.json \
  --output reports/qnn/validated-summary.json
```

The validator verifies every artifact checksum, requires exclusive
`QNNExecutionProvider` placement with zero CPU or unassigned nodes, rejects CPU
fallback, checks positive latency/throughput/memory measurements, and enforces
the declared FP32 drift limit. Add the same JSON to a release bundle with
`--qnn-evidence`; accepted bundles are labeled `validated_qnn_npu`. This label
is never generated from CI or an unvalidated directory.
