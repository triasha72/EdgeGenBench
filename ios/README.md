# Native iOS/Core ML runtime

This SwiftUI application exports the repository's current neural checkpoint to
a Core ML FP16 ML Program, applies the same preprocessing contract as Python,
and records one cold plus 100 warm application-level inference runs. The app
exports evidence JSON containing model provenance, device and OS identity,
thermal state, latency, and repeated-output drift.

The unsigned simulator build and XCTest suite run in CI. Simulator results
prove integration and compatibility only; they are not physical-iPhone
performance, Apple Neural Engine placement, energy, or power evidence.

## Generate the app

From the repository root on macOS:

```bash
python -m pip install -e '.[neural,coreml]'
python scripts/prepare_ios_resources.py
brew install xcodegen
cd ios
xcodegen generate
open EdgeGenBenchDemo.xcodeproj
```

The resource script uses the tracked current-model checkpoint and preprocessing
state. It embeds their SHA-256 values in `ModelContract.json`; the exported
device evidence is rejected if those values do not match the repository.

## Run on a physical iPhone

1. Install the full Xcode application from Apple and open
   `ios/EdgeGenBenchDemo.xcodeproj`.
2. Connect the iPhone by USB, unlock it, tap **Trust** if prompted, and enable
   Developer Mode if Xcode requests it.
3. Select the `EdgeGenBenchDemo` target. Under **Signing & Capabilities**, choose
   your personal development team. EdgeGenBench does not collect or store your
   signing identity.
4. Select the connected iPhone as the run destination and press **Run**.
5. Keep Low Power Mode off, close other foreground apps, and let the device
   reach a stable temperature.
6. In EdgeGenBench, tap **Run cold + warm benchmark**. Confirm it reports one
   cold and 100 warm runs.
7. Tap **Export evidence JSON**, AirDrop or save the file to the Mac, and retain
   a screenshot of the result screen.
8. Validate and render the report from the repository root:

```bash
python scripts/validate_ios_evidence.py \
  "$HOME/Downloads/EdgeGenBench-iOS-evidence.json" \
  --output-json reports/ios_device_summary.json \
  --output-markdown reports/ios_device_report.md
```

Commit the raw evidence JSON, generated summary, Markdown report, and screenshot
in a follow-up evidence PR. The validator rejects simulator evidence, fewer than
100 warm runs, excessive output drift, mismatched model/preprocessing hashes,
and unsupported ANE or power claims.

## Optional Instruments evidence

The app requests Core ML `MLComputeUnits.all`, which does **not** prove that the
Apple Neural Engine executed the graph. To claim ANE placement or energy, run a
Release build under Xcode Instruments with the Core ML and Energy Log templates,
retain the `.trace` bundle or exported summary, name the Xcode/iOS/device
versions, and report the measured boundary. Without that evidence the project
correctly reports `neuralEnginePlacement=not_measured` and
`powerMeasurement=not_measured`.
