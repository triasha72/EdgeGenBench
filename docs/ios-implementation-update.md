# iOS implementation and validation boundary

The SwiftUI app executes the generated aircraft-design Core ML surrogate. It does
not execute the separate DASHlink real-flight model. Benchmarking now runs off the
UI thread and saves raw warm latencies and inputs alongside model provenance,
thermal state and simulator identity. The contract rejects invalid normalization
and nonfinite inputs. The Python validator rejects nonfinite summaries and checks
raw sample consistency when samples are supplied.

From the repository root:

```sh
pip install -e '.[neural,coreml,dev]'
python scripts/prepare_ios_resources.py
cd ios
xcodegen generate
xcodebuild -project EdgeGenBenchDemo.xcodeproj -scheme EdgeGenBenchDemo -destination 'platform=iOS Simulator,name=iPhone 16' test
```

Choose an installed simulator from `xcrun simctl list devices available`. The new
CoreMLIntegrationTests loads the bundled model, executes inference and attaches
100-run evidence to the Xcode result. A simulator test checks integration only.
For hardware evidence, choose a physical iPhone and signing team in Xcode, run the
app and export its JSON through Share. Use `scripts/validate_ios_evidence.py --help`
to validate against the source artifact hashes. No physical iPhone timings have
been collected in this update. The local Xcode build was blocked by the execution
sandbox; the integration test has not yet been confirmed passing.

Core ML `.all` requests available compute units; it does not prove ANE placement.
Use retained Instruments measurements for placement or power claims. Cold latency
includes model loading plus first inference. This is a model benchmark, not a
validated aircraft-design recommendation system.
