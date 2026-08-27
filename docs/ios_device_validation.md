# iPhone evidence acceptance

EdgeGenBench separates three iOS proof levels:

| Level | What it proves | What it does not prove |
|---|---|---|
| CI simulator build + XCTest | Core ML resources compile, the Swift app links, and evidence contracts pass tests | Physical-device performance, ANE placement, power |
| Validated physical-iPhone JSON | Current-model Core ML app latency, device/OS identity, thermal boundary, deterministic repeated output | ANE placement or calibrated power |
| Retained Instruments capture | Only the metrics and placement visible in the named Instruments templates | Claims outside that measured boundary |

Use [`ios/README.md`](../ios/README.md) for the physical run. The evidence
validator compares the hashes embedded by the Core ML exporter with the tracked
checkpoint and preprocessing artifacts. A screenshot alone is supporting
visual evidence and cannot replace the JSON export.

Do not publish an Apple Neural Engine or energy-saving claim merely because the
app requests `MLComputeUnits.all`. Core ML remains free to choose an available
compute unit, and device power requires an appropriate named measurement tool.
