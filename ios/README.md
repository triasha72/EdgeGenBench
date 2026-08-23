# Native iOS/Core ML demo

This is the optional native route. The supported no-Xcode deliverable is the
installable [browser app](../web/README.md); it is the quickest way to use the
project on an iPhone without an Apple Developer membership or App Store review.

This SwiftUI app runs the compact neural surrogate through Core ML on an iOS
17 device. Python owns the trained model and preprocessing statistics; the
exporter writes both the FP16 ML Program and a JSON contract so Swift applies
the same feature normalization, category encoding, and target inverse scaling.

On macOS, export the resources:

```bash
python -m pip install -e '.[neural,coreml]'
python scripts/export_coreml.py \
  --model artifacts/neural/neural_surrogate.pt \
  --preprocessing artifacts/neural/preprocessing.npz
cp -R artifacts/coreml/NeuralSurrogate.mlpackage ios/EdgeGenBenchDemo/
cp artifacts/coreml/ModelContract.json ios/EdgeGenBenchDemo/
```

Generate and open the Xcode project:

```bash
brew install xcodegen
cd ios
xcodegen generate
open EdgeGenBenchDemo.xcodeproj
```

Select the two generated resources in Xcode and confirm that
`EdgeGenBenchDemo` appears under Target Membership. Choose a development team,
run on a physical iPhone, and use Instruments or MetricKit for device latency
and energy evidence. The repository supplies the native integration, but does
not claim a physical-device result without a signed build and captured run.

## Delivery choices

- **Installable browser app:** publish the repository's `web/` app with GitHub
  Pages and add it to the iPhone Home Screen. This route is implemented and
  needs neither Xcode nor Apple signing.
- **Cloud-built native app:** use Expo EAS or a hosted macOS CI runner to build
  and sign a native package. This avoids local Xcode but still needs Apple
  credentials and does not remove TestFlight/App Store requirements.
- **Local native app:** use this SwiftUI/Core ML target with Xcode. This is the
  right route when the goal is measured Core ML or Apple Neural Engine evidence
  on a physical iPhone.
