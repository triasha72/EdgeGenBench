# Native iOS/Core ML demo

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

