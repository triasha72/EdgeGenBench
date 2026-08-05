Configuration
     ↓
Synthetic physics model
     ↓
Dataset generation and deterministic split
     ↓
Surrogate training
     ├── Ridge regression
     ├── Random Forest
     └── HistGradientBoosting
     ↓
Model evaluation
     ├── Accuracy
     ├── Uncertainty
     └── Failure behavior
     ↓
Feasibility classifier
     ↓
Constrained multi-objective optimization
     ↓
Physics-based validation
     ↓
Deterministic edge feature encoder
     ↓
ONNX export
     ↓
ONNX Runtime
     ↓
Accuracy-size-latency benchmark