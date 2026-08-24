# EdgeGenBench native runtime

This C++17 target makes preprocessing, inference lifecycle, validation, and
benchmark reporting independently testable. The default `reference` backend is
deterministic and is **not** an NPU result.

```bash
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j
ctest --test-dir build/native --output-on-failure
./build/native/edgegenbench_benchmark --runs 100
./build/native/edgegenbench_benchmark --baseline-preprocess --runs 100
```

The fused path performs normalization in one pass and one allocation; the
baseline materializes a centered intermediate. Both use contiguous row-major
FP32 `[1,N]` data and are covered by an equivalence test. Compiler auto-
vectorization is intentionally preferred over architecture-specific intrinsics
until device profiles justify NEON-only code.

For QNN, use the device workflow in `docs/qnn_device_runbook.md`. It requires an
ORT build containing the QNN EP and fails when QNN cannot be selected. Do not
compare the reference backend's latency with device QNN latency.

```bash
cmake -S native -B build/qnn -DEDGEBENCH_ENABLE_ORT=ON \
  -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT" -DCMAKE_BUILD_TYPE=Release
cmake --build build/qnn --parallel
./build/qnn/edgegenbench_benchmark --backend qnn --model model.qdq.onnx \
  --qnn-backend-path libQnnHtp.so --qnn-context reports/device/model_ctx.onnx \
  --qnn-profile reports/device/qnn-profile.csv --runs 1000
```

`session.disable_cpu_ep_fallback=1` is set by default. Graph-I/O quantization
offload is also disabled, avoiding hidden CPU work around a QDQ graph. Session
creation fails if any operator cannot be assigned to QNN.
