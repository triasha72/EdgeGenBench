#include "edgegenbench/runtime.hpp"

#include <cmath>

namespace edgegenbench {
namespace {
class ReferenceSession final : public Session {
 public:
  Tensor run(const Tensor& input) override {
    input.validate();
    if (input.shape.size() != 2 || input.shape.front() != 1)
      throw RuntimeError("reference backend expects a [1,N] tensor");
    float score = 0.125F;
    for (std::size_t i = 0; i < input.data.size(); ++i)
      score += input.data[i] * (0.01F * static_cast<float>(i + 1));
    return {{1, 1}, {score}};
  }
  std::string placement_report() const override {
    return R"({"backend":"reference","operators":{"ReferenceLinear":1},"cpu_fallback":false,"hardware_measurement":false})";
  }
};
}  // namespace

std::unique_ptr<Session> create_session(const Config& config) {
  if (config.backend == "reference") return std::make_unique<ReferenceSession>();
#ifdef EDGEBENCH_ENABLE_ORT
  extern std::unique_ptr<Session> create_ort_session(const Config&);
  return create_ort_session(config);
#else
  throw RuntimeError("QNN backend unavailable: rebuild with -DEDGEBENCH_ENABLE_ORT=ON and ONNXRUNTIME_ROOT");
#endif
}
}  // namespace edgegenbench
