#include "edgegenbench/runtime.hpp"

#include <cmath>

namespace edgegenbench {
namespace {
class ReferenceSession final : public Session {
 public:
  Tensor run(const Tensor& input) override {
    input.validate();
    if (input.shape.size() != 2)
      throw RuntimeError("reference backend expects a [batch,N] tensor");
    const auto batch = static_cast<std::size_t>(input.shape[0]);
    const auto width = static_cast<std::size_t>(input.shape[1]);
    Tensor output{{input.shape[0], 6}, std::vector<float>(batch * 6)};
    for (std::size_t row = 0; row < batch; ++row) {
      for (std::size_t target = 0; target < 6; ++target) {
        float score = 0.125F * static_cast<float>(target + 1);
        for (std::size_t column = 0; column < width; ++column)
          score += input.data[row * width + column] *
                   (0.001F * static_cast<float>((target + 1) * (column + 1)));
        output.data[row * 6 + target] = score;
      }
    }
    return output;
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
