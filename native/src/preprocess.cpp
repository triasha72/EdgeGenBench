#include "edgegenbench/runtime.hpp"

#include <numeric>

namespace edgegenbench {

std::size_t Tensor::element_count() const {
  if (shape.empty()) return 0;
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                         [](std::size_t a, std::int64_t b) {
                           if (b <= 0) throw RuntimeError("tensor dimensions must be positive");
                           return a * static_cast<std::size_t>(b);
                         });
}

void Tensor::validate() const {
  if (element_count() != data.size()) throw RuntimeError("tensor shape/data size mismatch");
}

static void validate_params(const std::vector<float>& input,
                            const std::vector<float>& mean,
                            const std::vector<float>& scale) {
  if (input.empty()) throw RuntimeError("preprocess input is empty");
  if (mean.size() != input.size() || scale.size() != input.size())
    throw RuntimeError("mean and scale must match the input length");
  for (float value : scale) if (value == 0.0F) throw RuntimeError("scale contains zero");
}

Tensor preprocess_baseline(const std::vector<float>& input,
                           const std::vector<float>& mean,
                           const std::vector<float>& scale) {
  validate_params(input, mean, scale);
  std::vector<float> centered(input.size());
  std::vector<float> output(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) centered[i] = input[i] - mean[i];
  for (std::size_t i = 0; i < input.size(); ++i) output[i] = centered[i] / scale[i];
  return {{1, static_cast<std::int64_t>(output.size())}, std::move(output)};
}

Tensor preprocess_fused(const std::vector<float>& input,
                        const std::vector<float>& mean,
                        const std::vector<float>& scale) {
  validate_params(input, mean, scale);
  Tensor output{{1, static_cast<std::int64_t>(input.size())}, std::vector<float>(input.size())};
  for (std::size_t i = 0; i < input.size(); ++i) output.data[i] = (input[i] - mean[i]) / scale[i];
  return output;
}
}  // namespace edgegenbench
