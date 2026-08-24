#include "edgegenbench/runtime.hpp"

#include <cmath>
#include <iostream>
#include <limits>

namespace {
int failures = 0;
void check(bool condition, const char* name) {
  if (!condition) { std::cerr << "FAIL: " << name << '\n'; ++failures; }
}
template <typename F> void throws(F fn, const char* name) {
  try { fn(); check(false, name); } catch (const edgegenbench::RuntimeError&) {}
}
}  // namespace

int main() {
  const std::vector<float> input{1, 2, 3, 4}, mean{0.5F, 1, 1.5F, 2}, scale{0.5F, 0.5F, 0.5F, 0.5F};
  const auto baseline = edgegenbench::preprocess_baseline(input, mean, scale);
  const auto fused = edgegenbench::preprocess_fused(input, mean, scale);
  check(baseline.shape == fused.shape, "equivalent shapes");
  check(baseline.data == fused.data, "fused equals baseline");
  throws([&] { edgegenbench::preprocess_fused({}, {}, {}); }, "empty input rejected");
  throws([&] { edgegenbench::preprocess_fused(input, mean, {1}); }, "bad scale rejected");
  edgegenbench::Config config;
  auto session = edgegenbench::create_session(config);
  const auto result = session->run(fused);
  check(result.shape == std::vector<std::int64_t>({1, 6}), "output shape");
  check(std::isfinite(result.data.at(0)), "finite output");
  edgegenbench::Tensor invalid{{1, 2}, {1}};
  throws([&] { session->run(invalid); }, "shape mismatch rejected");
  const edgegenbench::AircraftDesign design{{65.0F, 950.0F, 535.0F, 527.0F, 0.57F, 0.24F},
                                             "parallel_hybrid"};
  const auto encoded = edgegenbench::encode_aircraft_designs({design});
  check(encoded.shape == std::vector<std::int64_t>({1, 10}), "production input shape");
  check(encoded.data[8] == 1.0F, "production category order");
  check(encoded.data[6] == 0.0F && encoded.data[7] == 0.0F && encoded.data[9] == 0.0F,
        "one hot encoding");
  edgegenbench::Tensor zero_predictions{{1, 6}, std::vector<float>(6, 0.0F)};
  const auto physical = edgegenbench::denormalize_predictions(zero_predictions);
  check(std::abs(physical.data[0] - 29978.1504F) < 0.01F, "target denormalization");
  throws([&] { edgegenbench::encode_aircraft_designs({{{1, 2, 3, 4, 5, 6}, "unknown"}}); },
         "unknown architecture rejected");
  edgegenbench::Tensor nonfinite{{1, 1}, {std::numeric_limits<float>::infinity()}};
  throws([&] { nonfinite.validate(); }, "non-finite tensor rejected");
  return failures == 0 ? 0 : 1;
}
