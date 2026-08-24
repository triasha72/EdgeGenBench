#include "edgegenbench/runtime.hpp"

#include <cmath>
#include <iostream>

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
  check(result.shape == std::vector<std::int64_t>({1, 1}), "output shape");
  check(std::isfinite(result.data.at(0)), "finite output");
  edgegenbench::Tensor invalid{{1, 2}, {1}};
  throws([&] { session->run(invalid); }, "shape mismatch rejected");
  return failures == 0 ? 0 : 1;
}
