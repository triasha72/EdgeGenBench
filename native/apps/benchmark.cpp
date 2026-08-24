#include "edgegenbench/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>

int main(int argc, char** argv) {
  try {
    const auto config = edgegenbench::parse_args(argc, argv);
    auto session = edgegenbench::create_session(config);
    std::vector<float> raw(config.batch_size * 10);
    for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = 0.01F * static_cast<float>((i % 10) + 1);
    const std::vector<float> mean(raw.size(), 0.05F), scale(raw.size(), 0.25F);
    auto prepare = [&] { return config.fused_preprocess
        ? edgegenbench::preprocess_fused(raw, mean, scale)
        : edgegenbench::preprocess_baseline(raw, mean, scale); };
    auto reshape = [&](edgegenbench::Tensor tensor) {
      tensor.shape = {static_cast<std::int64_t>(config.batch_size), 10};
      tensor.validate();
      return tensor;
    };
    for (std::size_t i = 0; i < config.warmup_runs; ++i) session->run(reshape(prepare()));
    std::vector<double> samples;
    samples.reserve(config.measured_runs);
    edgegenbench::Tensor output;
    for (std::size_t i = 0; i < config.measured_runs; ++i) {
      const auto start = std::chrono::steady_clock::now();
      output = session->run(reshape(prepare()));
      const auto end = std::chrono::steady_clock::now();
      samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(samples.begin(), samples.end());
    const double mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    const auto percentile = [&](double p) { return samples[static_cast<std::size_t>((samples.size() - 1) * p)]; };
    std::cout << std::fixed << std::setprecision(6)
              << "{\"schema_version\":1,\"backend\":\"" << config.backend
              << "\",\"preprocess\":\"" << (config.fused_preprocess ? "fused" : "baseline")
              << "\",\"warmup_runs\":" << config.warmup_runs
              << ",\"batch_size\":" << config.batch_size
              << ",\"measured_runs\":" << config.measured_runs
              << ",\"latency_ms\":{\"mean\":" << mean_ms << ",\"p50\":" << percentile(0.50)
              << ",\"p95\":" << percentile(0.95) << "},\"output\":" << output.data.at(0)
              << ",\"placement\":" << session->placement_report() << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "edgegenbench: " << error.what() << '\n';
    return 2;
  }
}
