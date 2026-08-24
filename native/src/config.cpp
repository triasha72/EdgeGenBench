#include "edgegenbench/runtime.hpp"

#include <charconv>
#include <limits>
#include <string_view>

namespace edgegenbench {
namespace {
std::size_t positive_size(std::string_view text, const char* flag) {
  std::size_t value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value == 0) {
    throw RuntimeError(std::string(flag) + " requires a positive integer");
  }
  return value;
}
}  // namespace

Config parse_args(int argc, char** argv) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto value = [&](const char* flag) -> std::string {
      if (++i >= argc) throw RuntimeError(std::string("missing value for ") + flag);
      return argv[i];
    };
    if (arg == "--model") config.model_path = value("--model");
    else if (arg == "--backend") config.backend = value("--backend");
    else if (arg == "--qnn-backend-path") config.qnn_backend_path = value("--qnn-backend-path");
    else if (arg == "--qnn-context") config.qnn_context_path = value("--qnn-context");
    else if (arg == "--qnn-profile") config.qnn_profiling_path = value("--qnn-profile");
    else if (arg == "--warmup") config.warmup_runs = positive_size(value("--warmup"), "--warmup");
    else if (arg == "--runs") config.measured_runs = positive_size(value("--runs"), "--runs");
    else if (arg == "--allow-cpu-fallback") config.disable_cpu_fallback = false;
    else if (arg == "--baseline-preprocess") config.fused_preprocess = false;
    else if (arg == "--help") throw RuntimeError("usage: edgegenbench_benchmark [--model FILE] [--backend reference|qnn] [--qnn-backend-path FILE] [--qnn-context FILE] [--qnn-profile FILE] [--warmup N] [--runs N] [--allow-cpu-fallback] [--baseline-preprocess]");
    else throw RuntimeError("unknown argument: " + arg);
  }
  if (config.backend != "reference" && config.model_path.empty())
    throw RuntimeError("--model is required for non-reference backends");
  return config;
}
}  // namespace edgegenbench
