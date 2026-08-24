#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace edgegenbench {

class RuntimeError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct Tensor {
  std::vector<std::int64_t> shape;
  std::vector<float> data;
  std::size_t element_count() const;
  void validate() const;
};

struct Config {
  std::string model_path;
  std::string backend{"reference"};
  std::size_t warmup_runs{5};
  std::size_t measured_runs{50};
  bool disable_cpu_fallback{true};
  bool fused_preprocess{true};
  std::string qnn_backend_path{"libQnnHtp.so"};
  std::string qnn_context_path;
  std::string qnn_profiling_path;
};

Config parse_args(int argc, char** argv);
Tensor preprocess_baseline(const std::vector<float>& input,
                           const std::vector<float>& mean,
                           const std::vector<float>& scale);
Tensor preprocess_fused(const std::vector<float>& input,
                        const std::vector<float>& mean,
                        const std::vector<float>& scale);

class Session {
 public:
  virtual ~Session() = default;
  virtual Tensor run(const Tensor& input) = 0;
  virtual std::string placement_report() const = 0;
};

std::unique_ptr<Session> create_session(const Config& config);

}  // namespace edgegenbench
