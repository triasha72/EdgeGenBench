#include "edgegenbench/runtime.hpp"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace edgegenbench {
namespace {
class OrtQnnSession final : public Session {
 public:
  explicit OrtQnnSession(const Config& config)
      : env_(ORT_LOGGING_LEVEL_WARNING, "edgegenbench"),
        memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    if (config.backend != "qnn") throw RuntimeError("ORT build currently supports only the qnn backend");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (config.disable_cpu_fallback)
      options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    if (!config.qnn_context_path.empty()) {
      options.AddConfigEntry("ep.context_enable", "1");
      options.AddConfigEntry("ep.context_file_path", config.qnn_context_path.c_str());
      options.AddConfigEntry("ep.context_embed_mode", "0");
    }
    std::unordered_map<std::string, std::string> provider_options{
        {"backend_path", config.qnn_backend_path},
        {"offload_graph_io_quantization", "0"},
        {"htp_graph_finalization_optimization_mode", "3"}};
    if (!config.qnn_profiling_path.empty()) {
      provider_options["profiling_level"] = "detailed";
      provider_options["profiling_file_path"] = config.qnn_profiling_path;
    }
    options.AppendExecutionProvider("QNN", provider_options);
    session_ = std::make_unique<Ort::Session>(env_, config.model_path.c_str(), options);

    Ort::AllocatorWithDefaultOptions allocator;
    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1)
      throw RuntimeError("EdgeGenBench native QNN path requires exactly one input and one output");
    input_name_ = session_->GetInputNameAllocated(0, allocator).get();
    output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
    report_ = std::string("{\"backend\":\"QNNExecutionProvider\",\"qnn_backend_path\":\"") +
              config.qnn_backend_path + "\",\"cpu_fallback\":" +
              (config.disable_cpu_fallback ? "false" : "true") +
              ",\"context_cache\":" + (!config.qnn_context_path.empty() ? "true" : "false") +
              ",\"hardware_measurement\":true}";
  }

  Tensor run(const Tensor& input) override {
    input.validate();
    auto input_value = Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(input.data.data()), input.data.size(),
        input.shape.data(), input.shape.size());
    const std::array<const char*, 1> input_names{input_name_.c_str()};
    const std::array<const char*, 1> output_names{output_name_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_value, 1,
                                 output_names.data(), 1);
    auto info = outputs.at(0).GetTensorTypeAndShapeInfo();
    Tensor output{info.GetShape(), {}};
    const std::size_t count = info.GetElementCount();
    const float* data = outputs.at(0).GetTensorData<float>();
    output.data.assign(data, data + count);
    output.validate();
    return output;
  }

  std::string placement_report() const override { return report_; }

 private:
  Ort::Env env_;
  Ort::MemoryInfo memory_info_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
  std::string report_;
};
}  // namespace

std::unique_ptr<Session> create_ort_session(const Config& config) {
  try {
    return std::make_unique<OrtQnnSession>(config);
  } catch (const Ort::Exception& error) {
    throw RuntimeError(std::string("QNN session creation/execution failed: ") + error.what());
  }
}
}  // namespace edgegenbench
