#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <vector>

#include "edgegenbench/runtime.hpp"

extern "C" JNIEXPORT jstring JNICALL
Java_dev_edgegenbench_MainActivity_runNativeBenchmark(JNIEnv* env, jobject,
                                                       jint warmup, jint runs) {
  try {
    if (warmup < 0 || runs <= 0) throw edgegenbench::RuntimeError("invalid run counts");
    edgegenbench::Config config;
    auto session = edgegenbench::create_session(config);
    const std::vector<float> raw{0.17F, 0.28F, 0.41F, 0.53F, 0.68F, 0.79F, 0.83F, 0.97F};
    const std::vector<float> mean(raw.size(), 0.5F), scale(raw.size(), 0.25F);
    constexpr std::size_t preprocess_elements = 262144;
    std::vector<float> preprocess_raw(preprocess_elements);
    for (std::size_t i = 0; i < preprocess_raw.size(); ++i) {
      preprocess_raw[i] = static_cast<float>(i % 251) / 250.0F;
    }
    const std::vector<float> preprocess_mean(preprocess_elements, 0.5F);
    const std::vector<float> preprocess_scale(preprocess_elements, 0.25F);
    auto baseline_tensor = edgegenbench::preprocess_baseline(
        preprocess_raw, preprocess_mean, preprocess_scale);
    auto fused_tensor = edgegenbench::preprocess_fused(
        preprocess_raw, preprocess_mean, preprocess_scale);
    double preprocess_max_abs_drift = 0.0;
    for (std::size_t i = 0; i < baseline_tensor.data.size(); ++i) {
      preprocess_max_abs_drift = std::max(
          preprocess_max_abs_drift,
          static_cast<double>(std::fabs(baseline_tensor.data[i] - fused_tensor.data[i])));
    }
    const auto baseline_output = session->run(edgegenbench::preprocess_baseline(raw, mean, scale));
    const auto fused_output = session->run(edgegenbench::preprocess_fused(raw, mean, scale));
    double output_max_abs_drift = 0.0;
    for (std::size_t i = 0; i < baseline_output.data.size(); ++i) {
      output_max_abs_drift = std::max(
          output_max_abs_drift,
          static_cast<double>(std::fabs(baseline_output.data[i] - fused_output.data[i])));
    }

    auto measure_preprocess = [&](bool fused) {
      std::vector<double> timings;
      timings.reserve(static_cast<std::size_t>(runs));
      edgegenbench::Tensor latest;
      for (jint i = 0; i < runs; ++i) {
        const auto start = std::chrono::steady_clock::now();
        latest = fused
            ? edgegenbench::preprocess_fused(preprocess_raw, preprocess_mean, preprocess_scale)
            : edgegenbench::preprocess_baseline(preprocess_raw, preprocess_mean, preprocess_scale);
        timings.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
      }
      const double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
      return sum / static_cast<double>(timings.size());
    };
    for (jint i = 0; i < warmup; ++i) {
      baseline_tensor = edgegenbench::preprocess_baseline(
          preprocess_raw, preprocess_mean, preprocess_scale);
      fused_tensor = edgegenbench::preprocess_fused(
          preprocess_raw, preprocess_mean, preprocess_scale);
    }
    const double baseline_preprocess_mean_ms = measure_preprocess(false);
    const double fused_preprocess_mean_ms = measure_preprocess(true);
    const double preprocess_speedup = fused_preprocess_mean_ms > 0.0
        ? baseline_preprocess_mean_ms / fused_preprocess_mean_ms : 0.0;

    auto one_run = [&] { return session->run(edgegenbench::preprocess_fused(raw, mean, scale)); };
    const auto cold_start = std::chrono::steady_clock::now();
    auto result = one_run();
    const double cold_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - cold_start).count();
    for (jint i = 0; i < warmup; ++i) result = one_run();
    std::vector<double> samples;
    for (jint i = 0; i < runs; ++i) {
      const auto start = std::chrono::steady_clock::now();
      result = one_run();
      samples.push_back(std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count());
    }
    std::sort(samples.begin(), samples.end());
    const double average = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    const auto p95 = samples[static_cast<std::size_t>((samples.size() - 1) * 0.95)];
    std::ostringstream report;
    report << std::fixed << std::setprecision(6)
           << "backend=reference (NOT QNN)\nCPU fallback=false (reference backend)\n"
           << "cold_ms=" << cold_ms << "\nwarm_mean_ms=" << average
           << "\nwarm_p95_ms=" << p95 << "\nruns=" << runs
           << "\nbaseline_preprocess_mean_ms=" << baseline_preprocess_mean_ms
           << "\nfused_preprocess_mean_ms=" << fused_preprocess_mean_ms
           << "\npreprocess_elements=" << preprocess_elements
           << "\npreprocess_speedup_x=" << preprocess_speedup
           << "\npreprocess_max_abs_drift=" << preprocess_max_abs_drift
           << "\noutput_max_abs_drift=" << output_max_abs_drift
           << "\noutput=" << result.data.at(0)
           << "\npower=not measured";
    __android_log_print(ANDROID_LOG_INFO, "EdgeGenBench", "%s", report.str().c_str());
    return env->NewStringUTF(report.str().c_str());
  } catch (const std::exception& error) {
    __android_log_print(ANDROID_LOG_ERROR, "EdgeGenBench", "%s", error.what());
    jclass cls = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(cls, error.what());
    return nullptr;
  }
}
