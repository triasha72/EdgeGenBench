#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <chrono>
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
