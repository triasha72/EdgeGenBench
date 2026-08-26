package dev.edgegenbench

import org.json.JSONObject

data class BenchmarkResult(
    val schemaVersion: Int,
    val backend: String,
    val cpuFallback: Boolean,
    val coldMs: Double,
    val warmMeanMs: Double,
    val warmP95Ms: Double,
    val runs: Int,
    val baselinePreprocessMeanMs: Double,
    val fusedPreprocessMeanMs: Double,
    val preprocessElements: Int,
    val preprocessSpeedup: Double,
    val preprocessMaxAbsDrift: Double,
    val outputMaxAbsDrift: Double,
    val output: Double,
    val runtimePageSizeBytes: Long,
    val power: String,
    val capturedAtEpochMs: Long,
) {
    fun toDisplayText(): String = buildString {
        appendLine(if (backend == "reference") "backend=reference (NOT QNN)" else "backend=$backend")
        appendLine("CPU fallback=$cpuFallback")
        appendLine("cold_ms=${coldMs.format(6)}")
        appendLine("warm_mean_ms=${warmMeanMs.format(6)}")
        appendLine("warm_p95_ms=${warmP95Ms.format(6)}")
        appendLine("runs=$runs")
        appendLine("baseline_preprocess_mean_ms=${baselinePreprocessMeanMs.format(6)}")
        appendLine("fused_preprocess_mean_ms=${fusedPreprocessMeanMs.format(6)}")
        appendLine("preprocess_elements=$preprocessElements")
        appendLine("preprocess_speedup_x=${preprocessSpeedup.format(3)}")
        appendLine("preprocess_max_abs_drift=${preprocessMaxAbsDrift.format(9)}")
        appendLine("output_max_abs_drift=${outputMaxAbsDrift.format(9)}")
        appendLine("output=${output.format(6)}")
        appendLine("runtime_page_size_bytes=$runtimePageSizeBytes")
        append("power=$power")
    }

    fun toJson(): String = JSONObject()
        .put("schema_version", schemaVersion)
        .put("backend", backend)
        .put("cpu_fallback", cpuFallback)
        .put("cold_ms", coldMs)
        .put("warm_mean_ms", warmMeanMs)
        .put("warm_p95_ms", warmP95Ms)
        .put("runs", runs)
        .put("baseline_preprocess_mean_ms", baselinePreprocessMeanMs)
        .put("fused_preprocess_mean_ms", fusedPreprocessMeanMs)
        .put("preprocess_elements", preprocessElements)
        .put("preprocess_speedup_x", preprocessSpeedup)
        .put("preprocess_max_abs_drift", preprocessMaxAbsDrift)
        .put("output_max_abs_drift", outputMaxAbsDrift)
        .put("output", output)
        .put("runtime_page_size_bytes", runtimePageSizeBytes)
        .put("power", power)
        .put("captured_at_epoch_ms", capturedAtEpochMs)
        .toString()

    companion object {
        fun parse(json: String): BenchmarkResult {
            val value = JSONObject(json)
            require(value.getInt("schema_version") == 1) { "Unsupported benchmark schema" }
            val result = BenchmarkResult(
                schemaVersion = 1,
                backend = value.getString("backend"),
                cpuFallback = value.getBoolean("cpu_fallback"),
                coldMs = value.nonNegativeDouble("cold_ms"),
                warmMeanMs = value.nonNegativeDouble("warm_mean_ms"),
                warmP95Ms = value.nonNegativeDouble("warm_p95_ms"),
                runs = value.getInt("runs"),
                baselinePreprocessMeanMs = value.nonNegativeDouble("baseline_preprocess_mean_ms"),
                fusedPreprocessMeanMs = value.nonNegativeDouble("fused_preprocess_mean_ms"),
                preprocessElements = value.getInt("preprocess_elements"),
                preprocessSpeedup = value.nonNegativeDouble("preprocess_speedup_x"),
                preprocessMaxAbsDrift = value.nonNegativeDouble("preprocess_max_abs_drift"),
                outputMaxAbsDrift = value.nonNegativeDouble("output_max_abs_drift"),
                output = value.getDouble("output"),
                runtimePageSizeBytes = value.optLong("runtime_page_size_bytes", 0),
                power = value.getString("power"),
                capturedAtEpochMs = value.optLong("captured_at_epoch_ms", System.currentTimeMillis()),
            )
            require(result.backend in setOf("reference", "QNNExecutionProvider")) {
                "Unexpected backend: ${result.backend}"
            }
            require(!result.cpuFallback) { "CPU fallback must remain disabled" }
            require(result.runs > 0 && result.preprocessElements > 0) { "Invalid benchmark counts" }
            require(result.runtimePageSizeBytes >= 0) { "Invalid runtime page size" }
            require(result.preprocessMaxAbsDrift <= 1e-6 && result.outputMaxAbsDrift <= 1e-6) {
                "Baseline/fused numerical parity failed"
            }
            return result
        }

        private fun JSONObject.nonNegativeDouble(name: String): Double =
            getDouble(name).also { require(it.isFinite() && it >= 0.0) { "$name must be finite and non-negative" } }

        private fun Double.format(decimals: Int) = java.lang.String.format(java.util.Locale.US, "%.${decimals}f", this)
    }
}
