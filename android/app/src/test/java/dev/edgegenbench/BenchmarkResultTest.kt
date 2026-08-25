package dev.edgegenbench

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class BenchmarkResultTest {
    private val validJson = """
        {
          "schema_version": 1,
          "backend": "reference",
          "cpu_fallback": false,
          "cold_ms": 0.01,
          "warm_mean_ms": 0.003,
          "warm_p95_ms": 0.004,
          "runs": 100,
          "baseline_preprocess_mean_ms": 0.2,
          "fused_preprocess_mean_ms": 0.1,
          "preprocess_elements": 262144,
          "preprocess_speedup_x": 2.0,
          "preprocess_max_abs_drift": 0.0,
          "output_max_abs_drift": 0.0,
          "output": 0.15616,
          "power": "not measured",
          "captured_at_epoch_ms": 1234
        }
    """.trimIndent()

    @Test fun parsesAndRoundTripsVersionedReport() {
        val result = BenchmarkResult.parse(validJson)
        assertEquals("reference", result.backend)
        assertEquals(100, result.runs)
        assertEquals(1234, BenchmarkResult.parse(result.toJson()).capturedAtEpochMs)
        assertTrue(result.toDisplayText().contains("backend=reference (NOT QNN)"))
        assertTrue(result.toDisplayText().contains("power=not measured"))
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsOutputDrift() {
        BenchmarkResult.parse(validJson.replace("\"output_max_abs_drift\": 0.0", "\"output_max_abs_drift\": 0.01"))
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsUnexpectedBackend() {
        BenchmarkResult.parse(validJson.replace("\"reference\"", "\"QNNExecutionProvider\""))
    }
}
