package dev.edgegenbench

import org.json.JSONArray
import org.json.JSONObject

data class EvidenceContext(
    val appVersionName: String,
    val appVersionCode: Long,
    val gitRevision: String,
    val manufacturer: String,
    val model: String,
    val androidRelease: String,
    val sdkInt: Int,
    val supportedAbis: List<String>,
)

object EvidenceBundle {
    fun build(context: EvidenceContext, results: List<BenchmarkResult>): String {
        require(context.appVersionName.isNotBlank()) { "App version is required" }
        require(context.gitRevision.isNotBlank()) { "Git revision is required" }
        require(context.model.isNotBlank() && context.sdkInt > 0) { "Device identity is required" }
        require(results.isNotEmpty()) { "At least one benchmark result is required" }
        val pageSizes = results.map { it.runtimePageSizeBytes }.filter { it > 0 }.distinct()
        return JSONObject()
            .put("schema_version", 1)
            .put("project", "EdgeGenBench")
            .put("app", JSONObject()
                .put("version_name", context.appVersionName)
                .put("version_code", context.appVersionCode)
                .put("git_revision", context.gitRevision))
            .put("device", JSONObject()
                .put("manufacturer", context.manufacturer)
                .put("model", context.model)
                .put("android_release", context.androidRelease)
                .put("sdk_int", context.sdkInt)
                .put("supported_abis", JSONArray(context.supportedAbis))
                .put("observed_runtime_page_sizes_bytes", JSONArray(pageSizes)))
            .put("measurement_claims", JSONObject()
                .put("backend", "reference")
                .put("qnn_npu_placement", "not tested")
                .put("power", "not measured")
                .put("thermal", "not included in app export"))
            .put("result_count", results.size)
            .put("results", JSONArray(results.map { JSONObject(it.toJson()) }))
            .toString(2)
    }
}
