package dev.edgegenbench

import android.content.SharedPreferences

class BenchmarkStore(private val preferences: SharedPreferences) {
    fun append(result: BenchmarkResult) {
        val updated = (history() + result).takeLast(MAX_RESULTS)
        preferences.edit().putString(HISTORY_KEY, updated.joinToString("\n") { it.toJson() }).apply()
    }

    fun history(): List<BenchmarkResult> = preferences.getString(HISTORY_KEY, "")
        .orEmpty()
        .lineSequence()
        .filter { it.isNotBlank() }
        .mapNotNull { runCatching { BenchmarkResult.parse(it) }.getOrNull() }
        .toList()

    fun latest(): BenchmarkResult? = history().lastOrNull()

    fun clear() = preferences.edit().remove(HISTORY_KEY).apply()

    companion object {
        const val MAX_RESULTS = 20
        private const val HISTORY_KEY = "benchmark_history_jsonl"
    }
}
