package dev.edgegenbench

import android.content.res.ColorStateList
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.os.Build
import android.util.Log
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    private external fun runNativeBenchmark(warmup: Int, runs: Int): String
    private external fun runNativeQnnBenchmark(
        modelPath: String,
        contextPath: String,
        profilePath: String,
        warmup: Int,
        runs: Int,
    ): String

    companion object {
        private const val TAG = "EdgeGenBench"
        init { System.loadLibrary("edgegenbench_jni") }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.i(TAG, "MainActivity.onCreate: rendering high-contrast benchmark UI")
        val output = TextView(this).apply {
            text = if (BuildConfig.QNN_COMPILED) {
                "QNN build ready\nTap Run to create a QNN-only session with CPU fallback disabled."
            } else {
                "Reference backend only\nTap Run to measure JNI + preprocessing + inference."
            }
            setTextColor(Color.rgb(20, 24, 31))
            textSize = 18f
            setTextIsSelectable(true)
            setPadding(32, 40, 32, 32)
        }
        val store = BenchmarkStore(getSharedPreferences("edgegenbench", MODE_PRIVATE))
        val run = Button(this).apply {
            text = "Run benchmark + compare preprocessing"
            contentDescription = "Run benchmark and compare baseline versus fused preprocessing"
            setTextColor(Color.WHITE)
            backgroundTintList = ColorStateList.valueOf(Color.rgb(9, 105, 218))
        }
        val export = Button(this).apply {
            text = "Export evidence bundle"
            isEnabled = store.latest() != null
        }
        val history = TextView(this).apply {
            text = "Saved runs: ${store.history().size}/${BenchmarkStore.MAX_RESULTS}"
            setTextColor(Color.DKGRAY)
            textSize = 14f
            setPadding(32, 16, 32, 0)
        }
        val heading = TextView(this).apply {
            text = "EdgeGenBench is ready"
            setTextColor(Color.rgb(20, 24, 31))
            textSize = 26f
            setPadding(0, 0, 0, 24)
        }
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.WHITE)
            addView(heading, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            addView(run, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            addView(export, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            addView(history, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            addView(output, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
        }
        ViewCompat.setOnApplyWindowInsetsListener(layout) { view, windowInsets ->
            val bars = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(32 + bars.left, 24 + bars.top, 32 + bars.right, 32 + bars.bottom)
            windowInsets
        }
        setContentView(layout)
        ViewCompat.requestApplyInsets(layout)
        Log.i(TAG, "MainActivity.onCreate: content view attached")
        run.setOnClickListener {
            run.isEnabled = false
            output.text = "Running…"
            thread {
                val result = try {
                    val json = if (BuildConfig.QNN_COMPILED) {
                        val model = copyAssetToFiles(BuildConfig.QNN_MODEL_ASSET)
                        runNativeQnnBenchmark(
                            model.absolutePath,
                            filesDir.resolve("edgegenbench-qnn-context.onnx").absolutePath,
                            filesDir.resolve("edgegenbench-qnn-profile.json").absolutePath,
                            10,
                            100,
                        )
                    } else {
                        runNativeBenchmark(10, 100)
                    }
                    BenchmarkResult.parse(json)
                }
                catch (error: Throwable) {
                    Log.e(TAG, "Benchmark failed", error)
                    null
                }
                runOnUiThread {
                    if (result == null) {
                        output.text = "ERROR: benchmark report was invalid; inspect logcat"
                    } else {
                        store.append(result)
                        output.text = result.toDisplayText()
                        export.isEnabled = true
                        history.text = "Saved runs: ${store.history().size}/${BenchmarkStore.MAX_RESULTS}"
                        Log.i(TAG, "benchmark_json=${result.toJson()}")
                    }
                    run.isEnabled = true
                }
            }
        }
        export.setOnClickListener {
            val latest = store.latest() ?: return@setOnClickListener
            val results = store.history().filter { it.backend == latest.backend }
            val evidence = EvidenceBundle.build(
                EvidenceContext(
                    appVersionName = BuildConfig.VERSION_NAME,
                    appVersionCode = BuildConfig.VERSION_CODE.toLong(),
                    gitRevision = BuildConfig.GIT_REVISION,
                    manufacturer = Build.MANUFACTURER,
                    model = Build.MODEL,
                    androidRelease = Build.VERSION.RELEASE,
                    sdkInt = Build.VERSION.SDK_INT,
                    supportedAbis = Build.SUPPORTED_ABIS.toList(),
                ),
                results,
            )
            startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
                type = "application/json"
                putExtra(Intent.EXTRA_SUBJECT, "EdgeGenBench benchmark evidence")
                putExtra(Intent.EXTRA_TEXT, evidence)
            }, "Export EdgeGenBench result"))
        }
        if (intent.getBooleanExtra("auto_run", false)) {
            Log.i(TAG, "MainActivity.onCreate: auto_run requested")
            run.performClick()
        }
    }

    private fun copyAssetToFiles(name: String) = filesDir.resolve(name).also { target ->
        require(name.isNotBlank()) { "QNN model asset is not configured" }
        assets.open(name).use { input -> target.outputStream().use(input::copyTo) }
    }
}
