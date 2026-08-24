package dev.edgegenbench

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    private external fun runNativeBenchmark(warmup: Int, runs: Int): String

    companion object {
        private const val TAG = "EdgeGenBench"
        init { System.loadLibrary("edgegenbench_jni") }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val output = TextView(this).apply {
            text = "Reference backend only\nTap Run to measure JNI + preprocessing + inference."
            setTextIsSelectable(true)
            setPadding(24, 24, 24, 24)
        }
        val run = Button(this).apply { text = "Run cold + warm benchmark" }
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(run)
            addView(output)
        }
        setContentView(layout)
        run.setOnClickListener {
            run.isEnabled = false
            output.text = "Running…"
            thread {
                val report = try { runNativeBenchmark(10, 100) }
                catch (error: Throwable) { "ERROR: ${error.message}" }
                Log.i(TAG, report)
                runOnUiThread { output.text = report; run.isEnabled = true }
            }
        }
    }
}
