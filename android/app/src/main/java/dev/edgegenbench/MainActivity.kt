package dev.edgegenbench

import android.content.res.ColorStateList
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.ViewGroup
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
        Log.i(TAG, "MainActivity.onCreate: rendering high-contrast benchmark UI")
        val output = TextView(this).apply {
            text = "EdgeGenBench is ready\n\nReference backend only\nTap Run to measure JNI + preprocessing + inference."
            setTextColor(Color.rgb(20, 24, 31))
            textSize = 18f
            setTextIsSelectable(true)
            setPadding(32, 40, 32, 32)
        }
        val run = Button(this).apply {
            text = "Run cold + warm benchmark"
            setTextColor(Color.WHITE)
            backgroundTintList = ColorStateList.valueOf(Color.rgb(9, 105, 218))
        }
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.WHITE)
            setPadding(32, 48, 32, 32)
            addView(run, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            addView(output, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
        }
        setContentView(layout)
        Log.i(TAG, "MainActivity.onCreate: content view attached")
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
