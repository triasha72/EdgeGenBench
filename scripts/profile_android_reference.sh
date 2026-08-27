#!/usr/bin/env bash
set -euo pipefail

repeats="${1:-10}"
output_dir="${2:-reports/device/reference-$(date -u +%Y%m%dT%H%M%SZ)}"
package_name="dev.edgegenbench"
activity_name="dev.edgegenbench/.MainActivity"

[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || {
  echo "REPEATS must be a positive integer" >&2
  exit 2
}
command -v adb >/dev/null || { echo "adb is required" >&2; exit 2; }

device_count="$(adb devices | awk 'NR > 1 && $2 == "device" {count++} END {print count+0}')"
[[ "$device_count" == "1" ]] || {
  echo "Exactly one authorized device is required" >&2
  exit 2
}
package_dump="$(adb shell dumpsys package "$package_name")"
grep -q 'versionName=' <<< "$package_dump" || {
  echo "$package_name is not installed" >&2
  exit 2
}

mkdir -p "$output_dir/runs"
printf 'run,cold_ms,warm_mean_ms,warm_p95_ms,baseline_preprocess_mean_ms,fused_preprocess_mean_ms,preprocess_speedup_x,preprocess_max_abs_drift,output_max_abs_drift,launch_total_ms,total_pss_kib,total_rss_kib\n' \
  > "$output_dir/latency-memory.csv"

{
  echo "captured_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "manufacturer=$(adb shell getprop ro.product.manufacturer | tr -d '\r')"
  echo "model=$(adb shell getprop ro.product.model | tr -d '\r')"
  echo "android=$(adb shell getprop ro.build.version.release | tr -d '\r')"
  echo "sdk=$(adb shell getprop ro.build.version.sdk | tr -d '\r')"
  echo "abi=$(adb shell getprop ro.product.cpu.abi | tr -d '\r')"
  echo "page_size=$(adb shell getconf PAGE_SIZE | tr -d '\r')"
  adb shell dumpsys package "$package_name" | grep -E 'versionCode|versionName' | tr -d '\r'
} > "$output_dir/device-report.txt"

adb shell dumpsys thermalservice > "$output_dir/thermal-before.txt"

for ((run = 1; run <= repeats; run++)); do
  run_dir="$output_dir/runs/run-$(printf '%03d' "$run")"
  mkdir -p "$run_dir"
  adb logcat -c
  adb shell am force-stop "$package_name"
  adb shell am start -W -n "$activity_name" --ez auto_run true > "$run_dir/activity-start.txt"

  completed=false
  for _ in {1..80}; do
    log_snapshot="$(adb logcat -d -s EdgeGenBench)"
    if grep -q 'benchmark_json=' <<< "$log_snapshot"; then
      completed=true
      break
    fi
    sleep 0.25
  done
  [[ "$completed" == true ]] || { echo "run $run timed out" >&2; exit 1; }

  adb logcat -d -v threadtime -s EdgeGenBench > "$run_dir/logcat.txt"
  adb shell dumpsys meminfo "$package_name" > "$run_dir/meminfo.txt"
  adb shell dumpsys thermalservice > "$run_dir/thermal.txt"

  # Match the stable machine-readable backend line emitted by v0.1.7+.  The
  # older UI label included "(NOT QNN)", but the structured log deliberately
  # reports only the backend identifier.
  grep -Eq 'EdgeGenBench: backend=reference\r?$' "$run_dir/logcat.txt"
  grep -q 'CPU fallback=false' "$run_dir/logcat.txt"
  grep -q 'power=not measured' "$run_dir/logcat.txt"
  grep -q 'benchmark_json=' "$run_dir/logcat.txt"
  grep -q 'baseline_preprocess_mean_ms=' "$run_dir/logcat.txt"
  grep -q 'fused_preprocess_mean_ms=' "$run_dir/logcat.txt"
  cold_ms="$(awk -F'cold_ms=' 'NF > 1 {print $2; exit}' "$run_dir/logcat.txt" | tr -d '\r')"
  warm_mean_ms="$(awk -F'warm_mean_ms=' 'NF > 1 {print $2; exit}' "$run_dir/logcat.txt" | tr -d '\r')"
  warm_p95_ms="$(awk -F'warm_p95_ms=' 'NF > 1 {print $2; exit}' "$run_dir/logcat.txt" | tr -d '\r')"
  baseline_preprocess_mean_ms="$(awk -F'baseline_preprocess_mean_ms=' 'NF > 1 {print $2; exit}' "$run_dir/logcat.txt" | tr -d '\r')"
  fused_preprocess_mean_ms="$(awk -F'fused_preprocess_mean_ms=' 'NF > 1 {print $2; exit}' "$run_dir/logcat.txt" | tr -d '\r')"
  preprocess_speedup_x="$(awk -F'preprocess_speedup_x=' 'NF > 1 {print $2; exit}' "$run_dir/logcat.txt" | tr -d '\r')"
  preprocess_max_abs_drift="$(awk -F'preprocess_max_abs_drift=' 'NF > 1 {print $2; exit}' "$run_dir/logcat.txt" | tr -d '\r')"
  output_max_abs_drift="$(awk -F'output_max_abs_drift=' 'NF > 1 {print $2; exit}' "$run_dir/logcat.txt" | tr -d '\r')"
  launch_total_ms="$(sed -n 's/^TotalTime: //p' "$run_dir/activity-start.txt" | tr -d '\r')"
  total_pss_kib="$(awk '/TOTAL PSS:/ {print $3; exit}' "$run_dir/meminfo.txt")"
  total_rss_kib="$(awk '/TOTAL RSS:/ {print $6; exit}' "$run_dir/meminfo.txt")"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$run" "$cold_ms" "$warm_mean_ms" "$warm_p95_ms" \
    "$baseline_preprocess_mean_ms" "$fused_preprocess_mean_ms" \
    "$preprocess_speedup_x" "$preprocess_max_abs_drift" "$output_max_abs_drift" \
    "$launch_total_ms" "$total_pss_kib" "$total_rss_kib" \
    >> "$output_dir/latency-memory.csv"
done

adb exec-out screencap -p > "$output_dir/final-screen.png"
adb shell dumpsys thermalservice > "$output_dir/thermal-after.txt"
cat > "$output_dir/measurement-status.txt" <<'EOF'
backend=reference
qnn-npu-placement=not-tested
power=not-measured
memory=post-run-snapshots-not-sampled-peaks
thermal=platform-sensor-snapshots-not-calibrated-power
EOF

echo "Repeated reference evidence written to $output_dir"
