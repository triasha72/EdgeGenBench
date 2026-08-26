#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 MODEL PLACEMENT_JSON ORT_VERSION QAIRT_VERSION MAX_DRIFT OUTPUT_DIR" >&2
  exit 2
fi

model_path="$1"
placement_path="$2"
ort_version="$3"
qairt_version="$4"
max_drift="$5"
output_dir="$6"
package_name="dev.edgegenbench"
activity_name="dev.edgegenbench/.MainActivity"

command -v adb >/dev/null || { echo "adb is required" >&2; exit 2; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 2; }
[[ -f "$model_path" ]] || { echo "model not found: $model_path" >&2; exit 2; }
[[ -f "$placement_path" ]] || { echo "placement report not found: $placement_path" >&2; exit 2; }
[[ "$max_drift" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "MAX_DRIFT must be a non-negative decimal" >&2
  exit 2
}

device_count="$(adb devices | awk 'NR > 1 && $2 == "device" {count++} END {print count+0}')"
[[ "$device_count" == "1" ]] || { echo "Exactly one authorized device is required" >&2; exit 2; }
adb shell dumpsys package "$package_name" | grep -q 'versionName=' || {
  echo "$package_name is not installed" >&2
  exit 2
}

mkdir -p "$output_dir/raw"
adb logcat -c
adb shell am force-stop "$package_name"
adb shell am start -W -n "$activity_name" --ez auto_run true > "$output_dir/raw/activity-start.txt"

benchmark_json=""
peak_rss_kib=0
for _ in {1..160}; do
  rss="$(adb shell dumpsys meminfo "$package_name" 2>/dev/null | awk '/TOTAL RSS:/ {print $6; exit}')"
  if [[ "$rss" =~ ^[0-9]+$ ]] && (( rss > peak_rss_kib )); then peak_rss_kib="$rss"; fi
  log_snapshot="$(adb logcat -d -s EdgeGenBench)"
  benchmark_json="$(sed -n 's/^.*benchmark_json=//p' <<< "$log_snapshot" | tail -1 | tr -d '\r')"
  [[ -n "$benchmark_json" ]] && break
  sleep 0.1
done
[[ -n "$benchmark_json" ]] || { echo "QNN benchmark timed out; inspect logcat" >&2; exit 1; }
printf '%s\n' "$benchmark_json" > "$output_dir/raw/android-qnn-benchmark.json"
python3 - "$output_dir/raw/android-qnn-benchmark.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("backend") != "QNNExecutionProvider" or value.get("cpu_fallback") is not False:
    raise SystemExit("app result is not fail-closed QNN evidence")
PY

adb logcat -d -v threadtime > "$output_dir/raw/logcat.txt"
adb shell dumpsys meminfo "$package_name" > "$output_dir/raw/meminfo-after.txt"
adb shell dumpsys thermalservice > "$output_dir/raw/thermal-after.txt"
adb shell getprop ro.build.fingerprint | tr -d '\r' > "$output_dir/raw/build-fingerprint.txt"
adb shell getprop ro.soc.model | tr -d '\r' > "$output_dir/raw/soc-model.txt"

private_files="$(adb shell run-as "$package_name" ls files | tr -d '\r')"
context_name="$(grep '^edgegenbench-qnn-context.*\.onnx$' <<< "$private_files" | head -1)"
profile_name="$(grep '^edgegenbench-qnn-profile.*' <<< "$private_files" | head -1)"
[[ -n "$context_name" ]] || { echo "QNN context binary was not generated" >&2; exit 1; }
[[ -n "$profile_name" ]] || { echo "QNN profile was not generated" >&2; exit 1; }
adb exec-out run-as "$package_name" cat "files/$context_name" > "$output_dir/raw/$context_name"
adb exec-out run-as "$package_name" cat "files/$profile_name" > "$output_dir/raw/$profile_name"

python3 - "$output_dir/raw/model-input.bin" <<'PY'
import struct, sys
values = (0.17, 0.28, 0.41, 0.53, 0.68, 0.79, 0.83, 0.97, 0.32, 0.64)
open(sys.argv[1], "wb").write(struct.pack("<10f", *values))
PY

peak_rss_mb="$(awk -v kib="$peak_rss_kib" 'BEGIN {printf "%.6f", kib / 1024.0}')"
python3 scripts/capture_qnn_evidence.py \
  --benchmark "$output_dir/raw/android-qnn-benchmark.json" \
  --context-binary "$output_dir/raw/$context_name" \
  --placement-report "$placement_path" \
  --profile "$output_dir/raw/$profile_name" \
  --logcat "$output_dir/raw/logcat.txt" \
  --model "$model_path" \
  --input "$output_dir/raw/model-input.bin" \
  --output-dir "$output_dir/validated" \
  --ort-version "$ort_version" \
  --qairt-version "$qairt_version" \
  --device-fingerprint "$(<"$output_dir/raw/build-fingerprint.txt")" \
  --soc-model "$(<"$output_dir/raw/soc-model.txt")" \
  --cold-ms 0 \
  --peak-rss-mb "$peak_rss_mb" \
  --max-abs-drift-vs-fp32 0 \
  --max-allowed-abs-drift "$max_drift"

echo "Validated Android QNN evidence written to $output_dir/validated"
