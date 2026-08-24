#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 APP_DEBUG_APK [OUTPUT_DIR]" >&2
  exit 2
fi

apk_path="$1"
output_dir="${2:-reports/device/android-$(date -u +%Y%m%dT%H%M%SZ)}"
package_name="dev.edgegenbench"
activity_name="dev.edgegenbench/.MainActivity"

[[ -f "$apk_path" ]] || { echo "APK not found: $apk_path" >&2; exit 2; }
command -v adb >/dev/null || { echo "adb is required" >&2; exit 2; }
mkdir -p "$output_dir"

device_count="$(adb devices | awk 'NR > 1 && $2 == "device" {count++} END {print count+0}')"
[[ "$device_count" == "1" ]] || { echo "Exactly one authorized device is required" >&2; exit 2; }

shasum -a 256 "$apk_path" > "$output_dir/apk-sha256.txt"
adb shell getprop ro.build.fingerprint > "$output_dir/build-fingerprint.txt"
adb shell getprop ro.soc.model > "$output_dir/soc-model.txt"
adb shell getprop ro.product.model > "$output_dir/device-model.txt"
adb shell dumpsys battery > "$output_dir/battery-before.txt"
adb install -r "$apk_path" | tee "$output_dir/install.txt"
adb logcat -c
adb shell am force-stop "$package_name"
adb shell am start -W "$activity_name" | tee "$output_dir/activity-start.txt"
sleep 5
adb shell dumpsys meminfo "$package_name" > "$output_dir/meminfo.txt"
adb shell dumpsys battery > "$output_dir/battery-after.txt"
adb logcat -d -v threadtime -s EdgeGenBench > "$output_dir/logcat.txt"

cat > "$output_dir/measurement-status.txt" <<'EOF'
backend=reference-unless-the-app-log-explicitly-says-QNNExecutionProvider
power=not-measured
thermal=battery-temperature-snapshot-only
exclusive-npu-placement=not-established-by-this-script
EOF

echo "Evidence written to $output_dir"
