#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 APP.apk ANDROID_NDK_ROOT" >&2
  exit 2
fi

apk_path="$1"
ndk_root="$2"
zipalign_bin="${ANDROID_SDK_ROOT:?ANDROID_SDK_ROOT is required}/build-tools/35.0.0/zipalign"
readelf_bin="$ndk_root/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf"

[[ -f "$apk_path" ]] || { echo "APK not found: $apk_path" >&2; exit 2; }
[[ -x "$zipalign_bin" ]] || { echo "zipalign not found: $zipalign_bin" >&2; exit 2; }
[[ -x "$readelf_bin" ]] || { echo "llvm-readelf not found: $readelf_bin" >&2; exit 2; }

"$zipalign_bin" -c -P 16 -v 4 "$apk_path"

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT
unzip -q "$apk_path" 'lib/*/*.so' -d "$work_dir"

found=0
while IFS= read -r library; do
  found=1
  echo "Checking $library"
  while IFS= read -r alignment; do
    if (( alignment < 0x4000 )); then
      echo "ELF LOAD alignment is below 16 KiB: $library ($alignment)" >&2
      exit 1
    fi
  done < <("$readelf_bin" -lW "$library" | awk '$1 == "LOAD" {print $NF}')
done < <(find "$work_dir/lib" -type f -name '*.so' -print | sort)

[[ "$found" == 1 ]] || { echo "No native libraries found in APK" >&2; exit 1; }
echo "APK ZIP and ELF 16 KiB alignment checks passed"
