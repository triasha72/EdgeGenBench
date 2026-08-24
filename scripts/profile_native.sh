#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build/native}"
out_dir="${2:-reports/native}"
mkdir -p "$out_dir"
cmake -S native -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" -j
ctest --test-dir "$build_dir" --output-on-failure
"$build_dir/edgegenbench_benchmark" --baseline-preprocess --runs 1000 > "$out_dir/baseline.json"
"$build_dir/edgegenbench_benchmark" --runs 1000 > "$out_dir/fused.json"
echo "Wrote $out_dir/baseline.json and $out_dir/fused.json"
